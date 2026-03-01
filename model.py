import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import CCDPA

# try:
#     from kan import KAN
#     KAN_AVAILABLE = True
# except ImportError:
#     KAN_AVAILABLE = False
#     print("Warning: pykan not installed. Install with: pip install pykan")

import torch.distributed as dist
import torch.distributed.nn as dist_nn
from einops import rearrange

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

def all_reduce_if_initialized(tensor):
    if dist.is_available() and dist.is_initialized():
        dist_nn.all_reduce(tensor, op=dist_nn.ReduceOp.SUM)
    return tensor

def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau
    
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y

class EPHS_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # B N C
        B, N, C = x.shape

        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)  # B H G
        all_reduce_if_initialized(slice_norm)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        all_reduce_if_initialized(slice_token)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

class HamMLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(HamMLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class EPHSLatentDynamics(nn.Module):
    """
    Learns EPHS dynamics in a latent space z
    """
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Learns the Hamiltonian Operator
        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Structure Matrices Parameterization
        self.J_param = nn.Sequential(nn.Linear(latent_dim, latent_dim**2), nn.Tanh())
        self.R_param = nn.Sequential(nn.Linear(latent_dim, latent_dim**2), nn.Sigmoid()) # R > 0
        
        # Input Port G(z)
        self.G_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

    def get_structure_matrices(self, z):
        batch_size = z.shape[0]
        
        J_raw = self.J_param(z).view(batch_size, self.latent_dim, self.latent_dim)
        R_raw = self.R_param(z).view(batch_size, self.latent_dim, self.latent_dim)
        
        # Enforce Skew-Symmetry for J
        J = (J_raw - J_raw.transpose(-1, -2)) / 2.0
        
        # ensuring semi-positive definiteness
        R = torch.matmul(R_raw, R_raw.transpose(-1, -2))
        
        return J, R

    def forward(self, z, u_ext=None):
        """
        Calculates dz/dt
        z: (Batch, Latent)
        u_ext: (Batch, 1) - External Input Power magnitude
        """
        batch_size = z.shape[0]
        
        with torch.enable_grad():
            z_in = z.clone().requires_grad_(True)
            E = self.H_net(z_in).sum() # Sum for batched gradient
            
            # Compute Gradient of Hamiltonian
            dH = torch.autograd.grad(E, z_in, create_graph=True)[0] # (Batch, Latent)
            dH = dH.unsqueeze(-1) # (Batch, Latent, 1)

        # Get EPHS matrices
        J, R = self.get_structure_matrices(z)
        
        # Internal Dynamics: (J - R) * dH
        # (B, L, L) @ (B, L, 1) -> (B, L, 1)
        dz_internal = torch.matmul(J - R, dH).squeeze(-1)
        
        # External Forcing: + G(z) * u
        if u_ext is not None:
            # G(z): (Batch, Latent)
            g_vec = self.G_net(z) 
            # u_ext: (Batch, 1)
            dz_external = g_vec * u_ext
        else:
            dz_external = 0.0
            
        return dz_internal + dz_external

class X_HAM(nn.Module):
    def __init__(self, img_size=(8, 128), latent_dim=32, dynamics_dim=2):
        super().__init__()
        self.img_size = img_size # (D, H) - We process 2D spatial slices
        self.latent_dim = latent_dim
        # Encoder (Spatial 2D -> Latent)
        # Input: (B, 1, D, H)
        self.encoder_backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # (4, 64)
            
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # (2, 32)
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Input: (B, DynDim, T)
        self.dyn_backbone = nn.Sequential(
            nn.Conv1d(dynamics_dim, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Temporal Pooling -> (B, 64, 1)
        )
        
        self.cmit = CCDPA(64) # Fuse Thermal (Q) and Dynamics (K, V)
        
        flat_dim = 64 * (img_size[0]//4) * (img_size[1]//4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim)
        )
        
        self.dynamics = EPHSLatentDynamics(latent_dim)
        
        self.N_spatial = self.img_size[0] * self.img_size[1]  # Number of spatial points
        hidden_attn = 128
        
        # Spatial position embeddings (fixed)
        d_coords = torch.linspace(0, 1, self.img_size[0])
        h_coords = torch.linspace(0, 1, self.img_size[1])
        grid_d, grid_h = torch.meshgrid(d_coords, h_coords, indexing="ij")
        spatial_coords = torch.stack([grid_d.flatten(), grid_h.flatten()], dim=-1)  # (N, 2)
        self.register_buffer("spatial_pos", spatial_coords)
        
        # Position embedding projection (2D coords → hidden_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )
        
        # Latent feature projection (latent_dim → spatial_feat_dim)
        self.latent_to_spatial = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )
        
        # Input embedding: combine [x_value, position, latent_feat]
        # Total input dim: 1 (field value) + 32 (pos) + 32 (latent) = 65
        self.input_projection = nn.Linear(1 + 32 + 32, hidden_attn)
        
        # EPHS-Attention layer
        self.physics_attn = EPHS_Attention(
            dim=hidden_attn,
            heads=4,
            dim_head=hidden_attn // 4,
            dropout=0.1,
            slice_num=32
        )
        
        # LayerNorm + MLP block
        self.attn_ln = nn.LayerNorm(hidden_attn)
        self.attn_mlp = HamMLP(hidden_attn, hidden_attn * 2, hidden_attn, n_layers=0, res=False, act='gelu')
        self.mlp_ln = nn.LayerNorm(hidden_attn)
        
        # Output: predict delta (residual)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_attn),
            nn.Linear(hidden_attn, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # Initialize delta head with small weights for stability
        nn.init.xavier_normal_(self.delta_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.delta_head[-1].bias)
        
        print(f"      Input: (x_prev, latent) → Delta prediction → x_next = x_prev + delta")
        

    def encode_z0(self, x_img, dynamics_seq):
        """
        Dual Encoder + Cross-Attention Fusion to predict initial latent state.
        
        x_img: (B, 1, D, H)
        dynamics_seq: (B, T, DynDim) -> Needs Permute for Conv1d
        """
        # Thermal Features
        # (B, 1, D, H) -> (B, 64, D/4, H/4)
        feat_temp = self.encoder_backbone(x_img)
        
        # Dynamics Features
        # (B, T, 2) -> (B, 2, T) for Conv1d
        dyn_in = dynamics_seq.permute(0, 2, 1)
        # (B, 2, T) -> (B, 64, 1)
        feat_dyn_global = self.dyn_backbone(dyn_in)
        
        # (B, 64, 1) -> (B, 64, D/4, H/4)
        B, C, Dp, Hp = feat_temp.shape
        feat_dyn_spatial = feat_dyn_global.unsqueeze(-1).expand(B, C, Dp, Hp)
        feat_temp_3d = feat_temp.unsqueeze(2) # (B, 64, 1, D', H')
        feat_dyn_3d = feat_dyn_spatial.unsqueeze(2)
        feat_fused_3d = self.cmit(feat_temp_3d, feat_dyn_3d)
        feat_fused = feat_fused_3d.squeeze(2)
        feat_final = feat_fused + feat_temp

        z0 = self.head(feat_final)
        return z0

    def decode_step(self, z: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """
        Decode one timestep using physics-attention residual prediction.
        
        Args:
            z: (B, latent_dim) - latent state
            x_prev: (B, 1, D, H) - previous spatial field
            
        Returns:
            x_next: (B, 1, D, H) - next spatial field
        """
        B = z.shape[0]
        D, H = self.img_size
        
        x_flat = x_prev.reshape(B, -1, 1)  # (B, N, 1)
        pos_feat = self.pos_embed(self.spatial_pos)  # (N, 32)
        pos_feat = pos_feat.unsqueeze(0).expand(B, -1, -1)  # (B, N, 32)
        
        latent_feat = self.latent_to_spatial(z)  # (B, 32)
        latent_feat = latent_feat.unsqueeze(1).expand(B, self.N_spatial, -1)  # (B, N, 32)

        combined = torch.cat([x_flat, pos_feat, latent_feat], dim=-1)  # (B, N, 1+32+32)
        
        tokens = self.input_projection(combined)
        
        tokens = self.physics_attn(self.attn_ln(tokens)) + tokens
        tokens = self.attn_mlp(self.mlp_ln(tokens)) + tokens
        
        delta_flat = self.delta_head(tokens)  # (B, N, 1)
        
        delta = delta_flat.reshape(B, 1, D, H)  # (B, 1, D, H)
        x_next = x_prev + delta
        
        return x_next

    def forward_sequence(self, x_init, u_seq, steps=10, dt=0.01, z_init=None):
        """
        Autoregressive rollout in latent space.
        z_init: Optional pre-calculated latent state
        x_init: Used if z_init is None (fallback to simple thermal encoding)
        """
        if z_init is not None:
             z_curr = z_init
        else:
             feat = self.encoder_backbone(x_init)
             z_curr = self.head(feat)
        
        x_curr = x_init  # (B, 1, D, H)
        predictions = []
        
        # Latent Integration Loop
        for t in range(steps):
             if t < u_seq.shape[1]: 
                u_t = u_seq[:, t, :] # (B, 1)
             else:
                u_t = u_seq[:, -1, :] # Repeat last if steps > input
            
             # ODE Step (Euler): z_new = z + dt * f(z, u)
             dz = self.dynamics(z_curr, u_t)
             z_curr = z_curr + dt * dz

             x_next = self.decode_step(z_curr, x_curr)
             predictions.append(x_next.unsqueeze(1))  # (B, 1, 1, D, H)
             x_curr = x_next

        
        out = torch.cat(predictions, dim=1)  # (B, Steps, 1, D, H)

        
        return out

    def forward(self, x_sparse, dynamics_seq):

        x0 = x_sparse[:, :, :, :, 0] # (B, 1, D, H)
        z0 = self.encode_z0(x0, dynamics_seq)
        u_mag = torch.norm(dynamics_seq, dim=2, keepdim=True) # (B, T, 1)
        steps = dynamics_seq.shape[1]
        return self.forward_sequence(x0, u_mag, steps=steps, z_init=z0)
