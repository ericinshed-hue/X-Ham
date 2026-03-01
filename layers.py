import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CCDPA(nn.Module):
    """
    Asymmetric Cross-Channel Dot-Product Attention (CCDPA).
    """
    def __init__(self, in_channels, reduction=8):
        super(CCDPA, self).__init__()
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv   = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_temp, x_dyn):
        """
        x_temp: (B, C, D, H, W) - Thermal features
        x_dyn: (B, C, D, H, W) - Dynamics features
        """
        m_batchsize, C, D, H, W = x_temp.size()
        
        # Projections
        proj_query = self.query_conv(x_temp).view(m_batchsize, -1, D*H*W) # (B, C', N)
        proj_key   = self.key_conv(x_dyn).view(m_batchsize, -1, D*H*W)    # (B, C', N)
        
        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1))
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x_dyn).view(m_batchsize, -1, D*H*W) # (B, C, N)
        
        Q = self.query_conv(x_temp).view(m_batchsize, -1, D*H*W) # (B, C_in//r, N)
        K = self.key_conv(x_dyn).view(m_batchsize, -1, D*H*W)   # (B, C_in//r, N)
        V = self.value_conv(x_dyn).view(m_batchsize, -1, D*H*W) # (B, C_in, N)
        
        return self._forward_no_reduction(x_temp, x_dyn)

    def _forward_no_reduction(self, x_temp, x_dyn):
        m_batchsize, C, D, H, W = x_temp.size()
        
        # Q: (B, C, N)
        proj_query = x_temp.view(m_batchsize, C, -1)
        # K: (B, C, N)
        proj_key   = x_dyn.view(m_batchsize, C, -1)
        # V: (B, C, N)
        proj_value = x_dyn.view(m_batchsize, C, -1)
        
        # (B, C, N) x (B, N, C) -> (B, C, C)
        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1))
        energy = energy / (D*H*W)**0.5
        
        attention = F.softmax(energy, dim=-1) # (B, C, C)
        
        # (B, C, C) x (B, C, N) -> (B, C, N)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, D, H, W)
        
        return self.gamma * out + x_temp # Residual connection to Thermal stream

class SelfAttentionBlock(nn.Module):
    """ Simple spatial self-attention for a 2D plane (H, W) """
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # x: (B', C, H, W)
        B, C, H, W = x.size()
        N = H * W
        
        proj_q = self.query(x).view(B, -1, N).permute(0, 2, 1) # B, N, C'
        proj_k = self.key(x).view(B, -1, N) # B, C', N
        
        energy = torch.bmm(proj_q, proj_k) # B, N, N
        attn = F.softmax(energy, dim=-1)
        
        proj_v = self.value(x).view(B, -1, N) # B, C, N
        
        out = torch.bmm(proj_v, attn.permute(0, 2, 1)) # B, C, N
        out = out.view(B, C, H, W)
        
        return self.gamma * self.out_conv(out) + x
