import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

class Thermal3DDataset(Dataset):
    def __init__(self, data_root, mode='train', split_ratio=0.7, img_size=(8, 128, 2000)):
        """
        Args:
            data_root (str): Path to the dataset folder.
            mode (str): 'train' or 'test'.
            split_ratio (float): Train split.
        """
        self.data_root = data_root
        self.mode = mode
        self.img_size = img_size # (D, H, W)
        
        # Paths
        self.profile_dir = os.path.join(data_root, 'input', 'profile')
        self.reading_dir = os.path.join(data_root, 'input', 'reading')
        self.field_dir = os.path.join(data_root, 'gt', 'field')
        
        # Sample List
        self.files = [f for f in os.listdir(self.profile_dir) if f.endswith('.csv')]
        # Sort by ID
        self.files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # Split
        split_idx = int(len(self.files) * split_ratio)
        if mode == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]
            
        print(f"[{mode.upper()}] Loaded {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        sample_id = file_name.split('_')[1].split('.')[0]
        
        # 1. Load Profile (Dynamics)
        # Format: (T, 2)
        profile_path = os.path.join(self.profile_dir, file_name)
        df_profile = pd.read_csv(profile_path, comment='#')
        
        # Standardize (RPM ~3000, Pos ~1) - Adjust based on data analysis if needed
        rpm = df_profile['RPM'].values / 3000.0
        pos = df_profile['Position'].values / 1.0 # Assuming max pos is around 1m
        dynamics = np.stack([rpm, pos], axis=1).astype(np.float32) # (2000, 2)
        
        # 2. Load Sparse Sensor Reading
        # Format: (T, 100) -> Transpose to (100, T) to match Image (Space, Time)
        reading_path = os.path.join(self.reading_dir, file_name)
        reading_data = np.loadtxt(reading_path, delimiter=',').astype(np.float32) # (2000, 100)
        reading_data = reading_data.T # (100, 2000)
        
        # Normalize Reading (Approx 15-60 deg C range based on typical data)
        # Map 15->-1, 65->1 (Same as DHc)
        reading_norm = (reading_data - 40.0) / 25.0
        # reading_norm = (reading_data - 27.5) / 7.5
        # 3. Load GT Field (Dense Target)
        # Try NPY first, else PNG
        # Note: Previous check showed PNGs.
        field_path_png = os.path.join(self.field_dir, f"sample_{sample_id}.png")
        
        if os.path.exists(field_path_png):
            # Load as HSV to recover temperature from Jet Colormap
            # Jet: Blue (Cold) -> Cyan -> Green -> Yellow -> Red (Hot)
            # Hue: 240 deg (~170) -> ... -> 0 deg (0)
            # So Temp is inversely proportional to Hue.
            
            gt_img = Image.open(field_path_png).convert('HSV')
            gt_hsv = np.array(gt_img).astype(np.float32)
            gt_h = gt_hsv[:, :, 0] # Hue channel (0..255)
            
            # Map Hue to [-1, 1]
            # Assumed Range: Blue (170) to Red (0)
            # x=170 -> y=-1 (Cold)
            # x=0   -> y=1  (Hot)
            # y = 1 - (x / 85.0)
            gt_norm = 1.0 - (gt_h / 85.0)
            
            # Clip to [-1, 1] just in case
            gt_norm = np.clip(gt_norm, -1.0, 1.0)
        else:
            # Fallback zero
            gt_norm = np.zeros((100, 2000), dtype=np.float32)

        # 4. Prepare Tensors & Resize to 3D Block (D, H, W)
        # H=100 -> Resize to target H (e.g. 128)
        # W=2000 -> Keep or Resize (2000 is div by 8)
        
        target_D, target_H, target_W = self.img_size
        
        # Resize helper
        def resize_2d(arr):
            # Input (H_in, W_in) -> (1, 1, H_in, W_in) for interpolate
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(target_H, target_W), mode='bilinear', align_corners=True)
            return t.squeeze(0) # (1, H, W)
            
        x_sparse_2d = resize_2d(reading_norm) # (1, 128, 2000)
        x_raw_2d = resize_2d(reading_data)    # (1, 128, 2000)
        gt_2d = resize_2d(gt_norm)            # (1, 128, 2000)
        
        # Expand to pseudo-3D (Repeat D times)
        # Shape: (1, D, H, W)
        x_sparse_3d = x_sparse_2d.expand(1, target_D, target_H, target_W).clone()
        x_raw_3d = x_raw_2d.expand(1, target_D, target_H, target_W).clone()
        gt_3d = gt_2d.expand(1, target_D, target_H, target_W).clone()
        
        # Capture raw sensors (100, T) for PIKAN
        x_sensors = torch.from_numpy(reading_norm)
        
        # Dynamics: (T, 2). If W changed, we must resample dynamics
        dyn_tensor = torch.from_numpy(dynamics).unsqueeze(0).permute(0, 2, 1) # (1, 2, T)
        if dynamics.shape[0] != target_W:
            dyn_tensor = F.interpolate(dyn_tensor, size=target_W, mode='linear', align_corners=True)
        
        dyn_tensor = dyn_tensor.squeeze(0).permute(1, 0) # (W, 2)
        
        return {
            'x_sparse': x_sparse_3d, # (1, D, H, W)
            'x_raw': x_raw_3d,       # (1, D, H, W) Unnormalized input for visualization
            'x_sensors': x_sensors,  # (100, T) Raw normalized sensors for PIKAN
            'dynamics': dyn_tensor,  # (W, 2) aka (T, 2)
            'gt': gt_3d              # (1, D, H, W)
        }
