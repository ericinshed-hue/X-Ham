import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from fetch_dataset import Thermal3DDataset
from model import X_HAM

CONFIG = {
    # Data
    "data_root": str(SCRIPT_DIR / "Data" / "dataset1"),
    "img_size": (8, 128, 2000),
    
    # Training
    "epochs": 200,
    "batch_size": 4,  
    "seq_len": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "num_workers": 0,
    "dt": 0.05,
    
    # Model
    "latent_dim": 32,

    # Loss weights
    "loss_l1_weight": 1.0,
    "loss_grad_weight": 0.1,  # Gradient loss for spatial smoothness
    
    # Paths
    "save_dir": str(SCRIPT_DIR / "Trained"),
    "log_dir": str(SCRIPT_DIR / "logs"),
}

class GradientLoss(nn.Module):
    """Penalize unrealistic spatial gradients"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Spatial gradients
        pred_dx = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        pred_dy = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        target_dx = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        target_dy = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        
        loss_dx = torch.abs(pred_dx - target_dx).mean()
        loss_dy = torch.abs(pred_dy - target_dy).mean()
        
        return loss_dx + loss_dy

def setup_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"train_residual_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)

def sample_chunk(item, seq_len: int, device: torch.device):
    "Sample a random chunk from full sequence"
    gt = item["gt"]          # (1, D, H, W)
    dynamics = item["dynamics"]  # (W, 2)
    
    W = gt.shape[-1]
    max_start = W - seq_len - 1
    start = np.random.randint(0, max(1, max_start + 1))
    
    # Initial condition
    x0 = gt[..., start]  # (1, D, H)
    
    # Dynamics sequence
    u_seq = dynamics[start : start + seq_len, :]  # (seq_len, 2)
    
    # next frame
    target_seq = gt[..., start + 1 : start + seq_len + 1]  # (1, D, H, seq_len)
    target_seq = target_seq.permute(3, 0, 1, 2)  # (seq_len, 1, D, H)
    
    x_sparse = x0.unsqueeze(-1).expand(-1, -1, -1, seq_len + 1)  # (1, D, H, seq_len+1)
    x_sparse = x_sparse.unsqueeze(0)  # (1, 1, D, H, seq_len+1)
    
    return (
        x_sparse.to(device),
        u_seq.to(device),
        target_seq.to(device),
    )

def train():
    logger = setup_logger(CONFIG["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")
    
    Path(CONFIG["save_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Dataset
    logger.info("Loading dataset...")
    train_dataset = Thermal3DDataset(
        CONFIG["data_root"], mode="train", img_size=tuple(CONFIG["img_size"])
    )
    test_dataset = Thermal3DDataset(
        CONFIG["data_root"], mode="test", img_size=tuple(CONFIG["img_size"])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    logger.info(f"Batches/epoch: {len(train_loader)}")
    
    # Model
    logger.info("Creating X_HAM with Physics-Attention Residual Decoder...")
    model = X_HAM(
        img_size=(CONFIG["img_size"][0], CONFIG["img_size"][1]),
        latent_dim=CONFIG["latent_dim"],
        dynamics_dim=2,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")
    
    # Loss & Optimizer
    criterion_l1 = nn.L1Loss()
    criterion_grad = GradientLoss()
    
    optimizer = AdamW(
        model.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CONFIG["lr"],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=1000.0,
    )
    
    # Training
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting training for {CONFIG['epochs']} epochs")
    logger.info(f"Sequence length: {CONFIG['seq_len']}, Batch size: {CONFIG['batch_size']}")
    logger.info(f"{'='*70}\n")
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_grad = 0.0
        n_batches = 0
        
        for batch_idx, batch_items in enumerate(train_loader):
            B = batch_items["gt"].shape[0]
            
            # Sample chunks
            x_sparse_list, u_list, tgt_list = [], [], []
            for b in range(B):
                item_b = {k: v[b] for k, v in batch_items.items()}
                x_sp, u_seq, tgt = sample_chunk(item_b, CONFIG["seq_len"], device)
                x_sparse_list.append(x_sp)
                u_list.append(u_seq)
                tgt_list.append(tgt)
            
            x_sparse = torch.cat(x_sparse_list, dim=0)  # (B, 1, D, H, T+1)
            u_seq = torch.stack(u_list, dim=0)          # (B, T, 2)
            target = torch.stack(tgt_list, dim=0)       # (B, T, 1, D, H)
            
            # Forward
            pred_seq = model(x_sparse, u_seq)  # (B, T, 1, D, H)
            
            # Loss
            loss_l1 = criterion_l1(pred_seq, target)
            loss_grad = criterion_grad(pred_seq, target)
            
            loss = (CONFIG["loss_l1_weight"] * loss_l1 + 
                   CONFIG["loss_grad_weight"] * loss_grad)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_l1 += loss_l1.item()
            epoch_grad += loss_grad.item()
            n_batches += 1
            
            # Progress
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                avg_loss = epoch_loss / n_batches
                avg_l1 = epoch_l1 / n_batches
                print(
                    f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.6f} (L1: {avg_l1:.6f})",
                    flush=True,
                )
        
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_l1 = epoch_l1 / max(n_batches, 1)
        avg_grad = epoch_grad / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        
        history["train_loss"].append(avg_loss)
        history["lr"].append(current_lr)
        
        logger.info(
            f"Epoch {epoch:03d}/{CONFIG['epochs']} | "
            f"Loss: {avg_loss:.6f} (L1: {avg_l1:.6f}, Grad: {avg_grad:.6f}) | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )
        
        # Validation
        if epoch % 10 == 0 or epoch == CONFIG["epochs"]:
            model.eval()
            val_loss = 0.0
            n_val = min(len(test_dataset), 5)
            
            with torch.no_grad():
                for vi in range(n_val):
                    item = test_dataset[vi]
                    x_sp, u_seq, tgt = sample_chunk(item, CONFIG["seq_len"], device)
                    
                    pred = model(x_sp, u_seq.unsqueeze(0))  # Add batch dim
                    val_loss += criterion_l1(pred, tgt.unsqueeze(0)).item()
            
            val_loss /= max(n_val, 1)
            history["val_loss"].append(val_loss)
            logger.info(f"  >> Validation L1: {val_loss:.6f}")
        
        # Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(CONFIG["save_dir"]) / "Marked_best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": CONFIG,
                },
                save_path,
            )
            logger.info(f"  >> Saved best model (Loss: {best_loss:.6f})")
        
        # Periodic Save
        if epoch % 50 == 0:
            ckpt_path = Path(CONFIG["save_dir"]) / f"Marked_model_ep{epoch:03d}.pth"
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), 
                 "loss": avg_loss, "config": CONFIG},
                ckpt_path,
            )
    
    # Save Final Model
    final_path = Path(CONFIG["save_dir"]) / "Marked_latest_model.pth"
    torch.save(
        {"epoch": CONFIG["epochs"], "model_state_dict": model.state_dict(),
         "loss": avg_loss, "config": CONFIG},
        final_path,
    )
    
    # Save history
    history_path = Path(CONFIG["save_dir"]) / "Marked_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training complete! Best Loss: {best_loss:.6f}")
    logger.info(f"Checkpoints: {CONFIG['save_dir']}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    train()
