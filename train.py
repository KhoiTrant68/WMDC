import argparse
import math
import random
import sys
import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Accelerate Imports
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# Import CompressAI datasets/utils
from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim

# Import your model
from models.WMDC import WMDC

# =========================================================
#  CONFIGURATION & UTILS
# =========================================================

def setup_logger(log_dir):
    # Only setup logger on main process
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pad_image(x, p=128):
    """Pads image to be divisible by p (required for Window Attention/Mamba)"""
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    if padding_left == 0 and padding_right == 0 and padding_top == 0 and padding_bottom == 0:
        return x, (0,0,0,0)
    x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop_image(x, padding):
    """Crops back to original size"""
    if sum(padding) == 0:
        return x
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))

def compute_bpp(out_net, num_pixels):
    """Calculates Bits Per Pixel based on likelihoods"""
    bpp_loss = sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                   for likelihoods in out_net["likelihoods"].values())
    return bpp_loss

# =========================================================
#  LOSS FUNCTION
# =========================================================

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, metric="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metric = metric

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # 1. Rate (Bpp)
        out["bpp_loss"] = compute_bpp(output, num_pixels)
        
        # 2. Distortion
        if self.metric == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]
        elif self.metric == "ms-ssim":
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            distortion = out["ms_ssim_loss"]
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # 3. Total Loss
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        
        return out

# =========================================================
#  OPTIMIZER SETUP
# =========================================================

def configure_optimizers(net, args):
    """
    Separates parameters into:
    1. Main parameters (weights, biases) -> optimizer
    2. Aux parameters (entropy bottleneck quantiles) -> aux_optimizer
    """
    parameters = {
        n: p for n, p in net.named_parameters() 
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n: p for n, p in net.named_parameters() 
        if n.endswith(".quantiles") and p.requires_grad
    }

    optimizer = optim.Adam(parameters.values(), lr=args.learning_rate)
    aux_optimizer = optim.Adam(aux_parameters.values(), lr=args.aux_learning_rate)

    return optimizer, aux_optimizer

# =========================================================
#  TRAINING LOOP
# =========================================================

def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger, writer, accelerator):
    model.train()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    
    # Only show progress bar on main process
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
    
    for i, d in pbar:
        # NOTE: No d.to(device) needed, Accelerator handles it
        
        # --- 1. Main Optimizer Step ---
        optimizer.zero_grad()
        out_net = model(d)
        
        out_criterion = criterion(out_net, d)
        
        # Accelerate backward
        accelerator.backward(out_criterion["loss"])
        
        if clip_max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)
            
        optimizer.step()

        # --- 2. Aux Optimizer Step (Entropy Bottleneck) ---
        # We need to unwrap the model to access .aux_loss() because DDP wraps it
        unwrapped_model = accelerator.unwrap_model(model)
        aux_loss = unwrapped_model.aux_loss()
            
        aux_optimizer.zero_grad()
        accelerator.backward(aux_loss)
        aux_optimizer.step()

        # --- 3. Logging ---
        # Gather metrics across GPUs for accurate logging (optional but good for debugging)
        # For speed, we often just log rank 0, but gathering ensures we see global stats
        loss_val = out_criterion["loss"].item()
        bpp_val = out_criterion["bpp_loss"].item()
        
        loss_meter.update(loss_val)
        bpp_meter.update(bpp_val)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", bpp=f"{bpp_meter.avg:.4f}")
        
        if accelerator.is_main_process:
            step = epoch * len(train_dataloader) + i
            if i % 100 == 0 and writer is not None:
                writer.add_scalar("Train/Total_Loss", loss_val, step)
                writer.add_scalar("Train/Bpp", bpp_val, step)

    return loss_meter.avg

def test_epoch(epoch, test_dataloader, model, criterion, logger, writer, accelerator):
    model.eval()
    
    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Test Epoch {epoch}", disable=not accelerator.is_local_main_process)
    count_images = 0
    with torch.no_grad():
        for i, d in pbar:
            d_padded, padding = pad_image(d, p=128) 
            
            out_net = model(d_padded)
            out_net["x_hat"].clamp_(0, 1)
            
            # Crop back
            x_hat = crop_image(out_net["x_hat"], padding)
            
            # Loss Calc (Calculate per-image locally first)
            num_pixels = d.size(0) * d.size(2) * d.size(3)
            bpp_val = compute_bpp(out_net, num_pixels)
            mse_val = F.mse_loss(x_hat, d)
            psnr_val = -10 * math.log10(mse_val.item()) if mse_val.item() > 0 else 100
            
            if criterion.metric == "mse":
                loss_val = criterion.lmbda * 255**2 * mse_val + bpp_val
            else:
                loss_val = criterion.lmbda * (1 - ms_ssim(x_hat, d, data_range=1.0)) + bpp_val

            metrics = torch.tensor([psnr_val, bpp_val, loss_val], device=accelerator.device)
            gathered_metrics = accelerator.gather(metrics) 
            
            if gathered_metrics.ndim == 1:
                gathered_metrics = gathered_metrics.view(-1, 3)
            
            avg_metrics = gathered_metrics.mean(dim=0)
            
            psnr_meter.update(avg_metrics[0].item())
            bpp_meter.update(avg_metrics[1].item())
            loss_meter.update(avg_metrics[2].item())
            count_images += d.size(0)
            if i == 0 and accelerator.is_main_process and writer is not None:
                comparison = torch.cat([d, x_hat], dim=0)
                grid = make_grid(comparison, nrow=d.size(0))
                writer.add_image("Test/Reconstruction", grid, epoch)

    if accelerator.is_main_process:
        total_saw = count_images * accelerator.num_processes
        logger.info(f"[Sanity Check] Processed approximately {total_saw} images across {accelerator.num_processes} GPUs.")
        logger.info(
            f"Test Epoch {epoch}: Loss: {loss_meter.avg:.4f} | PSNR: {psnr_meter.avg:.2f}dB | Bpp: {bpp_meter.avg:.4f}"
        )
        if writer is not None:
            writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
            writer.add_scalar("Val/Bpp", bpp_meter.avg, epoch)
            writer.add_scalar("Val/Loss", loss_meter.avg, epoch)

    return loss_meter.avg

# =========================================================
#  MAIN
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="WMDC Training (Accelerate)")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--aux-lr", dest="aux_learning_rate", type=float, default=1e-3)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.0018)
    parser.add_argument("--patch-size", type=int, default=256, help="Crop size for training")
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, help="Path to resume checkpoint")
    parser.add_argument("--metric", type=str, default="mse", choices=["mse", "ms-ssim"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    return parser.parse_args()

def main():
    args = parse_args()

    # --- ACCELERATOR SETUP ---
    # find_unused_parameters=True is CRITICAL for Mamba/VSS models in DDP
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)

    # Logging setup (Rank 0 only)
    save_dir = os.path.join(args.save_path, f"lambda_{args.lmbda}_{args.metric}")
    logger = None
    writer = None
    
    if accelerator.is_main_process:
        logger = setup_logger(save_dir)
        writer = SummaryWriter(os.path.join(save_dir, "tensorboard"))
        logger.info(f"Training started with Accelerate. Config: {args}")

    # Data Loading
    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(args.patch_size), 
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="valid", transform=test_transforms)
    
    # NOTE: accelerator handles samplers/shuffling automatically
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model Initialization
    if accelerator.is_main_process:
        logger.info("Initializing Model...")
        
    model = WMDC(N=args.N, M=args.M, num_slices=5)
    
    # Optimizers
    optimizer, aux_optimizer = configure_optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90], gamma=0.1)
    
    # Prepare everything with Accelerate
    # NOTE: Do NOT wrap model in DataParallel manually! Accelerate does DDP automatically.
    model, optimizer, aux_optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, aux_optimizer, train_loader, test_loader, lr_scheduler
    )

    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.metric)

    # Resume Training
    start_epoch = 0
    best_loss = float("inf")
    
    if args.checkpoint:
        if accelerator.is_main_process:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
        # Unwrapped load for safety across different devices
        unwrapped_model = accelerator.unwrap_model(model)
        ckpt = torch.load(args.checkpoint, map_location="cpu") # Load to CPU first
        
        unwrapped_model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))

    # Training Loop
    if accelerator.is_main_process:
        logger.info("Starting Training Loop...")
        
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, aux_optimizer, 
            epoch, args.clip_max_norm, logger, writer, accelerator
        )
        
        test_loss = test_epoch(epoch, test_loader, model, criterion, logger, writer, accelerator)
        lr_scheduler.step()

        # Save Checkpoint (Rank 0 only)
        if accelerator.is_main_process:
            state = {
                "epoch": epoch,
                "state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "loss": test_loss,
                "args": args
            }
            
            torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth.tar"))
            
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(state, os.path.join(save_dir, "checkpoint_best.pth.tar"))
                logger.info("New best model saved!")

    if writer is not None:
        writer.close()
    if accelerator.is_main_process:
        logger.info("Training complete.")

if __name__ == "__main__":
    main()