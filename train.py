import os
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import logging
import math
import sys
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from timm.utils import ModelEmaV2

from models.WMDC import WMDC

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================================================
#  UTILITIES
# =========================================================

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

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse) if mse > 0 else 100

# =========================================================
#  LOSS & OPTIMIZER
# =========================================================

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, type="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                             for likelihoods in output["likelihoods"].values())
        
        if self.type == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
            out["psnr"] = -10 * math.log10(out["mse_loss"].item()) if out["mse_loss"].item() > 0 else 100
        else:
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]
        return out

def configure_optimizers(net, args):
    parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=args.learning_rate)
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=args.aux_learning_rate)
    return optimizer, aux_optimizer

# =========================================================
#  TRAINING & TESTING LOGIC
# =========================================================

def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, 
                    clip_max_norm, accelerator, writer, ema_model, global_step):
    model.train()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    
    loop = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)

    for i, d in enumerate(loop):
        global_step += 1

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)

        unwrapped = accelerator.unwrap_model(model)
        aux_loss = unwrapped.aux_loss()

        if torch.isnan(out_criterion["loss"]) or torch.isnan(aux_loss):
            logging.warning(f"NaN loss detected at step {global_step}. Skipping step.")
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            continue

        total_loss = out_criterion["loss"] + aux_loss
        accelerator.backward(total_loss)

        if clip_max_norm > 0:
            main_params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
            accelerator.clip_grad_norm_(main_params, clip_max_norm)
        
        optimizer.step()
        aux_optimizer.step()

        if ema_model is not None:
            ema_model.update(unwrapped)

        batch_size = d.size(0)
        loss_meter.update(out_criterion["loss"].item(), batch_size)
        bpp_meter.update(out_criterion["bpp_loss"].item(), batch_size)
        if "mse_loss" in out_criterion:
            mse_meter.update(out_criterion["mse_loss"].item(), batch_size)

        if accelerator.is_main_process:
            if i % 10 == 0:
                writer.add_scalar("Train/Running_Loss", out_criterion["loss"].item(), global_step)
                writer.add_scalar("Train/Running_Bpp", out_criterion["bpp_loss"].item(), global_step)

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=f"{loss_meter.avg:.3f}", bpp=f"{bpp_meter.avg:.3f}")

    if accelerator.is_main_process:
        writer.add_scalar("Train/Epoch_Loss", loss_meter.avg, epoch)
        writer.add_scalar("Train/Epoch_Bpp", bpp_meter.avg, epoch)

    return global_step

def test_epoch(epoch, test_dataloader, model, criterion, accelerator, writer):
    model.eval()
    p = 128
    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    loss_meter = AverageMeter()

    # STABILITY FIX: Clear residual memory from training phase
    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d_padded, padding = pad(d, p)
            
            out_net = model(d_padded)
            out_net["x_hat"].clamp_(0, 1)
            out_net["x_hat"] = crop(out_net["x_hat"], padding)

            out_criterion = criterion(out_net, d)
            psnr_val = compute_psnr(d, out_net["x_hat"])
            bpp_val = out_criterion["bpp_loss"].item()
            loss_val = out_criterion["loss"].item()

            metrics = torch.tensor([psnr_val, bpp_val, loss_val], device=accelerator.device)
            metrics = metrics.expand(d.size(0), -1) 
            metrics = accelerator.gather_for_metrics(metrics)
            
            psnr_meter.update(metrics[:, 0].mean().item(), metrics.size(0))
            bpp_meter.update(metrics[:, 1].mean().item(), metrics.size(0))
            loss_meter.update(metrics[:, 2].mean().item(), metrics.size(0))

            # Logging Images
            if i == 0 and accelerator.is_main_process:
                n = min(d.size(0), 4) # Limit to 4 images max to save memory
                comparison = torch.cat([d[:n].cpu(), out_net["x_hat"][:n].cpu()], dim=0)
                grid = make_grid(comparison, nrow=n)
                writer.add_image("Test/Reconstruction", grid, epoch)
                del comparison, grid # Free CPU memory

            # STABILITY FIX: Aggressive VRAM cleanup per validation image
            del d, d_padded, out_net, out_criterion
            
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    if accelerator.is_main_process:
        logging.info(f"Test Epoch {epoch}: Loss {loss_meter.avg:.4f} | PSNR {psnr_meter.avg:.2f} | Bpp {bpp_meter.avg:.4f}")
        writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
        writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
        writer.add_scalar("Val/Bpp", bpp_meter.avg, epoch)
        
        return loss_meter.avg
    
    return float('inf')

# =========================================================
#  MAIN SCRIPT
# =========================================================

def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_file_handler = logging.FileHandler(log_dir, encoding="utf-8")
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="WMDC Training")
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=4)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.0018)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--type", type=str, default="mse", choices=["mse", "ms-ssim"])
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--lr_epoch", nargs="+", type=int, default=[35, 45])
    parser.add_argument("--continue_train", action="store_true", default=False)
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    accelerator = Accelerator()
    set_seed(args.seed)

    save_path = os.path.join(args.save_path, str(args.lmbda))
    writer = None
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        setup_logger(os.path.join(save_path, f"{time.strftime('%Y%m%d_%H%M%S')}.log"))
        writer = SummaryWriter(os.path.join(save_path, "tensorboard"))

    # Model + EMA Strategy
    net = WMDC(N=args.N, M=args.M, num_slices=5).to(accelerator.device)
    ema_model = ModelEmaV2(net, decay=0.999)
    ema_model.module.to(accelerator.device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_epoch, gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, type=args.type)

    # Dataset
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    net, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        net, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    last_epoch = 0
    best_loss = float("inf")
    global_step = 0

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        accelerator.unwrap_model(net).load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            
            if "aux_optimizer" in checkpoint:
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            if "ema_state_dict" in checkpoint:
                ema_model.module.load_state_dict(checkpoint["ema_state_dict"])
                
            global_step = last_epoch * len(train_dataloader)

    for epoch in range(last_epoch, args.epochs):
        global_step = train_one_epoch(
            net, criterion, train_dataloader, optimizer, aux_optimizer,
            epoch, args.clip_max_norm, accelerator, writer, ema_model, global_step
        )

        loss = test_epoch(epoch, test_dataloader, net, criterion, accelerator, writer)
        lr_scheduler.step()

        if accelerator.is_main_process:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            
            state = {
                "epoch": epoch,
                "state_dict": accelerator.unwrap_model(net).state_dict(),
                "ema_state_dict": ema_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "loss": loss,
            }
            torch.save(state, os.path.join(save_path, "checkpoint_latest.pth.tar"))
            if is_best:
                torch.save(state, os.path.join(save_path, "checkpoint_best.pth.tar"))

    if writer: writer.close()
    
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main(sys.argv[1:])