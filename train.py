import os

# STABILITY FIX: Prevent CUDA OOM Fragmentation at the system level
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import logging
import math
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

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
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse) if mse > 0 else 100


def compute_bpp(out_net):
    size = out_net["x_hat"].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["likelihoods"].values()
    ).item()


# =========================================================
#  LOGGING & CHECKPOINTING
# =========================================================


def setup_logger(log_dir):
    logger = logging.getLogger("Train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def save_checkpoint(state, is_best, save_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(save_path, filename),
            os.path.join(save_path, "checkpoint_best.pth.tar"),
        )


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
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.type == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
            out["psnr"] = (
                -10 * math.log10(out["mse_loss"].item())
                if out["mse_loss"].item() > 0
                else 100
            )
        else:
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]
        return out


def configure_optimizers(net, args):
    """Separates parameters for Main Optimizer (AdamW) and Aux Optimizer (Adam)."""
    params_dict = dict(net.named_parameters())

    params = {n: p for n, p in params_dict.items() if not n.endswith(".quantiles")}
    aux_params = {n: p for n, p in params_dict.items() if n.endswith(".quantiles")}

    # Main: AdamW (From sample code)
    optimizer = optim.AdamW(
        [p for p in params.values() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Aux: Standard Adam
    aux_optimizer = optim.Adam(
        [p for p in aux_params.values() if p.requires_grad],
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


# =========================================================
#  TRAINING & TESTING LOGIC
# =========================================================


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, 
                    clip_max_norm, accelerator, logger, writer, ema_model, global_step, args):
    model.train()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    dist_meter = AverageMeter()
    
    loop = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)

    for i, d in enumerate(loop):
        global_step += 1

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # 1. Forward Pass
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        loss = out_criterion["loss"]

        # 2. Auxiliary Loss
        unwrapped = accelerator.unwrap_model(model)
        aux_loss = unwrapped.aux_loss()

        # NaN protection
        if torch.isnan(loss) or torch.isnan(aux_loss):
            logging.warning(f"NaN loss detected at step {global_step}. Skipping step.")
            continue

        # ========================================================
        # DDP STABILITY FIX: Combine losses for a single backward pass
        # ========================================================
        total_loss = loss + aux_loss
        accelerator.backward(total_loss)

        # 3. Clip gradients for main parameters only (standard CompressAI behavior)
        if clip_max_norm > 0:
            main_params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
            accelerator.clip_grad_norm_(main_params, clip_max_norm)
        
        # 4. Step BOTH optimizers (they automatically update their respective parameters)
        optimizer.step()
        aux_optimizer.step()

        if ema_model is not None:
            ema_model.update(model)

        # 5. Logging metrics
        batch_size = d.size(0)
        loss_meter.update(loss.item(), batch_size)
        bpp_meter.update(out_criterion["bpp_loss"].item(), batch_size)
        
        dist_metric = out_criterion["mse_loss"].item() if "mse_loss" in out_criterion else out_criterion["ms_ssim_loss"].item()
        dist_meter.update(dist_metric, batch_size)

        if i % args.print_freq == 0 and accelerator.is_main_process:
            logger.info(
                f"Epoch [{epoch}][{i}/{len(train_dataloader)}] "
                f"Loss: {loss_meter.val:.4f} | Bpp: {bpp_meter.val:.4f} | Dist: {dist_meter.val:.6f}"
            )
            writer.add_scalar("Train/Loss", loss_meter.val, global_step)
            writer.add_scalar("Train/Bpp", bpp_meter.val, global_step)
            writer.add_scalar("Train/Dist", dist_meter.val, global_step)

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=f"{loss_meter.avg:.3f}", bpp=f"{bpp_meter.avg:.3f}")

    if accelerator.is_main_process:
        writer.add_scalar("Train/Epoch_Loss", loss_meter.avg, epoch)
        writer.add_scalar("Train/Epoch_Bpp", bpp_meter.avg, epoch)

    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, accelerator, logger, writer):
    model.eval()
    p = 128
    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    loss_meter = AverageMeter()

    # OOM FIX: Flush memory before evaluation
    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d_padded, padding = pad(d, p)
            out_net = model(d_padded)
            out_net["x_hat"] = out_net["x_hat"].clamp(0, 1)
            out_net["x_hat"] = crop(out_net["x_hat"], padding)

            out_criterion = criterion(out_net, d)
            psnr_val = compute_psnr(d, out_net["x_hat"])
            bpp_val = compute_bpp(out_net)
            loss_val = out_criterion["loss"].item()

            metrics = torch.tensor(
                [psnr_val, bpp_val, loss_val], device=accelerator.device
            )
            metrics = metrics.expand(d.size(0), -1)
            metrics = accelerator.gather_for_metrics(metrics)

            psnr_meter.update(metrics[:, 0].mean().item(), metrics.size(0))
            bpp_meter.update(metrics[:, 1].mean().item(), metrics.size(0))
            loss_meter.update(metrics[:, 2].mean().item(), metrics.size(0))

            if i == 0 and accelerator.is_main_process:
                n = min(d.size(0), 4)
                # OOM FIX: Move comparison to CPU to save VRAM before writing to Tensorboard
                comparison = torch.cat([d[:n].cpu(), out_net["x_hat"][:n].cpu()], dim=0)
                grid = make_grid(comparison, nrow=n)
                writer.add_image("Val/Reconstruction", grid, epoch)
                del comparison, grid

            # OOM FIX: Explicitly delete tensors to prevent caching fragmentation
            del d, d_padded, out_net, out_criterion
            if i % 5 == 0:
                torch.cuda.empty_cache()

    if accelerator.is_main_process:
        logger.info(
            f"Test Epoch [{epoch}] "
            f"Loss: {loss_meter.avg:.4f} | Bpp: {bpp_meter.avg:.4f} | PSNR: {psnr_meter.avg:.2f} dB"
        )
        writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
        writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
        writer.add_scalar("Val/Bpp", bpp_meter.avg, epoch)

        return loss_meter.avg

    return float("inf")


# =========================================================
#  MAIN SCRIPT
# =========================================================


def parse_args(argv):
    parser = argparse.ArgumentParser(description="WMDC Training")
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("-e", "--epochs", default=400, type=int)
    parser.add_argument("--batch-size", type=int, default=8)

    # Optimization
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.0018)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)

    # Data & Logging
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("-n", "--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1926)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--type", type=str, default="mse", choices=["mse", "ms-ssim"])

    # Model Architecture
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    set_seed(args.seed)

    # 1. Setup Accelerator (REMOVED find_unused_parameters=True)
    accelerator = Accelerator()

    # 2. Logging
    save_path = os.path.join(args.save_path, f"lambda_{args.lmbda}")
    logger = None
    writer = None
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        logger = setup_logger(save_path)
        writer = SummaryWriter(os.path.join(save_path, "tb"))
        logger.info(f"Training Config: {args}")

    # 3. Data Loading
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(args.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(512), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="valid", transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4. Initialize Model & EMA
    net = WMDC(N=args.N, M=args.M, num_slices=5)
    ema_model = ModelEmaV2(net, decay=0.999)

    # 5. Optimizers & Scheduler
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 6. Criterion
    criterion = RateDistortionLoss(lmbda=args.lmbda, type=args.type)

    # 7. Resume Checkpoint
    start_epoch = 0
    best_loss = float("inf")

    if args.checkpoint:
        if accelerator.is_main_process:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        net.load_state_dict(checkpoint["state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "aux_optimizer" in checkpoint:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        if "loss" in checkpoint:
            best_loss = checkpoint["loss"]
        if "ema_state_dict" in checkpoint:
            ema_model.module.load_state_dict(checkpoint["ema_state_dict"])

    # 8. Prepare via Accelerator
    net, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler = (
        accelerator.prepare(
            net,
            optimizer,
            aux_optimizer,
            train_dataloader,
            test_dataloader,
            lr_scheduler,
        )
    )

    # EMA to device (after prepare)
    ema_model.module.to(accelerator.device)

    # 9. Training Loop
    global_step = start_epoch * len(train_dataloader)

    for epoch in range(start_epoch, args.epochs):

        # Manually lower LR at 75% of epochs (Adapted from Sample Code)
        if epoch == int(args.epochs * 0.75):
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate * 0.1
            if accelerator.is_main_process:
                logger.info(">>> Switching to STE quantization phase and lowering LR")

        global_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            accelerator,
            logger,
            writer,
            ema_model,
            global_step,
            args,
        )

        loss = test_epoch(
            epoch, test_dataloader, net, criterion, accelerator, logger, writer
        )
        lr_scheduler.step()

        # 10. Checkpointing
        if accelerator.is_main_process:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": accelerator.unwrap_model(net).state_dict(),
                    "ema_state_dict": ema_model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "loss": loss,
                },
                is_best,
                save_path,
                filename="checkpoint_latest.pth.tar",
            )

            # Save EMA checkpoint
            torch.save(
                {"state_dict": ema_model.module.state_dict(), "epoch": epoch},
                os.path.join(save_path, "checkpoint_ema.pth.tar"),
            )

    if writer:
        writer.close()

    # STABILITY FIX: Clean Distributed Shutdown
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(sys.argv[1:])
