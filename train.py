import argparse
import logging
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from models.WMDC import WMDC


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class AverageMeter:
    """Tracks a running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pad_image(x: torch.Tensor, p: int = 64):
    """Pad image so H and W are divisible by p (accommodates 16× CNN + wavelet downsampling)."""
    h, w = x.size(2), x.size(3)
    new_h = math.ceil(h / p) * p
    new_w = math.ceil(w / p) * p
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    if sum(padding) == 0:
        return x, padding
    return F.pad(x, padding, mode="reflect"), padding


def crop_image(x: torch.Tensor, padding: tuple) -> torch.Tensor:
    if sum(padding) == 0:
        return x
    pl, pr, pt, pb = padding
    return F.pad(x, (-pl, -pr, -pt, -pb))


def compute_bpp(out_net: dict, num_pixels: int) -> torch.Tensor:
    return sum(
        torch.log(likelihoods.float()).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["likelihoods"].values()
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss: L = λ · D + R."""

    def __init__(self, lmbda: float = 1e-2, metric: str = "mse"):
        super().__init__()
        assert metric in ("mse", "ms-ssim"), f"Unknown metric: {metric}"
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metric = metric

    def forward(self, output: dict, target: torch.Tensor) -> dict:
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}
        out["bpp_loss"] = compute_bpp(output, num_pixels)

        if self.metric == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]
        else:
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            distortion = out["ms_ssim_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        return out


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


def configure_optimizers(net: nn.Module, args: argparse.Namespace):
    """
    Separate main parameters from entropy bottleneck quantile parameters.
    The quantiles are updated by aux_optimizer only, not by the main optimizer.
    """
    main_params = {
        n: p
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_params = {
        n: p
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    optimizer = optim.Adam(main_params.values(), lr=args.learning_rate)
    aux_optimizer = optim.Adam(aux_params.values(), lr=args.aux_learning_rate)
    return optimizer, aux_optimizer


# ---------------------------------------------------------------------------
# DataLoader worker seeding
# ---------------------------------------------------------------------------


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model,
    criterion: RateDistortionLoss,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    epoch: int,
    clip_max_norm: float,
    logger: logging.Logger,
    writer: SummaryWriter,
    accelerator: Accelerator,
) -> float:
    model.train()
    rd_meter = AverageMeter()
    aux_meter = AverageMeter()
    bpp_meter = AverageMeter()

    pbar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )

    for i, d in pbar:
        # ── Forward ────────────────────────────────────────────────────────
        out_net = model(d)
        out_criterion = criterion(out_net, d)

        # ── Main (RD) backward ─────────────────────────────────────────────
        optimizer.zero_grad()
        accelerator.backward(out_criterion["loss"])
        if clip_max_norm > 0:
            main_params = [
                p for n, p in model.named_parameters() if not n.endswith(".quantiles")
            ]
            accelerator.clip_grad_norm_(main_params, clip_max_norm)
        optimizer.step()

        # ── Auxiliary (entropy bottleneck quantiles) backward ──────────────
        aux_optimizer.zero_grad()
        aux_loss = accelerator.unwrap_model(model).aux_loss()
        accelerator.backward(aux_loss)
        aux_optimizer.step()

        # ── Logging ────────────────────────────────────────────────────────
        rd_val = out_criterion["loss"].item()
        aux_val = aux_loss.item()
        bpp_val = out_criterion["bpp_loss"].item()

        rd_meter.update(rd_val)
        aux_meter.update(aux_val)
        bpp_meter.update(bpp_val)

        pbar.set_postfix(
            rd=f"{rd_meter.avg:.4f}",
            aux=f"{aux_meter.avg:.6f}",
            bpp=f"{bpp_meter.avg:.4f}",
        )

        if accelerator.is_main_process and i % 100 == 0:
            step = epoch * len(train_dataloader) + i
            if writer:
                writer.add_scalar("Train/RD_Loss", rd_val, step)
                writer.add_scalar("Train/Aux_Loss", aux_val, step)
                writer.add_scalar("Train/Bpp", bpp_val, step)
            if logger:
                logger.info(
                    f"[Train] epoch {epoch} step {i}/{len(train_dataloader)} | "
                    f"RD={rd_val:.4f} Aux={aux_val:.6f} Bpp={bpp_val:.4f}"
                )

    return rd_meter.avg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_epoch(
    epoch: int,
    test_dataloader: DataLoader,
    model,
    criterion: RateDistortionLoss,
    logger: logging.Logger,
    writer: SummaryWriter,
    accelerator: Accelerator,
) -> float:
    model.eval()
    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    loss_meter = AverageMeter()

    pbar = tqdm(
        enumerate(test_dataloader),
        total=len(test_dataloader),
        desc=f"Val Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )

    with torch.no_grad():
        for i, d in pbar:
            d_padded, padding = pad_image(d)
            out_net = model(d_padded)
            out_net["x_hat"].clamp_(0, 1)
            x_hat = crop_image(out_net["x_hat"], padding)

            num_pixels = d.size(0) * d.size(2) * d.size(3)
            bpp_val = compute_bpp(out_net, num_pixels)
            mse_val = F.mse_loss(x_hat, d)
            psnr_val = -10 * math.log10(mse_val.item()) if mse_val.item() > 0 else 100.0

            if criterion.metric == "mse":
                loss_val = criterion.lmbda * 255**2 * mse_val + bpp_val
            else:
                loss_val = (
                    criterion.lmbda * (1 - ms_ssim(x_hat, d, data_range=1.0)) + bpp_val
                )

            # Gather across GPUs before averaging
            metrics = torch.tensor(
                [psnr_val, bpp_val, loss_val], device=accelerator.device
            )
            gathered = accelerator.gather(metrics).view(-1, 3).mean(dim=0)

            psnr_meter.update(gathered[0].item())
            bpp_meter.update(gathered[1].item())
            loss_meter.update(gathered[2].item())

            if i == 0 and accelerator.is_main_process and writer:
                grid = make_grid(torch.cat([d[:4], x_hat[:4]], dim=0), nrow=4)
                writer.add_image("Val/Reconstruction", grid, epoch)

    if accelerator.is_main_process:
        msg = (
            f"[Val] epoch {epoch} | "
            f"Loss={loss_meter.avg:.4f} PSNR={psnr_meter.avg:.2f}dB Bpp={bpp_meter.avg:.4f}"
        )
        if logger:
            logger.info(msg)
        if writer:
            writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
            writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
            writer.add_scalar("Val/Bpp", bpp_meter.avg, epoch)

    return loss_meter.avg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train WMDC learned image compression model"
    )
    p.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Path to ImageFolder dataset root",
    )
    p.add_argument("--save_path", type=str, default="checkpoints")
    p.add_argument("-e", "--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", dest="learning_rate", type=float, default=1e-4)
    p.add_argument("--aux-lr", dest="aux_learning_rate", type=float, default=1e-3)
    p.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0018,
        help="Rate-distortion tradeoff λ (higher = more quality, higher bitrate)",
    )
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--clip_max_norm", type=float, default=1.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--metric", type=str, default="mse", choices=["mse", "ms-ssim"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--N", type=int, default=192)
    p.add_argument("--M", type=int, default=320)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision="no")
    set_seed(args.seed)

    save_dir = os.path.join(args.save_path, f"lambda_{args.lmbda}_{args.metric}")
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir) if accelerator.is_main_process else None
    writer = (
        SummaryWriter(os.path.join(save_dir, "tensorboard"))
        if accelerator.is_main_process
        else None
    )

    if accelerator.is_main_process:
        logger.info(f"Training config: {vars(args)}")

    # ── Datasets ────────────────────────────────────────────────────────────
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(args.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.patch_size),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="valid", transform=test_transforms)

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=rng,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Model & optimizers ──────────────────────────────────────────────────
    model = WMDC(N=args.N, M=args.M, num_slices=5)
    optimizer, aux_optimizer = configure_optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 90], gamma=0.1
    )
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.metric)

    # Accelerate wraps model, optimizers, loaders, scheduler
    (model, optimizer, aux_optimizer, train_loader, test_loader, lr_scheduler) = (
        accelerator.prepare(
            model, optimizer, aux_optimizer, train_loader, test_loader, lr_scheduler
        )
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss = float("inf")

    if args.checkpoint:
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger,
            writer,
            accelerator,
        )
        test_loss = test_epoch(
            epoch,
            test_loader,
            model,
            criterion,
            logger,
            writer,
            accelerator,
        )
        lr_scheduler.step()

        if accelerator.is_main_process:
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)

            state = {
                "epoch": epoch,
                "state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "loss": test_loss,
                "best_loss": best_loss,
                "args": vars(args),
            }
            torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth.tar"))
            if is_best:
                torch.save(state, os.path.join(save_dir, "checkpoint_best.pth.tar"))
                logger.info(
                    f"  → New best loss: {best_loss:.4f} (saved checkpoint_best)"
                )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
