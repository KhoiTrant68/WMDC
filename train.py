import argparse
import logging
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
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
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_bpp(out_net: dict, num_pixels: int) -> torch.Tensor:
    """Differentiable training-time BPP estimate from likelihood tensors."""
    return sum(
        torch.log(lh.float()).sum() / (-math.log(2) * num_pixels)
        for lh in out_net["likelihoods"].values()
    )


def pad_to_multiple(x: torch.Tensor, p: int = 64):
    """Pad tensor to make H and W multiples of p using reflect padding."""
    H, W = x.size(2), x.size(3)
    pad_h = (p - H % p) % p
    pad_w = (p - W % p) % p
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, H, W


# ---------------------------------------------------------------------------
# Rate-Distortion Loss
# ---------------------------------------------------------------------------


class RateDistortionLoss(nn.Module):
    """
    Rate-distortion loss with adaptive dispersion regularisation.

    L = λ · distortion + bpp + w_disp · scale · dispersion
    """

    def __init__(
        self,
        lmbda: float = 1e-2,
        metric: str = "mse",
        disp_weight: float = 0.1,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metric = metric
        self.disp_weight = disp_weight
        self.register_buffer("ema_bpp", torch.tensor(1.0))
        self.register_buffer("ema_disp", torch.tensor(1.0))
        self.effective_disp_coeff: float = disp_weight

    def forward(self, output: dict, target: torch.Tensor) -> dict:
        num_pixels = target.size(0) * target.size(2) * target.size(3)
        bpp_loss = compute_bpp(output, num_pixels)

        out = {"bpp_loss": bpp_loss}

        if self.metric == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            distortion = 255.0**2 * out["mse_loss"]
        else:
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            distortion = out["ms_ssim_loss"]

        out["loss"] = self.lmbda * distortion + bpp_loss

        if "dispersion_loss" in output:
            disp = output["dispersion_loss"]
            out["dispersion_loss"] = disp

            if self.training:
                with torch.no_grad():
                    b_val = bpp_loss.abs().detach().squeeze()
                    d_val = disp.abs().detach().squeeze()
                    if dist.is_initialized():
                        dist.all_reduce(b_val, op=dist.ReduceOp.AVG)
                        dist.all_reduce(d_val, op=dist.ReduceOp.AVG)
                    self.ema_bpp.mul_(0.99).add_(b_val, alpha=0.01)
                    self.ema_disp.mul_(0.99).add_(d_val, alpha=0.01)

            scale = (self.ema_bpp / (self.ema_disp + 1e-8)).clamp(0.01, 10.0)
            effective = float(self.disp_weight * scale.item())
            self.effective_disp_coeff = effective
            out["loss"] = out["loss"] + effective * disp

        return out


# ---------------------------------------------------------------------------
# Optimiser configuration
# ---------------------------------------------------------------------------


def configure_optimizers(net: nn.Module, args):
    """Separate main params from entropy-bottleneck quantile params."""
    main_params = [
        p
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    ]
    aux_params = [
        p
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    ]
    optimizer = optim.Adam(main_params, lr=args.learning_rate)
    aux_optimizer = optim.Adam(aux_params, lr=args.aux_learning_rate)
    return optimizer, aux_optimizer


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
    logger,
    writer,
    accelerator,
):
    model.train()
    rd_meter = AverageMeter()
    aux_meter = AverageMeter()
    bpp_meter = AverageMeter()
    disp_meter = AverageMeter()

    pbar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )

    for i, d in pbar:
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)

        # ── Pass 1: RD loss (main parameters) ────────────────────────────
        accelerator.backward(out_criterion["loss"])
        if clip_max_norm > 0:
            main_params = [
                p for n, p in model.named_parameters() if not n.endswith(".quantiles")
            ]
            accelerator.clip_grad_norm_(main_params, clip_max_norm)
        optimizer.step()

        # ── Pass 2: aux loss (entropy bottleneck CDF approximation) ──────
        accelerator.backward(out_net["aux_loss"])
        aux_optimizer.step()

        rd_meter.update(out_criterion["loss"].item())
        aux_meter.update(out_net["aux_loss"].item())
        bpp_meter.update(out_criterion["bpp_loss"].item())
        if "dispersion_loss" in out_criterion:
            disp_meter.update(out_criterion["dispersion_loss"].item())

        pbar.set_postfix(
            rd=f"{rd_meter.avg:.4f}",
            aux=f"{aux_meter.avg:.5f}",
            bpp=f"{bpp_meter.avg:.4f}",
            disp=f"{disp_meter.avg:.4f}",
        )

        if accelerator.is_main_process and i % 100 == 0:
            step = epoch * len(train_dataloader) + i
            if writer:
                writer.add_scalar("Train/RD_Loss", rd_meter.avg, step)
                writer.add_scalar("Train/Aux_Loss", aux_meter.avg, step)
                writer.add_scalar("Train/Bpp", bpp_meter.avg, step)
                writer.add_scalar("Train/Dispersion_raw", disp_meter.avg, step)
                writer.add_scalar(
                    "Train/Dispersion_eff_coeff",
                    criterion.effective_disp_coeff,
                    step,
                )

    return rd_meter.avg


# ---------------------------------------------------------------------------
# Train/inference gap measurement
# ---------------------------------------------------------------------------


def measure_train_inference_gap(model, batch: torch.Tensor, device: str) -> float:
    """
    Measure PSNR gap between forward() (noise relaxation) and
    compress/decompress() (true quantisation) on the same batch.
    """
    model.eval()
    d_padded, H_orig, W_orig = pad_to_multiple(batch.to(device), p=64)

    with torch.no_grad():
        out_fwd = model(d_padded)
        x_hat_fwd = out_fwd["x_hat"][:, :, :H_orig, :W_orig].clamp(0, 1)

        mse_fwd = F.mse_loss(batch.to(device), x_hat_fwd)
        psnr_fwd = -10 * math.log10(mse_fwd.item()) if mse_fwd.item() > 0 else 100.0

        out_enc = model.compress(d_padded)
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        x_hat_dec = out_dec["x_hat"][:, :, :H_orig, :W_orig].clamp(0, 1)

        mse_dec = F.mse_loss(batch.to(device), x_hat_dec)
        psnr_dec = -10 * math.log10(mse_dec.item()) if mse_dec.item() > 0 else 100.0

    return psnr_fwd - psnr_dec


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------


def test_epoch(
    epoch,
    test_dataloader,
    model,
    criterion,
    logger,
    writer,
    accelerator,
    gap_check: bool = False,
):
    model.eval()
    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    loss_meter = AverageMeter()
    gap_psnr: float = 0.0
    gap_measured: bool = False

    with torch.no_grad():
        for i, d in tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            desc=f"Val Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
        ):
            d_padded, H_orig, W_orig = pad_to_multiple(d, p=64)
            out_net = model(d_padded)

            x_hat = out_net["x_hat"][:, :, :H_orig, :W_orig].clamp_(0, 1)
            d_orig = d[:, :, :H_orig, :W_orig]

            if i == 0 and accelerator.is_main_process and writer is not None:
                n = min(d_orig.size(0), 4)
                cmp = torch.cat([d_orig[:n], x_hat[:n]])
                grid = make_grid(cmp, nrow=n, normalize=True, value_range=(0, 1))
                writer.add_image("Val/Reconstruction", grid, epoch)

            B = d_padded.size(0)
            H_pad = d_padded.size(2)
            W_pad = d_padded.size(3)
            num_pixels_pad = B * H_pad * W_pad
            num_pixels_orig = B * H_orig * W_orig

            bpp_padded = compute_bpp(out_net, num_pixels_pad)
            bpp_val = bpp_padded * (num_pixels_pad / num_pixels_orig)

            mse_val = F.mse_loss(x_hat, d_orig)
            psnr_val = -10 * math.log10(mse_val.item()) if mse_val.item() > 0 else 100.0

            if criterion.metric == "mse":
                loss_val = criterion.lmbda * 255.0**2 * mse_val + bpp_val
            else:
                loss_val = (
                    criterion.lmbda * (1 - ms_ssim(x_hat, d_orig, data_range=1.0))
                    + bpp_val
                )

            metrics = (
                accelerator.gather(
                    torch.tensor(
                        [psnr_val, bpp_val.item(), loss_val.item()],
                        device=accelerator.device,
                    )
                )
                .view(-1, 3)
                .mean(dim=0)
            )
            psnr_meter.update(metrics[0].item())
            bpp_meter.update(metrics[1].item())
            loss_meter.update(metrics[2].item())

            if gap_check and not gap_measured and accelerator.is_main_process:
                try:
                    gap_psnr = measure_train_inference_gap(
                        accelerator.unwrap_model(model),
                        d[:1],
                        device=str(accelerator.device),
                    )
                    gap_measured = True
                except Exception as e:
                    gap_psnr = float("nan")
                    if logger:
                        logger.warning(f"Gap check failed: {e}")
                    gap_measured = True

    if accelerator.is_main_process:
        if logger:
            logger.info(
                f"[Val] Epoch {epoch} | Loss: {loss_meter.avg:.4f} | "
                f"PSNR: {psnr_meter.avg:.2f} dB | "
                f"BPP (original pixels): {bpp_meter.avg:.4f}"
                + (f" | Train/Infer ΔPSNR: {gap_psnr:+.3f} dB" if gap_measured else "")
            )
        if writer:
            writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
            writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
            writer.add_scalar("Val/Bpp_orig", bpp_meter.avg, epoch)
            if gap_measured and not math.isnan(gap_psnr):
                writer.add_scalar("Val/TrainInfer_PSNR_gap", gap_psnr, epoch)

    return loss_meter.avg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--routing_mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    p.add_argument("-d", "--dataset", type=str, required=True)
    p.add_argument("--save_path", type=str, default="checkpoints")
    p.add_argument("-e", "--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", dest="learning_rate", type=float, default=1e-4)
    p.add_argument("--aux-lr", dest="aux_learning_rate", type=float, default=1e-3)
    p.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        required=True,
        help="RD tradeoff λ. MSE: 0.0018,0.0035,0.013,0.05. "
        "MS-SSIM: 2.4,4.58,8.73,16.64,31.73,115.37",
    )
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--clip_max_norm", type=float, default=1.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--metric", type=str, default="mse", choices=["mse", "ms-ssim"])
    p.add_argument("--disp-weight", type=float, default=0.1)
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=20,
    )
    p.add_argument(
        "--lr-milestones",
        type=int,
        nargs="+",
        default=[60, 85],
    )
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument(
        "--gap-check-interval",
        type=int,
        default=10,
    )
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision="no")

    save_dir = os.path.join(args.save_path, f"lambda_{args.lmbda}_{args.metric}")
    logger = setup_logger(save_dir) if accelerator.is_main_process else None
    writer = (
        SummaryWriter(os.path.join(save_dir, "tb"))
        if accelerator.is_main_process
        else None
    )

    train_dataset = ImageFolder(
        args.dataset,
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomCrop(args.patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    test_dataset = ImageFolder(
        args.dataset,
        split="valid",
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = WMDC(
        N=192,
        M=320,
        num_slices=5,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
    )
    optimizer, aux_optimizer = configure_optimizers(model, args)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
    )
    criterion = RateDistortionLoss(
        lmbda=args.lmbda,
        metric=args.metric,
        disp_weight=args.disp_weight,
    ).to(accelerator.device)

    (
        model,
        optimizer,
        aux_optimizer,
        train_loader,
        test_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        aux_optimizer,
        train_loader,
        test_loader,
        lr_scheduler,
    )

    start_epoch = 0
    best_loss = float("inf")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        if "criterion" in ckpt:
            criterion.load_state_dict(ckpt["criterion"])

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

        run_gap = (
            args.gap_check_interval > 0 and (epoch + 1) % args.gap_check_interval == 0
        )

        test_loss = test_epoch(
            epoch,
            test_loader,
            model,
            criterion,
            logger,
            writer,
            accelerator,
            gap_check=run_gap,
        )
        lr_scheduler.step()

        if accelerator.is_main_process:
            state = {
                "epoch": epoch,
                "best_loss": best_loss,
                "state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "criterion": criterion.state_dict(),
                "args": vars(args),
            }
            torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth.tar"))
            if test_loss < best_loss:
                best_loss = test_loss
                state["best_loss"] = best_loss
                torch.save(state, os.path.join(save_dir, "checkpoint_best.pth.tar"))
                if logger:
                    logger.info(
                        f"New best checkpoint at epoch {epoch}: loss={best_loss:.4f}"
                    )


if __name__ == "__main__":
    main()
