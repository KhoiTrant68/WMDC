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
    """
    Differentiable training-time BPP estimate from likelihood tensors.

    num_pixels must be the number of *original* (unpadded) pixels so that
    the rate objective correctly reflects the cost per actual image pixel.
    Under DDP this is the per-device pixel count; gradients are averaged
    across devices by the DDP all-reduce on the loss, which is correct.
    """
    return sum(
        torch.log(lh.float()).sum() / (-math.log(2) * num_pixels)
        for lh in out_net["likelihoods"].values()
    )


# ---------------------------------------------------------------------------
# Rate-Distortion Loss
# ---------------------------------------------------------------------------


class RateDistortionLoss(nn.Module):
    """
    Rate-distortion loss with adaptive dispersion regularisation.

    L = λ · distortion + bpp  +  w_disp · scale · dispersion_loss

    The dispersion term is *added* because `_dispersion_loss` already returns
    negative entropy (-H). Minimizing -H is mathematically equivalent to
    maximizing the entropy (H) of dictionary usage, which encourages diverse
    codebook utilisation.

    EMA buffers track the running magnitudes of the BPP and dispersion terms
    so that `scale` auto-normalises their relative contribution, making the
    effective dispersion coefficient `disp_weight` interpretable as a
    fraction of the rate loss rather than an absolute value.

    Note on DDP:
        The criterion is NOT wrapped by accelerate.prepare() and therefore
        lives as an independent module on each device.  To keep EMA buffers
        synchronised across ranks we explicitly all-reduce the scalar values
        before updating, which mirrors the behaviour of a single-GPU run.
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
        # Use unpadded pixel count so BPP is always normalised to true image
        # size.  target is already the unpadded crop (patch_size × patch_size
        # during training, original dimensions at validation).
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
                    # Scalar tensors for all-reduce
                    b_val = bpp_loss.abs().detach().reshape(1)
                    d_val = disp.abs().detach().reshape(1)

                    # Synchronise EMA inputs across DDP ranks so all devices
                    # update their buffers with the same value.
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(b_val, op=dist.ReduceOp.AVG)
                        dist.all_reduce(d_val, op=dist.ReduceOp.AVG)

                    self.ema_bpp.lerp_(b_val.squeeze(), weight=0.01)
                    self.ema_disp.lerp_(d_val.squeeze(), weight=0.01)

            scale = (self.ema_bpp / (self.ema_disp + 1e-8)).clamp(0.01, 10.0)
            effective = float(self.disp_weight * scale.item())
            self.effective_disp_coeff = effective

            # Add dispersion loss: disp = −H, so + (effective * (−H)) = −H·effective
            # Minimising −H maximises dictionary entropy (diverse codebook usage).
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

        # d is a patch of shape (B, 3, patch_size, patch_size).
        # patch_size is a multiple of 64 by construction (default 256), so
        # no padding is needed during training.
        out_net = model(d)

        # Pass unpadded target (= d itself during training) to the loss so
        # that num_pixels is computed over the true image content.
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
            # aux=f"{aux_meter.avg:.5f}",
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
    Input is guaranteed to be a multiple of 64 from the test dataloader.
    """
    model.eval()
    d = batch.to(device)

    with torch.no_grad():
        # Forward pass (continuous noise relaxation)
        out_fwd = model(d)
        x_hat_fwd = out_fwd["x_hat"].clamp(0, 1)

        mse_fwd = F.mse_loss(d, x_hat_fwd)
        psnr_fwd = -10 * math.log10(mse_fwd.item()) if mse_fwd.item() > 0 else 100.0

        # Compress/Decompress pass (discrete quantization)
        out_enc = model.compress(d)
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        x_hat_dec = out_dec["x_hat"].clamp(0, 1)

        mse_dec = F.mse_loss(d, x_hat_dec)
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
            out_net = model(d)
            x_hat = out_net["x_hat"].clamp_(0, 1)

            if i == 0 and accelerator.is_main_process and writer is not None:
                n = min(d.size(0), 4)
                cmp = torch.cat([d[:n], x_hat[:n]])
                grid = make_grid(cmp, nrow=n, normalize=True, value_range=(0, 1))
                writer.add_image("Val/Reconstruction", grid, epoch)

            B, _, H, W = d.size()
            num_pixels = B * H * W

            total_bits = sum(
                torch.log(lh.float()).sum() / (-math.log(2))
                for lh in out_net["likelihoods"].values()
            )
            bpp_val = total_bits / num_pixels

            mse_val = F.mse_loss(x_hat, d)
            psnr_val = -10 * math.log10(mse_val.item()) if mse_val.item() > 0 else 100.0

            if criterion.metric == "mse":
                loss_val = criterion.lmbda * 255.0**2 * mse_val + bpp_val
            else:
                loss_val = (
                    criterion.lmbda * (1 - ms_ssim(x_hat, d, data_range=1.0)) + bpp_val
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
                f"BPP: {bpp_meter.avg:.4f}"
                + (f" | Train/Infer ΔPSNR: {gap_psnr:+.3f} dB" if gap_measured else "")
            )
        if writer:
            writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
            writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
            writer.add_scalar("Val/Bpp", bpp_meter.avg, epoch)
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

    # Validate that patch_size is divisible by 64 so forward() never sees
    # non-compliant input during training.
    if args.patch_size % 64 != 0:
        raise ValueError(
            f"--patch-size must be a multiple of 64, got {args.patch_size}."
        )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
        transform=transforms.Compose(
            [
                transforms.CenterCrop(args.patch_size),
                transforms.ToTensor(),
            ]
        ),
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
        batch_size=args.batch_size,
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

    # Criterion is placed on the accelerator device but NOT passed to
    # accelerate.prepare() — it holds no parameters that need DDP wrapping.
    # EMA buffers are synchronised manually via all_reduce inside forward().
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
