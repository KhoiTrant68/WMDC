import argparse
import json
import logging
import math
import os
import random
import sys
import warnings
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

from models.WMDC import WMDC, ste_mode

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{stamp}.log")
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
    return sum(
        torch.log(lh.float()).sum() / (-math.log(2) * num_pixels)
        for lh in out_net["likelihoods"].values()
    )


# ---------------------------------------------------------------------------
# Rate-Distortion Loss
# ---------------------------------------------------------------------------


class RateDistortionLoss(nn.Module):
    """
    Rate-distortion loss with routing regularisers.

        L = λ·D + R·rate_warmup
          + β_col · column_neg_entropy
          + β_row · row_entropy
          + γ · ReLU(−Pearson(row_mass, complexity))   [delayed by alignment_delay_epochs]
          + δ · dict_penalty
          + tv_loss

    Changes vs previous version
    ----------------------------
    1. rate_warmup parameter (passed from train loop, linear schedule over
       warmup_steps) prevents scale collapse in early epochs by ramping up
       the rate penalty gradually.
    2. alignment_delay_epochs prevents the alignment loss from destabilising
       the model before the basic RD objective has converged.
    3. _alignment_loss now has a variance floor to prevent gradient explosion
       at initialisation when row_mass is nearly uniform.
    4. warmup_steps and alignment_delay_epochs are stored as attributes so
       train_one_epoch can access them.
    """

    def __init__(
        self,
        lmbda: float = 1e-2,
        metric: str = "mse",
        column_entropy_weight: float = 0.01,
        row_entropy_weight: float = 0.05,
        alignment_weight: float = 0.2,
        dict_penalty_weight: float = 0.1,
        warmup_steps: int = 2000,
        alignment_delay_epochs: int = 50,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metric = metric
        self.column_entropy_weight = float(column_entropy_weight)
        self.row_entropy_weight = float(row_entropy_weight)
        self.alignment_weight = float(alignment_weight)
        self.dict_penalty_weight = float(dict_penalty_weight)
        self.warmup_steps = warmup_steps
        self.alignment_delay_epochs = alignment_delay_epochs

        if metric == "ms-ssim" and lmbda < 1.0:
            warnings.warn(
                f"λ={lmbda:.4f} is very small for metric='ms-ssim'. "
                "Recommended λ range: 2.4 – 115.37.",
                UserWarning,
                stacklevel=2,
            )
        if metric == "mse" and lmbda > 1.0:
            warnings.warn(
                f"λ={lmbda:.4f} is unusually large for metric='mse'. "
                "Recommended λ range: 0.0018 – 0.05.",
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def _alignment_loss(
        row_mass: torch.Tensor,
        complexity: torch.Tensor,
        var_floor: float = 1e-4,
    ) -> torch.Tensor:
        """
        Hinge on −Pearson(row_mass, complexity).
        """
        if row_mass.dim() == 3:
            B, S, HW = row_mass.shape
            rm = row_mass.reshape(B * S, HW)
            co = complexity.unsqueeze(1).expand(B, S, HW).reshape(B * S, HW)
        else:
            rm = row_mass
            co = complexity

        rm = rm - rm.mean(dim=-1, keepdim=True)
        co = co - co.mean(dim=-1, keepdim=True)
        num = (rm * co).sum(dim=-1)

        # Variance floor: prevents division by ~0 at init when P is uniform
        min_norm = math.sqrt(var_floor * rm.shape[-1])
        rm_norm = rm.norm(dim=-1).clamp(min=min_norm)
        co_norm = co.norm(dim=-1).clamp(min=min_norm)

        corr = (num / (rm_norm * co_norm)).clamp(-1.0, 1.0)
        return F.relu(-corr).mean()

    def forward(
        self,
        output: dict,
        target: torch.Tensor,
        rate_warmup: float = 1.0,
        current_epoch: int = 0,
    ) -> dict:
        """
        Parameters
        ----------
        rate_warmup    : scalar in [0.1, 1.0], linearly ramped over warmup_steps.
                         Prevents scale collapse by starting with a small rate penalty.
        current_epoch  : used to gate the alignment loss (only active after
                         alignment_delay_epochs).
        """
        num_pixels = target.size(0) * target.size(2) * target.size(3)
        bpp_loss = compute_bpp(output, num_pixels)

        out: dict = {"bpp_loss": bpp_loss}

        # ── Distortion ────────────────────────────────────────────────────
        if self.metric == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            distortion = 255.0**2 * out["mse_loss"]
        else:
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            distortion = out["ms_ssim_loss"]

        # rate_warmup ramps bpp_loss from 0.1× to 1.0× over warmup_steps
        loss = self.lmbda * distortion + bpp_loss * rate_warmup

        # ── Column-entropy bonus ────────────────────────────────────────
        col_neg_H = output.get("column_neg_entropy")
        if col_neg_H is not None:
            out["column_neg_entropy"] = col_neg_H.detach()
            if self.training and self.column_entropy_weight > 0.0:
                loss = loss + self.column_entropy_weight * col_neg_H

        # ── Row-entropy penalty ────────────────────────────────────────
        row_H = output.get("row_entropy")
        if row_H is not None:
            out["row_entropy"] = row_H.detach()
            if self.training and self.row_entropy_weight > 0.0:
                loss = loss + self.row_entropy_weight * row_H

        # ── Anti-leakage alignment (delayed) ─────────────────────────
        row_mass = output.get("row_mass")
        complexity = output.get("complexity")
        alignment_active = (
            self.training
            and self.alignment_weight > 0.0
            and current_epoch >= self.alignment_delay_epochs
            and row_mass is not None
            and complexity is not None
        )
        if alignment_active:
            align = self._alignment_loss(row_mass, complexity)
            out["alignment_loss"] = align.detach()
            loss = loss + self.alignment_weight * align
        else:
            out["alignment_loss"] = torch.zeros((), device=target.device)

        # ── Dictionary-token diversity penalty ─────────────────────────
        dp = output.get("dict_penalty")
        if dp is not None:
            out["dict_penalty"] = dp.detach() if dp.requires_grad else dp
            if self.training and self.dict_penalty_weight > 0.0:
                loss = loss + self.dict_penalty_weight * dp

        # ── Spatial TV on P ────────────────────────────────────────────
        tv = output.get("tv_loss")
        if tv is not None:
            out["tv_loss"] = tv.detach() if tv.requires_grad else tv
            if self.training:
                loss = loss + tv

        out["loss"] = loss
        return out


# ---------------------------------------------------------------------------
# Optimiser configuration
# ---------------------------------------------------------------------------


def configure_optimizers(net: nn.Module, args):
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
    global_step_offset: int = 0,
):
    model.train()
    criterion.train()

    rd_meter = AverageMeter()
    aux_meter = AverageMeter()
    bpp_meter = AverageMeter()
    col_H_meter = AverageMeter()
    row_H_meter = AverageMeter()
    align_meter = AverageMeter()
    dict_pen_meter = AverageMeter()
    warmup_meter = AverageMeter()

    pbar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )

    for i, d in pbar:
        global_step = global_step_offset + i

        # rate_warmup: linear ramp from 0.1 → 1.0 over criterion.warmup_steps
        # Prevents cc_scale_transforms from collapsing to min_clamp (0.11) in
        # the first few epochs when the rate gradient is very strong.
        rate_warmup = float(
            min(1.0, 0.1 + 0.9 * global_step / max(1, criterion.warmup_steps))
        )
        warmup_meter.update(rate_warmup)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(
            out_net,
            d,
            rate_warmup=rate_warmup,
            current_epoch=epoch,
        )

        # Pass 1: RD loss
        accelerator.backward(out_criterion["loss"])
        if clip_max_norm > 0:
            main_params = [
                p for n, p in model.named_parameters() if not n.endswith(".quantiles")
            ]
            accelerator.clip_grad_norm_(main_params, clip_max_norm)
        optimizer.step()

        # Pass 2: aux loss (entropy bottleneck CDF)
        accelerator.backward(out_net["aux_loss"])
        aux_optimizer.step()

        rd_meter.update(out_criterion["loss"].item())
        aux_meter.update(out_net["aux_loss"].item())
        bpp_meter.update(out_criterion["bpp_loss"].item())

        if "column_neg_entropy" in out_criterion:
            col_H_meter.update(-out_criterion["column_neg_entropy"].item())
        if "row_entropy" in out_criterion:
            row_H_meter.update(out_criterion["row_entropy"].item())
        if "alignment_loss" in out_criterion:
            align_meter.update(out_criterion["alignment_loss"].item())
        if "dict_penalty" in out_criterion:
            dict_pen_meter.update(out_criterion["dict_penalty"].item())

        align_active = epoch >= criterion.alignment_delay_epochs
        pbar.set_postfix(
            rd=f"{rd_meter.avg:.4f}",
            bpp=f"{bpp_meter.avg:.4f}",
            rw=f"{rate_warmup:.2f}",
            Hc=f"{col_H_meter.avg:.2f}",
            Hr=f"{row_H_meter.avg:.2f}",
            al=f"{align_meter.avg:.3f}" + ("" if align_active else "(off)"),
        )

        if accelerator.is_main_process and i % 100 == 0:
            step = epoch * len(train_dataloader) + i
            if writer:
                writer.add_scalar("Train/RD_Loss", rd_meter.avg, step)
                writer.add_scalar("Train/Aux_Loss", aux_meter.avg, step)
                writer.add_scalar("Train/Bpp", bpp_meter.avg, step)
                writer.add_scalar("Train/Rate_Warmup", warmup_meter.avg, step)
                writer.add_scalar("Train/H_col_bits", col_H_meter.avg, step)
                writer.add_scalar("Train/H_row_bits", row_H_meter.avg, step)
                writer.add_scalar("Train/Alignment_hinge", align_meter.avg, step)
                writer.add_scalar("Train/Dict_Penalty", dict_pen_meter.avg, step)

    return rd_meter.avg, global_step_offset + len(train_dataloader)


# ---------------------------------------------------------------------------
# Train/inference gap measurement
# ---------------------------------------------------------------------------


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_train_inference_gap(model, batch: torch.Tensor, device: str) -> float:
    d = batch.to(device).float()
    prev_training = model.training
    prev_use_ste = model.use_ste

    try:
        model.train(True)
        model.use_ste = False
        with torch.no_grad():
            out_train = model(d)
            x_hat_train = out_train["x_hat"].clamp(0, 1)
            if torch.isnan(x_hat_train).any():
                return float("nan")
            mse_train = F.mse_loss(d, x_hat_train).item()
            psnr_train = -10.0 * math.log10(mse_train) if mse_train > 0 else 100.0

        model.eval()
        model.use_ste = False
        with torch.no_grad():
            model.update(force=True)
            _sync()
            out_enc = model.compress(d)
            _sync()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            x_hat_dec = out_dec["x_hat"].clamp(0, 1)
            if torch.isnan(x_hat_dec).any():
                return float("nan")
            mse_dec = F.mse_loss(d, x_hat_dec).item()
            psnr_dec = -10.0 * math.log10(mse_dec) if mse_dec > 0 else 100.0

    finally:
        model.train(prev_training)
        model.use_ste = prev_use_ste

    return psnr_train - psnr_dec


def _bytes_from_strings(obj) -> int:
    """
    Recursively sum byte-lengths of all bytes objects in nested structure.
    Structure of out_enc["strings"]:
        [y_strings, z_strings]
        y_strings = [[b_s0_img0, b_s0_img1, ...],   # slice 0
                     [b_s1_img0, b_s1_img1, ...],   # slice 1
                     ...]                             # num_slices lists
        z_strings = [b_z_img0, b_z_img1, ...]
    """
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_bytes_from_strings(item) for item in obj)
    return 0


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
    real_bytes_batches: int = 5,
):
    model.eval()
    criterion.eval()

    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    loss_meter = AverageMeter()
    col_H_meter = AverageMeter()
    row_H_meter = AverageMeter()
    align_meter = AverageMeter()
    dict_pen_meter = AverageMeter()

    real_psnr_meter = AverageMeter()
    real_bpp_meter = AverageMeter()
    bpp_gap_meter = AverageMeter()

    gap_psnr: float = float("nan")
    nan_batches: int = 0

    accelerator.unwrap_model(model).update(force=True)

    real_done = 0
    try:
        with torch.no_grad():
            for i, d in tqdm(
                enumerate(test_dataloader),
                total=len(test_dataloader),
                desc=f"Val Epoch {epoch}",
                disable=not accelerator.is_local_main_process,
            ):
                # Fast pass
                out_net = model(d)
                out_criterion = criterion(
                    out_net,
                    d,
                    rate_warmup=1.0,  # full rate in val
                    current_epoch=epoch,
                )
                x_hat = out_net["x_hat"].clamp_(0, 1)

                # NaN guard: if decoder produces NaN, skip quality metrics
                if torch.isnan(x_hat).any():
                    nan_batches += 1
                    continue

                if i == 0 and accelerator.is_main_process and writer is not None:
                    n = min(d.size(0), 4)
                    cmp = torch.cat([d[:n], x_hat[:n]])
                    grid = make_grid(cmp, nrow=n, normalize=True, value_range=(0, 1))
                    writer.add_image("Val/Reconstruction", grid, epoch)

                mse_per_image = F.mse_loss(x_hat, d, reduction="none").mean(
                    dim=(1, 2, 3)
                )
                psnr_per_image = torch.where(
                    mse_per_image > 0,
                    -10.0 * mse_per_image.log10(),
                    torch.full_like(mse_per_image, 100.0),
                )

                bpp_val = out_criterion["bpp_loss"].item()
                loss_val = out_criterion["loss"].item()

                if "column_neg_entropy" in out_criterion:
                    col_H_meter.update(
                        -out_criterion["column_neg_entropy"].item(), d.size(0)
                    )
                if "row_entropy" in out_criterion:
                    row_H_meter.update(out_criterion["row_entropy"].item(), d.size(0))
                if "alignment_loss" in out_criterion:
                    align_meter.update(
                        out_criterion["alignment_loss"].item(), d.size(0)
                    )
                if "dict_penalty" in out_criterion:
                    dict_pen_meter.update(
                        out_criterion["dict_penalty"].item(), d.size(0)
                    )

                per_image_metrics = torch.stack(
                    [
                        psnr_per_image,
                        torch.full_like(psnr_per_image, bpp_val),
                        torch.full_like(psnr_per_image, loss_val),
                    ],
                    dim=1,
                )
                gathered = accelerator.gather_for_metrics(per_image_metrics)
                psnr_meter.update(gathered[:, 0].mean().item(), gathered.size(0))
                bpp_meter.update(gathered[:, 1].mean().item(), gathered.size(0))
                loss_meter.update(gathered[:, 2].mean().item(), gathered.size(0))

                # Slow pass: real arithmetic codec round-trip
                if real_done < real_bytes_batches:
                    unwrapped = accelerator.unwrap_model(model)
                    out_enc = unwrapped.compress(d)
                    out_dec = unwrapped.decompress(out_enc["strings"], out_enc["shape"])
                    x_hat_real = out_dec["x_hat"].clamp(0, 1)

                    if torch.isnan(x_hat_real).any():
                        real_done += 1
                        continue

                    mse_real = F.mse_loss(x_hat_real, d, reduction="none").mean(
                        dim=(1, 2, 3)
                    )
                    psnr_real = torch.where(
                        mse_real > 0,
                        -10.0 * mse_real.log10(),
                        torch.full_like(mse_real, 100.0),
                    )

                    num_pixels_per_image = d.size(2) * d.size(3)
                    total_bytes = _bytes_from_strings(out_enc["strings"])
                    real_bpp_batch = (
                        total_bytes * 8.0 / (d.size(0) * num_pixels_per_image)
                    )

                    real_metrics = torch.stack(
                        [
                            psnr_real,
                            torch.full_like(psnr_real, real_bpp_batch),
                            torch.full_like(psnr_real, real_bpp_batch - bpp_val),
                        ],
                        dim=1,
                    )
                    gathered_real = accelerator.gather_for_metrics(real_metrics)
                    real_psnr_meter.update(
                        gathered_real[:, 0].mean().item(), gathered_real.size(0)
                    )
                    real_bpp_meter.update(
                        gathered_real[:, 1].mean().item(), gathered_real.size(0)
                    )
                    bpp_gap_meter.update(
                        gathered_real[:, 2].mean().item(), gathered_real.size(0)
                    )
                    real_done += 1

        if gap_check and accelerator.is_main_process:
            try:
                sample = d[:1]
                gap_psnr = measure_train_inference_gap(
                    accelerator.unwrap_model(model),
                    sample,
                    device=str(accelerator.device),
                )
            except Exception as e:
                if logger:
                    logger.warning(f"Gap check failed: {e}")

    finally:
        criterion.train()

    if accelerator.is_main_process:
        real_str = ""
        if real_psnr_meter.count > 0:
            delta_sign = "+" if bpp_gap_meter.avg >= 0 else ""
            real_str = (
                f" | REAL PSNR: {real_psnr_meter.avg:.2f} dB"
                f" | REAL BPP: {real_bpp_meter.avg:.4f}"
                f" | ΔBPP(real-est): {delta_sign}{bpp_gap_meter.avg:.4f}"
            )
        nan_str = f" | NaN batches: {nan_batches}" if nan_batches > 0 else ""
        gap_str = (
            f" | Train/Infer ΔPSNR: {gap_psnr:+.3f} dB"
            if not math.isnan(gap_psnr)
            else ""
        )

        # Sanity check: ΔBPP(real-est) must be positive (real ≥ estimated)
        if real_bpp_meter.count > 0 and bpp_gap_meter.avg < -0.05:
            if logger:
                logger.warning(
                    f"ΔBPP(real-est) = {bpp_gap_meter.avg:.4f} < 0. "
                    "This indicates _bytes_from_strings is undercounting. "
                    "Check that the recursive version is in use."
                )

        if logger:
            logger.info(
                f"[Val] Epoch {epoch} | Loss: {loss_meter.avg:.4f} "
                f"| PSNR(est): {psnr_meter.avg:.2f} dB "
                f"| BPP(est): {bpp_meter.avg:.4f}"
                f"{real_str}{gap_str}{nan_str}"
            )
        if writer:
            writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
            writer.add_scalar("Val/PSNR_est", psnr_meter.avg, epoch)
            writer.add_scalar("Val/Bpp_est", bpp_meter.avg, epoch)
            writer.add_scalar("Val/H_col_bits", col_H_meter.avg, epoch)
            writer.add_scalar("Val/H_row_bits", row_H_meter.avg, epoch)
            writer.add_scalar("Val/Alignment_hinge", align_meter.avg, epoch)
            writer.add_scalar("Val/Dict_Penalty", dict_pen_meter.avg, epoch)
            writer.add_scalar("Val/NaN_batches", nan_batches, epoch)
            if real_psnr_meter.count > 0:
                writer.add_scalar("Val/PSNR_real", real_psnr_meter.avg, epoch)
                writer.add_scalar("Val/Bpp_real", real_bpp_meter.avg, epoch)
                writer.add_scalar(
                    "Val/Bpp_gap_real_minus_est", bpp_gap_meter.avg, epoch
                )
            if not math.isnan(gap_psnr):
                writer.add_scalar("Val/TrainInfer_PSNR_gap", gap_psnr, epoch)

    return loss_meter.avg, gap_psnr


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--routing-mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    p.add_argument("-d", "--dataset", type=str, required=True)
    p.add_argument("--save_path", type=str, default="checkpoints")
    p.add_argument("-e", "--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", dest="learning_rate", type=float, default=1e-4)
    p.add_argument("--aux-lr", dest="aux_learning_rate", type=float, default=1e-3)
    p.add_argument("--lambda", dest="lmbda", type=float, required=True)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--clip_max_norm", type=float, default=1.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--metric", type=str, default="mse", choices=["mse", "ms-ssim"])

    # Dictionary-routing regularisers
    p.add_argument("--column-entropy-weight", type=float, default=0.01)
    p.add_argument("--row-entropy-weight", type=float, default=0.05)
    p.add_argument("--alignment-weight", type=float, default=0.2)
    p.add_argument("--dict-penalty-weight", type=float, default=0.1)

    # Training stability
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        dest="warmup_steps",
        help=(
            "Number of global steps over which to ramp the BPP loss weight "
            "from 0.1× to 1.0×. Prevents cc_scale_transforms from collapsing "
            "to min_clamp in early training. Default: 2000."
        ),
    )
    p.add_argument(
        "--alignment-delay-epochs",
        type=int,
        default=50,
        dest="alignment_delay_epochs",
        help=(
            "Epoch at which to activate the anti-leakage alignment loss. "
            "Keep at 0 to disable the delay. Default: 50."
        ),
    )

    # Model architecture
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument("--marginal-div", type=str, default="kl", choices=["kl", "tv"])
    p.add_argument(
        "--backbone",
        type=str,
        default="fdm",
        choices=["fdm", "cnn", "swin", "ss2d", "fdm_reversed"],
    )
    p.add_argument(
        "--use-dense-concat",
        action="store_true",
        default=False,
        dest="use_dense_concat",
    )
    p.add_argument(
        "--memory-init",
        type=str,
        default="bootstrap",
        choices=["bootstrap", "zero"],
        dest="memory_init",
    )
    p.add_argument(
        "--content-adaptive",
        action="store_true",
        default=False,
        dest="use_content_adaptive",
    )
    p.add_argument("--cluster-num", type=int, default=8, dest="cluster_num")
    p.add_argument("--sinkhorn-iters", type=int, default=20)

    # LR schedule
    p.add_argument("--lr-milestones", type=int, nargs="+", default=[360, 380])
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument(
        "--last-epochs-with-ste",
        type=int,
        default=20,
        help="Final N epochs to train with STE. 0=disabled.",
    )
    p.add_argument(
        "--gap-check-interval",
        type=int,
        default=10,
        help="Measure train/infer PSNR gap every N epochs. 0=disabled.",
    )
    p.add_argument(
        "--real-bytes-batches",
        type=int,
        default=5,
        help="Val batches per rank for full compress→decompress round-trip. 0=disabled.",
    )
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def worker_init_fn(worker_id: int):
    """
    Add worker_id offset so each DataLoader worker
    gets a distinct seed. Without the offset, torch.initial_seed() returns
    the same value for all workers → identical augmentation sequences →
    effective dataset diversity reduced by num_workers×.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    set_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    # ── Datasets ──────────────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WMDC(
        N=192,
        M=320,
        num_slices=5,
        routing_mode=args.routing_mode,
        marginal_div=args.marginal_div,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        backbone=args.backbone,
        use_dense_concat=args.use_dense_concat,
        memory_init=args.memory_init,
        use_content_adaptive=args.use_content_adaptive,
        cluster_num=args.cluster_num,
    )

    optimizer, aux_optimizer = configure_optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    )
    aux_scheduler = optim.lr_scheduler.MultiStepLR(
        aux_optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    )

    criterion = RateDistortionLoss(
        lmbda=args.lmbda,
        metric=args.metric,
        column_entropy_weight=args.column_entropy_weight,
        row_entropy_weight=args.row_entropy_weight,
        alignment_weight=args.alignment_weight,
        dict_penalty_weight=args.dict_penalty_weight,
        warmup_steps=args.warmup_steps,
        alignment_delay_epochs=args.alignment_delay_epochs,
    ).to(accelerator.device)

    (
        model,
        optimizer,
        aux_optimizer,
        train_loader,
        test_loader,
        lr_scheduler,
        aux_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        aux_optimizer,
        train_loader,
        test_loader,
        lr_scheduler,
        aux_scheduler,
    )

    start_epoch = 0
    best_loss = float("inf")
    global_step = 0  # tracks total training steps across epochs for rate_warmup

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = accelerator.unwrap_model(model).load_state_dict(
            ckpt["state_dict"], strict=False
        )
        if accelerator.is_main_process and (missing or unexpected):
            print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "aux_scheduler" in ckpt:
            aux_scheduler.load_state_dict(ckpt["aux_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        global_step = ckpt.get("global_step", start_epoch * len(train_loader))
        if "criterion" in ckpt:
            criterion.load_state_dict(ckpt["criterion"], strict=False)
        if logger:
            logger.info(
                f"Resumed from epoch {start_epoch - 1} "
                f"(best_loss={best_loss:.4f}, global_step={global_step})"
            )

    summary_path = os.path.join(save_dir, "training_summary.json")
    summary: dict = {}

    # ── Log training config ───────────────────────────────────────────────────
    if accelerator.is_main_process and logger:
        logger.info(
            f"Training config: λ={args.lmbda}, metric={args.metric}, "
            f"routing={args.routing_mode}, backbone={args.backbone}, "
            f"warmup_steps={args.warmup_steps}, "
            f"alignment_delay={args.alignment_delay_epochs} epochs"
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        # STE activation
        ste_active = (
            args.last_epochs_with_ste > 0
            and epoch >= args.epochs - args.last_epochs_with_ste
        )
        accelerator.unwrap_model(model).use_ste = ste_active
        if (
            accelerator.is_main_process
            and ste_active
            and epoch == args.epochs - args.last_epochs_with_ste
        ):
            print(
                f"Turning on STE for the last {args.last_epochs_with_ste} epochs "
                f"(epoch {epoch} → {args.epochs - 1})."
            )

        _, global_step = train_one_epoch(
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
            global_step_offset=global_step,
        )

        run_gap = (
            args.gap_check_interval > 0 and (epoch + 1) % args.gap_check_interval == 0
        )

        test_loss, gap_psnr = test_epoch(
            epoch,
            test_loader,
            model,
            criterion,
            logger,
            writer,
            accelerator,
            gap_check=run_gap,
            real_bytes_batches=args.real_bytes_batches,
        )
        lr_scheduler.step()
        aux_scheduler.step()

        if accelerator.is_main_process:
            state = {
                "epoch": epoch,
                "best_loss": best_loss,
                "global_step": global_step,
                "state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "aux_scheduler": aux_scheduler.state_dict(),
                "criterion": criterion.state_dict(),
                "args": vars(args),
            }
            torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth.tar"))

            if test_loss < best_loss:
                best_loss = test_loss
                state["best_loss"] = best_loss
                torch.save(state, os.path.join(save_dir, "checkpoint_best.pth.tar"))
                if logger:
                    logger.info(f"New best at epoch {epoch}: loss={best_loss:.4f}")

            if run_gap and not math.isnan(gap_psnr):
                summary[f"epoch_{epoch}"] = {
                    "test_loss": round(test_loss, 4),
                    "gap_psnr_db": round(gap_psnr, 3),
                    "global_step": global_step,
                }
                with open(summary_path, "w") as f:
                    json.dump(
                        {
                            "lambda": args.lmbda,
                            "metric": args.metric,
                            "history": summary,
                        },
                        f,
                        indent=4,
                    )

    if accelerator.is_main_process and writer:
        writer.close()


if __name__ == "__main__":
    main()
