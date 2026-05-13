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
    Mathematically principled rate-distortion loss with explicit routing
    regularisers for dictionary-based neural image compression.

        L =  λ · distortion                                  # RD trade-off
           + bpp                                              #
           + β_col · column_neg_entropy(P)                    # = −β_col · H_col
           + β_row · row_entropy(P)                           # = +β_row · H_row
           + γ      · ReLU( −Pearson(row_mass, complexity) )  # anti-leakage
           + δ      · dict_penalty                            # token diversity
           + tv_loss                                          # spatial TV on P

    Information-theoretic derivation of the dictionary regularisers
    ---------------------------------------------------------------
    For the side-information channel "pixel → token" implemented by the
    transport plan P, the mutual information decomposes as

        I(pixel ; token)  =  H_col  −  E_pixel[ H_row ]

    Maximising I therefore requires SIMULTANEOUSLY:
      (a) maximising H_col   ⇒  no dead codes (every token used somewhere),
      (b) minimising H_row   ⇒  each pixel selects FEW tokens (specialisation).

    The pre-refactor loss carried only (a) and a token-orthogonality penalty,
    leaving the trivial uniform routing P[i,j] = 1/N as a global optimum —
    matching the observed 100 % per-slice utilisation and the 0.5 dB rate gap.

    Anti-leakage alignment
    ----------------------
    Under unbalanced entropic OT the row marginal m_i = Σ_j P_ij carries the
    spatial gating Aprox_KL(1) = ρ_i/(ρ_i + ε) ∈ (0, 1].  Compression theory
    says the dictionary side-information should help MOST where the latent
    is high-entropy (i.e. content-complex regions).  Audit on the failed
    checkpoint showed Pearson( row_mass , complexity ) = −0.66 — the optimizer
    chose the OPPOSITE alignment, dropping mass at complex regions and
    laundering the saved rate into the Gaussian channel.  The hinge term
    ReLU( −corr(row_mass, complexity) ) penalises only the wrong-sign regime
    and is identically zero on data already in the correct alignment.

    Why a fixed (not EMA-adaptive) weight on the entropy terms
    ----------------------------------------------------------
    Earlier code scaled β_col by EMA(bpp) / EMA(H_col) clamped to [0.01, 2.0]
    in an attempt to keep entropy commensurate with bpp.  This conflated two
    quantities that change on different time scales and introduced a feedback
    loop with the optimiser (high bpp → larger entropy weight → faster bpp
    reduction → smaller weight → …).  In bits-vs-bits both H_col and H_row
    are already commensurate with bpp_loss, so a constant weight is principled.
    """

    def __init__(
        self,
        lmbda: float = 1e-2,
        metric: str = "mse",
        column_entropy_weight: float = 0.01,
        row_entropy_weight: float = 0.05,
        alignment_weight: float = 0.2,
        dict_penalty_weight: float = 0.1,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metric = metric
        self.column_entropy_weight = float(column_entropy_weight)
        self.row_entropy_weight = float(row_entropy_weight)
        self.alignment_weight = float(alignment_weight)
        self.dict_penalty_weight = float(dict_penalty_weight)

        if metric == "ms-ssim" and lmbda < 1.0:
            warnings.warn(
                f"λ={lmbda:.4f} is very small for metric='ms-ssim'. "
                "MS-SSIM distortion is in [0, 1], so tiny λ values cause the "
                "model to optimise rate only. Recommended λ range: 2.4 – 115.37. "
                "Did you mean to use --metric mse?",
                UserWarning,
                stacklevel=2,
            )
        if metric == "mse" and lmbda > 1.0:
            warnings.warn(
                f"λ={lmbda:.4f} is unusually large for metric='mse'. "
                "MSE distortion is scaled by 255²≈65025. "
                "Recommended λ range: 0.0018 – 0.05.",
                UserWarning,
                stacklevel=2,
            )

    # -----------------------------------------------------------------------
    # Anti-leakage alignment regulariser
    # -----------------------------------------------------------------------

    @staticmethod
    def _alignment_loss(
        row_mass: torch.Tensor, complexity: torch.Tensor
    ) -> torch.Tensor:
        """
        Hinge on the NEGATIVE Pearson correlation between per-pixel row mass
        and per-pixel content complexity:

            L_align = E_batch,slice [ ReLU( −corr(row_mass, complexity) ) ]
                    ∈ [0, 1]

        Parameters
        ----------
        row_mass    : (B, S, HW) or (B, HW)   grad-bearing
        complexity  : (B, HW)                 detached, ∈ [0, 1]

        Returns
        -------
        scalar tensor ∈ [0, 1].  Zero if every (batch, slice) pair already
        has non-negative correlation between routing mass and complexity.
        """
        if row_mass.dim() == 3:
            B, S, HW = row_mass.shape
            rm = row_mass.reshape(B * S, HW)
            co = complexity.unsqueeze(1).expand(B, S, HW).reshape(B * S, HW)
        else:
            rm = row_mass
            co = complexity

        # Constant-row guard: if every position has the same mass (e.g. softmax
        # / balanced modes), Pearson is undefined; treat as zero correlation.
        rm = rm - rm.mean(dim=-1, keepdim=True)
        co = co - co.mean(dim=-1, keepdim=True)
        num = (rm * co).sum(dim=-1)
        den = rm.norm(dim=-1) * co.norm(dim=-1) + 1e-8
        corr = num / den  # (B*S,)
        return F.relu(-corr).mean()

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, output: dict, target: torch.Tensor) -> dict:
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

        loss = self.lmbda * distortion + bpp_loss

        # ── Column-entropy bonus (anti-dead-code) ─────────────────────────
        # column_neg_entropy = −H_col.  Adding it with a POSITIVE weight to
        # the loss thus MAXIMISES H_col (every dictionary token gets used).
        col_neg_H = output.get("column_neg_entropy")
        if col_neg_H is not None:
            out["column_neg_entropy"] = col_neg_H.detach()
            if self.training and self.column_entropy_weight > 0.0:
                loss = loss + self.column_entropy_weight * col_neg_H

        # ── Row-entropy penalty (sparsity / specialisation) ────────────────
        # row_entropy = H_row ≥ 0.  Adding it with a POSITIVE weight MINIMISES
        # H_row, driving each pixel toward a peaked choice over tokens.
        row_H = output.get("row_entropy")
        if row_H is not None:
            out["row_entropy"] = row_H.detach()
            if self.training and self.row_entropy_weight > 0.0:
                loss = loss + self.row_entropy_weight * row_H

        # ── Anti-leakage alignment ────────────────────────────────────────
        row_mass = output.get("row_mass")
        complexity = output.get("complexity")
        if (
            self.training
            and self.alignment_weight > 0.0
            and row_mass is not None
            and complexity is not None
        ):
            align = self._alignment_loss(row_mass, complexity)
            out["alignment_loss"] = align.detach()
            loss = loss + self.alignment_weight * align

        # ── Dictionary-token diversity penalty ────────────────────────────
        dp = output.get("dict_penalty")
        if dp is not None:
            out["dict_penalty"] = dp.detach() if dp.requires_grad else dp
            if self.training and self.dict_penalty_weight > 0.0:
                loss = loss + self.dict_penalty_weight * dp

        # ── Spatial TV on P (already weight-scaled inside the attention) ───
        tv = output.get("tv_loss")
        if tv is not None:
            out["tv_loss"] = tv.detach() if tv.requires_grad else tv
            if self.training:
                loss = loss + tv  # tv_weight is baked in

        out["loss"] = loss
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
    criterion.train()

    rd_meter = AverageMeter()
    aux_meter = AverageMeter()
    bpp_meter = AverageMeter()
    col_H_meter = AverageMeter()  # H_col   = −column_neg_entropy
    row_H_meter = AverageMeter()  # H_row   (sparsity penalty target)
    align_meter = AverageMeter()  # alignment hinge ∈ [0, 1]
    dict_pen_meter = AverageMeter()

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

        # ── Pass 1: RD loss ────────────────────────────────────────────────
        accelerator.backward(out_criterion["loss"])
        if clip_max_norm > 0:
            main_params = [
                p for n, p in model.named_parameters() if not n.endswith(".quantiles")
            ]
            accelerator.clip_grad_norm_(main_params, clip_max_norm)
        optimizer.step()

        # ── Pass 2: aux loss (entropy bottleneck CDF) ──────────────────────
        accelerator.backward(out_net["aux_loss"])
        aux_optimizer.step()

        rd_meter.update(out_criterion["loss"].item())
        aux_meter.update(out_net["aux_loss"].item())
        bpp_meter.update(out_criterion["bpp_loss"].item())

        if "column_neg_entropy" in out_criterion:
            # log H_col (positive) for readability
            col_H_meter.update(-out_criterion["column_neg_entropy"].item())
        if "row_entropy" in out_criterion:
            row_H_meter.update(out_criterion["row_entropy"].item())
        if "alignment_loss" in out_criterion:
            align_meter.update(out_criterion["alignment_loss"].item())
        if "dict_penalty" in out_criterion:
            dict_pen_meter.update(out_criterion["dict_penalty"].item())

        pbar.set_postfix(
            rd=f"{rd_meter.avg:.4f}",
            bpp=f"{bpp_meter.avg:.4f}",
            Hc=f"{col_H_meter.avg:.2f}",
            Hr=f"{row_H_meter.avg:.2f}",
            al=f"{align_meter.avg:.3f}",
        )

        if accelerator.is_main_process and i % 100 == 0:
            step = epoch * len(train_dataloader) + i
            if writer:
                writer.add_scalar("Train/RD_Loss", rd_meter.avg, step)
                writer.add_scalar("Train/Aux_Loss", aux_meter.avg, step)
                writer.add_scalar("Train/Bpp", bpp_meter.avg, step)
                writer.add_scalar("Train/H_col_bits", col_H_meter.avg, step)
                writer.add_scalar("Train/H_row_bits", row_H_meter.avg, step)
                writer.add_scalar("Train/Alignment_hinge", align_meter.avg, step)
                writer.add_scalar("Train/Dict_Penalty", dict_pen_meter.avg, step)

    return rd_meter.avg


# ---------------------------------------------------------------------------
# Train/inference gap measurement
# ---------------------------------------------------------------------------


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_train_inference_gap(model, batch: torch.Tensor, device: str) -> float:
    """
    PSNR gap between the EXACT signal seen by the optimiser and the EXACT
    signal produced by the deployed codec.

        gap = psnr_train_forward  −  psnr_compressed

    where
      • psnr_train_forward  : model.train(), use_ste=False  → additive uniform
                              noise quantisation (the gradient path).
      • psnr_compressed     : model.eval(),                  → real arithmetic
                              encode → decode → reconstruct (the bitstream).

    Why this signature
    ------------------
    The pre-refactor implementation compared model.eval()+STE (hard rounding)
    against compress/decompress (also hard rounding) — by construction the
    two paths use the SAME quantiser, so the gap was reported as 0.000 dB
    every val while the true train-vs-eval gap was ~1.7 dB.  Comparing
    train()+noise against eval()+arith makes the metric report the gap that
    the audit found:  the cost of training under a relaxed quantiser.

    Returns
    -------
    gap_db : float (positive → optimiser is more optimistic than deployment)
    """
    d = batch.to(device).float()

    prev_training = model.training
    prev_use_ste = model.use_ste

    try:
        # ── Pass 1: training-mode forward (noise quantisation) ───────────
        # This is the EXACT pipeline the optimiser sees.  No torch.no_grad
        # wrapper around the forward itself would be fine for PSNR-only
        # measurement, but we keep grad off to avoid building the graph.
        model.train(True)
        model.use_ste = False
        with torch.no_grad():
            out_train = model(d)
            x_hat_train = out_train["x_hat"].clamp(0, 1)
            mse_train = F.mse_loss(d, x_hat_train).item()
            psnr_train = -10.0 * math.log10(mse_train) if mse_train > 0 else 100.0

        # ── Pass 2: eval-mode compress / decompress (real arith coding) ──
        model.eval()
        model.use_ste = False
        with torch.no_grad():
            model.update(force=True)  # rebuild CDF tables before any compress()
            _sync()
            out_enc = model.compress(d)
            _sync()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            x_hat_dec = out_dec["x_hat"].clamp(0, 1)
            mse_dec = F.mse_loss(d, x_hat_dec).item()
            psnr_dec = -10.0 * math.log10(mse_dec) if mse_dec > 0 else 100.0

    finally:
        # Always restore original state — even on exception
        model.train(prev_training)
        model.use_ste = prev_use_ste

    return psnr_train - psnr_dec


def _bytes_from_strings(strings) -> int:
    """Sum byte length over the nested 'strings' structure returned by compress()."""
    n = 0
    for s_or_list in strings:
        if isinstance(s_or_list, (list, tuple)):
            for s in s_or_list:
                n += len(s)
        else:
            n += len(s_or_list)
    return n


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
    """
    Validation epoch with TWO measurements per batch:

      • Fast pass (every batch): forward() in eval mode → PSNR_est, BPP_est
        from likelihood entropy.  This is what the original implementation
        reported and what the val log used to track.  Cheap, gradient-free.

      • Real pass (first `real_bytes_batches` batches only):
            compress(d) → arithmetic encode → byte_strings, shape
            decompress(strings, shape) → x_hat_real
            BPP_real = total_bits / num_pixels
            PSNR_real = −10·log10(MSE(x, x_hat_real))
        This is the EXACT bitstream the deployed codec produces and is
        what eval.py reports — putting it inside the val loop closes the
        feedback loop on the optimiser (the audit found a 1.66 dB gap
        between the fast PSNR and the real one).

    real_bytes_batches=0 disables the slow pass entirely.
    """
    model.eval()
    criterion.eval()

    psnr_meter = AverageMeter()           # forward (estimated) PSNR
    bpp_meter = AverageMeter()            # estimated BPP from likelihoods
    loss_meter = AverageMeter()
    col_H_meter = AverageMeter()          # H_col
    row_H_meter = AverageMeter()          # H_row
    align_meter = AverageMeter()          # alignment hinge
    dict_pen_meter = AverageMeter()

    # Real-bytes meters (rank-0 only, since compress is sequential per image)
    real_psnr_meter = AverageMeter()
    real_bpp_meter = AverageMeter()
    bpp_gap_meter = AverageMeter()        # real_bpp − est_bpp (sign of mis-cal)

    gap_psnr: float = float("nan")

    # Rebuild entropy tables on the underlying model on every rank.
    # Required before ANY compress()/decompress() call below.
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
                # ── Fast pass: forward only, on every rank ────────────────
                out_net = model(d)
                out_criterion = criterion(out_net, d)
                x_hat = out_net["x_hat"].clamp_(0, 1)

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
                    row_H_meter.update(
                        out_criterion["row_entropy"].item(), d.size(0)
                    )
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

                # ── Slow pass: arithmetic-coded round-trip ────────────────
                # Run on every rank; gather later.  The first
                # `real_bytes_batches` batches per rank participate.
                if real_done < real_bytes_batches:
                    unwrapped = accelerator.unwrap_model(model)
                    out_enc = unwrapped.compress(d)
                    out_dec = unwrapped.decompress(
                        out_enc["strings"], out_enc["shape"]
                    )
                    x_hat_real = out_dec["x_hat"].clamp(0, 1)
                    mse_real = F.mse_loss(
                        x_hat_real, d, reduction="none"
                    ).mean(dim=(1, 2, 3))
                    psnr_real = torch.where(
                        mse_real > 0,
                        -10.0 * mse_real.log10(),
                        torch.full_like(mse_real, 100.0),
                    )

                    num_pixels_per_image = d.size(2) * d.size(3)
                    total_bytes = _bytes_from_strings(out_enc["strings"])
                    real_bpp_batch = total_bytes * 8.0 / (
                        d.size(0) * num_pixels_per_image
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
            real_str = (
                f" | REAL PSNR: {real_psnr_meter.avg:.2f} dB"
                f" | REAL BPP: {real_bpp_meter.avg:.4f}"
                f" | ΔBPP(real-est): {bpp_gap_meter.avg:+.4f}"
            )
        gap_str = (
            f" | Train/Infer ΔPSNR: {gap_psnr:+.3f} dB"
            if not math.isnan(gap_psnr)
            else ""
        )
        if logger:
            logger.info(
                f"[Val] Epoch {epoch} | Loss: {loss_meter.avg:.4f} "
                f"| PSNR(est): {psnr_meter.avg:.2f} dB "
                f"| BPP(est): {bpp_meter.avg:.4f}"
                f"{real_str}{gap_str}"
            )
        if writer:
            writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
            writer.add_scalar("Val/PSNR_est", psnr_meter.avg, epoch)
            writer.add_scalar("Val/Bpp_est", bpp_meter.avg, epoch)
            writer.add_scalar("Val/H_col_bits", col_H_meter.avg, epoch)
            writer.add_scalar("Val/H_row_bits", row_H_meter.avg, epoch)
            writer.add_scalar("Val/Alignment_hinge", align_meter.avg, epoch)
            writer.add_scalar("Val/Dict_Penalty", dict_pen_meter.avg, epoch)
            if real_psnr_meter.count > 0:
                writer.add_scalar("Val/PSNR_real", real_psnr_meter.avg, epoch)
                writer.add_scalar("Val/Bpp_real", real_bpp_meter.avg, epoch)
                writer.add_scalar("Val/Bpp_gap_real_minus_est", bpp_gap_meter.avg, epoch)
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
    p.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        required=True,
        help=(
            "RD tradeoff λ.\n"
            "  MSE    (--metric mse)    : 0.0018, 0.0035, 0.013, 0.05\n"
            "  MS-SSIM (--metric ms-ssim): 2.4, 4.58, 8.73, 16.64, 31.73, 115.37\n"
            "WARNING: using a wrong λ range silently trains a broken model."
        ),
    )
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--clip_max_norm", type=float, default=1.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--metric",
        type=str,
        default="mse",
        choices=["mse", "ms-ssim"],
    )
    # ── Dictionary-routing regularisers ─────────────────────────────────
    # All four are in commensurate units (bits or [0,1]) so the weights are
    # fixed — no EMA adaptive scaling.  See RateDistortionLoss docstring for
    # the mutual-information derivation.
    p.add_argument(
        "--column-entropy-weight",
        type=float,
        default=0.01,
        help=(
            "β_col: weight on column_neg_entropy = −H_col (bits). "
            "Positive → MAXIMISE H_col (anti-dead-code).  Set 0 to disable."
        ),
    )
    p.add_argument(
        "--row-entropy-weight",
        type=float,
        default=0.05,
        help=(
            "β_row: weight on row_entropy = H_row (bits). "
            "Positive → MINIMISE H_row (sparse per-pixel selection).  "
            "Together with β_col this maximises I(pixel ; token) and was "
            "MISSING in the pre-refactor loss → dictionary collapsed to "
            "100% utilisation on every slice.  Set 0 to disable."
        ),
    )
    p.add_argument(
        "--alignment-weight",
        type=float,
        default=0.2,
        help=(
            "γ: weight on the anti-leakage hinge "
            "ReLU(−Pearson(row_mass, complexity)).  Penalises only the "
            "wrong-sign correlation — zero on data already aligned. "
            "The failed checkpoint had corr = −0.66 (UEOT leakage).  "
            "Set 0 to disable."
        ),
    )
    p.add_argument(
        "--dict-penalty-weight",
        type=float,
        default=0.1,
        help=(
            "δ: weight on the off-diagonal cosine-similarity penalty on the "
            "dictionary tokens (∈ [0, 1] by construction).  Encourages "
            "TOKEN diversity (orthogonal basis vectors) — orthogonal concern "
            "from per-pixel sparsity.  Set 0 to disable."
        ),
    )
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument(
        "--marginal-div",
        type=str,
        default="kl",
        choices=["kl", "tv"],
        help="Marginal divergence for unbalanced_eot: 'kl' (smooth) or 'tv' (sharp gating).",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="fdm",
        choices=["fdm", "cnn", "swin", "ss2d", "fdm_reversed"],
        help="Encoder/decoder backbone block (default: fdm = FrequencyDisentangledMamba).",
    )
    p.add_argument(
        "--use-dense-concat",
        action="store_true",
        default=False,
        dest="use_dense_concat",
        help="Replace stateful Markov memory with dense channel-autoregressive concat.",
    )
    p.add_argument(
        "--memory-init",
        type=str,
        default="bootstrap",
        choices=["bootstrap", "zero"],
        dest="memory_init",
        help="How to initialise M_1: 'bootstrap' (learned Conv projection) or 'zero'.",
    )
    p.add_argument("--sinkhorn-iters", type=int, default=20)
    p.add_argument("--lr-milestones", type=int, nargs="+", default=[360, 380])
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument(
        "--last-epochs-with-ste",
        type=int,
        default=20,
        help="Final epochs to train with STE instead of noise relaxation. 0=disabled.",
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
        help=(
            "Number of val batches PER RANK on which to run the full "
            "compress→decompress arithmetic-coding round-trip during "
            "every val epoch.  Reports REAL bpp and REAL PSNR alongside "
            "the entropy-estimate metrics so the audit-blindness fixed "
            "by this refactor stays fixed.  Set 0 to disable (faster val)."
        ),
    )
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
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
    find_unused = (
        args.column_entropy_weight == 0.0
        or args.row_entropy_weight == 0.0
        or args.alignment_weight == 0.0
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)
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

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "aux_scheduler" in ckpt:
            aux_scheduler.load_state_dict(ckpt["aux_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        if "criterion" in ckpt:
            # strict=False: old checkpoints stored EMA buffers (ema_bpp,
            # ema_disp, ema_steps) that the refactored criterion no longer
            # owns.  Drop them silently rather than aborting the resume.
            criterion.load_state_dict(ckpt["criterion"], strict=False)
        if logger:
            logger.info(
                f"Resumed from epoch {start_epoch - 1} (best_loss={best_loss:.4f})"
            )

    # ── Per-lambda summary ─────────────────────────────────────────────────────
    summary_path = os.path.join(save_dir, "training_summary.json")
    summary: dict = {}

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        ste_active = epoch >= args.epochs - args.last_epochs_with_ste
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
