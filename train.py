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
    Rate-distortion loss with adaptive entropy-dispersion bonus AND a
    separate, fixed-weight dictionary diversity penalty.

        L =  λ · distortion
           + bpp
           − effective · H(m)
           + dict_penalty_weight · dict_penalty

    where
      H(m)         is Shannon entropy (bits) of the column-marginal of the
                   transport plan — maximised → uniform dictionary usage.
      dict_penalty is the off-diagonal cosine-similarity penalty on the
                   dictionary tokens (≥ 0) — minimised → diverse tokens.

    Sign convention for the entropy bonus:
      - model.forward() returns  dispersion_loss = −H   (negative entropy)
      - here:  dispersion_bonus = −dispersion_loss = H  (positive)
      - loss = RD − effective · H

    Why dict_penalty is handled separately from dispersion_loss
    -----------------------------------------------------------
    Earlier versions packed `dict_penalty * 10` into `dispersion_loss`,
    so the EMA-based adaptive scale had to track a mixed signal whose sign
    can flip during training (H − 10·dict_penalty).  With a fixed weight on
    dict_penalty and EMA only over the routing-entropy term, both signals
    have well-defined gradient directions and the EMA is no longer noisy.
    """

    def __init__(
        self,
        lmbda: float = 1e-2,
        metric: str = "mse",
        disp_weight: float = 0.1,
        dict_penalty_weight: float = 1.0,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metric = metric
        self.disp_weight = disp_weight
        self.dict_penalty_weight = dict_penalty_weight

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

        # EMA for adaptive dispersion coefficient.
        # Bias correction: 1 - 0.99^t → 1 after ~300 steps.
        self.register_buffer("ema_bpp", torch.tensor(0.0))
        self.register_buffer("ema_disp", torch.tensor(0.0))
        self.register_buffer("ema_steps", torch.tensor(0, dtype=torch.long))

        # Exposed for TensorBoard logging
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

        # ── Routing entropy bonus  (−effective · H) ─────────────────────────
        if "dispersion_loss" in output:
            disp_bonus = -output["dispersion_loss"]  # H ≥ 0 (bits)
            out["dispersion_bonus"] = disp_bonus

            if self.training:
                with torch.no_grad():
                    b_val = bpp_loss.abs().detach().reshape(1)
                    d_val = disp_bonus.abs().detach().reshape(1)

                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(b_val, op=dist.ReduceOp.AVG)
                        dist.all_reduce(d_val, op=dist.ReduceOp.AVG)

                    self.ema_steps += 1
                    self.ema_bpp.lerp_(b_val.squeeze(), weight=0.01)
                    self.ema_disp.lerp_(d_val.squeeze(), weight=0.01)

                t = float(self.ema_steps.item())
                bc = max(1.0 - 0.99**t, 1e-6)  # bias correction
                ema_bpp_hat = self.ema_bpp / bc
                ema_disp_hat = self.ema_disp / bc

                scale = (ema_bpp_hat / (ema_disp_hat + 1e-8)).clamp(0.01, 2.0)
                effective = float(self.disp_weight * scale.item())
                self.effective_disp_coeff = effective
            else:
                effective = self.effective_disp_coeff

            out["loss"] = out["loss"] - effective * disp_bonus

        # ── Dictionary diversity penalty  (+ fixed_w · dict_penalty) ─────────
        # Always added with a constant weight: dict_penalty is already bounded
        # in [0, 1] by construction (squared ReLU on cosine similarities), so
        # no adaptive scaling is needed.  Only contributes during training,
        # since QueryDictionaryGenerator returns 0.0 in eval mode.
        if "dict_penalty" in output:
            dp = output["dict_penalty"]
            out["dict_penalty"] = dp
            if self.training:
                out["loss"] = out["loss"] + self.dict_penalty_weight * dp

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
    disp_meter = AverageMeter()
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

        if "dispersion_bonus" in out_criterion:
            disp_meter.update(out_criterion["dispersion_bonus"].item())
        if "dict_penalty" in out_criterion:
            dict_pen_meter.update(out_criterion["dict_penalty"].item())

        pbar.set_postfix(
            rd=f"{rd_meter.avg:.4f}",
            bpp=f"{bpp_meter.avg:.4f}",
            H=f"{disp_meter.avg:.3f}",
            dp=f"{dict_pen_meter.avg:.4f}",
        )

        if accelerator.is_main_process and i % 100 == 0:
            step = epoch * len(train_dataloader) + i
            if writer:
                writer.add_scalar("Train/RD_Loss", rd_meter.avg, step)
                writer.add_scalar("Train/Aux_Loss", aux_meter.avg, step)
                writer.add_scalar("Train/Bpp", bpp_meter.avg, step)
                writer.add_scalar("Train/Dict_Entropy_H", disp_meter.avg, step)
                writer.add_scalar("Train/Dict_Penalty", dict_pen_meter.avg, step)
                writer.add_scalar(
                    "Train/Disp_eff_coeff",
                    criterion.effective_disp_coeff,
                    step,
                )

    return rd_meter.avg


# ---------------------------------------------------------------------------
# Train/inference gap measurement
# ---------------------------------------------------------------------------


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_train_inference_gap(model, batch: torch.Tensor, device: str) -> float:
    """
    Measure PSNR gap between forward() (noise relaxation / STE) and
    compress→decompress (true hard quantisation) on the same image patch.

    Returns: psnr_forward − psnr_quantised  (positive → forward is optimistic)
    """
    d = batch.to(device).float()

    prev_training = model.training
    model.eval()

    try:
        # model.use_ste is restored to its prior value on exit, even
        # if an exception is raised inside the block.
        with torch.no_grad(), ste_mode(model) as m:
            # Rebuild entropy tables BEFORE codec calls
            m.update(force=True)

            # Forward pass with STE (hard rounding approximation)
            out_fwd = m(d)
            x_hat_fwd = out_fwd["x_hat"].clamp(0, 1)
            mse_fwd = F.mse_loss(d, x_hat_fwd)
            psnr_fwd = (
                -10.0 * math.log10(mse_fwd.item()) if mse_fwd.item() > 0 else 100.0
            )

            # Compress / decompress (true hard quantisation via arithmetic coding)
            _sync()
            out_enc = m.compress(d)
            _sync()
            out_dec = m.decompress(out_enc["strings"], out_enc["shape"])
            x_hat_dec = out_dec["x_hat"].clamp(0, 1)
            mse_dec = F.mse_loss(d, x_hat_dec)
            psnr_dec = (
                -10.0 * math.log10(mse_dec.item()) if mse_dec.item() > 0 else 100.0
            )

    finally:
        # Always restore original training state
        model.train(prev_training)

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
    criterion.eval()

    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    loss_meter = AverageMeter()
    disp_meter = AverageMeter()
    dict_pen_meter = AverageMeter()
    gap_psnr: float = float("nan")

    # Sync entropy tables on all ranks before eval loop
    if gap_check:
        accelerator.unwrap_model(model).update(force=True)

    try:
        with torch.no_grad():
            for i, d in tqdm(
                enumerate(test_dataloader),
                total=len(test_dataloader),
                desc=f"Val Epoch {epoch}",
                disable=not accelerator.is_local_main_process,
            ):
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
                if "dispersion_bonus" in out_criterion:
                    disp_meter.update(
                        out_criterion["dispersion_bonus"].item(), d.size(0)
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
        gap_str = (
            f" | Train/Infer ΔPSNR: {gap_psnr:+.3f} dB"
            if not math.isnan(gap_psnr)
            else ""
        )
        if logger:
            logger.info(
                f"[Val] Epoch {epoch} | Loss: {loss_meter.avg:.4f} | PSNR: {psnr_meter.avg:.2f} dB | BPP: {bpp_meter.avg:.4f}{gap_str}"
            )
        if writer:
            writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
            writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
            writer.add_scalar("Val/Bpp", bpp_meter.avg, epoch)
            writer.add_scalar("Val/Dict_H", disp_meter.avg, epoch)
            writer.add_scalar("Val/Dict_Penalty", dict_pen_meter.avg, epoch)
            if not math.isnan(gap_psnr):
                writer.add_scalar("Val/TrainInfer_PSNR_gap", gap_psnr, epoch)

    return loss_meter.avg, gap_psnr


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
    p.add_argument("--disp-weight", type=float, default=0.1)
    p.add_argument(
        "--dict-penalty-weight",
        type=float,
        default=1.0,
        help="Fixed weight on the dictionary-token diversity penalty "
        "(off-diagonal cosine similarity, ≥0).  Bounded in [0,1] so a "
        "constant weight is sufficient — no EMA scaling needed.  "
        "Set to 0 to disable.",
    )
    p.add_argument("--ot-eps", type=float, default=0.1)
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
        disp_weight=args.disp_weight,
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
            criterion.load_state_dict(ckpt["criterion"])
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
