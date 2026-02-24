import argparse
import logging
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.WMDC import WMDC

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.0)


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
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim_loss(a, b):
    return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.0).item())


def compute_bpp(out_net):
    size = out_net["x_hat"].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["likelihoods"].values()
    ).item()


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

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
            out["psnr"] = -10 * math.log10(out["mse_loss"].item())
            out["msssim"] = ms_ssim(
                torch.round(output["x_hat"] * 255),
                torch.round(target * 255),
                data_range=255,
                size_average=True,
            )
        else:
            out["ms_ssim_loss"] = compute_msssim_loss(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out["ms_ssim_loss"]) + out["bpp_loss"]

        return out


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

    logging.info("Logging file is %s" % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer."""
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
    type,
    accelerator,
):
    model.train()

    loop = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)

    for i, d in enumerate(loop):
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)

        if torch.isnan(out_criterion["loss"]):
            del out_net, out_criterion
            torch.cuda.empty_cache()
            continue

        accelerator.backward(out_criterion["loss"])

        if clip_max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()

        aux_loss = accelerator.unwrap_model(model).aux_loss()
        accelerator.backward(aux_loss)
        aux_optimizer.step()

        if i % 50 == 0 and accelerator.is_main_process:
            if type == "mse":
                logging.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.4f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                    f'\tPSNR: {out_criterion["psnr"]:.2f} |'
                    f'\tMS-SSIM: {out_criterion["msssim"].item():.4f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                logging.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )


def test_epoch(epoch, test_dataloader, model, criterion, type, accelerator):
    model.eval()
    p = 128

    total_loss, total_bpp, total_psnr, total_ssim = 0, 0, 0, 0
    count = 0

    with torch.no_grad():
        for d in test_dataloader:
            d_padded, padding = pad(d, p)
            out_net = model(d_padded)
            out_net["x_hat"].clamp_(0, 1)
            out_net["x_hat"] = crop(out_net["x_hat"], padding)

            if type == "mse":
                psnr_val = compute_psnr(d, out_net["x_hat"])
                ssim_val = compute_msssim(d, out_net["x_hat"])
                bpp_val = compute_bpp(out_net)

                metrics = torch.tensor(
                    [psnr_val, ssim_val, bpp_val], device=accelerator.device
                ).unsqueeze(0)
                metrics = accelerator.gather_for_metrics(metrics)

                total_psnr += metrics[:, 0].sum().item()
                total_ssim += metrics[:, 1].sum().item()
                total_bpp += metrics[:, 2].sum().item()
                count += metrics.shape[0]

            else:
                out_criterion = criterion(out_net, d)
                loss_val = out_criterion["loss"].item()
                ms_ssim_val = out_criterion["ms_ssim_loss"].item()
                bpp_val = out_criterion["bpp_loss"].item()

                metrics = torch.tensor(
                    [loss_val, ms_ssim_val, bpp_val], device=accelerator.device
                ).unsqueeze(0)
                metrics = accelerator.gather_for_metrics(metrics)

                total_loss += metrics[:, 0].sum().item()
                total_ssim += metrics[:, 1].sum().item()
                total_bpp += metrics[:, 2].sum().item()
                count += metrics.shape[0]

    # Only log on the main process
    if accelerator.is_main_process:
        if type == "mse":
            logging.info(
                f"Test epoch {epoch}: Average losses:"
                f"\tBpp loss: {total_bpp/count:.3f} |"
                f"\tPSNR: {total_psnr/count:.2f} |"
                f"\tMS-SSIM: {total_ssim/count:.4f} |"
            )
            return total_bpp / count
        else:
            logging.info(
                f"Test epoch {epoch}: Average losses:"
                f"\tLoss: {total_loss/count:.3f} |"
                f"\tMS_SSIM loss: {total_ssim/count:.3f} |"
                f"\tBpp loss: {total_bpp/count:.2f} |"
            )
            return total_loss / count

    return float("inf")


def save_checkpoint(state, is_best, epoch, save_path):
    torch.save(state, os.path.join(save_path, "checkpoint_latest.pth.tar"))
    if is_best:
        torch.save(state, os.path.join(save_path, "checkpoint_best.pth.tar"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m", "--model", choices=models.keys(), help="Model architecture"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument(
        "-n", "--num-workers", type=int, default=8
    )  # Lowered default for safety in DDP
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0018,
        help="Bit-rate distortion trade-off parameter",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--type", type=str, default="mse", choices=["mse", "ms-ssim"])
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--skip_epoch", type=int, default=0)
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--lr_epoch", nargs="+", type=int, default=[35, 45])
    parser.add_argument("--continue_train", action="store_true", default=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    accelerator = Accelerator()

    if args.seed is not None:
        set_seed(args.seed)

    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))

    # Setup Logger only on main process
    if accelerator.is_main_process:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        setup_logger(os.path.join(save_path, time.strftime("%Y%m%d_%H%M%S") + ".log"))

        for arg in vars(args):
            print(arg, ":", getattr(args, arg))
        for k in args.__dict__:
            logging.info(k + ":" + str(args.__dict__[k]))

    # Datasets and Transforms
    train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Model Init
    net = WMDC(N=args.N, M=args.M, num_slices=5).to(accelerator.device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=0.1, last_epoch=-1
    )

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0

    # Load checkpoint BEFORE accelerator.prepare
    if args.checkpoint:
        if accelerator.is_main_process:
            print("Loading", args.checkpoint)

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        net.load_state_dict(checkpoint["state_dict"])

        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # Prepare everything with accelerator
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

    best_loss = float("inf")

    for epoch in range(last_epoch, args.epochs):
        if accelerator.is_main_process:
            logging.info("====== Current epoch %s ======" % epoch)
            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            type,
            accelerator,
        )

        loss = test_epoch(epoch, test_dataloader, net, criterion, type, accelerator)
        lr_scheduler.step()

        # Checkpoint Saving (Only executed by the main process)
        accelerator.wait_for_everyone()  # sync before saving

        if accelerator.is_main_process:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                # Unwrap model to strip DDP wrappers before saving
                unwrapped_model = accelerator.unwrap_model(net)

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": unwrapped_model.state_dict(),
                        "loss": loss,
                        "optimizer": (
                            optimizer.optimizer.state_dict()
                            if hasattr(optimizer, "optimizer")
                            else optimizer.state_dict()
                        ),
                        "aux_optimizer": (
                            aux_optimizer.optimizer.state_dict()
                            if hasattr(aux_optimizer, "optimizer")
                            else aux_optimizer.state_dict()
                        ),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    epoch,
                    save_path,
                )


if __name__ == "__main__":
    main(sys.argv[1:])
