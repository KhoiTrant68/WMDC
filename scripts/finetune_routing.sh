#!/usr/bin/env bash
# scripts/finetune_routing.sh
# ===========================
# STE fine-tune of a noise-trained checkpoint.
#
# CONTEXT — why this script exists
# --------------------------------
# The previous training run (`scripts/train_routing.sh`) ran 400 epochs with
# `--last-epochs-with-ste 20` and `--lr-milestones 360 380`.  Two events
# collide at epoch 380:
#   • STE switch (use_ste=True)              → loss spike 0.585 → 0.887
#   • 2nd MultiStepLR drop (gamma=0.1)       → LR=1e-6
# Consequence: the STE phase never reduces loss enough to update best_loss,
# so `checkpoint_best.pth.tar` stays frozen at epoch 379 — meaning the
# deployed weights were NEVER fine-tuned for hard quantization.  The Kodak
# eval result (bpp 0.30, PSNR 28.9, std/mean=56%) is a direct consequence.
#
# This script:
#   1. Loads checkpoint_best.pth.tar with the `--finetune` flag (weights only,
#      no optimizer/scheduler/epoch state — see train.py).
#   2. Enables STE from epoch 0 (--last-epochs-with-ste >= --epochs).
#   3. LR=5e-5 (50× higher than the previous run's final LR but half the
#      original LR — enough to move weights without breaking the converged
#      architecture).
#   4. Writes to a NEW directory so the original checkpoints are preserved.
#
# NOTE on the dict-util fix (2026-05-31):
# ---------------------------------------
# After the cost-matrix / eps / entropy-weight refactor, the saved
# `log_eps` parameter and the cost-matrix scale have CHANGED MEANING.
# Pre-fix checkpoints will load but the eps mapped from the old `log_eps`
# is not what the old run was using — this is by design, but it means
# fine-tuning a pre-fix checkpoint is closer to training-from-warm-init
# than to a faithful resume.  Expect 10-20 epochs of re-stabilisation
# before metrics resemble the old run's trajectory.  Also: --eps-warmup
# is set to 0 below since the saved features are not "random" any more.
#
# Usage on Kaggle:
#   bash scripts/finetune_routing.sh                       # default 40 epochs
#   bash scripts/finetune_routing.sh --epochs 60           # tune epoch count
#   bash scripts/finetune_routing.sh --src /path/to/ckpt   # alternate checkpoint
#
# Output:
#   $CHECKPOINT_ROOT/routing_finetune/<mode>/lambda_<lam>_mse/
#       checkpoint_best.pth.tar      ← new deploy file
#       checkpoint_latest.pth.tar
#       train_*.log
#       training_summary.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# ── Fine-tune hyperparams (CLI-overridable) ──────────────────────────────
MODE="unbalanced_eot"
LAMBDA=0.0036
FT_EPOCHS=40                  # 40 epochs is enough; monitor loss and stop early if needed
FT_LR=5e-5                    # 50× the previous final LR (1e-6), 0.5× the original LR (1e-4)
FT_LR_MILESTONES="25 35"      # 2 LR drops within 40 epochs
FT_LR_GAMMA=0.3               # softer than the original 0.1 → preserves momentum
FT_GAP_CHECK_INTERVAL=2       # measure Train/Infer ΔPSNR every 2 epochs for early detection
FT_REAL_BYTES_BATCHES=10      # measure REAL BPP on more batches for a stable estimate

# Default source checkpoint — change if the file lives elsewhere
DEFAULT_SRC="$CHECKPOINT_ROOT/routing/${MODE}/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar"
SRC_CKPT="$DEFAULT_SRC"

# ── Parse args ───────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --epochs)     FT_EPOCHS="$2"; shift 2 ;;
        --lr)         FT_LR="$2"; shift 2 ;;
        --src)        SRC_CKPT="$2"; shift 2 ;;
        --mode)       MODE="$2"; shift 2 ;;
        --lambda)     LAMBDA="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Recompute default src if --mode or --lambda changed and --src was not set
if [ "$SRC_CKPT" = "$DEFAULT_SRC" ]; then
    SRC_CKPT="$CHECKPOINT_ROOT/routing/${MODE}/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar"
fi

# ── Sanity check ─────────────────────────────────────────────────────────
if [ ! -f "$SRC_CKPT" ]; then
    echo "ERROR: source checkpoint not found:" >&2
    echo "       $SRC_CKPT" >&2
    echo "Pass --src <path> or check config.sh::CHECKPOINT_ROOT" >&2
    exit 1
fi

SAVE_DIR="$CHECKPOINT_ROOT/routing_finetune/$MODE"
mkdir -p "$SAVE_DIR"
cd "$REPO"

# ── Banner ───────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  STE FINE-TUNE"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Mode              : $MODE"
echo "  Lambda            : $LAMBDA"
echo "  Source ckpt       : $SRC_CKPT"
echo "  Save dir          : $SAVE_DIR"
echo "  Epochs            : $FT_EPOCHS  (all with STE on)"
echo "  LR                : $FT_LR"
echo "  LR milestones     : $FT_LR_MILESTONES  (gamma=$FT_LR_GAMMA)"
echo "  Gap check         : every $FT_GAP_CHECK_INTERVAL epochs"
echo "  Real-bytes batches: $FT_REAL_BYTES_BATCHES (val)"
echo "═══════════════════════════════════════════════════════════════════"

arch_flags="$(common_arch_flags)"

# shellcheck disable=SC2086
$LAUNCHER train.py \
    -d "$TRAIN_DATA" \
    --save_path "$SAVE_DIR" \
    --checkpoint "$SRC_CKPT" \
    --finetune \
    --lambda "$LAMBDA" \
    --metric mse \
    --routing-mode "$MODE" \
    --epochs "$FT_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$FT_LR" \
    --lr-milestones $FT_LR_MILESTONES \
    --lr-gamma "$FT_LR_GAMMA" \
    --last-epochs-with-ste "$FT_EPOCHS" \
    --gap-check-interval "$FT_GAP_CHECK_INTERVAL" \
    --real-bytes-batches "$FT_REAL_BYTES_BATCHES" \
    --ortho-weight "${ORTHO_WEIGHT:-0.01}" \
    --eps-warmup-epochs 0 \
    $arch_flags

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Fine-tune done"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Best ckpt   : $SAVE_DIR/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar"
echo ""
echo "  Eval now with:"
echo "    python eval.py \\"
echo "      --dataset $KODAK_DIR \\"
echo "      --checkpoint $SAVE_DIR/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar \\"
echo "      --output $RESULTS_DIR/routing_finetune/$MODE \\"
echo "      --routing-mode $MODE \\"
echo "      --cuda --measure-dict-util"
echo ""
echo "  Expected vs. baseline eval:"
echo "    bpp     0.299  →  ~0.18-0.22"
echo "    PSNR    28.90  →  ~29.5-30.0 dB"
echo "    std/mean 56%   →  ~30-40%"
echo "═══════════════════════════════════════════════════════════════════"
