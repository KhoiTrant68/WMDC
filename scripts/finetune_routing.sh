#!/usr/bin/env bash
# scripts/finetune_routing.sh
# ===========================
# STE fine-tune một checkpoint đã trained-with-noise.
#
# CONTEXT — tại sao cần script này
# --------------------------------
# Lần train trước (`scripts/train_routing.sh`) chạy 400 epoch với
# `--last-epochs-with-ste 20` và `--lr-milestones 360 380`.  Hai sự kiện
# trùng nhau ở epoch 380:
#   • STE switch (use_ste=True)        → loss spike 0.585 → 0.887
#   • MultiStepLR drop thứ 2 (gamma=0.1) → LR=1e-6
# Hậu quả: STE phase không bao giờ giảm loss đủ để cập nhật best_loss,
# `checkpoint_best.pth.tar` vĩnh viễn ở epoch 379 — tức weights CHƯA HỀ
# được fine-tune cho hard quantization. Eval Kodak thấy bpp 0.30 PSNR 28.9
# với std/mean=56% chính là hậu quả.
#
# Script này:
#   1. Load checkpoint_best.pth.tar bằng cờ `--finetune` (chỉ lấy weights,
#      bỏ optimizer/scheduler/epoch state — xem train.py:1113).
#   2. Bật STE từ epoch 0 (--last-epochs-with-ste >= --epochs).
#   3. LR=5e-5 (cao hơn LR cuối phiên cũ 50× nhưng nửa LR gốc — đủ để di
#      chuyển weights mà không phá kiến trúc đã hội tụ).
#   4. Lưu vào thư mục MỚI để không ghi đè checkpoint gốc.
#
# Usage trên Kaggle:
#   bash scripts/finetune_routing.sh                       # mặc định 40 epoch
#   bash scripts/finetune_routing.sh --epochs 60           # tinh chỉnh số epoch
#   bash scripts/finetune_routing.sh --src /path/to/ckpt   # checkpoint khác
#
# Output:
#   $CHECKPOINT_ROOT/routing_finetune/<mode>/lambda_<lam>_mse/
#       checkpoint_best.pth.tar      ← deploy file mới
#       checkpoint_latest.pth.tar
#       train_*.log
#       training_summary.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# ── Fine-tune hyperparams (override-able qua CLI args) ───────────────────
MODE="unbalanced_eot"
LAMBDA=0.0036
FT_EPOCHS=40                  # 40 epoch là đủ; theo dõi loss & dừng sớm nếu cần
FT_LR=5e-5                    # 50× LR cuối phiên cũ (1e-6), 0.5× LR gốc (1e-4)
FT_LR_MILESTONES="25 35"      # 2 lần drop trong 40 epoch
FT_LR_GAMMA=0.3               # drop nhẹ hơn 0.1 cũ → giữ momentum tiến bộ
FT_GAP_CHECK_INTERVAL=2       # đo Train/Infer ΔPSNR mỗi 2 epoch để bắt sớm
FT_REAL_BYTES_BATCHES=10      # đo REAL BPP trên nhiều batch hơn để estimate ổn

# Nguồn checkpoint mặc định — sửa nếu file ở chỗ khác
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

# Recompute default src nếu --mode hoặc --lambda thay đổi và --src không set
if [ "$SRC_CKPT" = "$DEFAULT_SRC" ]; then
    SRC_CKPT="$CHECKPOINT_ROOT/routing/${MODE}/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar"
fi

# ── Sanity check ─────────────────────────────────────────────────────────
if [ ! -f "$SRC_CKPT" ]; then
    echo "ERROR: source checkpoint không tồn tại:" >&2
    echo "       $SRC_CKPT" >&2
    echo "Truyền --src <path> hoặc kiểm tra config.sh::CHECKPOINT_ROOT" >&2
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
echo "  Epochs            : $FT_EPOCHS  (toàn bộ với STE on)"
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
    $arch_flags

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Fine-tune done"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Best ckpt   : $SAVE_DIR/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar"
echo ""
echo "  Eval ngay với:"
echo "    python eval.py \\"
echo "      --dataset $KODAK_DIR \\"
echo "      --checkpoint $SAVE_DIR/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar \\"
echo "      --output $RESULTS_DIR/routing_finetune/$MODE \\"
echo "      --routing-mode $MODE \\"
echo "      --cuda --measure-dict-util"
echo ""
echo "  So với baseline (eval cũ), kỳ vọng:"
echo "    bpp     0.299  →  ~0.18-0.22"
echo "    PSNR    28.90  →  ~29.5-30.0 dB"
echo "    std/mean 56%   →  ~30-40%"
echo "═══════════════════════════════════════════════════════════════════"
