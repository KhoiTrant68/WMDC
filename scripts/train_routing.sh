#!/usr/bin/env bash
# scripts/train_routing.sh
# ========================
# Train the 3 routing-mode variants (Table 1): softmax, balanced_eot,
# unbalanced_eot. All other hyperparams are held at their default values.
#
# Usage:
#   bash scripts/train_routing.sh                    # train all 3 modes
#   bash scripts/train_routing.sh unbalanced_eot     # train one mode only
#   bash scripts/train_routing.sh --resume           # resume from latest checkpoint
#
# Output:
#   $CHECKPOINT_ROOT/routing/<mode>/lambda_<lam>_mse/
#       checkpoint_best.pth.tar
#       checkpoint_latest.pth.tar
#       train_*.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

LAMBDA=0.0036   # single lambda for routing comparison
RESUME=0

# Parse args
MODES_TO_RUN=("${ROUTING_MODES[@]}")
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME=1 ;;
        softmax|balanced_eot|unbalanced_eot) MODES_TO_RUN=("$arg") ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

train_mode() {
    local mode="$1"
    local save_dir="$CHECKPOINT_ROOT/routing/$mode"
    local latest="$save_dir/lambda_${LAMBDA}_mse/checkpoint_latest.pth.tar"

    echo ""
    echo "=== Training: routing-mode=$mode  λ=$LAMBDA ==="

    local resume_flag=""
    if [ "$RESUME" -eq 1 ] && [ -f "$latest" ]; then
        echo "[resume] loading $latest"
        resume_flag="--checkpoint $latest"
    fi

    mkdir -p "$save_dir"
    cd "$REPO"

    # shellcheck disable=SC2086
    $LAUNCHER train.py \
        -d "$TRAIN_DATA" \
        --save_path "$save_dir" \
        --lambda "$LAMBDA" \
        --metric mse \
        --routing-mode "$mode" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr-milestones $LR_MILESTONES \
        --last-epochs-with-ste "$LAST_EPOCHS_STE" \
        $resume_flag

    echo "[done] $mode → $save_dir"
}

for mode in "${MODES_TO_RUN[@]}"; do
    train_mode "$mode"
done

echo ""
echo "=== All routing training done ==="
