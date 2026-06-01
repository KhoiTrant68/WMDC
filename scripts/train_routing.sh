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
        unbalanced_eot|balanced_eot|softmax) MODES_TO_RUN=("$arg") ;;
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

    local arch_flags; arch_flags="$(common_arch_flags)"

    # Entropy weights overridden for the adaptive-eps patch
    # (modules/dictionary_blocks.py: _adaptive_eps + log_adaptive_range).
    #
    # History
    # -------
    #   v1 (pre-patch defaults col=0.1, row=0.3) → dict collapse, util 26 %,
    #     5/24 catastrophic noise images.
    #   v2 (col=0.3, row=0.0 + adaptive eps) → no noise, but routing
    #     OVER-corrected to fully uniform: H_col = 7 bits (util 100 %),
    #     BPP +19 %.  Cold-start cycle: cold dict → small margins →
    #     adaptive eps maxes out soft → weak gradient to differentiate
    #     atoms → atoms stay similar.
    #   v3 (current) col=0.15, row=0.05 + adaptive eps:
    #     β_row = 0.05 is a SMALL global sharpness baseline that
    #       bootstraps atom differentiation during early training;
    #       adaptive eps then modulates per-pixel around this baseline.
    #     β_col = 0.15 still pushes atom dispersion but not so hard the
    #       loss demands fully uniform H_col.
    #     Hypothesis: the right operating point is β_col mildly > β_row,
    #       both nonzero — the v2 "row=0" failure shows adaptive eps
    #       CANNOT entirely replace β_row in cold-start.
    #
    # Adaptive-eps cụm 1–3 patches:
    #   --slice-coherence-weight 0.01 : B3 — per-slice k_dict orthogonality.
    #                                   Direct counter to the 2-cluster
    #                                   degeneracy at slices 1, 2.
    #   --revive-every-n-steps 500    : B2 — dead-atom revival.  Atoms
    #                                   below 5 % of fair share in EVERY
    #                                   slice get re-init from a live row.
    #   --use-adaptive-eps            : C2 — per-pixel ε with entropy-
    #                                   based ambiguity, fixed CLARITY_REF.
    #   --image-conditional-range     : C1 — per-image bias on
    #                                   log_adaptive_range from pooled hp.
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
        --ortho-weight "${ORTHO_WEIGHT:-0.01}" \
        --column-entropy-weight 0.3 \
        --row-entropy-weight 0.1 \
        --slice-coherence-weight 0.01 \
        --revive-every-n-steps 500 \
        --use-adaptive-eps \
        --image-conditional-range \
        $arch_flags \
        $resume_flag

    echo "[done] $mode → $save_dir"
}

for mode in "${MODES_TO_RUN[@]}"; do
    train_mode "$mode"
done

echo ""
echo "=== All routing training done ==="
