#!/usr/bin/env bash
# scripts/config.sh
# =================
# Shared configuration — sourced by all other scripts.
# Edit the paths here; everything else picks them up automatically.

# ── Paths (edit me) ─────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DATA="/kaggle/input/datasets/tranjohan/data-1000-test/dataset_1000_test"
KODAK_DIR="/kaggle/input/datasets/khitrnminh/kodak-test"
CHECKPOINT_ROOT="$REPO/wmdc_runs"
RESULTS_DIR="$REPO/results"

# ── Training hyperparams ─────────────────────────────────────────────
EPOCHS=2
BATCH_SIZE=4
LR_MILESTONES="1"
LAST_EPOCHS_STE=1

# ── Lambdas ──────────────────────────────────────────────────────────
# LAMBDAS_FULL=(0.0018 0.0036 0.0067 0.013 0.025 0.0483)   # full RD curve
LAMBDAS_FULL=(0.0036)   # full RD curve
# LAMBDAS_ABLATION=(0.0036 0.013 0.0483)                    # 3-point ablation
LAMBDAS_ABLATION=(0.0036)                    # 3-point ablation


# ── Model defaults ───────────────────────────────────────────────────
ROUTING_MODE="unbalanced_eot"
ROUTING_MODES=(unbalanced_eot balanced_eot softmax)
CONTENT_ADAPTIVE=1 # set to 0 to disable ContentAdaptiveVSSBlock
CLUSTER_NUM=8

# Derived CLI flags for eval/visualize scripts. Empty if content-adaptive
# is disabled. Must match the value used during training so that
# load_state_dict sees the same parameter shapes / buffer keys.
CA_FLAGS=""
[ "${CONTENT_ADAPTIVE:-0}" -eq 1 ] && CA_FLAGS="--content-adaptive --cluster-num ${CLUSTER_NUM:-8}"

# ── Ablation variants ────────────────────────────────────────────────
VARIANTS=(full no_ueot no_fdm no_stateful_mem no_bootstrap_M1 no_disp_bonus no_dict_penalty)

# ── Tools ────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"

# Auto-detect GPU count → LAUNCHER.
#   • LAUNCHER env var (if set) wins.
#   • else: 0 GPUs → "python"; 1 GPU → "accelerate launch"; ≥2 GPUs → multi_gpu.
# Override examples:
#   LAUNCHER=python bash scripts/train_routing.sh                # single GPU / CPU
#   LAUNCHER="accelerate launch --multi_gpu --num_processes 4" …  # 4-GPU
if [ -z "${LAUNCHER:-}" ]; then
    NUM_GPUS=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    case "$NUM_GPUS" in
        0) LAUNCHER="$PYTHON" ;;
        1) LAUNCHER="accelerate launch" ;;
        *) LAUNCHER="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}" ;;
    esac
fi
export LAUNCHER

# ── Helper: extra flags per ablation variant ─────────────────────────
variant_flags() {
    case "$1" in
        full)             echo "" ;;
        no_ueot)          echo "--routing-mode softmax" ;;
        no_fdm)           echo "--backbone ss2d" ;;
        no_stateful_mem)  echo "--use-dense-concat" ;;
        no_bootstrap_M1)  echo "--memory-init zero" ;;
        no_disp_bonus)    echo "--column-entropy-weight 0.0 --row-entropy-weight 0.0 --alignment-weight 0.0" ;;
        no_dict_penalty)  echo "--dict-penalty-weight 0.0" ;;
        *) echo "ERROR: unknown variant $1" >&2; return 1 ;;
    esac
}

# ── Helper: eval-time architecture flags per ablation variant ────────
# Returns the flags that affect model structure / param set at inference
# time.  These MUST match the corresponding training flags or
# load_state_dict(strict=False) will skip params silently (e.g. loading
# a softmax-trained checkpoint into an unbalanced_eot model leaves
# rho_predictors at random init — eval is then meaningless).
variant_eval_flags() {
    case "$1" in
        no_ueot)         echo "--routing-mode softmax" ;;
        no_fdm)          echo "--backbone ss2d" ;;
        no_stateful_mem) echo "--use-dense-concat" ;;
        no_bootstrap_M1) echo "--memory-init zero" ;;
        *)               echo "" ;;
    esac
}

# ── Helper: checkpoint path for a given variant + lambda ─────────────
ckpt_path() {
    local dir="$1" lam="$2"
    echo "$CHECKPOINT_ROOT/$dir/lam_${lam}/lambda_${lam}_mse/checkpoint_best.pth.tar"
}
