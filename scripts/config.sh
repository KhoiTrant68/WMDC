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
EPOCHS=400
BATCH_SIZE=16
LR_MILESTONES="360 380"
LAST_EPOCHS_STE=20

# ── Lambdas ──────────────────────────────────────────────────────────
# LAMBDAS_FULL=(0.0018 0.0036 0.0067 0.013 0.025 0.0483)   # full RD curve
LAMBDAS_FULL=(0.0036)   # full RD curve
# LAMBDAS_ABLATION=(0.0036 0.013 0.0483)                    # 3-point ablation
LAMBDAS_ABLATION=(0.0036)                    # 3-point ablation


# ── Model defaults ───────────────────────────────────────────────────
ROUTING_MODE="unbalanced_eot"
ROUTING_MODES=(unbalanced_eot balanced_eot softmax)

# ── Ablation variants ────────────────────────────────────────────────
VARIANTS=(full no_ueot no_fdm no_stateful_mem no_bootstrap_M1 no_disp_bonus no_dict_penalty)

# ── Tools ────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
LAUNCHER="${LAUNCHER:-accelerate launch}"   # set LAUNCHER=python for single GPU
# LAUNCHER="${LAUNCHER:-accelerate launch --multi_gpu --num_processes 2}" #Uncomment if use 2 GPUs

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
# Returns only the flags that affect model structure at inference time.
# Used by eval.sh and run_ablation.sh to match the checkpoint's architecture.
variant_eval_flags() {
    case "$1" in
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
