#!/usr/bin/env bash
# scripts/config.sh
# =================
# Shared configuration — sourced by all other scripts.
# Edit the paths here; everything else picks them up automatically.

# ── Paths (edit me) ─────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DATA="/kaggle/input/datasets/tranjohan/data-1000-test/dataset_1000_test"
KODAK_DIR="/kaggle/input/datasets/khitrnminh/kodak-test"
CHECKPOINT_ROOT="$HOME/wmdc_runs"
RESULTS_DIR="$REPO/results"

# ── Training hyperparams ─────────────────────────────────────────────
EPOCHS=2
BATCH_SIZE=4
LR_MILESTONES="1"
LAST_EPOCHS_STE=0

# ── Lambdas ──────────────────────────────────────────────────────────
# LAMBDAS_FULL=(0.0018 0.0036 0.0067 0.013 0.025 0.0483)   # full RD curve
LAMBDAS_FULL=(0.0036)   # full RD curve
# LAMBDAS_ABLATION=(0.0036 0.013 0.0483)                    # 3-point ablation
LAMBDAS_ABLATION=(0.0036)                    # 3-point ablation


# ── Model defaults ───────────────────────────────────────────────────
ROUTING_MODE="unbalanced_eot"
ROUTING_MODES=(softmax balanced_eot unbalanced_eot)

# ── Ablation variants ────────────────────────────────────────────────
VARIANTS=(full no_ueot no_fdm no_stateful_mem no_bootstrap_M1 no_disp_bonus no_dict_penalty)

# ── Tools ────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
LAUNCHER="${LAUNCHER:-accelerate launch}"   # set LAUNCHER=python for single GPU

# ── Helper: extra flags per ablation variant ─────────────────────────
variant_flags() {
    case "$1" in
        full)             echo "" ;;
        no_ueot)          echo "--routing-mode softmax" ;;
        no_fdm)           echo "--backbone ss2d" ;;
        no_stateful_mem)  echo "--use-dense-concat" ;;
        no_bootstrap_M1)  echo "--memory-init zero" ;;
        no_disp_bonus)    echo "--disp-weight 0.0" ;;
        no_dict_penalty)  echo "--dict-penalty-weight 0.0" ;;
        *) echo "ERROR: unknown variant $1" >&2; return 1 ;;
    esac
}

# ── Helper: checkpoint path for a given variant + lambda ─────────────
ckpt_path() {
    local dir="$1" lam="$2"
    echo "$CHECKPOINT_ROOT/$dir/lam_${lam}/lambda_${lam}_mse/checkpoint_best.pth.tar"
}
