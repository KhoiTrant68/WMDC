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
# WARNING: these are SMOKE-TEST defaults so a 5-minute "did everything wire
# up?" run completes.  For the PAPER results use the commented values:
#
#     EPOCHS=400
#     BATCH_SIZE=8
#     LR_MILESTONES="360 380"
#     LAST_EPOCHS_STE=20
#     LAMBDAS_FULL=(0.0018 0.0036 0.0067 0.013 0.025 0.0483)   # full RD curve
#     LAMBDAS_ABLATION=(0.0036 0.013 0.0483)                    # 3-point ablation
#
# Training in earnest with the values below will produce nonsense numbers.
EPOCHS=6
BATCH_SIZE=4
LR_MILESTONES="1"
LAST_EPOCHS_STE=1

# ── Lambdas ──────────────────────────────────────────────────────────
# LAMBDAS_FULL=(0.0018 0.0036 0.0067 0.013 0.025 0.0483)   # full RD curve
LAMBDAS_FULL=(0.0036)   # full RD curve   ← smoke-test value, see WARNING above
# LAMBDAS_ABLATION=(0.0036 0.013 0.0483)                    # 3-point ablation
LAMBDAS_ABLATION=(0.0036)                    # ← smoke-test value


# ── Model defaults ───────────────────────────────────────────────────
ROUTING_MODE="unbalanced_eot"
ROUTING_MODES=(unbalanced_eot balanced_eot softmax)
CONTENT_ADAPTIVE=1 # set to 0 to disable ContentAdaptiveVSSBlock
CLUSTER_NUM=8
USE_WLS_SHORTCUT=1 # CMIC-style wavelet multi-scale shortcut (1=on, 0=off)
ORTHO_WEIGHT=0.01  # OLP orthogonality regulariser weight (0 to disable)

# ── Ablation variants ────────────────────────────────────────────────
# Original 7 (Table 4) + CMIC components + Phase-B8 cond. marginals.
#   no_wls                — disable WLS/iWLS shortcuts (uses __SKIP_WLS__ sentinel)
#   no_olp                — disable OLP orthogonality regulariser
#   no_ste_y              — disable STE on y in the last training epochs
#   no_col_entropy        — drop only −H_col (split of no_disp_bonus)
#   no_row_entropy        — drop only +H_row (split of no_disp_bonus)
#   no_alignment          — drop only the complexity-alignment hinge
#   no_alignment_margin   — keep alignment, set margin κ = 0 (sign-only)
#   full_cond_marg        — toggle ON multi-marginal OT (α = 0.5)
#   full_cond_marg_strong — multi-marginal OT with α = 0.9
VARIANTS=(
    full
    no_ueot no_fdm no_stateful_mem no_bootstrap_M1
    no_disp_bonus no_col_entropy no_row_entropy no_alignment no_alignment_margin
    no_dict_penalty no_wls no_olp no_ste_y
    full_cond_marg full_cond_marg_strong
)

# ── Tools ────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
# LAUNCHER="${LAUNCHER:-accelerate launch}"   # set LAUNCHER=python for single GPU
LAUNCHER="${LAUNCHER:-accelerate launch --multi_gpu --num_processes 2}" #Uncomment if use 2 GPUs

# ── Helper: extra flags per ablation variant ─────────────────────────
variant_flags() {
    case "$1" in
        full)                    echo "" ;;
        no_ueot)                 echo "--routing-mode softmax" ;;
        no_fdm)                  echo "--backbone ss2d" ;;
        no_stateful_mem)         echo "--use-dense-concat" ;;
        no_bootstrap_M1)         echo "--memory-init zero" ;;
        no_disp_bonus)           echo "--column-entropy-weight 0.0 --row-entropy-weight 0.0 --alignment-weight 0.0" ;;
        no_col_entropy)          echo "--column-entropy-weight 0.0" ;;
        no_row_entropy)          echo "--row-entropy-weight 0.0" ;;
        no_alignment)            echo "--alignment-weight 0.0" ;;
        no_alignment_margin)     echo "--alignment-margin 0.0" ;;
        no_dict_penalty)         echo "--dict-penalty-weight 0.0" ;;
        no_wls)                  echo "__SKIP_WLS__" ;;          # consumed by common_arch_flags
        no_olp)                  echo "--ortho-weight 0.0" ;;
        no_ste_y)                echo "--last-epochs-with-ste 0" ;;
        full_cond_marg)          echo "--use-conditional-marginals --cond-alpha 0.5" ;;
        full_cond_marg_strong)   echo "--use-conditional-marginals --cond-alpha 0.9" ;;
        *) echo "ERROR: unknown variant $1" >&2; return 1 ;;
    esac
}

# ── Helper: eval-time architecture flags per ablation variant ────────
# Returns only the flags that affect model structure at inference time.
# Used by eval.sh and run_ablation.sh to match the checkpoint's architecture.
#
# IMPORTANT: callers that hard-code a default --routing-mode (eval.sh,
# bd_rate.sh, run_ablation.sh::run_evaluate) must place the variant flags
# AFTER their hardcoded default so the variant override wins.  This is
# required for `no_ueot` whose checkpoint was trained with softmax.
variant_eval_flags() {
    case "$1" in
        no_ueot)                 echo "--routing-mode softmax" ;;
        no_fdm)                  echo "--backbone ss2d" ;;
        no_stateful_mem)         echo "--use-dense-concat" ;;
        no_bootstrap_M1)         echo "--memory-init zero" ;;
        no_wls)                  echo "__SKIP_WLS__" ;;
        full_cond_marg)          echo "--use-conditional-marginals --cond-alpha 0.5" ;;
        full_cond_marg_strong)   echo "--use-conditional-marginals --cond-alpha 0.9" ;;
        # Loss-only ablations (no_col_entropy, no_row_entropy, no_alignment,
        # no_alignment_margin, no_dict_penalty, no_olp, no_ste_y, no_disp_bonus)
        # do not change inference architecture — empty default is correct.
        *)                       echo "" ;;
    esac
}

# ── Helper: shared content-adaptive + WLS flags (gated by env) ───────
# Returns the flags every script should pass to train.py / eval.py /
# analyze scripts so model architecture matches the checkpoint.
#
# `__SKIP_WLS__` sentinel
# -----------------------
# `variant_flags` and `variant_eval_flags` echo the literal string
# `__SKIP_WLS__` for variants that need WLS disabled.  Callers must:
#   1. Pass that variant string in as `variant_extra` HERE, so the helper
#      knows not to append `--use-wls-shortcut`.
#   2. Strip the sentinel from the variant_extra they pass downstream
#      (the standard idiom is `extra_flags="${extra_flags//__SKIP_WLS__/}"`
#      right after they call `variant_*_flags`).
# Using a real flag would require an extra `--no-wls-shortcut` to be wired
# into train.py / eval.py / etc.; the sentinel keeps all the WLS plumbing
# in this file.
common_arch_flags() {
    local variant_extra="${1:-}"
    local out=""
    [ "${CONTENT_ADAPTIVE:-0}" -eq 1 ] && out="$out --content-adaptive --cluster-num ${CLUSTER_NUM:-8}"
    if [ "${USE_WLS_SHORTCUT:-0}" -eq 1 ] && [[ "$variant_extra" != *"__SKIP_WLS__"* ]]; then
        out="$out --use-wls-shortcut"
    fi
    echo "$out"
}

# ── Helper: checkpoint path for a given variant + lambda ─────────────
ckpt_path() {
    local dir="$1" lam="$2"
    echo "$CHECKPOINT_ROOT/$dir/lam_${lam}/lambda_${lam}_mse/checkpoint_best.pth.tar"
}
