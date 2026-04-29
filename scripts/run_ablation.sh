#!/usr/bin/env bash
# scripts/run_ablation.sh
# =======================
# Launches all training jobs for the component-removal ablation (Table 4)
# and the routing ablation (Table 1). Designed for a SLURM cluster but
# works on a single multi-GPU node (just remove `sbatch` and the SBATCH
# directives below).
#
# Usage:
#   ./scripts/run_ablation.sh launch        # submit all jobs
#   ./scripts/run_ablation.sh dry-run       # print commands only
#   ./scripts/run_ablation.sh evaluate      # run evaluation on completed checkpoints
#
# Configuration:
#   See `configs/ablation_variants.py` for variant definitions.
#   Edit DATA_DIR, CHECKPOINT_ROOT, WANDB_ENTITY at the top of this file.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

SBATCH="${SBATCH:-sbatch}"   # set SBATCH=echo for dry-run

# Local aliases for names used in the heredoc below
DATA_DIR="$TRAIN_DATA"
LAMBDAS=("${LAMBDAS_ABLATION[@]}")

mkdir -p "$CHECKPOINT_ROOT"

# ── Submit one training job ─────────────────────────────────────────
submit_train_job() {
    local variant="$1"
    local lam="$2"
    local out_dir="$CHECKPOINT_ROOT/${variant}/lam_${lam}"
    local extra_flags
    extra_flags="$(variant_flags "$variant")"

    mkdir -p "$out_dir"

    cat > "$out_dir/launch.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=wmdc_${variant}_lam${lam}
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=$out_dir/slurm-%j.out

set -euo pipefail
source ~/.bashrc

cd \$SLURM_SUBMIT_DIR
$PYTHON train.py \\
    --dataset "$DATA_DIR" \\
    --epochs $EPOCHS \\
    --batch-size $BATCH_SIZE \\
    --lambda $lam \\
    --metric mse \\
    --save_path "$out_dir" \\
    $extra_flags
EOF

    echo "[submit] $variant @ λ=$lam → $out_dir"
    if [ "$SBATCH" = "echo" ]; then
        cat "$out_dir/launch.sbatch"
    else
        $SBATCH "$out_dir/launch.sbatch"
    fi
}

# ── Submit one evaluation job ───────────────────────────────────────
submit_eval_jobs() {
    # Per-variant, build the list of checkpoints (one per lambda) and
    # call compute_bd_rate.py evaluate.
    for variant in "${VARIANTS[@]}"; do
        local rd_json="$CHECKPOINT_ROOT/${variant}/rd_kodak.json"
        local ckpts=()
        for lam in "${LAMBDAS[@]}"; do
            local ck="$CHECKPOINT_ROOT/${variant}/lam_${lam}/checkpoint_best.pth.tar"
            if [ -f "$ck" ]; then
                ckpts+=("$ck")
            else
                echo "[skip-eval] missing $ck"
            fi
        done
        if [ "${#ckpts[@]}" -eq 0 ]; then
            echo "[no-eval] $variant: no checkpoints"
            continue
        fi
        echo "[evaluate] $variant → $rd_json"
        $PYTHON analyze/compute_bd_rate.py evaluate \
            --variant-name "$variant" \
            --checkpoints "${ckpts[@]}" \
            --dataset "$KODAK_DIR" \
            --output "$rd_json" \
            --cuda
    done

    # Aggregate BD-rate of all variants vs. the "full" variant as anchor.
    local anchor="$CHECKPOINT_ROOT/full/rd_kodak.json"
    if [ ! -f "$anchor" ]; then
        echo "[ERR] anchor RD JSON not found at $anchor; cannot compute BD-rate."
        return 1
    fi
    local var_jsons=()
    for variant in "${VARIANTS[@]}"; do
        if [ "$variant" != "full" ]; then
            local rd="$CHECKPOINT_ROOT/${variant}/rd_kodak.json"
            [ -f "$rd" ] && var_jsons+=("$rd")
        fi
    done
    $PYTHON analyze/compute_bd_rate.py bd-rate \
        --anchor-json "$anchor" \
        --variant-jsons "${var_jsons[@]}" \
        --output "$CHECKPOINT_ROOT/bd_rate_table.json"
    echo "[done] BD-rate table → $CHECKPOINT_ROOT/bd_rate_table.json"
}

# ── Main dispatcher ─────────────────────────────────────────────────
case "${1:-}" in
    launch)
        for variant in "${VARIANTS[@]}"; do
            for lam in "${LAMBDAS[@]}"; do
                submit_train_job "$variant" "$lam"
            done
        done
        ;;
    dry-run)
        SBATCH=echo
        for variant in "${VARIANTS[@]}"; do
            for lam in "${LAMBDAS[@]}"; do
                submit_train_job "$variant" "$lam"
            done
        done
        ;;
    evaluate)
        submit_eval_jobs
        ;;
    *)
        echo "Usage: $0 {launch|dry-run|evaluate}"
        echo ""
        echo "  launch     — submit all training jobs to the cluster"
        echo "  dry-run    — print the sbatch scripts without submitting"
        echo "  evaluate   — evaluate completed checkpoints + BD-rate table"
        exit 1
        ;;
esac
