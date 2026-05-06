#!/usr/bin/env bash
# scripts/run_ablation.sh
# =======================
# Component-removal ablation (Table 4): train & evaluate 7 variants.
#
# Usage:
#   bash scripts/run_ablation.sh train        # train all variants sequentially (Kaggle/single-node)
#   bash scripts/run_ablation.sh train full   # train one variant only
#   bash scripts/run_ablation.sh dry-run      # print commands without running
#   bash scripts/run_ablation.sh launch       # submit to SLURM cluster
#   bash scripts/run_ablation.sh evaluate     # eval checkpoints + BD-rate table
#
# Configuration: edit scripts/config.sh (paths, epochs, lambdas).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

SBATCH="${SBATCH:-sbatch}"

# ── Core training function (used by both train and launch) ───────────
run_train_job() {
    local variant="$1"
    local lam="$2"
    local dry="${3:-0}"
    local extra_flags
    extra_flags="$(variant_flags "$variant")"
    local out_dir="$CHECKPOINT_ROOT/ablation/$variant/lam_$lam"

    mkdir -p "$out_dir"

    # Build the command
    local cmd="$LAUNCHER train.py \
        -d \"$TRAIN_DATA\" \
        --save_path \"$out_dir\" \
        --lambda $lam \
        --metric mse \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr-milestones $LR_MILESTONES \
        --last-epochs-with-ste $LAST_EPOCHS_STE"

    [ -n "$extra_flags" ] && cmd="$cmd $extra_flags"

    if [ "$dry" -eq 1 ]; then
        echo "[dry-run] $variant @ λ=$lam"
        echo "  $cmd"
        echo ""
        return
    fi

    echo ""
    echo "=== Training: $variant  λ=$lam ==="
    echo "    save_path: $out_dir"
    [ -n "$extra_flags" ] && echo "    extra: $extra_flags"

    # Resume if latest checkpoint exists
    local latest="$out_dir/lambda_${lam}_mse/checkpoint_latest.pth.tar"
    if [ -f "$latest" ]; then
        echo "    [resume] $latest"
        cmd="$cmd --checkpoint \"$latest\""
    fi

    eval "$cmd"
    echo "[done] $variant @ λ=$lam"
}

# ── SLURM submission (cluster) ───────────────────────────────────────
submit_slurm_job() {
    local variant="$1"
    local lam="$2"
    local extra_flags
    extra_flags="$(variant_flags "$variant")"
    local out_dir="$CHECKPOINT_ROOT/ablation/$variant/lam_$lam"

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
cd "\$SLURM_SUBMIT_DIR"

$LAUNCHER train.py \\
    -d "$TRAIN_DATA" \\
    --save_path "$out_dir" \\
    --lambda $lam \\
    --metric mse \\
    --epochs $EPOCHS \\
    --batch-size $BATCH_SIZE \\
    --lr-milestones $LR_MILESTONES \\
    --last-epochs-with-ste $LAST_EPOCHS_STE \\
    $extra_flags
EOF

    echo "[submit] $variant @ λ=$lam → $out_dir"
    $SBATCH "$out_dir/launch.sbatch"
}

# ── Evaluate all variants → RD JSONs + BD-rate table ────────────────
run_evaluate() {
    echo ""
    echo "=== Evaluating ablation variants on Kodak ==="
    cd "$REPO"
    mkdir -p "$RESULTS_DIR/bd_rate"

    for variant in "${VARIANTS[@]}"; do
        local ckpts=()
        for lam in "${LAMBDAS_ABLATION[@]}"; do
            local ck; ck="$(ckpt_path "ablation/$variant" "$lam")"
            if [ -f "$ck" ]; then
                ckpts+=("$ck")
            else
                echo "[skip] missing $ck"
            fi
        done

        if [ "${#ckpts[@]}" -eq 0 ]; then
            echo "[no-eval] $variant: no checkpoints found"
            continue
        fi

        local out="$RESULTS_DIR/bd_rate/${variant}.json"
        local eval_flags; eval_flags="$(variant_eval_flags "$variant")"
        echo ""
        echo "-- $variant (${#ckpts[@]} checkpoints)"

        # shellcheck disable=SC2086  # word-split intentional for flags
        $PYTHON analyze/compute_bd_rate.py evaluate \
            --variant-name "$variant" \
            --checkpoints "${ckpts[@]}" \
            --dataset "$KODAK_DIR" \
            --output "$out" \
            --routing-mode unbalanced_eot \
            $eval_flags \
            --cuda

        echo "[done] $variant → $out"
    done

    # BD-rate table vs. "full" anchor
    local anchor="$RESULTS_DIR/bd_rate/full.json"
    if [ ! -f "$anchor" ]; then
        echo "[ERR] anchor JSON not found: $anchor (run 'evaluate' after training 'full')"
        return 1
    fi

    local var_jsons=()
    for variant in "${VARIANTS[@]}"; do
        [ "$variant" = "full" ] && continue
        local f="$RESULTS_DIR/bd_rate/${variant}.json"
        [ -f "$f" ] && var_jsons+=("$f")
    done

    local table="$RESULTS_DIR/bd_rate/bd_rate_table.json"
    $PYTHON analyze/compute_bd_rate.py bd-rate \
        --anchor-json "$anchor" \
        --variant-jsons "${var_jsons[@]}" \
        --output "$table" \
        --method akima

    echo ""
    echo "[done] BD-rate table → $table"
}

# ── Main dispatcher ──────────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "$CMD" in
    train)
        # Optional: pass a single variant name as argument
        TARGET="${1:-all}"
        for variant in "${VARIANTS[@]}"; do
            [ "$TARGET" != "all" ] && [ "$TARGET" != "$variant" ] && continue
            for lam in "${LAMBDAS_ABLATION[@]}"; do
                run_train_job "$variant" "$lam" 0
            done
        done
        echo ""
        echo "=== All ablation training done ==="
        ;;

    dry-run)
        TARGET="${1:-all}"
        for variant in "${VARIANTS[@]}"; do
            [ "$TARGET" != "all" ] && [ "$TARGET" != "$variant" ] && continue
            for lam in "${LAMBDAS_ABLATION[@]}"; do
                run_train_job "$variant" "$lam" 1
            done
        done
        ;;

    launch)
        for variant in "${VARIANTS[@]}"; do
            for lam in "${LAMBDAS_ABLATION[@]}"; do
                submit_slurm_job "$variant" "$lam"
            done
        done
        ;;

    evaluate)
        run_evaluate
        ;;

    *)
        echo "Usage: $0 {train|dry-run|launch|evaluate} [variant]"
        echo ""
        echo "  train [variant]  — train all (or one) variant sequentially  ← Kaggle"
        echo "  dry-run          — print commands without running"
        echo "  launch           — submit all jobs to SLURM cluster"
        echo "  evaluate         — eval checkpoints on Kodak + BD-rate table"
        echo ""
        echo "Variants: ${VARIANTS[*]}"
        exit 1
        ;;
esac
