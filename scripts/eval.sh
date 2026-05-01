#!/usr/bin/env bash
# scripts/eval.sh
# ===============
# Evaluate trained checkpoints on the Kodak dataset.
# Covers both the routing-mode ablation and component-removal ablation.
#
# Usage:
#   bash scripts/eval.sh routing          # eval 3 routing modes (Table 1)
#   bash scripts/eval.sh ablation         # eval 7 ablation variants (Table 4)
#   bash scripts/eval.sh routing ablation # eval both

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

LAMBDA=0.0036   # routing comparison lambda

eval_routing() {
    echo ""
    echo "=== Evaluating routing-mode ablation ==="
    cd "$REPO"

    for mode in "${ROUTING_MODES[@]}"; do
        local ckpt="$CHECKPOINT_ROOT/routing/$mode/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar"
        local out="$RESULTS_DIR/routing/$mode"

        if [ ! -f "$ckpt" ]; then
            echo "[skip] missing checkpoint: $ckpt"
            continue
        fi

        echo ""
        echo "-- routing-mode=$mode"
        mkdir -p "$out"

        $PYTHON eval.py \
            --dataset "$KODAK_DIR" \
            --checkpoint "$ckpt" \
            --output "$out" \
            --routing-mode "$mode" \
            --cuda \
            --measure-dict-util

        echo "[done] $mode → $out/RD_report.json"
    done
}

eval_ablation() {
    echo ""
    echo "=== Evaluating component-removal ablation ==="
    cd "$REPO"

    for variant in "${VARIANTS[@]}"; do
        for lam in "${LAMBDAS_ABLATION[@]}"; do
            local ckpt
            ckpt="$(ckpt_path "ablation/$variant" "$lam")"
            local out="$RESULTS_DIR/ablation/$variant/lam_$lam"

            if [ ! -f "$ckpt" ]; then
                echo "[skip] missing checkpoint: $ckpt"
                continue
            fi

            local eval_flags; eval_flags="$(variant_eval_flags "$variant")"
            echo ""
            echo "-- variant=$variant  λ=$lam"
            mkdir -p "$out"

            # shellcheck disable=SC2086  # word-split intentional for flags
            $PYTHON eval.py \
                --dataset "$KODAK_DIR" \
                --checkpoint "$ckpt" \
                --output "$out" \
                --routing-mode unbalanced_eot \
                $eval_flags \
                --cuda \
                --measure-dict-util

            echo "[done] $variant/lam_$lam → $out/RD_report.json"
        done
    done
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 {routing|ablation} [routing|ablation]"
    exit 1
fi

for cmd in "$@"; do
    case "$cmd" in
        routing)  eval_routing ;;
        ablation) eval_ablation ;;
        *) echo "Unknown subcommand: $cmd" >&2; exit 1 ;;
    esac
done

echo ""
echo "=== Evaluation done. Results in: $RESULTS_DIR ==="
