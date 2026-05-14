#!/usr/bin/env bash
# scripts/scale_audit.sh
# ======================
# Per-λ audit of Gaussian-scale coverage by the entropy coder's scale_table.
# Runs analyze/verify_scale_table.py on every checkpoint under the routing
# (or ablation) directory and aggregates the verdicts.
#
# Usage:
#   bash scripts/scale_audit.sh routing       # audit routing/<mode>/lambda_*
#   bash scripts/scale_audit.sh ablation      # audit ablation/<variant>/lam_*
#   bash scripts/scale_audit.sh both
#
# Output:
#   $RESULTS_DIR/scale_audit/<group>/<name>/lambda_<lam>.json
# A summary of pass/fail verdicts is printed at the end.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

AUDIT_DIR="$RESULTS_DIR/scale_audit"
MAX_IMAGES="${MAX_IMAGES:-24}"   # override via env if needed

audit_routing() {
    echo ""
    echo "=== Scale-table audit (routing) ==="
    cd "$REPO"
    for mode in "${ROUTING_MODES[@]}"; do
        for lam in "${LAMBDAS_FULL[@]}"; do
            local ck="$CHECKPOINT_ROOT/routing/$mode/lambda_${lam}_mse/checkpoint_best.pth.tar"
            local out_dir="$AUDIT_DIR/routing/$mode"
            local out="$out_dir/lambda_${lam}.json"
            if [ ! -f "$ck" ]; then
                echo "[skip] missing $ck"
                continue
            fi
            mkdir -p "$out_dir"
            echo "-- routing=$mode λ=$lam"

            # shellcheck disable=SC2086  # word-split intentional for CA_FLAGS
            $PYTHON analyze/verify_scale_table.py \
                --checkpoint "$ck" \
                --dataset "$KODAK_DIR" \
                --routing-mode "$mode" \
                --max-images "$MAX_IMAGES" \
                $CA_FLAGS \
                --cuda \
                --output "$out" \
                || echo "[warn] verify_scale_table failed for $mode @ $lam"
        done
    done
}

audit_ablation() {
    echo ""
    echo "=== Scale-table audit (ablation) ==="
    cd "$REPO"
    for variant in "${VARIANTS[@]}"; do
        for lam in "${LAMBDAS_ABLATION[@]}"; do
            local ck; ck="$(ckpt_path "ablation/$variant" "$lam")"
            local out_dir="$AUDIT_DIR/ablation/$variant"
            local out="$out_dir/lambda_${lam}.json"
            if [ ! -f "$ck" ]; then
                echo "[skip] missing $ck"
                continue
            fi
            mkdir -p "$out_dir"
            local eval_flags; eval_flags="$(variant_eval_flags "$variant")"
            local routing="unbalanced_eot"
            [ "$variant" = "no_ueot" ] && routing="softmax"
            echo "-- variant=$variant λ=$lam"

            # shellcheck disable=SC2086
            $PYTHON analyze/verify_scale_table.py \
                --checkpoint "$ck" \
                --dataset "$KODAK_DIR" \
                --routing-mode "$routing" \
                --max-images "$MAX_IMAGES" \
                $eval_flags \
                $CA_FLAGS \
                --cuda \
                --output "$out" \
                || echo "[warn] verify_scale_table failed for $variant @ $lam"
        done
    done
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 {routing|ablation|both}"
    exit 1
fi

for cmd in "$@"; do
    case "$cmd" in
        routing)  audit_routing ;;
        ablation) audit_ablation ;;
        both)
            audit_routing
            audit_ablation
            ;;
        *) echo "Unknown subcommand: $cmd" >&2; exit 1 ;;
    esac
done

echo ""
echo "=== Scale-table audit done. JSONs in: $AUDIT_DIR ==="
