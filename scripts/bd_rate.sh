#!/usr/bin/env bash
# scripts/bd_rate.sh
# ==================
# Evaluate ablation-variant checkpoints on Kodak, then compute BD-rate
# relative to the "full" model anchor. Fills Tables 2, 3, 4 in the paper.
#
# Usage:
#   bash scripts/bd_rate.sh evaluate    # run eval → per-variant RD JSONs
#   bash scripts/bd_rate.sh compute     # compute BD-rate from RD JSONs
#   bash scripts/bd_rate.sh latex       # print LaTeX table rows
#   bash scripts/bd_rate.sh all         # run all three steps

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

BD_DIR="$RESULTS_DIR/bd_rate"

# ── Step 1: Evaluate each variant → RD JSON ──────────────────────────
run_evaluate() {
    echo ""
    echo "=== Step 1: Evaluate variants on Kodak ==="
    cd "$REPO"
    mkdir -p "$BD_DIR"

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
            echo "[no-eval] $variant: no checkpoints found, skipping"
            continue
        fi

        local out="$BD_DIR/${variant}.json"
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
            $CA_FLAGS \
            --cuda

        echo "[done] $variant → $out"
    done
}

# ── Step 2: BD-rate vs. "full" anchor ────────────────────────────────
run_compute() {
    echo ""
    echo "=== Step 2: Compute BD-rate vs. full anchor ==="
    cd "$REPO"

    local anchor="$BD_DIR/full.json"
    if [ ! -f "$anchor" ]; then
        echo "[ERR] anchor JSON not found: $anchor" >&2
        echo "      Run '$0 evaluate' first." >&2
        exit 1
    fi

    local var_jsons=()
    for variant in "${VARIANTS[@]}"; do
        [ "$variant" = "full" ] && continue
        local f="$BD_DIR/${variant}.json"
        if [ -f "$f" ]; then
            var_jsons+=("$f")
        else
            echo "[skip] missing $f"
        fi
    done

    if [ "${#var_jsons[@]}" -eq 0 ]; then
        echo "[ERR] no variant JSONs found in $BD_DIR" >&2
        exit 1
    fi

    local out="$BD_DIR/bd_rate_table.json"
    $PYTHON analyze/compute_bd_rate.py bd-rate \
        --anchor-json "$anchor" \
        --variant-jsons "${var_jsons[@]}" \
        --output "$out" \
        --method akima

    echo "[done] BD-rate table → $out"
}

# ── Step 3: LaTeX table ───────────────────────────────────────────────
run_latex() {
    echo ""
    echo "=== Step 3: Print LaTeX table ==="
    cd "$REPO"

    local table="$BD_DIR/bd_rate_table.json"
    if [ ! -f "$table" ]; then
        echo "[ERR] BD-rate table not found: $table" >&2
        echo "      Run '$0 compute' first." >&2
        exit 1
    fi

    $PYTHON analyze/aggregate_to_latex.py \
        --bd-rate-json "$table"
}

# ── dispatcher ────────────────────────────────────────────────────────
case "${1:-all}" in
    evaluate) run_evaluate ;;
    compute)  run_compute ;;
    latex)    run_latex ;;
    all)
        run_evaluate
        run_compute
        run_latex
        ;;
    *)
        echo "Usage: $0 {all|evaluate|compute|latex}"
        echo ""
        echo "  evaluate  — eval each variant on Kodak, produce RD JSONs"
        echo "  compute   — compute BD-rate from RD JSONs"
        echo "  latex     — print LaTeX table from BD-rate JSON"
        exit 1
        ;;
esac
