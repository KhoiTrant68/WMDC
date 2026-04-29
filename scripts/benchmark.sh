#!/usr/bin/env bash
# scripts/benchmark.sh
# ====================
# Measure VRAM usage and FLOPs for the full WMDC model and backbone variants.
#
# Usage:
#   bash scripts/benchmark.sh          # run all benchmarks
#   bash scripts/benchmark.sh vram     # VRAM only
#   bash scripts/benchmark.sh flops    # FLOPs / throughput only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

LAMBDA=0.0036
BENCH_DIR="$RESULTS_DIR/benchmark"

run_vram() {
    echo ""
    echo "=== VRAM measurement ==="
    cd "$REPO"
    mkdir -p "$BENCH_DIR"

    local ck="$CHECKPOINT_ROOT/routing/unbalanced_eot/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar"
    [ -f "$ck" ] || { echo "[skip] checkpoint not found: $ck"; return; }

    $PYTHON analyze/measure_vram.py \
        --checkpoint "$ck" \
        --routing-mode unbalanced_eot \
        --cuda \
        --output "$BENCH_DIR/vram_report.json"

    echo "[done] VRAM report → $BENCH_DIR/vram_report.json"
}

run_flops() {
    echo ""
    echo "=== FLOPs / throughput benchmark ==="
    cd "$REPO"
    mkdir -p "$BENCH_DIR"

    $PYTHON analyze/benchmark_backbone.py \
        --routing-mode unbalanced_eot \
        --cuda \
        --output "$BENCH_DIR/flops_report.json"

    echo "[done] FLOPs report → $BENCH_DIR/flops_report.json"
}

case "${1:-all}" in
    vram)  run_vram ;;
    flops) run_flops ;;
    all)
        run_vram
        run_flops
        ;;
    *)
        echo "Usage: $0 {all|vram|flops}"
        exit 1
        ;;
esac
