#!/usr/bin/env bash
# scripts/save_results.sh
# =======================
# Collect all generated outputs into a single zip archive.
#
# Usage:
#   bash scripts/save_results.sh              # zip to results_save.zip
#   bash scripts/save_results.sh my_run.zip   # zip to custom filename

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

OUTPUT="${1:-result_save.zip}"
STAGING="$REPO/result_save"

echo "=== Collecting results ==="
rm -rf "$STAGING"
mkdir -p "$STAGING"

copy_if_exists() {
    local src="$1" dst="$2"
    if [ -e "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp -r "$src" "$dst"
        echo "[copy] $src"
    else
        echo "[skip] not found: $src"
    fi
}

# PDF figures
if [ -d "$RESULTS_DIR/viz" ]; then
    mkdir -p "$STAGING/figures"
    find "$RESULTS_DIR/viz" -name "*.pdf" -exec cp {} "$STAGING/figures/" \;
    echo "[copy] figures ($(find "$STAGING/figures" -name "*.pdf" | wc -l) PDFs)"
fi

# Eval reports (routing ablation)
copy_if_exists "$RESULTS_DIR/routing" "$STAGING/eval_routing"

# Eval reports (component ablation)
copy_if_exists "$RESULTS_DIR/ablation" "$STAGING/eval_ablation"

# BD-rate table
copy_if_exists "$RESULTS_DIR/bd_rate" "$STAGING/bd_rate"

# Benchmark reports
copy_if_exists "$RESULTS_DIR/benchmark" "$STAGING/benchmark"

# JSON data files from viz
if [ -d "$RESULTS_DIR/viz" ]; then
    find "$RESULTS_DIR/viz" -name "*.json" -exec cp {} "$STAGING/" \; 2>/dev/null || true
fi

echo ""
echo "=== Zipping → $OUTPUT ==="
cd "$REPO"
zip -r "$OUTPUT" result_save/
rm -rf "$STAGING"

echo "[done] $(du -sh "$OUTPUT" | cut -f1)  $OUTPUT"
