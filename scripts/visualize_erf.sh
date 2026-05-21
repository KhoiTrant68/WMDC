#!/usr/bin/env bash
# scripts/visualize_erf.sh
# ========================
# Per-image Effective Receptive Field (ERF) visualization for WMDC, in the
# layout of Figure 16: one column per Kodak image, ERF heatmap overlaid on
# the input.
#
# Usage:
#   bash scripts/visualize_erf.sh                       # default: 5 Kodak images, two_rows layout
#   bash scripts/visualize_erf.sh overlay_only          # 1 row, overlay only
#   bash scripts/visualize_erf.sh heatmap_only          # 1 row, heatmap only
#
# Override defaults via env vars:
#   LAMBDA=0.0036 CENTER_RADIUS=2 CMAP=inferno bash scripts/visualize_erf.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

LAMBDA="${LAMBDA:-0.0036}"
LAYOUT="${1:-two_rows}"
CENTER_RADIUS="${CENTER_RADIUS:-1}"
CMAP="${CMAP:-viridis}"
ALPHA="${ALPHA:-0.6}"

DEFAULT_IMAGES=(
    kodim17.png
    kodim21.png
    kodim19.png
    kodim18.png
    kodim06.png
)

if [ -n "${IMAGES:-}" ]; then
    read -r -a IMAGES <<< "$IMAGES"
else
    IMAGES=("${DEFAULT_IMAGES[@]}")
fi

VIZ_DIR="$RESULTS_DIR/viz"

# CKPT="${CKPT:-$CHECKPOINT_ROOT/routing/$ROUTING_MODE/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar}"
CKPT="/kaggle/input/datasets/khitrnminh/18052026/checkpoint_best.pth.tar"
CA_FLAGS="$(common_arch_flags)"

if [ ! -f "$CKPT" ]; then
    echo "[error] checkpoint not found: $CKPT" >&2
    exit 1
fi

mkdir -p "$VIZ_DIR"

OUT="$VIZ_DIR/erf_wmdc_${LAYOUT}_lambda_${LAMBDA}.pdf"

cd "$REPO"

echo "=== Per-image ERF visualization ==="
echo "  checkpoint   : $CKPT"
echo "  routing mode : $ROUTING_MODE"
echo "  layout       : $LAYOUT"
echo "  images       : ${IMAGES[*]}"
echo "  center radius: $CENTER_RADIUS"

# shellcheck disable=SC2086
$PYTHON analyze/visualize_erf.py \
    -d "$KODAK_DIR" \
    -c "$CKPT" \
    --routing-mode "$ROUTING_MODE" \
    --image_names "${IMAGES[@]}" \
    --layout "$LAYOUT" \
    --center_radius "$CENTER_RADIUS" \
    --cmap "$CMAP" \
    --alpha "$ALPHA" \
    --cuda $CA_FLAGS \
    -o "$OUT"

echo "[done] ERF figure → $OUT"
