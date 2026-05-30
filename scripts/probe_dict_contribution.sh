#!/usr/bin/env bash
# scripts/probe_dict_contribution.sh
# ==================================
# Run the "is the dictionary actually used?" diagnostic from
# analyze/probe_dict_contribution.py on a trained routing checkpoint.
#
# The probe runs every Kodak image twice — once normally, once with every
# UnifiedDictionaryAttention output forced to zero via a forward hook —
# and reports the average bpp / PSNR swing.  A near-zero swing means the
# dictionary branch contributes nothing and the routing collapse diagnosis
# is confirmed.
#
# Usage:
#   bash scripts/probe_dict_contribution.sh                     # mode=unbalanced_eot, λ=0.0036
#   bash scripts/probe_dict_contribution.sh --mode softmax
#   bash scripts/probe_dict_contribution.sh --lambda 0.013
#   bash scripts/probe_dict_contribution.sh --ckpt /path/to/ckpt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODE="unbalanced_eot"
LAMBDA=0.0036
CKPT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --mode)   MODE="$2"; shift 2 ;;
        --lambda) LAMBDA="$2"; shift 2 ;;
        --ckpt)   CKPT="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$CKPT" ]; then
    CKPT="$CHECKPOINT_ROOT/routing/${MODE}/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar"
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

OUT_DIR="$RESULTS_DIR/probe_dict_contribution/$MODE"
OUT_JSON="$OUT_DIR/lambda_${LAMBDA}.json"
mkdir -p "$OUT_DIR"
cd "$REPO"

arch_flags="$(common_arch_flags)"

echo "═══════════════════════════════════════════════════════════════════"
echo "  DICTIONARY-CONTRIBUTION PROBE"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Mode       : $MODE"
echo "  Lambda     : $LAMBDA"
echo "  Checkpoint : $CKPT"
echo "  Output     : $OUT_JSON"
echo "═══════════════════════════════════════════════════════════════════"

# shellcheck disable=SC2086
$PYTHON analyze/probe_dict_contribution.py \
    --dataset "$KODAK_DIR" \
    --checkpoint "$CKPT" \
    --routing-mode "$MODE" \
    --output "$OUT_JSON" \
    --cuda \
    $arch_flags
