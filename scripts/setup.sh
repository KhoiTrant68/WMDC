#!/usr/bin/env bash
# scripts/setup.sh
# ================
# Install all dependencies and build the selective_scan CUDA kernel.
#
# Usage:
#   bash scripts/setup.sh           # full setup
#   bash scripts/setup.sh kernel    # rebuild CUDA kernel only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

install_deps() {
    echo "=== Installing Python dependencies ==="
    pip install \
        compressai \
        accelerate \
        einops \
        timm \
        pytorch-msssim \
        flops-profiler \
        bjontegaard

    # Wavelet transforms: pytorch_wavelets provides bior4.4 with proper
    # boundary handling (replaces the hand-rolled DWT/IDWT).  PyWavelets is
    # a transitive dep of pytorch_wavelets but is pinned here so version
    # drift does not silently change filter coefficients.
    pip install pytorch-wavelets "PyWavelets>=1.4.0"

    # Pin tensorboard to avoid protobuf conflicts
    pip install tensorboard==2.14 protobuf==4.25.3
    echo "[ok] dependencies installed"
}

build_kernel() {
    echo "=== Building selective_scan CUDA kernel ==="
    cd "$REPO/vmamba"
    pip install .
    cd "$REPO"
    echo "[ok] kernel built"
}

verify() {
    echo "=== Verifying wavelet module (PR at boundaries, no crop) ==="
    cd "$REPO"
    $PYTHON -m modules.wavelet_blocks
    echo "[ok] wavelet module OK"

    echo "=== Importing core model classes ==="
    $PYTHON -c "
from models.WMDC import WMDC
from modules.dictionary_blocks import UnifiedDictionaryAttention
from modules.content_adaptive_blocks import TokenClustering
m = WMDC(routing_mode='softmax')
print('  softmax        params:', sum(p.numel() for p in m.parameters()))
m = WMDC(routing_mode='balanced_eot')
print('  balanced_eot   params:', sum(p.numel() for p in m.parameters()))
m = WMDC(routing_mode='unbalanced_eot')
print('  unbalanced_eot params:', sum(p.numel() for p in m.parameters()))
print('[ok] all three routing modes instantiate cleanly')
"
}

case "${1:-all}" in
    kernel)  build_kernel ;;
    verify)  verify ;;
    all)
        install_deps
        build_kernel
        verify
        ;;
    *)
        echo "Usage: $0 {all|kernel|verify}"
        exit 1
        ;;
esac
