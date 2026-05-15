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
    echo "=== Verifying wavelet module ==="
    cd "$REPO"
    $PYTHON -m modules.wavelet_blocks
    echo "[ok] wavelet module OK"
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
