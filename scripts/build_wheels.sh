#!/usr/bin/env bash
# Build gpuNUFFT binary wheels locally with cibuildwheel (requires Docker or Podman).
#
# Usage:
#   scripts/build_wheels.sh                # CUDA 13.0, all Python versions
#   CUDA=12.8 scripts/build_wheels.sh      # CUDA 12 variant (gpuNUFFT-cu12)
#   CIBW_BUILD=cp312-* scripts/build_wheels.sh   # only one Python version
#
# Wheels are written to ./wheelhouse
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA="${CUDA:-13.0}"
export CIBW_ENVIRONMENT_LINUX="GPUNUFFT_WHEEL_CUDA=${CUDA} PATH=/usr/local/cuda/bin:\$PATH"

if command -v uvx >/dev/null 2>&1; then
    RUN=(uvx --python 3.12 cibuildwheel==3.4.1)
else
    python3 -m pip install --user "cibuildwheel==3.4.1"
    RUN=(python3 -m cibuildwheel)
fi

"${RUN[@]}" --platform linux --output-dir wheelhouse .
ls -lh wheelhouse
