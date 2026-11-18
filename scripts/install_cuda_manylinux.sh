#!/usr/bin/env bash
# Install a minimal CUDA toolkit inside a manylinux_2_28 (AlmaLinux 8) container.
# Used by cibuildwheel as `before-all` (see pyproject.toml).
#
# Usage: install_cuda_manylinux.sh [CUDA_VERSION]   (default: $GPUNUFFT_WHEEL_CUDA or 13.0)
set -euo pipefail

CUDA_VERSION="${1:-${GPUNUFFT_WHEEL_CUDA:-13.0}}"
CUDA_MAJOR="${CUDA_VERSION%%.*}"
CUDA_MINOR="${CUDA_VERSION#*.}"
CUDA_MINOR="${CUDA_MINOR%%.*}"
PKG_SUFFIX="${CUDA_MAJOR}-${CUDA_MINOR}"

case "$(uname -m)" in
    x86_64)  REPO_ARCH=x86_64 ;;
    aarch64) REPO_ARCH=sbsa ;;
    *) echo "Unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

echo "Installing CUDA ${CUDA_MAJOR}.${CUDA_MINOR} (${REPO_ARCH})"

dnf install -y dnf-plugins-core
dnf config-manager --add-repo \
    "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${REPO_ARCH}/cuda-rhel8.repo"

# Only what gpuNUFFT needs: compiler, runtime headers + static lib, and cuFFT
# headers + link library. At runtime cuFFT comes from NVIDIA's pip package.
dnf install -y --setopt=install_weak_deps=False \
    "cuda-nvcc-${PKG_SUFFIX}" \
    "cuda-cudart-devel-${PKG_SUFFIX}" \
    "cuda-driver-devel-${PKG_SUFFIX}" \
    "libcufft-devel-${PKG_SUFFIX}"
dnf clean all

ln -sfn "/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}" /usr/local/cuda
/usr/local/cuda/bin/nvcc --version
