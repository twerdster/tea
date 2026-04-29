#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-cmake}"
CUDA_ARCH="${CUDA_ARCH:-86}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TEA_ENABLE_OPENMP="${TEA_ENABLE_OPENMP:-ON}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DTEA_ENABLE_OPENMP="${TEA_ENABLE_OPENMP}"

cmake --build "${BUILD_DIR}" -j

echo "Built Tea at ${BUILD_DIR}/Tea"
