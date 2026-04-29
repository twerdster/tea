#!/usr/bin/env bash
set -euo pipefail

status=0

check_command() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    printf '[ok]   %s -> %s\n' "${name}" "$(command -v "${name}")"
  else
    printf '[miss] %s\n' "${name}"
    status=1
  fi
}

echo "Tea environment check"
echo

check_command nvcc
check_command g++
check_command python3
check_command cmake

echo
if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc:"
  nvcc --version | tail -n 4
  echo
fi

if command -v g++ >/dev/null 2>&1; then
  echo "g++:"
  g++ --version | head -n 1
  echo
fi

if command -v python3 >/dev/null 2>&1; then
  echo "python3:"
  python3 --version
  echo
fi

if command -v cmake >/dev/null 2>&1; then
  echo "cmake:"
  cmake --version | head -n 1
  echo
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi:"
  if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
    status=1
  fi
else
  echo "[miss] nvidia-smi"
  status=1
fi

echo
echo "Suggested next steps:"
echo "  make Tea"
echo "  make smoke"
echo "  or: bash scripts/build.sh && bash scripts/run_smoke.sh ./build-cmake/Tea"

exit "${status}"
