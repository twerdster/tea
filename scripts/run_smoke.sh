#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEA_BIN="${1:-${ROOT_DIR}/Tea}"
WORK_DIR="${WORK_DIR:-/tmp/tea-smoke-tests}"

python3 "${ROOT_DIR}/tests/run_smoke_tests.py" --tea "${TEA_BIN}" --work-dir "${WORK_DIR}"
