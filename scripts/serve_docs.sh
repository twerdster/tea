#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADDR="${ADDR:-127.0.0.1:8000}"

python3 -m mkdocs serve -f "${ROOT_DIR}/mkdocs.yml" -a "${ADDR}"
