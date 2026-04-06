#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}

log_dir="logs/schedules"
mkdir -p "$log_dir"

echo "[1/2] Train RAW C16 for 300k"
$PYTHON_BIN scripts/train_online.py --config configs/main/cat_c16_raw_300k.yaml | tee "$log_dir/cat_c16_raw_300k.log"

echo "[2/2] Train PREINIT C16 for 300k"
$PYTHON_BIN scripts/train_online.py --config configs/main/cat_c16_preinit_300k.yaml | tee "$log_dir/cat_c16_preinit_300k.log"

echo "C16 runs finished."
