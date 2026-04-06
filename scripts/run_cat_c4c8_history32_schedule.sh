#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}

log_dir="logs/schedules"
mkdir -p "$log_dir"

echo "[1/4] Continue RAW C4 for +250k"
$PYTHON_BIN scripts/train_online.py --config configs/main/cat_c4_raw_continue250k.yaml | tee "$log_dir/cat_c4_raw_continue250k.log"

echo "[2/4] Continue RAW C8 for +250k"
$PYTHON_BIN scripts/train_online.py --config configs/main/cat_c8_raw_continue250k.yaml | tee "$log_dir/cat_c8_raw_continue250k.log"

echo "[3/4] Restart PREINIT C4 for 300k"
$PYTHON_BIN scripts/train_online.py --config configs/main/cat_c4_preinit_300k.yaml | tee "$log_dir/cat_c4_preinit_300k.log"

echo "[4/4] Restart PREINIT C8 for 300k"
$PYTHON_BIN scripts/train_online.py --config configs/main/cat_c8_preinit_300k.yaml | tee "$log_dir/cat_c8_preinit_300k.log"

echo "All runs finished."
