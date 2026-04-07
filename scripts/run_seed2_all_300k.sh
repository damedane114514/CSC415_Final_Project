#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
LOG_DIR="logs/schedules"
mkdir -p "$LOG_DIR"

echo "[seed2 1/9] baseline PPO+MLP 300k"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed2_ppo_mlp_c1_300k.yaml | tee "$LOG_DIR/seed2_ppo_mlp_c1_300k.log"

echo "[seed2 2/9] baseline PPO-Lagrangian+MLP 300k"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed2_ppolag_mlp_c1_300k.yaml | tee "$LOG_DIR/seed2_ppolag_mlp_c1_300k.log"

echo "[seed2 3/9] baseline PPO-Lagrangian+Transformer(C1) 300k"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed2_ppolag_trans_c1_300k.yaml | tee "$LOG_DIR/seed2_ppolag_trans_c1_300k.log"

echo "[seed2 4/9] CAT C4 raw 300k"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed2_cat_c4_raw_300k.yaml | tee "$LOG_DIR/seed2_cat_c4_raw_300k.log"

echo "[seed2 5/9] CAT C8 raw 300k"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed2_cat_c8_raw_300k.yaml | tee "$LOG_DIR/seed2_cat_c8_raw_300k.log"

echo "[seed2 6/9] CAT C16 raw 300k"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed2_cat_c16_raw_300k.yaml | tee "$LOG_DIR/seed2_cat_c16_raw_300k.log"

echo "[seed2 7/9] CAT C4 preinit 300k"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed2_cat_c4_preinit_300k.yaml | tee "$LOG_DIR/seed2_cat_c4_preinit_300k.log"

echo "[seed2 8/9] CAT C8 preinit 300k"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed2_cat_c8_preinit_300k.yaml | tee "$LOG_DIR/seed2_cat_c8_preinit_300k.log"

echo "[seed2 9/9] CAT C16 preinit 300k"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed2_cat_c16_preinit_300k.yaml | tee "$LOG_DIR/seed2_cat_c16_preinit_300k.log"

echo "All seed2 300k runs finished."
