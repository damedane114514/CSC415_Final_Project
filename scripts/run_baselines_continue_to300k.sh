#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
LOG_DIR="logs/schedules"
mkdir -p "$LOG_DIR"

echo "[1/3] Continue baseline PPO+MLP to 300k total"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed1_ppo_mlp_c1_continue_to300k.yaml | tee "$LOG_DIR/seed1_ppo_mlp_c1_continue_to300k.log"

echo "[2/3] Continue baseline PPO-Lagrangian+MLP to 300k total"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed1_ppolag_mlp_c1_continue_to300k.yaml | tee "$LOG_DIR/seed1_ppolag_mlp_c1_continue_to300k.log"

echo "[3/3] Continue baseline PPO-Lagrangian+Transformer(C1) to 300k total"
$PYTHON_BIN scripts/train_online.py --config configs/runs/seed1_ppolag_trans_c1_continue_to300k.yaml | tee "$LOG_DIR/seed1_ppolag_trans_c1_continue_to300k.log"

echo "Baseline continuation runs finished."
