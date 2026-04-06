#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
LOG_DIR="logs/schedules"
mkdir -p "$LOG_DIR"

echo "[1/3] Train baseline: PPO + MLP (C1)"
$PYTHON_BIN scripts/train_online.py --config configs/baselines/ppo_mlp_c1.yaml | tee "$LOG_DIR/baseline_ppo_mlp_c1.log"

echo "[2/3] Train baseline: PPO-Lagrangian + MLP (C1)"
$PYTHON_BIN scripts/train_online.py --config configs/baselines/ppolag_mlp_c1.yaml | tee "$LOG_DIR/baseline_ppolag_mlp_c1.log"

echo "[3/3] Train baseline: PPO-Lagrangian + Transformer (C1)"
$PYTHON_BIN scripts/train_online.py --config configs/baselines/ppolag_trans_c1.yaml | tee "$LOG_DIR/baseline_ppolag_trans_c1.log"

echo "All baseline runs finished."
