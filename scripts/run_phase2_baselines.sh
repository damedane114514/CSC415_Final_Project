#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_ENTRY="scripts/train_online.py"
BASELINES=(
  "configs/baselines/ppo_mlp_c1.yaml"
  "configs/baselines/ppolag_mlp_c1.yaml"
  "configs/baselines/ppolag_trans_c1.yaml"
)

if [[ ! -f "$TRAIN_ENTRY" ]]; then
  echo "[Phase2][ERROR] Missing training entrypoint: ${TRAIN_ENTRY}"
  echo "[Phase2][ERROR] This repository currently has offline pretraining only (scripts/pretrain.py)."
  echo "[Phase2][ERROR] Add online trainer implementation first, then rerun this script."
  exit 2
fi

for cfg in "${BASELINES[@]}"; do
  echo "=== Running baseline ${cfg} ==="
  bash -lc "python -u ${TRAIN_ENTRY} --config ${cfg}"
done

echo "[Phase2] Baseline runs completed."
