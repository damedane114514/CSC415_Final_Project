#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET_PATH=${DATASET_PATH:-"data/offline/highway_mixed_v1.npz"}
SAMPLES=${SAMPLES:-12000}
HISTORY_LEN=${HISTORY_LEN:-8}
FUTURE_STEPS=${FUTURE_STEPS:-16}

CONFIGS=(
  "configs/pretraining/cat_c4_pretrain.yaml"
  "configs/pretraining/cat_c8_pretrain.yaml"
  "configs/pretraining/cat_c16_pretrain.yaml"
)

echo "[Phase1] Collecting pure IDM offline dataset..."
bash -lc "python -u scripts/collect_offline_data.py \
  --policy idm \
  --config configs/pretraining/cat_c4_pretrain.yaml \
  --output ${DATASET_PATH} \
  --samples ${SAMPLES} \
  --history-len ${HISTORY_LEN} \
  --future-steps ${FUTURE_STEPS}"

echo "[Phase1] Offline pretraining (CAT C4/C8/C16) using ${DATASET_PATH}"
for cfg in "${CONFIGS[@]}"; do
  echo "=== Running ${cfg} ==="
  bash -lc "python -u scripts/pretrain.py --config ${cfg} --dataset-path ${DATASET_PATH}"
done

echo "[Phase1] Done. Check logs/*/pretraining/pretrain_metrics.csv and checkpoints/*.pt"
