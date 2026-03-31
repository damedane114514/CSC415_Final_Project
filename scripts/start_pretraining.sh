#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/start_pretraining.sh /absolute/path/to/offline_data.npz
#
# Optional env vars:
#   PYTHON_BIN=python3.10
#   EPOCHS=2
#   BATCH_SIZE=64

PYTHON_BIN=${PYTHON_BIN:-python3.10}
DATASET_PATH=${1:-}
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-64}

if [[ -z "${DATASET_PATH}" ]]; then
  echo "[ERROR] Please provide dataset path (.npz) as first argument."
  echo "Example: bash scripts/start_pretraining.sh /data/highway_mixed_v1.npz"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d ".venv" ]]; then
  ${PYTHON_BIN} -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pretrain.txt

declare -a CFGS=(
  "configs/pretraining/cat_c4_pretrain.yaml"
  "configs/pretraining/cat_c8_pretrain.yaml"
  "configs/pretraining/cat_c16_pretrain.yaml"
)

for cfg in "${CFGS[@]}"; do
  echo "=== Running ${cfg} (epochs=${EPOCHS}, batch=${BATCH_SIZE}) ==="
  python scripts/pretrain.py \
    --config "${cfg}" \
    --dataset-path "${DATASET_PATH}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}"
done

echo "Pretraining starter run completed."
