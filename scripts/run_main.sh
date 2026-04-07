#!/usr/bin/env bash
set -euo pipefail

NUM_SEEDS="${1:-1}"
DEVICE="${2:-auto}"

python scripts/run_group.py --group baselines   --profile main --num-seeds "${NUM_SEEDS}" --device "${DEVICE}"
python scripts/run_group.py --group main        --profile main --num-seeds "${NUM_SEEDS}" --device "${DEVICE}"
python scripts/run_group.py --group pretraining --profile main --num-seeds "${NUM_SEEDS}" --device "${DEVICE}"
python scripts/run_group.py --group replanning  --profile main --num-seeds "${NUM_SEEDS}" --device "${DEVICE}"

python scripts/summarize_results.py --profile main