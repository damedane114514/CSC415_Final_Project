#!/usr/bin/env bash
set -euo pipefail

NUM_SEEDS="${1:-1}"
DEVICE="${2:-auto}"

python scripts/run_group.py --group baselines   --profile debug --num-seeds "${NUM_SEEDS}" --device "${DEVICE}"
python scripts/run_group.py --group main        --profile debug --num-seeds "${NUM_SEEDS}" --device "${DEVICE}"
python scripts/run_group.py --group pretraining --profile debug --num-seeds "${NUM_SEEDS}" --device "${DEVICE}"
python scripts/run_group.py --group replanning  --profile debug --num-seeds "${NUM_SEEDS}" --device "${DEVICE}"

python scripts/summarize_results.py --profile debug