#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs/schedules

launch() {
  local name="$1"
  local cfg="$2"
  nohup /root/miniconda3/bin/conda run -p /root/miniconda3 --no-capture-output \
    python scripts/train_online.py --config "$cfg" \
    > "logs/schedules/${name}.log" 2>&1 &
  echo "$name:$!"
}

# seed2_ppo_mlp_c1_300k is already running; launch the rest in parallel.
launch "seed2_ppolag_mlp_c1_300k" "configs/runs/seed2_ppolag_mlp_c1_300k.yaml"
launch "seed2_ppolag_trans_c1_300k" "configs/runs/seed2_ppolag_trans_c1_300k.yaml"
launch "seed2_cat_c4_raw_300k" "configs/runs/seed2_cat_c4_raw_300k.yaml"
launch "seed2_cat_c8_raw_300k" "configs/runs/seed2_cat_c8_raw_300k.yaml"
launch "seed2_cat_c16_raw_300k" "configs/runs/seed2_cat_c16_raw_300k.yaml"
launch "seed2_cat_c4_preinit_300k" "configs/runs/seed2_cat_c4_preinit_300k.yaml"
launch "seed2_cat_c8_preinit_300k" "configs/runs/seed2_cat_c8_preinit_300k.yaml"
launch "seed2_cat_c16_preinit_300k" "configs/runs/seed2_cat_c16_preinit_300k.yaml"
