#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/run_baselines_continue_to300k.sh
bash scripts/run_seed2_all_300k.sh

echo "Baseline continuation + seed2 suite completed."