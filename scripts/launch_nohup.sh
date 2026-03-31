#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-debug}"
NUM_SEEDS="${2:-1}"
DEVICE="${3:-auto}"

LOG_DIR="/root/autodl-tmp/csc415_runs/launcher_logs"
mkdir -p "${LOG_DIR}"
STAMP=$(date +"%Y%m%d_%H%M%S")

if [[ "${PROFILE}" == "debug" ]]; then
  nohup bash scripts/run_debug.sh "${NUM_SEEDS}" "${DEVICE}" > "${LOG_DIR}/debug_${STAMP}.log" 2>&1 &
  echo "Launched debug run -> ${LOG_DIR}/debug_${STAMP}.log"
elif [[ "${PROFILE}" == "main" ]]; then
  nohup bash scripts/run_main.sh "${NUM_SEEDS}" "${DEVICE}" > "${LOG_DIR}/main_${STAMP}.log" 2>&1 &
  echo "Launched main run -> ${LOG_DIR}/main_${STAMP}.log"
else
  echo "Unknown profile: ${PROFILE}"
  exit 1
fi