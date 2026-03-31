#!/usr/bin/env bash
set -euo pipefail

echo "=== Running train.py processes ==="
ps -ef | grep "scripts/train.py" | grep -v grep || true

echo
echo "=== Recent launcher logs ==="
ls -lt /root/autodl-tmp/csc415_runs/launcher_logs 2>/dev/null | head || true