#!/usr/bin/env python3
"""One-step runner that chains the commands documented in README."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ALL_STAGES = ["collect", "pretrain", "raw", "baseline", "online"]


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> int:
    rendered = " ".join(cmd)
    print(f"\n$ {rendered}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def _build_commands(python_exec: str) -> dict[str, list[list[str]]]:
    py = python_exec
    return {
        "collect": [
            [
                py,
                "scripts/collect_offline_data.py",
                "--policy",
                "idm",
                "--config",
                "configs/pretraining/cat_c4_pretrain.yaml",
                "--output",
                "data/offline/highway_mixed_v1.npz",
                "--samples",
                "12000",
                "--history-len",
                "32",
                "--future-steps",
                "16",
                "--seed",
                "1",
            ]
        ],
        "pretrain": [
            [py, "scripts/pretrain.py", "--config", "configs/pretraining/cat_c4_pretrain.yaml"],
            [py, "scripts/pretrain.py", "--config", "configs/pretraining/cat_c8_pretrain.yaml"],
            [py, "scripts/pretrain.py", "--config", "configs/pretraining/cat_c16_pretrain.yaml"],
        ],
        "raw": [
            ["bash", "scripts/run_cat_c4c8_history32_schedule.sh"],
            ["bash", "scripts/run_cat_c16_history32_schedule.sh"],
        ],
        "baseline": [
            ["bash", "scripts/run_baselines_fresh.sh"],
        ],
        "online": [
            [py, "scripts/train_online.py", "--config", "configs/main/cat_c4.yaml"],
            [py, "scripts/train_online.py", "--config", "configs/main/cat_c8.yaml"],
            [py, "scripts/train_online.py", "--config", "configs/main/cat_c16.yaml"],
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="One-step pipeline runner")
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    command_map = _build_commands(args.python)

    failures: list[tuple[str, str, int]] = []

    print("=== One-step pipeline ===")
    print(f"Repo root: {repo_root}")
    print(f"Stages: {', '.join(args.stages)}")

    for stage in args.stages:
        print(f"\n--- {stage} ---")
        for cmd in command_map[stage]:
            code = _run(cmd, repo_root, args.dry_run)
            if code != 0:
                failures.append((stage, " ".join(cmd), code))
                if not args.continue_on_error:
                    break
        if failures and not args.continue_on_error:
            break

    print("\n=== Done ===")
    if failures:
        print("Failed commands:")
        for stage, cmd, code in failures:
            print(f"- stage={stage}, exit_code={code}, cmd={cmd}")
        return 1

    print("All selected stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
