#!/usr/bin/env python3
"""One-step runner that chains the commands documented in README.

Default behavior runs 5 seeds: 1..5.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ALL_STAGES = ["collect", "pretrain", "raw", "baseline", "online"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]

PRETRAIN_CONFIGS = [
    "configs/pretraining/cat_c4_pretrain.yaml",
    "configs/pretraining/cat_c8_pretrain.yaml",
    "configs/pretraining/cat_c16_pretrain.yaml",
]
BASELINE_CONFIGS = [
    "configs/baselines/ppo_mlp_c1.yaml",
    "configs/baselines/ppolag_mlp_c1.yaml",
    "configs/baselines/ppolag_trans_c1.yaml",
]
RAW_CONFIGS = [
    "configs/main/cat_c4_raw_continue250k.yaml",
    "configs/main/cat_c8_raw_continue250k.yaml",
    "configs/main/cat_c4_preinit_300k.yaml",
    "configs/main/cat_c8_preinit_300k.yaml",
    "configs/main/cat_c16_raw_300k.yaml",
    "configs/main/cat_c16_preinit_300k.yaml",
]
ONLINE_CONFIGS = [
    "configs/main/cat_c4.yaml",
    "configs/main/cat_c8.yaml",
    "configs/main/cat_c16.yaml",
]


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> int:
    rendered = " ".join(cmd)
    print(f"\n$ {rendered}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def _seeded_config_path(config_rel: str, seed: int, temp_dir: Path) -> Path:
    cfg_name = Path(config_rel).stem
    return temp_dir / f"{cfg_name}_seed{seed}.yaml"


def _write_seeded_config(config_rel: str, seed: int, repo_root: Path, temp_dir: Path) -> Path | None:
    src = repo_root / config_rel
    if not src.exists():
        print(f"[WARN] Missing config, skipped: {config_rel}")
        return None

    with src.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    meta = cfg.get("meta", {})
    meta["seed"] = int(seed)
    exp_name = str(meta.get("experiment_name", Path(config_rel).stem))
    if not exp_name.endswith(f"_seed{seed}"):
        exp_name = f"{exp_name}_seed{seed}"
    meta["experiment_name"] = exp_name
    cfg["meta"] = meta

    out = _seeded_config_path(config_rel, seed, temp_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def _seed_dataset_path(seed: int) -> str:
    return f"data/offline/highway_mixed_v1_seed{seed}.npz"


def _build_stage_commands(stage: str, seed: int, python_exec: str, repo_root: Path, temp_dir: Path) -> list[list[str]]:
    py = python_exec

    if stage == "collect":
        return [
            [
                py,
                "scripts/collect_offline_data.py",
                "--policy",
                "idm",
                "--config",
                "configs/pretraining/cat_c4_pretrain.yaml",
                "--output",
                _seed_dataset_path(seed),
                "--samples",
                "12000",
                "--history-len",
                "32",
                "--future-steps",
                "16",
                "--seed",
                str(seed),
            ]
        ]

    if stage == "pretrain":
        cmds: list[list[str]] = []
        for cfg in PRETRAIN_CONFIGS:
            seeded_cfg = _write_seeded_config(cfg, seed, repo_root, temp_dir)
            if seeded_cfg is None:
                continue
            cmds.append(
                [
                    py,
                    "scripts/pretrain.py",
                    "--config",
                    str(seeded_cfg),
                    "--dataset-path",
                    _seed_dataset_path(seed),
                ]
            )
        return cmds

    if stage == "baseline":
        cmds = []
        for cfg in BASELINE_CONFIGS:
            seeded_cfg = _write_seeded_config(cfg, seed, repo_root, temp_dir)
            if seeded_cfg is None:
                continue
            cmds.append([py, "scripts/train_online.py", "--config", str(seeded_cfg)])
        return cmds

    if stage == "raw":
        cmds = []
        for cfg in RAW_CONFIGS:
            seeded_cfg = _write_seeded_config(cfg, seed, repo_root, temp_dir)
            if seeded_cfg is None:
                continue
            cmds.append([py, "scripts/train_online.py", "--config", str(seeded_cfg)])
        return cmds

    if stage == "online":
        cmds = []
        for cfg in ONLINE_CONFIGS:
            seeded_cfg = _write_seeded_config(cfg, seed, repo_root, temp_dir)
            if seeded_cfg is None:
                continue
            cmds.append([py, "scripts/train_online.py", "--config", str(seeded_cfg)])
        return cmds

    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="One-step pipeline runner")
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    temp_dir = repo_root / ".tmp" / "one_step_run"

    failures: list[tuple[str, str, int]] = []

    print("=== One-step pipeline ===")
    print(f"Repo root: {repo_root}")
    print(f"Stages: {', '.join(args.stages)}")
    print(f"Seeds: {', '.join(str(s) for s in args.seeds)}")

    for stage in args.stages:
        print(f"\n--- {stage} ---")
        for seed in args.seeds:
            print(f"\n[seed={seed}]")
            stage_cmds = _build_stage_commands(stage, seed, args.python, repo_root, temp_dir)
            for cmd in stage_cmds:
                code = _run(cmd, repo_root, args.dry_run)
                if code != 0:
                    failures.append((f"{stage}/seed{seed}", " ".join(cmd), code))
                    if not args.continue_on_error:
                        break
            if failures and not args.continue_on_error:
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
