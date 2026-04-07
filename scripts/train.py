from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.trainers.entrypoint import run_training
from src.utils.config import deep_update, load_yaml, save_yaml
from src.utils.io import get_git_hash, now_str, save_json, save_text, safe_mkdir


DEBUG_OVERRIDES: Dict[str, Any] = {
    "algo": {
        "total_timesteps": 50000,
        "rollout_steps": 1024,
    },
    "evaluation": {
        "eval_interval": 5000,
        "eval_episodes": 5,
    },
}

MAIN_OVERRIDES: Dict[str, Any] = {
    # keep config as-is
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", type=str, choices=["debug", "main"], default="debug")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-root", type=str, default="/root/autodl-tmp/csc415_runs")
    return parser.parse_args()


def build_final_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml(args.config)
    profile_overrides = DEBUG_OVERRIDES if args.profile == "debug" else MAIN_OVERRIDES
    cfg = deep_update(cfg, profile_overrides)

    cfg.setdefault("meta", {})
    cfg["meta"]["seed"] = args.seed
    cfg["meta"]["profile"] = args.profile
    if args.device is not None:
        cfg["meta"]["device"] = args.device
    elif "device" not in cfg["meta"]:
        cfg["meta"]["device"] = "auto"

    return cfg


def build_run_dir(cfg: Dict[str, Any], run_root: str) -> Path:
    exp_name = cfg["meta"]["experiment_name"]
    seed = cfg["meta"]["seed"]
    profile = cfg["meta"]["profile"]

    run_dir = Path(run_root) / profile / exp_name / f"seed_{seed}"
    safe_mkdir(run_dir)
    safe_mkdir(run_dir / "checkpoints")
    return run_dir


def main() -> None:
    args = parse_args()
    cfg = build_final_cfg(args)
    run_dir = build_run_dir(cfg, args.run_root)

    # rewrite logging dirs to absolute run-specific paths where useful
    cfg.setdefault("logging", {})
    cfg["logging"]["save_dir"] = str(run_dir)
    cfg["logging"]["checkpoint_dir"] = str(run_dir / "checkpoints")

    cfg["meta"]["config_path"] = args.config
    cfg["meta"]["git_hash"] = get_git_hash()
    cfg["meta"]["launch_time"] = now_str()
    cfg["meta"]["run_dir"] = str(run_dir)

    save_yaml(cfg, run_dir / "used_config.yaml")
    save_text(cfg["meta"]["git_hash"], run_dir / "git_hash.txt")

    try:
        summary = run_training(cfg, run_dir)
        save_json(summary, run_dir / "summary.json")
        print(f"[DONE] {cfg['meta']['experiment_name']} seed={cfg['meta']['seed']}")
    except Exception as e:
        error_summary = {
            "experiment_name": cfg["meta"]["experiment_name"],
            "seed": cfg["meta"]["seed"],
            "status": "failed",
            "error": str(e),
        }
        save_json(error_summary, run_dir / "summary.json")
        raise


if __name__ == "__main__":
    main()