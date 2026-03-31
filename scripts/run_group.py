from __future__ import annotations

import argparse
import subprocess
from typing import Dict, List


CONFIG_GROUPS: Dict[str, List[str]] = {
    "baselines": [
        "configs/baselines/ppo_mlp_c1.yaml",
        "configs/baselines/ppolag_mlp_c1.yaml",
        "configs/baselines/ppolag_trans_c1.yaml",
    ],
    "main": [
        "configs/main/cat_c4.yaml",
        "configs/main/cat_c8.yaml",
        "configs/main/cat_c16.yaml",
    ],
    "pretraining": [
        "configs/pretraining/cat_c4_pretrain.yaml",
        "configs/pretraining/cat_c8_pretrain.yaml",
        "configs/pretraining/cat_c16_pretrain.yaml",
    ],
    "replanning": [
        "configs/replanning/cat_c4_replan.yaml",
        "configs/replanning/cat_c8_replan.yaml",
        "configs/replanning/cat_c16_replan.yaml",
    ],
}
CONFIG_GROUPS["all"] = (
    CONFIG_GROUPS["baselines"]
    + CONFIG_GROUPS["main"]
    + CONFIG_GROUPS["pretraining"]
    + CONFIG_GROUPS["replanning"]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, choices=["baselines", "main", "pretraining", "replanning", "all"], required=True)
    parser.add_argument("--profile", type=str, choices=["debug", "main"], default="debug")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-root", type=str, default="/root/autodl-tmp/csc415_runs")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = CONFIG_GROUPS[args.group]

    for cfg in configs:
        for seed in range(1, args.num_seeds + 1):
            cmd = [
                "python",
                "scripts/train.py",
                "--config",
                cfg,
                "--seed",
                str(seed),
                "--profile",
                args.profile,
                "--device",
                args.device,
                "--run-root",
                args.run_root,
            ]
            print("[RUN]", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()