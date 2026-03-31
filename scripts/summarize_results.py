from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, choices=["debug", "main"], required=True)
    parser.add_argument("--run-root", type=str, default="/root/autodl-tmp/csc415_runs")
    return parser.parse_args()


def stderr(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values) / (len(values) ** 0.5)


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    root = Path(args.run_root) / args.profile
    summary_files = list(root.glob("*/seed_*/summary.json"))

    rows: List[Dict[str, Any]] = []
    for sf in summary_files:
        row = load_summary(sf)
        row["run_dir"] = str(sf.parent)
        rows.append(row)

    if not rows:
        print("No summaries found.")
        return

    out_dir = Path(args.run_root) / "summaries" / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "all_runs.csv", index=False)

    metric_cols = [
        "final_eval_return",
        "final_eval_collision_rate",
        "final_eval_episode_cost",
        "final_eval_episode_length",
        "best_eval_return",
        "wallclock_sec",
    ]

    aggs: List[Dict[str, Any]] = []
    for exp_name, sub in raw_df.groupby("experiment_name"):
        row: Dict[str, Any] = {"experiment_name": exp_name, "num_seeds": len(sub)}
        for metric in metric_cols:
            if metric in sub.columns:
                vals = [float(v) for v in sub[metric].dropna().tolist()]
                if vals:
                    row[f"{metric}_mean"] = mean(vals)
                    row[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
                    row[f"{metric}_stderr"] = stderr(vals)
        aggs.append(row)

    agg_df = pd.DataFrame(aggs)
    agg_df.to_csv(out_dir / "aggregated.csv", index=False)

    print(f"Saved: {out_dir / 'all_runs.csv'}")
    print(f"Saved: {out_dir / 'aggregated.csv'}")


if __name__ == "__main__":
    main()