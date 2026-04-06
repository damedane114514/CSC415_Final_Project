from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import imageio.v2 as imageio
import numpy as np
import yaml
from stable_baselines3 import PPO

from train_online import FlatObsWrapper, ObservationHistoryWrapper, _as_project_path


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_demo_env(cfg: dict[str, Any]) -> gym.Env:
    env_name = str(cfg["env"]["name"])
    env_cfg = cfg.get("env", {}).get("config", None)
    env = gym.make(env_name, config=env_cfg, render_mode="rgb_array") if env_cfg else gym.make(env_name, render_mode="rgb_array")
    if env_cfg and hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure(env_cfg)

    env = FlatObsWrapper(env)
    if bool(cfg.get("history", {}).get("enabled", False)):
        env = ObservationHistoryWrapper(env, history_len=int(cfg.get("history", {}).get("history_len", 1)))
    return env


def resolve_checkpoint_path(cfg: dict[str, Any], project_root: Path, explicit_checkpoint: str | None) -> Path:
    if explicit_checkpoint:
        return _as_project_path(explicit_checkpoint, project_root)
    exp_name = str(cfg["meta"]["experiment_name"])
    ckpt_dir = _as_project_path(str(cfg["logging"]["checkpoint_dir"]), project_root)
    return ckpt_dir / f"{exp_name}_online_final.zip"


def run_demo(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    episodes: int,
    fps: int,
    deterministic: bool,
    max_steps: int | None,
    base_seed_override: int | None,
) -> None:
    cfg = load_config(config_path)
    model = PPO.load(str(checkpoint_path), device="cpu")
    env = build_demo_env(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    base_seed = int(base_seed_override if base_seed_override is not None else cfg.get("meta", {}).get("seed", 1))
    exp_name = str(cfg.get("meta", {}).get("experiment_name", "demo"))

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        ep_return = 0.0
        ep_collisions = 0
        ep_steps = 0
        frames: list[np.ndarray] = []

        while not done:
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_steps += 1
            if bool(info.get("crashed", False)):
                ep_collisions += 1

            if max_steps is not None and ep_steps >= max_steps:
                done = True

        end_frame = env.render()
        if isinstance(end_frame, np.ndarray):
            frames.append(end_frame)

        video_path = output_dir / f"{exp_name}_ep{ep + 1:02d}.mp4"
        imageio.mimwrite(video_path, frames, fps=fps, codec="libx264")

        row = {
            "episode": ep + 1,
            "return": ep_return,
            "collisions": ep_collisions,
            "steps": ep_steps,
            "video": str(video_path),
        }
        summary.append(row)
        print(
            f"[demo] ep={row['episode']} return={row['return']:.3f} collisions={row['collisions']} "
            f"steps={row['steps']} video={row['video']}"
        )

    env.close()
    summary_path = output_dir / f"{exp_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render demo videos from a trained online checkpoint.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config used for training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .zip checkpoint; optional")
    parser.add_argument("--output-dir", type=str, default="logs/demo_videos", help="Where to save MP4 files")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=15, help="Video FPS")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy instead of deterministic")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap for steps per episode")
    parser.add_argument("--base-seed", type=int, default=None, help="Override seed for reproducible demo episodes")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = _as_project_path(args.config, project_root)
    cfg = load_config(config_path)
    checkpoint_path = resolve_checkpoint_path(cfg, project_root, args.checkpoint)
    output_dir = _as_project_path(args.output_dir, project_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_demo(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        episodes=max(1, args.episodes),
        fps=max(1, args.fps),
        deterministic=not args.stochastic,
        max_steps=args.max_steps,
        base_seed_override=args.base_seed,
    )


if __name__ == "__main__":
    main()
