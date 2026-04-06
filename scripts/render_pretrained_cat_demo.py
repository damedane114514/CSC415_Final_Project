from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from pretrain import ChunkTransformerPolicy, _as_project_path


def _flatten_obs(obs: Any) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(cfg: dict[str, Any]) -> gym.Env:
    env_name = str(cfg["env"]["name"])
    env_cfg = cfg.get("env", {}).get("config", None)
    env = gym.make(env_name, config=env_cfg, render_mode="rgb_array") if env_cfg else gym.make(env_name, render_mode="rgb_array")
    if env_cfg and hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure(env_cfg)
    return env


def load_pretrained_model(ckpt_path: Path, fallback_cfg: dict[str, Any], device: str = "cpu"):
    payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid pretrain checkpoint format: {ckpt_path}")

    saved_cfg = payload.get("config", fallback_cfg)
    obs_dim = int(payload["obs_dim"])
    action_dim = int(payload["action_dim"])
    chunk_size = int(payload["chunk_size"])

    model = ChunkTransformerPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        d_model=int(saved_cfg["transformer"]["d_model"]),
        n_heads=int(saved_cfg["transformer"]["n_heads"]),
        n_layers=int(saved_cfg["transformer"]["n_layers"]),
        dropout=float(saved_cfg["transformer"]["dropout"]),
        use_positional_encoding=bool(saved_cfg["transformer"].get("use_positional_encoding", True)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, saved_cfg, chunk_size, action_dim


@torch.no_grad()
def predict_chunk(model: ChunkTransformerPolicy, obs_hist: np.ndarray, device: str = "cpu") -> np.ndarray:
    inp = torch.as_tensor(obs_hist[None, ...], dtype=torch.float32, device=device)
    out = model(inp).squeeze(0).cpu().numpy()
    return out.astype(np.float32)


def run_demo(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    episodes: int,
    fps: int,
    base_seed: int,
    max_steps: int | None,
) -> None:
    cfg = load_yaml(config_path)
    model, saved_cfg, chunk_size, action_dim = load_pretrained_model(checkpoint_path, cfg, device="cpu")

    history_len = int(saved_cfg.get("history", {}).get("history_len", cfg.get("history", {}).get("history_len", 8)))
    env = build_env(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    env_action_dim = int(action_low.shape[0])
    if action_dim % env_action_dim != 0:
        raise ValueError(f"Action dim mismatch: env={env_action_dim}, checkpoint={action_dim}")
    sub_actions_per_slot = action_dim // env_action_dim

    exp_name = str(cfg.get("meta", {}).get("experiment_name", "cat_pretrain_demo"))
    summary: list[dict[str, Any]] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        obs_vec = _flatten_obs(obs)

        hist: deque[np.ndarray] = deque(maxlen=history_len)
        for _ in range(history_len):
            hist.append(obs_vec.copy())

        done = False
        ep_return = 0.0
        ep_collisions = 0
        ep_steps = 0
        frames: list[np.ndarray] = []

        while not done:
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)

            obs_hist = np.stack(hist, axis=0).astype(np.float32)
            chunk = predict_chunk(model, obs_hist, device="cpu")

            for i in range(chunk_size):
                slot_vec = chunk[i]
                slot_actions = slot_vec.reshape(sub_actions_per_slot, env_action_dim)
                for a in slot_actions:
                    action = np.clip(a, action_low, action_high).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)

                    ep_return += float(reward)
                    ep_steps += 1
                    if bool(info.get("crashed", False)):
                        ep_collisions += 1

                    obs_vec = _flatten_obs(obs)
                    hist.append(obs_vec)
                    done = bool(terminated or truncated)

                    if max_steps is not None and ep_steps >= max_steps:
                        done = True
                    if done:
                        break
                if done:
                    break

        end_frame = env.render()
        if isinstance(end_frame, np.ndarray):
            frames.append(end_frame)

        video_path = output_dir / f"{exp_name}_pretrain_ep{ep + 1:02d}.mp4"
        imageio.mimwrite(video_path, frames, fps=fps, codec="libx264")

        row = {
            "episode": ep + 1,
            "return": ep_return,
            "collisions": ep_collisions,
            "steps": ep_steps,
            "chunk_size": chunk_size,
            "slot_action_dim": action_dim,
            "env_action_dim": env_action_dim,
            "sub_actions_per_slot": sub_actions_per_slot,
            "video": str(video_path),
        }
        summary.append(row)
        print(
            f"[demo-pretrain] ep={row['episode']} return={row['return']:.3f} collisions={row['collisions']} "
            f"steps={row['steps']} chunk={chunk_size} video={row['video']}"
        )

    env.close()
    summary_path = output_dir / f"{exp_name}_pretrain_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render demo videos using CAT pretraining checkpoint (.pt).")
    parser.add_argument("--config", type=str, required=True, help="Config YAML (cat_c4/c8/c16)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to *_pretrain_best.pt or *_pretrain_last.pt")
    parser.add_argument("--output-dir", type=str, default="logs/demo_videos/pretrain_cat", help="Directory to save mp4 files")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--fps", type=int, default=12, help="Video FPS")
    parser.add_argument("--base-seed", type=int, default=1, help="Base seed")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap for steps per episode")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = _as_project_path(args.config, project_root)
    checkpoint_path = _as_project_path(args.checkpoint, project_root)
    output_dir = _as_project_path(args.output_dir, project_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_demo(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        episodes=max(1, args.episodes),
        fps=max(1, args.fps),
        base_seed=args.base_seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
