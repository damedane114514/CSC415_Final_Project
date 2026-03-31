from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
import gymnasium as gym
import highway_env  # noqa: F401

from pretrain import ChunkTransformerPolicy


CONFIGS = [
    "configs/baselines/ppo_mlp_c1.yaml",
    "configs/baselines/ppolag_mlp_c1.yaml",
    "configs/baselines/ppolag_trans_c1.yaml",
    "configs/main/cat_c4.yaml",
    "configs/main/cat_c8.yaml",
    "configs/main/cat_c16.yaml",
]


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_obs_vec(obs: Any) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def smoke_config(cfg_path: Path, rollout_steps: int = 12) -> None:
    cfg = load_yaml(cfg_path)
    env_name = cfg["env"]["name"]
    env = gym.make(env_name)
    if "config" in cfg["env"] and hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure(cfg["env"]["config"])

    obs, _ = env.reset(seed=int(cfg["meta"]["seed"]))
    obs_vec = _extract_obs_vec(obs)
    action_sample = env.action_space.sample()
    action_dim = int(np.asarray(action_sample).reshape(-1).shape[0])

    policy_family = str(cfg["model"]["policy_family"]).lower()
    hidden_dim = int(cfg["model"]["hidden_dim"])
    chunk_size = int(cfg["chunking"]["chunk_size"])

    if policy_family in {"cat", "transformer"}:
        model = ChunkTransformerPolicy(
            obs_dim=obs_vec.shape[0],
            action_dim=action_dim,
            chunk_size=max(chunk_size, 1),
            d_model=int(cfg["transformer"]["d_model"]),
            n_heads=int(cfg["transformer"]["n_heads"]),
            n_layers=int(cfg["transformer"]["n_layers"]),
            dropout=float(cfg["transformer"]["dropout"]),
            use_positional_encoding=bool(cfg["transformer"]["use_positional_encoding"]),
        )
        history_len = int(cfg["history"]["history_len"]) if cfg["history"]["enabled"] else 1
        batch = torch.randn(2, max(history_len, 1), obs_vec.shape[0])
        with torch.no_grad():
            out = model(batch)
        print(f"[{cfg_path.name}] model forward OK -> {tuple(out.shape)}")
    else:
        model = MLPPolicy(obs_dim=obs_vec.shape[0], action_dim=action_dim, hidden_dim=hidden_dim)
        batch = torch.randn(2, obs_vec.shape[0])
        with torch.no_grad():
            out = model(batch)
        print(f"[{cfg_path.name}] model forward OK -> {tuple(out.shape)}")

    total_reward = 0.0
    for _ in range(rollout_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print(f"[{cfg_path.name}] env rollout OK for {rollout_steps} steps | reward_sum={total_reward:.3f}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print("Running smoke verification for 6 configs...")
    for rel in CONFIGS:
        smoke_config(root / rel)
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
