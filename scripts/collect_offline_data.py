from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import yaml
from tqdm.auto import tqdm


OBS_KEYS = ("observations", "obs", "states", "state", "x")
ACT_KEYS = ("actions", "action", "acts", "y")


def _obs_vec(obs: Any) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _sample_action(prev_action: np.ndarray, rng: np.random.Generator, aggressive_prob: float) -> np.ndarray:
    noise = rng.normal(0.0, 0.22, size=prev_action.shape).astype(np.float32)
    action = 0.88 * prev_action + noise
    if rng.random() < aggressive_prob:
        action += rng.normal(0.0, 0.45, size=prev_action.shape).astype(np.float32)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def collect_dataset(
    env_name: str,
    output_path: Path,
    env_config: dict[str, Any] | None,
    n_samples: int,
    history_len: int,
    future_steps: int,
    seed: int,
    max_steps_per_episode: int,
    aggressive_prob: float,
) -> None:
    if history_len < 1:
        raise ValueError("history_len must be >= 1")
    if future_steps < 1:
        raise ValueError("future_steps must be >= 1")

    env = gym.make(env_name)
    if env_config and hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure(env_config)
    rng = np.random.default_rng(seed)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    pbar = tqdm(total=n_samples, desc="Collecting offline samples", dynamic_ncols=True)
    episode_idx = 0

    while len(observations) < n_samples:
        obs, _ = env.reset(seed=seed + episode_idx)
        episode_idx += 1

        # Episode buffers indexed by environment time t.
        # obs_timeline[t] is observation at time t, before taking action a_t.
        obs_timeline: list[np.ndarray] = [_obs_vec(obs)]
        act_timeline: list[np.ndarray] = []

        prev_action = np.zeros_like(np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1))

        for _ in range(max_steps_per_episode):
            action_t = _sample_action(prev_action, rng, aggressive_prob)
            next_obs, _reward, terminated, truncated, _info = env.step(action_t)

            act_timeline.append(action_t.copy())
            obs_timeline.append(_obs_vec(next_obs))
            prev_action = action_t

            if terminated or truncated:
                break

        # Correct alignment:
        # input obs_seq ends at current time t, target is future actions [t, t+future_steps-1].
        # This avoids the bug where targets accidentally become past actions.
        episode_actions = len(act_timeline)
        for t in range(history_len - 1, episode_actions - future_steps + 1):
            obs_seq = np.stack(obs_timeline[t - history_len + 1 : t + 1], axis=0).astype(np.float32)
            fut = np.concatenate(act_timeline[t : t + future_steps], axis=0).astype(np.float32)

            observations.append(obs_seq)
            actions.append(fut)
            pbar.update(1)

            if len(observations) >= n_samples:
                break

    pbar.close()
    env.close()

    obs_arr = np.stack(observations, axis=0)
    act_arr = np.stack(actions, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, observations=obs_arr, actions=act_arr)

    print(f"Saved dataset: {output_path}")
    print(f"observations: shape={obs_arr.shape}, dtype={obs_arr.dtype}, min={obs_arr.min():.4f}, max={obs_arr.max():.4f}")
    print(f"actions: shape={act_arr.shape}, dtype={act_arr.dtype}, min={act_arr.min():.4f}, max={act_arr.max():.4f}")
    print(f"future_steps={future_steps} -> flattened target dim={act_arr.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect offline highway-env data with correct future-action alignment.")
    parser.add_argument("--env", type=str, default="highway-v0", help="Gymnasium env id")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretraining/cat_c4_pretrain.yaml",
        help="Optional YAML config path used to apply env.config (e.g., ContinuousAction)",
    )
    parser.add_argument("--output", type=str, default="data/offline/highway_mixed_v1.npz", help="Output .npz path")
    parser.add_argument("--samples", type=int, default=12000, help="Number of training samples to collect")
    parser.add_argument("--history-len", type=int, default=8, help="Observation history length")
    parser.add_argument("--future-steps", type=int, default=16, help="Number of future actions to flatten as target")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--max-steps-per-episode", type=int, default=400, help="Episode step cap")
    parser.add_argument("--aggressive-prob", type=float, default=0.10, help="Probability of aggressive action perturbation")

    args = parser.parse_args()

    env_cfg: dict[str, Any] | None = None
    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        env_cfg = cfg.get("env", {}).get("config", None)

    collect_dataset(
        env_name=args.env,
        output_path=Path(args.output).resolve(),
        env_config=env_cfg,
        n_samples=args.samples,
        history_len=args.history_len,
        future_steps=args.future_steps,
        seed=args.seed,
        max_steps_per_episode=args.max_steps_per_episode,
        aggressive_prob=args.aggressive_prob,
    )


if __name__ == "__main__":
    main()
