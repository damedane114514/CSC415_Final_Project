from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np


@dataclass
class PrimitiveStep:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    step_cost: float


class HighwayEnvWrapper:
    def __init__(
        self,
        env_name: str,
        env_config: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> None:
        self.env_name = env_name
        self.env_config = env_config
        self.seed = seed
        self.env = gym.make(env_name, config=env_config)

        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_collided = False

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        use_seed = self.seed if seed is None else seed
        obs, info = self.env.reset(seed=use_seed)
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_collided = False
        return self._flatten_obs(obs), info

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        return arr.reshape(-1)

    def step(self, action: np.ndarray) -> PrimitiveStep:
        obs, reward, terminated, truncated, info = self.env.step(action)
        crashed = bool(info.get("crashed", False))
        step_cost = 1.0 if crashed else 0.0

        self.episode_return += float(reward)
        self.episode_length += 1
        self.episode_collided = self.episode_collided or crashed

        return PrimitiveStep(
            obs=self._flatten_obs(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            step_cost=step_cost,
        )

    def execute_actions(
        self,
        action_chunk: np.ndarray,
        execute_len: int,
    ) -> Tuple[List[PrimitiveStep], Dict[str, Any]]:
        """
        Execute the first execute_len primitive actions from action_chunk.
        action_chunk shape: [C, action_dim]
        """
        results: List[PrimitiveStep] = []
        for i in range(min(execute_len, len(action_chunk))):
            step_res = self.step(action_chunk[i])
            results.append(step_res)
            if step_res.terminated or step_res.truncated:
                break

        summary = {
            "segment_reward_sum": float(sum(x.reward for x in results)),
            "segment_any_collision": float(any(x.step_cost > 0 for x in results)),
            "segment_len": len(results),
            "episode_return_so_far": self.episode_return,
            "episode_length_so_far": self.episode_length,
            "episode_collision_so_far": float(self.episode_collided),
        }
        return results, summary

    def get_episode_metrics(self) -> Dict[str, Any]:
        return {
            "episode_return": float(self.episode_return),
            "episode_length": int(self.episode_length),
            "episode_collision": float(self.episode_collided),
        }

    def close(self) -> None:
        self.env.close()


def build_obs_history(
    history_buffer: Deque[np.ndarray],
    history_len: int,
    obs_dim: int,
) -> np.ndarray:
    out = np.zeros((history_len, obs_dim), dtype=np.float32)
    buf_list = list(history_buffer)
    start = max(history_len - len(buf_list), 0)
    for i, x in enumerate(buf_list[-history_len:]):
        out[start + i] = x
    return out