from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv


def _as_project_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _require(cfg: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required config key: {'.'.join(keys)}")
        cur = cur[key]
    return cur


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FlatObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        sample = np.asarray(self.observation(self.env.observation_space.sample()), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        arr = np.asarray(observation, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr.reshape(-1)


class ObservationHistoryWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, history_len: int):
        super().__init__(env)
        self.history_len = history_len
        self.obs_hist: deque[np.ndarray] = deque(maxlen=history_len)
        obs_dim = int(np.prod(self.env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(history_len, obs_dim),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.obs_hist.clear()
        for _ in range(self.history_len):
            self.obs_hist.append(obs_vec)
        return np.stack(self.obs_hist, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.obs_hist.append(obs_vec)
        return np.stack(self.obs_hist, axis=0), reward, terminated, truncated, info


class CostPenaltyWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        lagrangian_enabled: bool,
        lambda_ref: list[float],
        offroad_penalty: float,
        terminate_on_offroad: bool,
    ):
        super().__init__(env)
        self.lagrangian_enabled = lagrangian_enabled
        self.lambda_ref = lambda_ref
        self.offroad_penalty = float(offroad_penalty)
        self.terminate_on_offroad = bool(terminate_on_offroad)

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        info = dict(info)

        crashed = bool(info.get("crashed", False))

        vehicle = getattr(self.env.unwrapped, "vehicle", None)
        if not crashed and vehicle is not None:
            crashed = bool(getattr(vehicle, "crashed", False))

        on_road = True
        rewards_dict = info.get("rewards", {})
        if isinstance(rewards_dict, dict) and "on_road_reward" in rewards_dict:
            on_road = bool(float(rewards_dict["on_road_reward"]) > 0.0)
        elif vehicle is not None and hasattr(vehicle, "on_road"):
            on_road = bool(getattr(vehicle, "on_road"))
        offroad = not on_road

        cost = 1.0 if (crashed or offroad) else 0.0
        info["cost"] = cost
        info["offroad"] = offroad

        if offroad and self.offroad_penalty > 0.0:
            reward = float(reward) - self.offroad_penalty

        if offroad and self.terminate_on_offroad:
            terminated = True

        if self.lagrangian_enabled:
            reward = float(reward) - float(self.lambda_ref[0]) * cost
        return obs, float(reward), terminated, truncated, info


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, d_model: int, n_heads: int, n_layers: int, dropout: float):
        assert isinstance(observation_space, gym.spaces.Box)
        assert len(observation_space.shape) == 2
        history_len, obs_dim = observation_space.shape
        super().__init__(observation_space, features_dim=d_model)
        self.obs_embed = nn.Linear(obs_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        pe = torch.zeros(history_len, d_model)
        pos = torch.arange(0, history_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.obs_embed(observations)
        x = x + self.pe[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)
        x = self.encoder(x)
        return self.norm(x[:, -1, :])


def build_env(cfg: dict[str, Any], lagrangian_enabled: bool, lambda_ref: list[float]) -> gym.Env:
    env_cfg = cfg.get("env", {}).get("config", None)
    if env_cfg:
        env_cfg = dict(env_cfg)

    safety_cfg = cfg.get("safety", {})
    env_offroad_terminal = bool(safety_cfg.get("env_offroad_terminal", True))
    if env_cfg is None:
        env_cfg = {"offroad_terminal": env_offroad_terminal}
    else:
        env_cfg["offroad_terminal"] = bool(env_cfg.get("offroad_terminal", env_offroad_terminal))

    offroad_penalty = float(safety_cfg.get("offroad_penalty", 5.0))
    terminate_on_offroad = bool(safety_cfg.get("terminate_on_offroad", True))

    env_name = _require(cfg, ["env", "name"])
    env = gym.make(env_name, config=env_cfg) if env_cfg else gym.make(env_name)
    if env_cfg and hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure(env_cfg)

    max_episode_steps = int(cfg.get("env", {}).get("max_episode_steps", 0))
    if max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    env = FlatObsWrapper(env)

    history_enabled = bool(cfg.get("history", {}).get("enabled", False))
    if history_enabled:
        history_len = int(cfg.get("history", {}).get("history_len", 1))
        env = ObservationHistoryWrapper(env, history_len=history_len)

    env = CostPenaltyWrapper(
        env,
        lagrangian_enabled=lagrangian_enabled,
        lambda_ref=lambda_ref,
        offroad_penalty=offroad_penalty,
        terminate_on_offroad=terminate_on_offroad,
    )
    return env


def build_env_factory(cfg: dict[str, Any], lagrangian_enabled: bool, lambda_ref: list[float]):
    def _factory():
        return build_env(cfg, lagrangian_enabled=lagrangian_enabled, lambda_ref=lambda_ref)

    return _factory


def evaluate(model: PPO, cfg: dict[str, Any], episodes: int = 10) -> tuple[float, float]:
    lambda_ref = [0.0]
    env = build_env(cfg, lagrangian_enabled=False, lambda_ref=lambda_ref)
    returns: list[float] = []
    costs: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(cfg.get("meta", {}).get("seed", 1)) + 10000 + ep)
        done = False
        ep_ret = 0.0
        ep_cost = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_cost += float(info.get("cost", 0.0))
            done = terminated or truncated
        returns.append(ep_ret)
        costs.append(ep_cost)
    env.close()
    return float(np.mean(returns)), float(np.mean(costs))


def write_baseline_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timesteps", "lambda", "eval_return", "eval_cost"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_training_curve(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        return
    xs = [int(r["timesteps"]) for r in rows]
    rets = [float(r["eval_return"]) for r in rows]
    costs = [float(r["eval_cost"]) for r in rows]
    lambdas = [float(r["lambda"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(xs, rets, marker="o", label="eval_return")
    axes[0].plot(xs, costs, marker="s", label="eval_cost")
    axes[0].set_ylabel("Return / Cost")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, lambdas, marker="^", color="tab:red", label="lambda")
    axes[1].set_ylabel("Lambda")
    axes[1].set_xlabel("Timesteps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def warmstart_features_from_pretrain(model: PPO, cfg: dict[str, Any], project_root: Path) -> None:
    pre_cfg = cfg.get("pretraining", {})
    if not bool(pre_cfg.get("enabled", False)):
        return

    ckpt_path_str = pre_cfg.get("init_checkpoint", None)
    if not ckpt_path_str:
        exp_name = str(cfg.get("meta", {}).get("experiment_name", ""))
        if "_cat_" in exp_name:
            base = exp_name.replace("_cat_", "_cat_")
        base = exp_name
        inferred = f"checkpoints/{base}_pretrain_pretrain_best.pt"
        ckpt_path = _as_project_path(inferred, project_root)
    else:
        ckpt_path = _as_project_path(str(ckpt_path_str), project_root)

    if not ckpt_path.exists():
        print(f"[WARN] pretraining.enabled=true but init checkpoint not found: {ckpt_path}")
        return

    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model_state_dict", {}) if isinstance(payload, dict) else {}
    if not isinstance(state, dict):
        print(f"[WARN] Invalid pretrain checkpoint format: {ckpt_path}")
        return

    fe = model.policy.features_extractor
    if not hasattr(fe, "obs_embed") or not hasattr(fe, "encoder"):
        print("[WARN] Current policy extractor is not transformer-like; skipping pretrain warm-start.")
        return

    obs_embed_sd = {k.replace("obs_embed.", "", 1): v for k, v in state.items() if k.startswith("obs_embed.")}
    encoder_sd = {k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")}

    loaded_any = False
    if obs_embed_sd:
        miss, unexp = fe.obs_embed.load_state_dict(obs_embed_sd, strict=False)
        loaded_any = True
        if miss or unexp:
            print(f"[INFO] obs_embed partial load. missing={len(miss)} unexpected={len(unexp)}")
    if encoder_sd:
        miss, unexp = fe.encoder.load_state_dict(encoder_sd, strict=False)
        loaded_any = True
        if miss or unexp:
            print(f"[INFO] encoder partial load. missing={len(miss)} unexpected={len(unexp)}")

    if loaded_any:
        print(f"[INFO] Loaded pretraining init weights from: {ckpt_path}")
    else:
        print(f"[WARN] No compatible transformer weights found in pretrain checkpoint: {ckpt_path}")


def run(config_path: Path, project_root: Path) -> None:
    cfg = load_config(config_path)

    torch_num_threads = int(cfg.get("meta", {}).get("torch_num_threads", 1))
    torch_num_threads = max(torch_num_threads, 1)
    torch.set_num_threads(torch_num_threads)

    experiment_name = str(_require(cfg, ["meta", "experiment_name"]))
    requested_device = str(_require(cfg, ["meta", "device"])).lower()
    policy_family = str(cfg.get("model", {}).get("policy_family", "mlp")).lower()
    force_cpu = bool(cfg.get("meta", {}).get("force_cpu", False))

    if force_cpu:
        device = "cpu"
        print("[INFO] force_cpu=true, overriding device to CPU.")
    elif requested_device in {"cpu", "cuda"}:
        device = requested_device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
            device = "cpu"
    elif requested_device in {"auto", "fastest"}:
        if policy_family in {"cat", "transformer"} and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"[INFO] auto device selection: policy_family={policy_family}, selected={device}")
    else:
        device = "cpu"
        print(f"[WARN] Unknown device setting '{requested_device}', defaulting to CPU.")

    save_dir = _as_project_path(str(_require(cfg, ["logging", "save_dir"])), project_root)
    ckpt_dir = _as_project_path(str(_require(cfg, ["logging", "checkpoint_dir"])), project_root)
    run_dir = save_dir / experiment_name / "online"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.get("logging", {}).get("save_used_config", True)):
        with (run_dir / "used_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    algo_name = str(_require(cfg, ["algo", "name"])).lower()
    lagrangian_enabled = bool(cfg.get("lagrangian", {}).get("enabled", False)) and algo_name == "ppo_lagrangian"
    lambda_ref = [float(cfg.get("lagrangian", {}).get("init_lambda", 0.0))]
    lambda_lr = float(cfg.get("lagrangian", {}).get("lambda_lr", 0.01))
    lambda_max = float(cfg.get("lagrangian", {}).get("lambda_max", 100.0))
    cost_threshold = float(cfg.get("safety", {}).get("cost_threshold", 0.1))

    parallel_envs = int(cfg.get("algo", {}).get("parallel_envs", 1))
    parallel_envs = max(parallel_envs, 1)
    train_env = make_vec_env(
        build_env_factory(cfg, lagrangian_enabled=lagrangian_enabled, lambda_ref=lambda_ref),
        n_envs=parallel_envs,
        vec_env_cls=DummyVecEnv,
        seed=int(_require(cfg, ["meta", "seed"])),
    )

    rollout_steps = int(_require(cfg, ["algo", "rollout_steps"]))
    minibatches = int(_require(cfg, ["algo", "minibatches"]))
    rollout_steps_per_env = max(rollout_steps // parallel_envs, 64)
    batch_size = max(rollout_steps // max(minibatches, 1), 32)

    hidden_dim = int(cfg.get("model", {}).get("hidden_dim", 256))

    policy_kwargs: dict[str, Any] = {
        "activation_fn": nn.Tanh,
        "net_arch": dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim]),
    }
    if policy_family in {"transformer", "cat"}:
        trans = cfg.get("transformer", {})
        policy_kwargs = {
            "features_extractor_class": TransformerFeaturesExtractor,
            "features_extractor_kwargs": {
                "d_model": int(trans.get("d_model", 256)),
                "n_heads": int(trans.get("n_heads", 4)),
                "n_layers": int(trans.get("n_layers", 4)),
                "dropout": float(trans.get("dropout", 0.1)),
            },
            "activation_fn": nn.Tanh,
            "net_arch": dict(pi=[hidden_dim], vf=[hidden_dim]),
        }

    resume_from = cfg.get("training", {}).get("resume_from", None)
    if resume_from:
        resume_path = _as_project_path(str(resume_from), project_root)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        model = PPO.load(
            str(resume_path),
            env=train_env,
            device=device,
            custom_objects={
                "observation_space": train_env.observation_space,
                "action_space": train_env.action_space,
            },
        )
        print(f"[INFO] Resumed online training from: {resume_path}")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=float(_require(cfg, ["algo", "learning_rate"])),
            n_steps=rollout_steps_per_env,
            batch_size=batch_size,
            n_epochs=int(_require(cfg, ["algo", "update_epochs"])),
            gamma=float(_require(cfg, ["algo", "gamma"])),
            gae_lambda=float(_require(cfg, ["algo", "gae_lambda"])),
            clip_range=float(_require(cfg, ["algo", "clip_ratio"])),
            ent_coef=float(_require(cfg, ["algo", "entropy_coef"])),
            vf_coef=float(_require(cfg, ["algo", "value_coef"])),
            max_grad_norm=float(_require(cfg, ["algo", "max_grad_norm"])),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(run_dir / "tb") if bool(cfg.get("logging", {}).get("tensorboard", True)) else None,
            device=device,
            verbose=1,
            seed=int(_require(cfg, ["meta", "seed"])),
        )

        warmstart_features_from_pretrain(model, cfg, project_root)

    total_timesteps = int(_require(cfg, ["algo", "total_timesteps"]))
    eval_interval = int(cfg.get("evaluation", {}).get("eval_interval", 10000))
    eval_episodes = int(cfg.get("evaluation", {}).get("eval_episodes", 10))

    metrics_rows: list[dict[str, Any]] = []
    use_progress_bar = bool(cfg.get("logging", {}).get("progress_bar", False))

    print(
        f"[INFO] train setup: parallel_envs={parallel_envs}, n_steps_per_env={rollout_steps_per_env}, "
        f"rollout_batch={parallel_envs * rollout_steps_per_env}, progress_bar={use_progress_bar}, "
        f"torch_num_threads={torch_num_threads}"
    )

    done_steps = 0
    while done_steps < total_timesteps:
        chunk = min(eval_interval, total_timesteps - done_steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=use_progress_bar)
        done_steps += chunk

        eval_ret, eval_cost = evaluate(model, cfg, episodes=eval_episodes)
        if lagrangian_enabled:
            lambda_ref[0] = float(np.clip(lambda_ref[0] + lambda_lr * (eval_cost - cost_threshold), 0.0, lambda_max))
        metrics_rows.append(
            {
                "timesteps": done_steps,
                "lambda": lambda_ref[0] if lagrangian_enabled else 0.0,
                "eval_return": eval_ret,
                "eval_cost": eval_cost,
            }
        )
        print(
            f"[Eval {done_steps}/{total_timesteps}] return={eval_ret:.3f} cost={eval_cost:.3f} "
            f"lambda={(lambda_ref[0] if lagrangian_enabled else 0.0):.4f}"
        )

    model_path = ckpt_dir / f"{experiment_name}_online_final"
    model.save(str(model_path))
    metrics_path = run_dir / "baseline_metrics.csv"
    curve_path = run_dir / "training_curve.png"
    write_baseline_metrics(metrics_path, metrics_rows)
    plot_training_curve(curve_path, metrics_rows, title=experiment_name)
    train_env.close()

    print("\nOnline training finished.")
    print(f"Model: {model_path}.zip")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Training curve: {curve_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Online baseline trainer for PPO / PPO-Lagrangian (C=1).")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline config YAML")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = _as_project_path(args.config, project_root)
    run(config_path=config_path, project_root=project_root)


if __name__ == "__main__":
    main()
