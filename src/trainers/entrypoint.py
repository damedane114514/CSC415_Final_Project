from __future__ import annotations

import time
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.highway_wrapper import HighwayEnvWrapper, build_obs_history
from src.models.actor_critic import MLPActorCritic, TransformerActorCritic
from src.utils.io import append_csv_row


@dataclass
class Transition:
    obs_input: np.ndarray
    action_flat: np.ndarray
    logprob: float
    reward_value: float
    cost_value: float
    reward: float
    done: float
    episode_cost: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def build_model(cfg: Dict[str, Any], obs_dim: int, action_dim: int, device: torch.device):
    family = str(cfg["model"]["policy_family"]).lower()
    chunk_size = int(cfg["chunking"]["chunk_size"])
    if family == "mlp":
        model = MLPActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            chunk_size=1,
        )
    elif family in {"transformer", "cat"}:
        model = TransformerActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            chunk_size=chunk_size if family == "cat" else 1,
            d_model=int(cfg["transformer"]["d_model"]),
            n_heads=int(cfg["transformer"]["n_heads"]),
            n_layers=int(cfg["transformer"]["n_layers"]),
            dropout=float(cfg["transformer"]["dropout"]),
            use_positional_encoding=bool(cfg["transformer"]["use_positional_encoding"]),
        )
    else:
        raise ValueError(f"Unknown policy family: {family}")
    return model.to(device)


def maybe_load_pretrain(model: torch.nn.Module, cfg: Dict[str, Any], run_dir: Path) -> None:
    if not bool(cfg["pretraining"]["enabled"]):
        return

    ckpt_path = cfg["pretraining"].get("checkpoint_path", None)
    if not ckpt_path:
        ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
        exp_name = cfg["meta"]["experiment_name"]
        ckpt_path = ckpt_dir / f"{exp_name}_pretrain_best.pt"

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretraining enabled but checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    print(f"[INFO] Loaded pretrain checkpoint: {ckpt_path}")
    print(f"[INFO] Missing keys after load: {missing}")
    print(f"[INFO] Unexpected keys after load: {unexpected}")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        adv[t] = last_gae
        next_value = values[t]
    returns = adv + values
    return adv, returns


def collect_batch(
    env: HighwayEnvWrapper,
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[List[Transition], Dict[str, float]]:
    history_enabled = bool(cfg["history"]["enabled"])
    history_len = int(cfg["history"]["history_len"])
    chunk_size = int(cfg["chunking"]["chunk_size"])
    partial_replanning = bool(cfg["chunking"]["partial_replanning"])
    replanning_interval = cfg["chunking"]["replanning_interval"]
    execute_mode = str(cfg["chunking"]["execute_mode"]).lower()
    rollout_steps = int(cfg["algo"]["rollout_steps"])

    obs, _ = env.reset(seed=int(cfg["meta"]["seed"]))
    obs_dim = obs.shape[0]
    hist_buf: Deque[np.ndarray] = deque(maxlen=history_len)
    hist_buf.append(obs)

    transitions: List[Transition] = []
    primitive_steps = 0
    episode_returns: List[float] = []
    episode_collisions: List[float] = []
    episode_lengths: List[int] = []

    current_episode_start_idx = 0

    while primitive_steps < rollout_steps:
        if history_enabled:
            obs_input = build_obs_history(hist_buf, history_len, obs_dim)
            obs_tensor = torch.as_tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            obs_input = obs.astype(np.float32)
            obs_tensor = torch.as_tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model.act(obs_tensor, deterministic=False)

        action_chunk = out.action_chunk.squeeze(0).cpu().numpy()

        if execute_mode == "stepwise":
            execute_len = 1
        else:
            if partial_replanning and replanning_interval is not None:
                execute_len = int(replanning_interval)
            else:
                execute_len = chunk_size

        action_chunk = np.clip(action_chunk, -1.0, 1.0)

        primitive_results, seg_summary = env.execute_actions(action_chunk, execute_len=execute_len)
        seg_reward = seg_summary["segment_reward_sum"]
        seg_done = 0.0
        for step_res in primitive_results:
            hist_buf.append(step_res.obs)
            obs = step_res.obs
            primitive_steps += 1
            if step_res.terminated or step_res.truncated:
                seg_done = 1.0
                break

        transitions.append(
            Transition(
                obs_input=obs_input.copy(),
                action_flat=out.action_flat.squeeze(0).cpu().numpy().copy(),
                logprob=float(out.logprob.squeeze(0).cpu().item()),
                reward_value=float(out.reward_value.squeeze(0).cpu().item()),
                cost_value=float(out.cost_value.squeeze(0).cpu().item()),
                reward=float(seg_reward),
                done=float(seg_done),
                episode_cost=0.0,
            )
        )

        if seg_done > 0.0:
            ep = env.get_episode_metrics()
            episode_returns.append(ep["episode_return"])
            episode_collisions.append(ep["episode_collision"])
            episode_lengths.append(ep["episode_length"])

            # assign episode-level cost to all transitions of this episode
            for i in range(current_episode_start_idx, len(transitions)):
                transitions[i].episode_cost = float(ep["episode_collision"])
            current_episode_start_idx = len(transitions)

            obs, _ = env.reset()
            hist_buf.clear()
            hist_buf.append(obs)

    stats = {
        "batch_mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "batch_collision_rate": float(np.mean(episode_collisions)) if episode_collisions else 0.0,
        "batch_mean_ep_len": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "primitive_steps": float(primitive_steps),
        "num_episodes": float(len(episode_returns)),
    }
    return transitions, stats


def evaluate_policy(
    env_name: str,
    env_config: Dict[str, Any],
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    num_episodes: int,
) -> Dict[str, float]:
    history_enabled = bool(cfg["history"]["enabled"])
    history_len = int(cfg["history"]["history_len"])
    chunk_size = int(cfg["chunking"]["chunk_size"])
    partial_replanning = bool(cfg["chunking"]["partial_replanning"])
    replanning_interval = cfg["chunking"]["replanning_interval"]
    execute_mode = str(cfg["chunking"]["execute_mode"]).lower()

    env = HighwayEnvWrapper(env_name=env_name, env_config=env_config, seed=int(cfg["meta"]["seed"]))
    returns, collisions, lengths = [], [], []

    for ep_i in range(num_episodes):
        obs, _ = env.reset(seed=int(cfg["meta"]["seed"]) + ep_i + 1000)
        obs_dim = obs.shape[0]
        hist_buf: Deque[np.ndarray] = deque(maxlen=history_len)
        hist_buf.append(obs)
        done = False

        while not done:
            if history_enabled:
                obs_input = build_obs_history(hist_buf, history_len, obs_dim)
                obs_tensor = torch.as_tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs_input = obs.astype(np.float32)
                obs_tensor = torch.as_tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                out = model.act(obs_tensor, deterministic=True)

            action_chunk = out.action_chunk.squeeze(0).cpu().numpy()
            action_chunk = np.clip(action_chunk, -1.0, 1.0)

            if execute_mode == "stepwise":
                execute_len = 1
            else:
                if partial_replanning and replanning_interval is not None:
                    execute_len = int(replanning_interval)
                else:
                    execute_len = chunk_size

            primitive_results, _ = env.execute_actions(action_chunk, execute_len=execute_len)
            for step_res in primitive_results:
                hist_buf.append(step_res.obs)
                obs = step_res.obs
                done = step_res.terminated or step_res.truncated
                if done:
                    break

        ep = env.get_episode_metrics()
        returns.append(ep["episode_return"])
        collisions.append(ep["episode_collision"])
        lengths.append(ep["episode_length"])

    env.close()

    return {
        "final_eval_return": float(np.mean(returns)) if returns else 0.0,
        "final_eval_collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "final_eval_episode_cost": float(np.mean(collisions)) if collisions else 0.0,
        "final_eval_episode_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def run_training(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    start_time = time.time()
    seed = int(cfg["meta"]["seed"])
    set_seed(seed)

    device = resolve_device(str(cfg["meta"]["device"]))
    env_name = cfg["env"]["name"]
    env_config = cfg["env"]["config"]

    env = HighwayEnvWrapper(env_name=env_name, env_config=env_config, seed=seed)
    obs0, _ = env.reset(seed=seed)
    obs_dim = obs0.shape[0]
    action_dim = int(np.asarray(env.action_space.sample()).reshape(-1).shape[0])

    model = build_model(cfg, obs_dim, action_dim, device)
    maybe_load_pretrain(model, cfg, run_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["algo"]["learning_rate"]))
    total_timesteps = int(cfg["algo"]["total_timesteps"])
    rollout_steps = int(cfg["algo"]["rollout_steps"])
    update_epochs = int(cfg["algo"]["update_epochs"])
    minibatches = int(cfg["algo"]["minibatches"])
    gamma = float(cfg["algo"]["gamma"])
    gae_lambda = float(cfg["algo"]["gae_lambda"])
    clip_ratio = float(cfg["algo"]["clip_ratio"])
    value_coef = float(cfg["algo"]["value_coef"])
    entropy_coef = float(cfg["algo"]["entropy_coef"])
    max_grad_norm = float(cfg["algo"]["max_grad_norm"])

    cost_threshold = float(cfg["safety"]["cost_threshold"])
    lag_enabled = bool(cfg["lagrangian"]["enabled"])
    lag_lambda = float(cfg["lagrangian"]["init_lambda"])
    lambda_lr = float(cfg["lagrangian"]["lambda_lr"])
    lambda_max = float(cfg["lagrangian"]["lambda_max"])

    metrics_path = run_dir / "metrics.csv"
    ckpt_dir = run_dir / "checkpoints"
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"

    best_eval_return = -1e18
    total_primitive_steps = 0
    update_idx = 0

    while total_primitive_steps < total_timesteps:
        transitions, batch_stats = collect_batch(env, model, cfg, device)
        total_primitive_steps += int(batch_stats["primitive_steps"])
        update_idx += 1

        obs_batch = np.stack([t.obs_input for t in transitions]).astype(np.float32)
        action_batch = np.stack([t.action_flat for t in transitions]).astype(np.float32)
        old_logprob_batch = np.array([t.logprob for t in transitions], dtype=np.float32)
        reward_val_batch = np.array([t.reward_value for t in transitions], dtype=np.float32)
        cost_val_batch = np.array([t.cost_value for t in transitions], dtype=np.float32)
        reward_batch = np.array([t.reward for t in transitions], dtype=np.float32)
        done_batch = np.array([t.done for t in transitions], dtype=np.float32)
        ep_cost_batch = np.array([t.episode_cost for t in transitions], dtype=np.float32)

        reward_adv, reward_ret = compute_gae(
            rewards=reward_batch,
            values=reward_val_batch,
            dones=done_batch,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        cost_adv, cost_ret = compute_gae(
            rewards=ep_cost_batch,
            values=cost_val_batch,
            dones=done_batch,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        reward_adv = (reward_adv - reward_adv.mean()) / (reward_adv.std() + 1e-8)
        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)

        combined_adv = reward_adv - (lag_lambda * cost_adv if lag_enabled else 0.0)

        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        act_tensor = torch.as_tensor(action_batch, dtype=torch.float32, device=device)
        old_logprob_tensor = torch.as_tensor(old_logprob_batch, dtype=torch.float32, device=device)
        reward_ret_tensor = torch.as_tensor(reward_ret, dtype=torch.float32, device=device)
        cost_ret_tensor = torch.as_tensor(cost_ret, dtype=torch.float32, device=device)
        comb_adv_tensor = torch.as_tensor(combined_adv, dtype=torch.float32, device=device)

        n = obs_tensor.shape[0]
        mb_size = max(n // minibatches, 1)

        policy_loss_v = 0.0
        reward_value_loss_v = 0.0
        cost_value_loss_v = 0.0
        entropy_v = 0.0

        for _ in range(update_epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb_idx = idx[start:start + mb_size]

                mb_obs = obs_tensor[mb_idx]
                mb_act = act_tensor[mb_idx]
                mb_old_logp = old_logprob_tensor[mb_idx]
                mb_rret = reward_ret_tensor[mb_idx]
                mb_cret = cost_ret_tensor[mb_idx]
                mb_adv = comb_adv_tensor[mb_idx]

                logp, entropy, r_v, c_v = model.evaluate_actions(mb_obs, mb_act)
                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                reward_value_loss = F.mse_loss(r_v, mb_rret)
                cost_value_loss = F.mse_loss(c_v, mb_cret)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + value_coef * reward_value_loss
                    + value_coef * cost_value_loss
                    + entropy_coef * entropy_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                policy_loss_v = float(policy_loss.item())
                reward_value_loss_v = float(reward_value_loss.item())
                cost_value_loss_v = float(cost_value_loss.item())
                entropy_v = float(entropy.mean().item())

        if lag_enabled:
            observed_cost = float(batch_stats["batch_collision_rate"])
            lag_lambda = min(lambda_max, max(0.0, lag_lambda + lambda_lr * (observed_cost - cost_threshold)))

        eval_every = int(cfg["evaluation"]["eval_interval"])
        do_eval = (
            total_primitive_steps >= eval_every
            and (total_primitive_steps // eval_every) > ((total_primitive_steps - int(batch_stats["primitive_steps"])) // eval_every)
        )

        eval_stats = {
            "final_eval_return": np.nan,
            "final_eval_collision_rate": np.nan,
            "final_eval_episode_cost": np.nan,
            "final_eval_episode_length": np.nan,
        }

        if do_eval:
            eval_stats = evaluate_policy(
                env_name=env_name,
                env_config=env_config,
                model=model,
                cfg=cfg,
                device=device,
                num_episodes=int(cfg["evaluation"]["eval_episodes"]),
            )

            if eval_stats["final_eval_return"] > best_eval_return:
                best_eval_return = eval_stats["final_eval_return"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg,
                        "update_idx": update_idx,
                        "total_primitive_steps": total_primitive_steps,
                    },
                    best_ckpt,
                )

        append_csv_row(
            metrics_path,
            {
                "update_idx": update_idx,
                "total_primitive_steps": total_primitive_steps,
                "batch_mean_return": batch_stats["batch_mean_return"],
                "batch_collision_rate": batch_stats["batch_collision_rate"],
                "batch_mean_ep_len": batch_stats["batch_mean_ep_len"],
                "num_episodes": batch_stats["num_episodes"],
                "lambda": lag_lambda,
                "policy_loss": policy_loss_v,
                "reward_value_loss": reward_value_loss_v,
                "cost_value_loss": cost_value_loss_v,
                "entropy": entropy_v,
                "eval_return": eval_stats["final_eval_return"],
                "eval_collision_rate": eval_stats["final_eval_collision_rate"],
                "eval_episode_cost": eval_stats["final_eval_episode_cost"],
                "eval_episode_length": eval_stats["final_eval_episode_length"],
            },
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "update_idx": update_idx,
                "total_primitive_steps": total_primitive_steps,
            },
            last_ckpt,
        )

        print(
            f"[Update {update_idx:04d}] "
            f"steps={total_primitive_steps} "
            f"batch_return={batch_stats['batch_mean_return']:.3f} "
            f"batch_collision={batch_stats['batch_collision_rate']:.3f} "
            f"lambda={lag_lambda:.4f}"
        )

    final_eval = evaluate_policy(
        env_name=env_name,
        env_config=env_config,
        model=model,
        cfg=cfg,
        device=device,
        num_episodes=int(cfg["evaluation"]["eval_episodes"]),
    )
    env.close()

    summary = {
        "experiment_name": cfg["meta"]["experiment_name"],
        "seed": seed,
        "status": "ok",
        "final_eval_return": final_eval["final_eval_return"],
        "final_eval_collision_rate": final_eval["final_eval_collision_rate"],
        "final_eval_episode_cost": final_eval["final_eval_episode_cost"],
        "final_eval_episode_length": final_eval["final_eval_episode_length"],
        "best_eval_return": best_eval_return if best_eval_return > -1e17 else final_eval["final_eval_return"],
        "best_eval_collision_rate": float("nan"),
        "total_timesteps": total_primitive_steps,
        "wallclock_sec": time.time() - start_time,
    }
    return summary