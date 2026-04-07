from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


@dataclass
class ActionOutput:
    action_flat: torch.Tensor
    action_chunk: torch.Tensor
    logprob: torch.Tensor
    reward_value: torch.Tensor
    cost_value: torch.Tensor
    entropy: torch.Tensor


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, chunk_size: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.flat_action_dim = action_dim * chunk_size

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.flat_action_dim),
        )
        self.reward_value = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.cost_value = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(self.flat_action_dim))

    def _dist(self, obs_flat: torch.Tensor) -> Normal:
        mean = self.actor(obs_flat)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs_input: torch.Tensor, deterministic: bool = False) -> ActionOutput:
        # obs_input shape: [B, obs_dim]
        dist = self._dist(obs_input)
        action_flat = dist.mean if deterministic else dist.rsample()
        logprob = dist.log_prob(action_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        rv = self.reward_value(obs_input).squeeze(-1)
        cv = self.cost_value(obs_input).squeeze(-1)
        action_chunk = action_flat.view(action_flat.shape[0], self.chunk_size, self.action_dim)
        return ActionOutput(action_flat, action_chunk, logprob, rv, cv, entropy)

    def evaluate_actions(self, obs_input: torch.Tensor, action_flat: torch.Tensor):
        dist = self._dist(obs_input)
        logprob = dist.log_prob(action_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        rv = self.reward_value(obs_input).squeeze(-1)
        cv = self.cost_value(obs_input).squeeze(-1)
        return logprob, entropy, rv, cv


class TransformerActorCritic(nn.Module):
    """
    Names intentionally aligned with pretrain.py:
      obs_embed, pos_enc, encoder, head
    so that pretraining checkpoints can be loaded with strict=False.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_positional_encoding: bool,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.flat_action_dim = action_dim * chunk_size

        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model) if use_positional_encoding else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.flat_action_dim),
        )
        self.reward_value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.cost_value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(self.flat_action_dim))

    def _encode(self, obs_seq: torch.Tensor) -> torch.Tensor:
        x = self.obs_embed(obs_seq)
        x = self.pos_enc(x)
        x = self.encoder(x)
        pooled = x[:, -1, :]
        return pooled

    def _dist(self, obs_seq: torch.Tensor) -> Normal:
        pooled = self._encode(obs_seq)
        mean = self.head(pooled)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs_input: torch.Tensor, deterministic: bool = False) -> ActionOutput:
        pooled = self._encode(obs_input)
        mean = self.head(pooled)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)

        action_flat = dist.mean if deterministic else dist.rsample()
        logprob = dist.log_prob(action_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        rv = self.reward_value_head(pooled).squeeze(-1)
        cv = self.cost_value_head(pooled).squeeze(-1)
        action_chunk = action_flat.view(action_flat.shape[0], self.chunk_size, self.action_dim)
        return ActionOutput(action_flat, action_chunk, logprob, rv, cv, entropy)

    def evaluate_actions(self, obs_input: torch.Tensor, action_flat: torch.Tensor):
        pooled = self._encode(obs_input)
        mean = self.head(pooled)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        logprob = dist.log_prob(action_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        rv = self.reward_value_head(pooled).squeeze(-1)
        cv = self.cost_value_head(pooled).squeeze(-1)
        return logprob, entropy, rv, cv