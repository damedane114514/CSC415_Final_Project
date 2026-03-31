from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _as_project_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _require(cfg: Dict[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            joined = ".".join(keys)
            raise KeyError(f"Missing required config key: {joined}")
        cur = cur[key]
    return cur


def _pick_npz_key(data: Dict[str, np.ndarray], candidates: Iterable[str], name: str) -> np.ndarray:
    for key in candidates:
        if key in data:
            return data[key]
    raise KeyError(
        f"Could not find {name} key in npz file. "
        f"Tried {list(candidates)}. Available keys: {list(data.keys())}"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ChunkDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.as_tensor(observations, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"PositionalEncoding requires even d_model, got {d_model}")
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len), persistent=False)

    def _build_pe(self, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            self.pe = self._build_pe(seq_len).to(device=self.pe.device)
        return x + self.pe[:, :seq_len, :].to(device=x.device, dtype=x.dtype)


class ChunkTransformerPolicy(nn.Module):
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
        self.chunk_size = chunk_size
        self.action_dim = action_dim

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
            nn.Linear(d_model, chunk_size * action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.obs_embed(obs)
        x = self.pos_enc(x)
        x = self.encoder(x)
        pooled = x[:, -1, :]
        out = self.head(pooled)
        return out.view(obs.shape[0], self.chunk_size, self.action_dim)


@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    val_loss: float


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(npz_path: Path, chunk_size: int, actions_2d_mode: str = "flattened_chunked") -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as npz_data:
        data = {k: npz_data[k] for k in npz_data.files}

    obs = _pick_npz_key(data, ["observations", "obs", "states", "state", "x"], "observations")
    actions = _pick_npz_key(data, ["actions", "action", "acts", "y"], "actions")

    if obs.ndim == 2:
        obs = obs[:, None, :]
    if obs.ndim != 3:
        raise ValueError(f"Expected observations with 2 or 3 dims, got shape {obs.shape}")

    if actions.ndim == 3:
        if actions.shape[1] != chunk_size:
            raise ValueError(
                f"Action chunk axis does not match chunk_size. actions.shape={actions.shape}, chunk_size={chunk_size}"
            )
    elif actions.ndim == 2:
        if actions_2d_mode != "flattened_chunked":
            raise ValueError(
                "2D actions were provided but actions_2d_mode is not 'flattened_chunked'. "
                "This script only reshapes pre-flattened chunked actions. "
                "If your data is single-step [N, action_dim], please pre-build chunked targets first."
            )
        if actions.shape[1] % chunk_size != 0:
            raise ValueError(
                f"actions second dim must be divisible by chunk_size when flattened. "
                f"actions.shape={actions.shape}, chunk_size={chunk_size}"
            )
        actions = actions.reshape(actions.shape[0], chunk_size, actions.shape[1] // chunk_size)
    else:
        raise ValueError(f"Expected actions with 2 or 3 dims, got shape {actions.shape}")

    if obs.shape[0] != actions.shape[0]:
        raise ValueError(f"Mismatched sample counts: obs={obs.shape[0]} actions={actions.shape[0]}")

    return obs.astype(np.float32), actions.astype(np.float32)


def maybe_freeze_backbone(model: ChunkTransformerPolicy) -> None:
    for p in model.obs_embed.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False
    if isinstance(model.pos_enc, nn.Module) and not isinstance(model.pos_enc, nn.Identity):
        for p in model.pos_enc.parameters():
            p.requires_grad = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    progress_desc: Optional[str] = None,
    show_progress: bool = False,
) -> float:
    model.train()
    running = 0.0
    count = 0
    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, total=len(loader), desc=progress_desc, leave=False, dynamic_ncols=True)

    for obs, actions in iterator:
        obs = obs.to(device)
        actions = actions.to(device)

        pred = model(obs)
        loss = criterion(pred, actions)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = obs.size(0)
        running += loss.item() * bs
        count += bs
    return running / max(count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    running = 0.0
    count = 0
    for obs, actions in loader:
        obs = obs.to(device)
        actions = actions.to(device)

        pred = model(obs)
        loss = criterion(pred, actions)

        bs = obs.size(0)
        running += loss.item() * bs
        count += bs
    return running / max(count, 1)


def write_metrics_csv(path: Path, stats: Iterable[TrainStats]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for s in stats:
            writer.writerow({"epoch": s.epoch, "train_loss": s.train_loss, "val_loss": s.val_loss})


def save_used_config(config: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run(
    config_path: Path,
    project_root: Path,
    dataset_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
) -> None:
    cfg = load_config(config_path)

    pre_enabled = bool(_require(cfg, ["pretraining", "enabled"]))
    if not pre_enabled:
        raise ValueError(f"Config has pretraining.enabled=false: {config_path}")

    seed = int(_require(cfg, ["meta", "seed"]))
    set_seed(seed)

    chunk_size = int(_require(cfg, ["chunking", "chunk_size"]))
    if chunk_size < 1:
        raise ValueError("chunking.chunk_size must be >= 1")

    dataset_path = dataset_override if dataset_override is not None else _require(cfg, ["pretraining", "dataset_path"])
    if dataset_override is not None:
        cfg["pretraining"]["dataset_path"] = dataset_override
    if not dataset_path:
        raise ValueError("pretraining.dataset_path must be set for pretraining configs")

    npz_path = _as_project_path(str(dataset_path), project_root)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    cfg["pretraining"]["dataset_path"] = str(npz_path)

    batch_size = int(batch_size_override if batch_size_override is not None else _require(cfg, ["pretraining", "batch_size"]))
    if batch_size_override is not None:
        cfg["pretraining"]["batch_size"] = batch_size

    epochs = int(epochs_override if epochs_override is not None else _require(cfg, ["pretraining", "epochs"]))
    if epochs_override is not None:
        cfg["pretraining"]["epochs"] = epochs
    if epochs < 1:
        raise ValueError("pretraining.epochs must be >= 1 when pretraining.enabled=true")

    actions_2d_mode = str(cfg.get("pretraining", {}).get("actions_2d_mode", "flattened_chunked"))
    if actions_2d_mode not in {"flattened_chunked"}:
        raise ValueError("pretraining.actions_2d_mode must be 'flattened_chunked'")

    obs, actions = load_dataset(npz_path, chunk_size, actions_2d_mode=actions_2d_mode)

    history_cfg = cfg.get("history", {})
    history_enabled = bool(history_cfg.get("enabled", False))
    configured_history_len = int(history_cfg.get("history_len", 1))
    dataset_history_len = int(obs.shape[1])
    if history_enabled:
        if configured_history_len != dataset_history_len:
            raise ValueError(
                "History length mismatch: "
                f"config history.history_len={configured_history_len}, "
                f"dataset sequence length={dataset_history_len}."
            )
    elif dataset_history_len != 1:
        raise ValueError(
            "Config has history.enabled=false but dataset observations contain sequence length "
            f"{dataset_history_len}. Expected sequence length 1."
        )

    obs_dim = obs.shape[-1]
    action_dim = actions.shape[-1]

    experiment_name = str(_require(cfg, ["meta", "experiment_name"]))
    save_dir = _as_project_path(str(_require(cfg, ["logging", "save_dir"])), project_root)
    ckpt_dir = _as_project_path(str(_require(cfg, ["logging", "checkpoint_dir"])), project_root)
    run_dir = save_dir / experiment_name / "pretraining"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if bool(_require(cfg, ["logging", "save_used_config"])):
        save_used_config(cfg, run_dir / "used_config.yaml")

    dataset = ChunkDataset(obs, actions)
    val_size = max(int(len(dataset) * 0.1), 1)
    train_size = max(len(dataset) - val_size, 1)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    device_str = str(_require(cfg, ["meta", "device"]))
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    model = ChunkTransformerPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        d_model=int(_require(cfg, ["transformer", "d_model"])),
        n_heads=int(_require(cfg, ["transformer", "n_heads"])),
        n_layers=int(_require(cfg, ["transformer", "n_layers"])),
        dropout=float(_require(cfg, ["transformer", "dropout"])),
        use_positional_encoding=bool(_require(cfg, ["transformer", "use_positional_encoding"])),
    ).to(device)

    if bool(_require(cfg, ["pretraining", "freeze_backbone"])):
        maybe_freeze_backbone(model)

    lr = float(_require(cfg, ["pretraining", "learning_rate"]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_path = ckpt_dir / f"{experiment_name}_pretrain_best.pt"
    last_path = ckpt_dir / f"{experiment_name}_pretrain_last.pt"

    history: list[TrainStats] = []
    show_progress = bool(cfg.get("logging", {}).get("progress_bar", True))

    epoch_iterator = range(1, epochs + 1)
    if show_progress and tqdm is not None:
        epoch_iterator = tqdm(epoch_iterator, total=epochs, desc="Epochs", dynamic_ncols=True)

    for epoch in epoch_iterator:
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            progress_desc=f"Train {epoch}/{epochs}",
            show_progress=show_progress,
        )
        val_loss = evaluate(model, val_loader, device, criterion)
        history.append(TrainStats(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        print(f"[Epoch {epoch:03d}/{epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if show_progress and tqdm is not None and hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "chunk_size": chunk_size,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "epoch": epochs,
            "val_loss": history[-1].val_loss,
        },
        last_path,
    )

    write_metrics_csv(run_dir / "pretrain_metrics.csv", history)

    print("\nPretraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Metrics CSV: {run_dir / 'pretrain_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline BC pretraining for CAT/Transformer policies from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config, e.g. configs/pretraining/cat_c4_pretrain.yaml")
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional override for pretraining.dataset_path")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for pretraining.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for pretraining.batch_size")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = _as_project_path(args.config, project_root)
    run(
        config_path=config_path,
        project_root=project_root,
        dataset_override=args.dataset_path,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
    )


if __name__ == "__main__":
    main()
