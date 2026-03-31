# CSC415 Final Project

Offline pretraining and smoke verification for the CSC415 final project.

## Environment

This project is configured for:

- Python 3.12
- PyTorch 2.7.0
- highway-env 1.10.2

## Setup

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pretrain.txt
```

### PowerShell (Windows)

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-pretrain.txt
```

## Pretraining scripts

- `scripts/pretrain.py`: run one config
- `scripts/run_pretraining.ps1`: run all pretraining configs (PowerShell)
- `scripts/start_pretraining.sh`: Linux starter script
- `scripts/smoke_verify.py`: quick smoke verification

## Dataset format

Set `pretraining.dataset_path` in config YAML to an `.npz` file that contains:

- observation key: one of `observations`, `obs`, `states`, `state`, `x`
- action key: one of `actions`, `action`, `acts`, `y`

Supported tensor shapes:

- observations: `[N, obs_dim]` or `[N, H, obs_dim]`
- actions: `[N, chunk_size, action_dim]` or `[N, chunk_size * action_dim]`

## Usage

### Run one pretraining config

```bash
python scripts/pretrain.py --config configs/pretraining/cat_c4_pretrain.yaml
```

### Run all pretraining configs (PowerShell)

```powershell
./scripts/run_pretraining.ps1
```

### Smoke verify the core configs

```bash
python scripts/smoke_verify.py
```

### Linux starter script

```bash
bash scripts/start_pretraining.sh /absolute/path/to/offline_data.npz
```

Optional environment variables:

- `PYTHON_BIN=python3.12`
- `EPOCHS=2`
- `BATCH_SIZE=64`

## Outputs

- checkpoints are written to `checkpoints/` as `*_pretrain_best.pt` and `*_pretrain_last.pt`
- metrics and resolved config are saved under `logs/<experiment_name>/pretraining/`
