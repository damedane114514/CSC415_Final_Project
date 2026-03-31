# CSC415_Final_Project
A repository that use for storing CSC415 Final Project. The project was completed collaboratively by Hongbin Gao, and Shijun Chen.

## Pretraining scripts

## Recommended environment (local / remote)

For current runs, use:

- Python 3.12
- PyTorch 2.7.0 with CUDA 12.8 (`cu128`)
- `highway-env==1.10.2`

Quick setup (Linux):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-pretrain.txt
```

Quick setup (PowerShell):

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-pretrain.txt
```

This repo now includes a config-driven offline pretraining entrypoint:

- `scripts/pretrain.py` (single config)
- `scripts/run_pretraining.ps1` (runs all three pretraining configs)
- `scripts/start_pretraining.sh` (Linux remote starter script)
- `scripts/smoke_verify.py` (quick functionality check for the 6 main experiment configs)

### Install dependencies

```bash
pip install -r requirements-pretrain.txt
```

### Dataset expectation

Set `pretraining.dataset_path` in each YAML to an `.npz` file containing:

- observations key: one of `observations`, `obs`, `states`, `state`, `x`
- actions key: one of `actions`, `action`, `acts`, `y`

Supported shapes:

- observations: `[N, obs_dim]` or `[N, H, obs_dim]`
- actions: `[N, chunk_size, action_dim]` or flattened `[N, chunk_size * action_dim]`

### Run one pretraining config

```bash
python scripts/pretrain.py --config configs/pretraining/cat_c4_pretrain.yaml
```

### Run all pretraining configs (PowerShell)

```powershell
./scripts/run_pretraining.ps1
```

### Smoke verify the 6 core configs (few steps)

```bash
python scripts/smoke_verify.py
```

This checks:

- PPO + MLP, `C=1`
- PPO-Lagrangian + MLP, `C=1`
- PPO-Lagrangian + Transformer, `C=1`
- CAT, `C=4/8/16`

### Linux remote starter (transfer-friendly)

```bash
bash scripts/start_pretraining.sh /absolute/path/to/highway_mixed_v1.npz
```

Optional env vars before running:

- `PYTHON_BIN=python3.10`
- `EPOCHS=2`
- `BATCH_SIZE=64`

Outputs:

- checkpoints in `checkpoints/` named `*_pretrain_best.pt` and `*_pretrain_last.pt`
- metrics CSV and used config under `logs/<experiment_name>/pretraining/`
