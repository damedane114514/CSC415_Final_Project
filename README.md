# CSC415 Final Project

This repository contains the full experiment pipeline for CAT / PPO experiments on `highway-v0`:
- offline data collection
- offline CAT pretraining
- baseline online training (C1)
- CAT raw training
- CAT pretraining-initialized training
- demo rendering from checkpoints

## One-step Run

Added one-step runner scripts that directly chain the commands documented below:

- `scripts/one_step_run.py`
- `scripts/one_step_run.ps1`

Run all stages:

```bash
python scripts/one_step_run.py
```

Or on Windows PowerShell:

```powershell
./scripts/one_step_run.ps1
```

Supported stages:
- `collect`
- `pretrain`
- `raw`
- `baseline`
- `online`

Examples:

```bash
python scripts/one_step_run.py --stages collect pretrain baseline
python scripts/one_step_run.py --dry-run
python scripts/one_step_run.py --continue-on-error
```

---
## Environment

Recommended:

- Python 3.12
- PyTorch 2.7.0
- highway-env 1.10.2

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Important Convention

- `history_len` is set to **32** in current configs.
- Online continuation uses `training.resume_from` and an existing `.zip` checkpoint.
- Pretraining outputs `.pt` checkpoints; online PPO uses `.zip` checkpoints.

---

## 1) Collect Offline Dataset

Default command (history = 32, future_steps = 16):

```bash
python scripts/collect_offline_data.py \
	--policy idm \
	--config configs/pretraining/cat_c4_pretrain.yaml \
	--output data/offline/highway_mixed_v1.npz \
	--samples 12000 \
	--history-len 32 \
	--future-steps 16 \
	--seed 1
```

Output:

- `data/offline/highway_mixed_v1.npz`

---

## 2) Offline Pretraining Experiments

Run one pretraining config:

```bash
python scripts/pretrain.py --config configs/pretraining/cat_c4_pretrain.yaml
python scripts/pretrain.py --config configs/pretraining/cat_c8_pretrain.yaml
python scripts/pretrain.py --config configs/pretraining/cat_c16_pretrain.yaml
```

Configs:

- `configs/pretraining/cat_c4_pretrain.yaml`
- `configs/pretraining/cat_c8_pretrain.yaml`
- `configs/pretraining/cat_c16_pretrain.yaml`

Outputs:

- `checkpoints/<experiment_name>_pretrain_best.pt`
- `checkpoints/<experiment_name>_pretrain_last.pt`
- `logs/<experiment_name>/pretraining/pretrain_metrics.csv`

---

## 3) Baseline Experiments (C1)

### 3.1 Fresh 50k baseline runs

Use the provided script:

```bash
bash scripts/run_baselines_fresh.sh
```

This runs:

1. `configs/baselines/ppo_mlp_c1.yaml`
2. `configs/baselines/ppolag_mlp_c1.yaml`
3. `configs/baselines/ppolag_trans_c1.yaml`

### 3.2 Continue a baseline run from checkpoint

To continue from 50k to 300k total:

1. Copy the original baseline config.
2. Set `algo.total_timesteps: 250000`.
3. Add:

```yaml
training:
	resume_from: "checkpoints/<original_experiment_name>_online_final.zip"
```

4. Optionally change `meta.experiment_name` for a new output folder.
5. Run:

```bash
python scripts/train_online.py --config <your_continue_config>.yaml
```

---

## 4) CAT Raw vs CAT Pretraining-Initialized (Main Experiments)

### C4 / C8 schedule

```bash
bash scripts/run_cat_c4c8_history32_schedule.sh
```

This script runs in order:

1. `configs/main/cat_c4_raw_250k.yaml`
2. `configs/main/cat_c8_raw_250k.yaml`
3. `configs/main/cat_c4_preinit_300k.yaml`
4. `configs/main/cat_c8_preinit_300k.yaml`

### C16 schedule

```bash
bash scripts/run_cat_c16_history32_schedule.sh
```

This script runs in order:

1. `configs/main/cat_c16_raw_300k.yaml`
2. `configs/main/cat_c16_preinit_300k.yaml`

---

## 5) Run a Single Online Experiment Manually

```bash
python scripts/train_online.py --config configs/main/cat_c4.yaml
python scripts/train_online.py --config configs/main/cat_c8.yaml
python scripts/train_online.py --config configs/main/cat_c16.yaml
```

---

## 6) Run with Another Seed

Recommended approach:

1. Duplicate the config (for example `cat_c4_preinit_300k_seed2.yaml`).
2. Change:

```yaml
meta:
	seed: 2
```

3. (Optional) change `meta.experiment_name` to avoid overwriting outputs.
4. Run `train_online.py` / `pretrain.py` with the new config.

---

## 7) Demo Rendering

From PPO `.zip` checkpoint:

```bash
python scripts/render_checkpoint_demo.py \
	--config configs/baselines/ppolag_mlp_c1.yaml \
	--checkpoint checkpoints/highwayv0_ppolag_mlp_c1_online_final.zip \
	--output-dir logs/demo_videos/ppolag_mlp_c1 \
	--episodes 2 --fps 15
```

From CAT pretraining `.pt` checkpoint:

```bash
python scripts/render_pretrained_cat_demo.py \
	--config configs/main/cat_c8.yaml \
	--checkpoint checkpoints/highwayv0_cat_c8_pretrain_pretrain_best.pt \
	--output-dir logs/demo_videos/pretrain_cat_c8 \
	--episodes 1 --fps 12 --base-seed 1
```

---

## 8) Quick Validation

```bash
python scripts/smoke_verify.py
```

---

## Outputs Summary

- Online checkpoints: `checkpoints/*_online_final.zip`
- Pretraining checkpoints: `checkpoints/*_pretrain_best.pt`, `checkpoints/*_pretrain_last.pt`
- Pretraining metrics: `logs/<experiment_name>/pretraining/pretrain_metrics.csv`
- Online metrics: `logs/<experiment_name>/online/baseline_metrics.csv`
- Curves: `logs/<experiment_name>/online/training_curve.png`
- Scheduler logs: `logs/schedules/*.log`
- Demo videos: `logs/demo_videos/**/*.mp4`
