# Tabero OpenPI Training and Evaluation

This repository is a clean, Tabero-focused staging repo extracted from an internal OpenPI fork. It keeps the code needed to:

- convert Tabero HDF5/MP4 demonstrations to LeRobot datasets,
- fine-tune pi0/pi05 OpenPI LoRA baselines for vision and vision+tactile policies,
- serve trained checkpoints through the OpenPI policy server,
- run simulator evaluation through the Tabero/Isaac Sim client.

Large artifacts are intentionally excluded. Do not commit raw demonstrations, LeRobot caches, checkpoints, logs, W&B runs, or simulator debug videos.

## Included Pipeline

| Stage | Entry point |
|---|---|
| Dataset download | `scripts/download_tabero_dataset.py` |
| Data conversion | `examples/tabero/convert_tabero_vision_data_to_lerobot.py`, `examples/tabero/convert_tabero_tactile_data_to_lerobot.py`, `scripts/prepare_tabero_dataset.sh` |
| Data alignment check | `examples/tabero/verify_tabero_frame_alignment.py` |
| Norm stats | `scripts/compute_norm_stats.py`, `scripts/compute_tabero_norm_stats.sh` |
| Training | `scripts/train.py`, `scripts/train_tabero_baseline.sh` |
| Serving | `scripts/serve_policy.py` |
| Simulator evaluation | `scripts/evaluate_tabero_simulator.sh`, `scripts/tabero_dynamic_eval/run_one_ckpt_eval.sh` |
| Tactile model | `src/openpi/models/pi0_tabero.py` |
| Tabero transforms | `src/openpi/policies/tabero_policy.py` |

## Policies

The repository supports four Tabero baselines:

| Policy | Config | Inputs | Effective action |
|---|---|---|---|
| pi0 vision | `pi0_lora_vision_tabero` | base RGB, wrist RGB, state, prompt | `50 x 7` |
| pi05 vision | `pi05_lora_vision_tabero` | base RGB, wrist RGB, state, prompt | `50 x 7` |
| pi0 vision+tactile | `pi0_lora_tacall_tabero` | base RGB, wrist RGB, tactile image, marker motion, force history, state, prompt | `50 x 13` |
| pi05 vision+tactile | `pi05_lora_tacall_tabero` | base RGB, wrist RGB, tactile image, marker motion, force history, state, prompt | `50 x 13` |

OpenPI internally pads actions to width 32. The Tabero output transform slices predictions back to 7D or 13D actions.

## Quick Start

Install dependencies:

```bash
uv sync
```

Prepare environment variables:

```bash
cp configs/training/tabero_env.example configs/training/tabero_env.local
```

Edit `configs/training/tabero_env.local` for your local paths, then source it:

```bash
source configs/training/tabero_env.local
```

Download the dataset from ModelScope:

```bash
scripts/download_tabero_dataset.py --local-dir data_tabero/raw/SoftTacWorld-v0
```

If the script prints a `Detected RAW_ROOT` different from `data_tabero/raw/SoftTacWorld-v0`, put that path in `configs/training/tabero_env.local`:

```bash
export RAW_ROOT=/detected/raw/root
```

Convert data:

```bash
scripts/prepare_tabero_dataset.sh vision
scripts/prepare_tabero_dataset.sh tactile
```

Compute norm stats:

```bash
scripts/compute_tabero_norm_stats.sh pi0 vision
scripts/compute_tabero_norm_stats.sh pi0 tactile
scripts/compute_tabero_norm_stats.sh pi05 vision
scripts/compute_tabero_norm_stats.sh pi05 tactile
```

Train:

```bash
scripts/train_tabero_baseline.sh pi0 vision
scripts/train_tabero_baseline.sh pi0 tactile
scripts/train_tabero_baseline.sh pi05 vision
scripts/train_tabero_baseline.sh pi05 tactile
```

Evaluate on a Tabero simulator host:

```bash
CONFIG=pi0_lora_tacall_tabero \
CKPT=/path/to/checkpoint/30000 \
MODE=tactile \
scripts/evaluate_tabero_simulator.sh
```

## Documentation

- [Install](INSTALL.md)
- [Dataset](docs/dataset.md)
- [Tasks](docs/tasks.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- Internal extraction notes are preserved under `docs/reference/`.

## Artifact Policy

Keep these out of Git:

- `.cache/`
- `data_tabero/`
- `checkpoints/`
- `assets/`
- `logs/`
- `wandb/`
- `evaluation_results/`
- raw `.hdf5` and `.mp4` files
