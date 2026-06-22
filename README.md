# SoftTacWorld OpenPI Training and Evaluation

SoftTacWorld is the public-facing name for this OpenPI pipeline. The repository covers the full training and simulator-evaluation chain:

- download the public dataset from ModelScope,
- convert raw demonstrations into LeRobot datasets,
- compute normalization statistics,
- train pi0 / pi05 LoRA baselines for vision and vision+tactile policies,
- evaluate trained checkpoints in the simulator.

The runtime config ids and checkpoint layout keep the historical `tabero_*` identifiers for compatibility with existing artifacts and the reference server docs.

## What Is Included

| Stage | Entry point |
|---|---|
| Dataset download | `scripts/download_softtacworld_dataset.py` |
| Data conversion | `examples/tabero/convert_tabero_vision_data_to_lerobot.py`, `examples/tabero/convert_tabero_tactile_data_to_lerobot.py`, `scripts/prepare_softtacworld_dataset.sh` |
| Frame alignment check | `examples/tabero/verify_tabero_frame_alignment.py` |
| Norm stats | `scripts/compute_norm_stats.py`, `scripts/compute_softtacworld_norm_stats.sh` |
| Training | `scripts/train.py`, `scripts/train_softtacworld_baseline.sh` |
| Serving | `scripts/serve_policy.py` |
| Simulator evaluation | `scripts/evaluate_softtacworld_simulator.sh`, `scripts/softtacworld_dynamic_eval/run_one_ckpt_eval.sh` |
| Tactile model | `src/openpi/models/pi0_tabero.py` |
| Policy transforms | `src/openpi/policies/tabero_policy.py` |

## Baselines

| Policy | Config | Inputs | Effective action |
|---|---|---|---|
| pi0 vision | `pi0_lora_vision_tabero` | base RGB, wrist RGB, state, prompt | `50 x 7` |
| pi05 vision | `pi05_lora_vision_tabero` | base RGB, wrist RGB, state, prompt | `50 x 7` |
| pi0 vision+tactile | `pi0_lora_tacall_tabero` | base RGB, wrist RGB, tactile image, marker motion, force history, state, prompt | `50 x 13` |
| pi05 vision+tactile | `pi05_lora_tacall_tabero` | base RGB, wrist RGB, tactile image, marker motion, force history, state, prompt | `50 x 13` |

OpenPI pads actions internally to width 32. The policy output transform slices predictions back to 7D or 13D actions before execution.

## Quick Start

Install dependencies:

```bash
uv sync
```

Prepare environment variables:

```bash
cp configs/training/softtacworld_env.example configs/training/softtacworld_env.local
source configs/training/softtacworld_env.local
```

Download the public dataset:

```bash
scripts/download_softtacworld_dataset.py --local-dir data_softtacworld/raw/SoftTacWorld-v0
```

If the script prints a different `Detected RAW_ROOT`, set `RAW_ROOT` in `configs/training/softtacworld_env.local` to that path.

Convert data:

```bash
scripts/prepare_softtacworld_dataset.sh vision
scripts/prepare_softtacworld_dataset.sh tactile
```

Compute norm stats:

```bash
scripts/compute_softtacworld_norm_stats.sh pi0 vision
scripts/compute_softtacworld_norm_stats.sh pi0 tactile
scripts/compute_softtacworld_norm_stats.sh pi05 vision
scripts/compute_softtacworld_norm_stats.sh pi05 tactile
```

Train:

```bash
scripts/train_softtacworld_baseline.sh pi0 vision
scripts/train_softtacworld_baseline.sh pi0 tactile
scripts/train_softtacworld_baseline.sh pi05 vision
scripts/train_softtacworld_baseline.sh pi05 tactile
```

Evaluate on a simulator host:

```bash
CONFIG=pi0_lora_tacall_tabero \
CKPT=/path/to/checkpoint/30000 \
MODE=tactile \
scripts/evaluate_softtacworld_simulator.sh
```

## Documentation

- [Install](INSTALL.md)
- [Dataset](docs/dataset.md)
- [Tasks](docs/tasks.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- Reference notes are preserved under `docs/reference/`

## Artifacts

Keep these out of Git:

- `.cache/`
- `data_softtacworld/`
- `checkpoints/`
- `assets/`
- `logs/`
- `wandb/`
- `evaluation_results/`
- raw `.hdf5` and `.mp4` files
