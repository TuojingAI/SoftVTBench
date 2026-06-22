# Training

This repo supports four Tabero OpenPI LoRA training configs:

```text
pi0_lora_vision_tabero
pi05_lora_vision_tabero
pi0_lora_tacall_tabero
pi05_lora_tacall_tabero
```

All configs use:

```text
action_horizon = 50
batch_size = 256 by default
num_train_steps = 30000 by default
save_interval = 1000 by default
```

## Environment

Copy and edit:

```bash
cp configs/training/tabero_env.example configs/training/tabero_env.local
source configs/training/tabero_env.local
```

## Convert Data

```bash
scripts/prepare_tabero_dataset.sh vision
scripts/prepare_tabero_dataset.sh tactile
```

## Compute Norm Stats

```bash
scripts/compute_tabero_norm_stats.sh pi0 vision
scripts/compute_tabero_norm_stats.sh pi0 tactile
scripts/compute_tabero_norm_stats.sh pi05 vision
scripts/compute_tabero_norm_stats.sh pi05 tactile
```

## Train

Examples:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train_tabero_baseline.sh pi0 vision
CUDA_VISIBLE_DEVICES=4,5,6,7 scripts/train_tabero_baseline.sh pi0 tactile
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train_tabero_baseline.sh pi05 vision
CUDA_VISIBLE_DEVICES=4,5,6,7 scripts/train_tabero_baseline.sh pi05 tactile
```

The wrapper selects:

| Arguments | Config | Dataset | Effective action |
|---|---|---|---|
| `pi0 vision` | `pi0_lora_vision_tabero` | `REPO_VISION` / `ROOT_VISION` | 7D |
| `pi05 vision` | `pi05_lora_vision_tabero` | `REPO_VISION` / `ROOT_VISION` | 7D |
| `pi0 tactile` | `pi0_lora_tacall_tabero` | `REPO_TACTILE` / `ROOT_TACTILE` | 13D |
| `pi05 tactile` | `pi05_lora_tacall_tabero` | `REPO_TACTILE` / `ROOT_TACTILE` | 13D |

Checkpoints are written to:

```text
checkpoints/<config>/<exp>/<step>
```

Pass a step directory to the policy server or simulator evaluation.

