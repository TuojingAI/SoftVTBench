# Simulator Evaluation

Training runs in this repository. Simulator evaluation runs on a host that contains:

```text
OpenPI checkout
This repository or another checkout containing the policy/model code
Isaac Sim / conda evaluation environment
LIBERO assembled HDF5 data
```

The original reference process is preserved in:

```text
docs/reference/eval_tabero_on_simulator.md
```

## Action Interfaces

| Policy | Config | Server output | Client execution |
|---|---|---:|---|
| vision | `pi0_lora_vision_tabero`, `pi05_lora_vision_tabero` | `50 x 7` | tactile env with `--abs7d`, padded to 13D |
| vision+tactile | `pi0_lora_tacall_tabero`, `pi05_lora_tacall_tabero` | `50 x 13` | tactile env native 13D |

The server predicts 50 actions per request. The evaluation client executes only the first `REPLAN_STEPS` actions, then replans.

Reference settings:

```text
N = 10 episodes per task
REPLAN_STEPS = 10
MAX_INFERENCE_STEPS = 30
NUM_SUCCESS_STEPS = 8
```

## Single Evaluation Run

On the simulator host:

```bash
cp configs/evaluation/softtacworld_simulator_env.example configs/evaluation/softtacworld_simulator_env.local
source configs/evaluation/softtacworld_simulator_env.local
```

Set at least:

```bash
export TABERO_DIR=/path/to/simulator-workspace
export OPENPI_DIR=/path/to/softtacworld-github
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export DATA_DIR=/path/to/datasets/Isaaclab_Libero
export WARP_EXT=/path/to/omni.warp.core-1.5.0+lx64
```

Then run:

```bash
CONFIG=pi0_lora_tacall_tabero \
CKPT=/path/to/checkpoints/pi0_lora_tacall_tabero/<EXP>/30000 \
MODE=tactile \
scripts/evaluate_softtacworld_simulator.sh
```

For a vision checkpoint:

```bash
CONFIG=pi0_lora_vision_tabero \
CKPT=/path/to/checkpoints/pi0_lora_vision_tabero/<EXP>/30000 \
MODE=vision_abs7d \
scripts/evaluate_softtacworld_simulator.sh
```

By default, the wrapper evaluates `libero_object` and `libero_spatial` tasks `0 1 2 3 4 5 6 7 8 9`.

## Outputs

The evaluation wrapper writes:

```text
evaluation_results/<variant>/<exp>/<step>/replan_<steps>_n<N>/
  logs/
  debug/
  progress.tsv
  summary.csv
  run.info
```
