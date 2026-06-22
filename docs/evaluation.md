# Simulator Evaluation

Training runs in this repository. Simulator evaluation runs in a Tabero/Isaac Sim environment that contains:

```text
Tabero checkout
OpenPI checkout
Isaac Sim / conda tabero environment
LIBERO assembled HDF5 data
```

The original internal reference process is preserved in:

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
cp configs/evaluation/tabero_simulator_env.example configs/evaluation/tabero_simulator_env.local
source configs/evaluation/tabero_simulator_env.local
```

Then run:

```bash
CONFIG=pi0_lora_tacall_tabero \
CKPT=/path/to/checkpoints/pi0_lora_tacall_tabero/<EXP>/30000 \
MODE=tactile \
scripts/evaluate_tabero_simulator.sh
```

For a vision checkpoint:

```bash
CONFIG=pi0_lora_vision_tabero \
CKPT=/path/to/checkpoints/pi0_lora_vision_tabero/<EXP>/30000 \
MODE=vision_abs7d \
scripts/evaluate_tabero_simulator.sh
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

