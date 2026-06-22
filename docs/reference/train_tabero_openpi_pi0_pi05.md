# Tabero -> OpenPI pi0/pi05 训练流程

目标：在本机 `/data/mingxinwang/openpi-univtac` 里训练 Tabero baseline 的四个 OpenPI LoRA 变体。

- pi0 纯视觉：`pi0_lora_vision_tabero`
- pi05 纯视觉：`pi05_lora_vision_tabero`
- pi0 视觉 + 触觉：`pi0_lora_tacall_tabero`
- pi05 视觉 + 触觉：`pi05_lora_tacall_tabero`

当前已转换好的 LeRobot 数据：

```bash
/data/mingxinwang/openpi-univtac/.cache/lerobot/local/tabero_vision_object_spatial_success_only_20260614
/data/mingxinwang/openpi-univtac/.cache/lerobot/local/tabero_tactile_object_spatial_success_only_20260614
```

原始 Tabero 成功轨迹数据：

```bash
/data/mingxinwang/openpi-univtac/data_tabero/train_ready_success_only_copy_20260614
/data/mingxinwang/openpi-univtac/data_tabero/.stage/tabero_object_spatial_success_only_20260614
```

说明：

- 当前数据集包含 `libero_object + libero_spatial` 成功轨迹，共 628 episodes / 88104 frames / 20 FPS。
- 纯视觉数据字段：`image [224,224,3]`、`wrist_image [224,224,3]`、`state [7]`、`actions [7]`。
- 触觉数据字段：额外包含 `tactile_image [224,224,3]`、`tactile_gripper_force [8,6]`、`tactile_marker_motion [9,198,2]`，`actions [13]`。
- 所有 Tabero config 都设为 `action_horizon=50`，训练 chunk 预测 50 步；评测端执行多少步由 simulator client 控制。
- 4 卡训练统一使用 global batch size `256`，`--fsdp-devices=4`，`--num-workers=8`。
- 默认训练步数来自 config：`num_train_steps=30000`，保存间隔：`save_interval=1000`。
- W&B 默认 online；如果不想上传，临时设置 `WANDB_MODE=offline`。

## 1. 环境

```bash
cd /data/mingxinwang/openpi-univtac

export PYTHON=/data/environment/miniconda3/envs/openpi/bin/python
export PYTHONPATH=/data/mingxinwang/openpi-univtac/src:/data/mingxinwang/openpi-univtac/packages/openpi-client/src:${PYTHONPATH:-}

export HF_LEROBOT_HOME=/data/mingxinwang/openpi-univtac/.cache/lerobot
export HF_HOME=/data/mingxinwang/openpi-univtac/.cache/huggingface
export HF_DATASETS_CACHE=/data/mingxinwang/openpi-univtac/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/data/mingxinwang/openpi-univtac/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online

mkdir -p logs/tabero_training_manual
```

可选检查：

```bash
$PYTHON - <<'PY'
import inspect
import jax
import openpi
import openpi.training.config as config

print(openpi.__file__)
print(inspect.getfile(config))
print(jax.devices())
PY
```

## 2. 数据和 checkpoint 变量

```bash
RUN_ID=tabero_object_spatial_success_only_20260614

REPO_VISION=local/tabero_vision_object_spatial_success_only_20260614
ROOT_VISION=/data/mingxinwang/openpi-univtac/.cache/lerobot/local/tabero_vision_object_spatial_success_only_20260614

REPO_TACTILE=local/tabero_tactile_object_spatial_success_only_20260614
ROOT_TACTILE=/data/mingxinwang/openpi-univtac/.cache/lerobot/local/tabero_tactile_object_spatial_success_only_20260614

PI0_BASE=/data/mingxinwang/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_base/params
PI05_BASE=/data/mingxinwang/openpi-univtac/.cache/openpi/openpi-assets/checkpoints/pi05_base/params

ASSET_PI0_VISION=/data/mingxinwang/openpi-univtac/assets/${RUN_ID}/pi0_vision
ASSET_PI05_VISION=/data/mingxinwang/openpi-univtac/assets/${RUN_ID}/pi05_vision
ASSET_PI0_TACTILE=/data/mingxinwang/openpi-univtac/assets/${RUN_ID}/pi0_tactile
ASSET_PI05_TACTILE=/data/mingxinwang/openpi-univtac/assets/${RUN_ID}/pi05_tactile

ASSET_ID_PI0_VISION=tabero_vision_pi0_h50
ASSET_ID_PI05_VISION=tabero_vision_pi05_h50
ASSET_ID_PI0_TACTILE=tabero_tactile_pi0_h50
ASSET_ID_PI05_TACTILE=tabero_tactile_pi05_h50
```

注意：`openpi-univtac/.cache/openpi/openpi-assets/checkpoints/pi0_base` 目前是空目录。完整 `pi0_base` 在：

```bash
/data/mingxinwang/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_base/params
```

所以 pi0 训练命令里显式传 `--weight-loader.params-path=${PI0_BASE}`。不要再复制一份 12G 权重到 `openpi-univtac`，会让目录更乱。

## 3. 数据转换：HDF5+MP4 -> LeRobot

当前已经转好了 LeRobot 数据，所以正常继续训练时可以直接跳到本节最后的“检查数据”。如果原始 HDF5/MP4 有更新，按下面命令重建 stage 目录并重新转换。

Tabero 原始训练数据不是直接可喂给 OpenPI 的 LeRobot 格式，而是：

```text
data_tabero/train_ready_success_only_copy_20260614/
  libero_object/
    replayed_demos/*.hdf5
    video_datasets/libero_object_task*/videos/*.mp4
    video_datasets/libero_object_task*/tactile_outputs/*.mp4
  libero_spatial/
    replayed_demos/*.hdf5
    video_datasets/libero_spatial_task*/videos/*.mp4
    video_datasets/libero_spatial_task*/tactile_outputs/*.mp4
```

转换脚本要求一个 Isaaclab_Libero 风格的单 root，所以先用软链接把 `libero_object` 和 `libero_spatial` 合并到 stage root：

```bash
RAW_ROOT=/data/mingxinwang/openpi-univtac/data_tabero/train_ready_success_only_copy_20260614
STAGE_ROOT=/data/mingxinwang/openpi-univtac/data_tabero/.stage/tabero_object_spatial_success_only_20260614
TASK_SUBSET=/data/mingxinwang/openpi-univtac/examples/tabero/config/tabero_object_spatial_success_only_20260614.json

mkdir -p "${STAGE_ROOT}/replayed_demos" "${STAGE_ROOT}/video_datasets"

for SUITE in libero_object libero_spatial; do
  for H5 in "${RAW_ROOT}/${SUITE}/replayed_demos/"*.hdf5; do
    ln -sfn "${H5}" "${STAGE_ROOT}/replayed_demos/$(basename "${H5}")"
  done
  for TASK_DIR in "${RAW_ROOT}/${SUITE}/video_datasets/${SUITE}"_task*; do
    ln -sfn "${TASK_DIR}" "${STAGE_ROOT}/video_datasets/$(basename "${TASK_DIR}")"
  done
done

find -L "${STAGE_ROOT}/replayed_demos" -maxdepth 1 -name '*.hdf5' | wc -l
find -L "${STAGE_ROOT}/video_datasets" -mindepth 1 -maxdepth 1 -type d | wc -l
```

期望两个计数都是 `20`：`libero_object` 10 个任务 + `libero_spatial` 10 个任务。

重新转换纯视觉 LeRobot 数据：

```bash
$PYTHON examples/tabero/convert_tabero_vision_data_to_lerobot.py \
  --data-root "${STAGE_ROOT}" \
  --repo-name "${REPO_VISION}" \
  --output-dir "${HF_LEROBOT_HOME}" \
  --task-suites libero_object libero_spatial \
  --task-subset-path "${TASK_SUBSET}" \
  2>&1 | tee logs/tabero_training_manual/convert_vision_${RUN_ID}.log
```

重新转换视觉 + 触觉 LeRobot 数据：

```bash
$PYTHON examples/tabero/convert_tabero_tactile_data_to_lerobot.py \
  --data-root "${STAGE_ROOT}" \
  --repo-name "${REPO_TACTILE}" \
  --output-dir "${HF_LEROBOT_HOME}" \
  --task-suites libero_object libero_spatial \
  --task-subset-path "${TASK_SUBSET}" \
  --tactile-output-type tactile_rgb \
  --force-history-len 8 \
  --marker-history-len 8 \
  2>&1 | tee logs/tabero_training_manual/convert_tactile_${RUN_ID}.log
```

注意：

- 不要加 `--overwrite`；当前两个 converter 没有这个参数。
- converter 如果发现输出目录已存在，会先清理再重建：`${HF_LEROBOT_HOME}/${REPO_VISION}` 或 `${HF_LEROBOT_HOME}/${REPO_TACTILE}`。
- `--task-subset-path` 必须用 `tabero_object_spatial_success_only_20260614.json`；默认 `tabero_tasks.json` 是早期评测子集，不会转换完整 object+spatial 20 个任务。
- 纯视觉 converter 会把 13D action 截成前 7D；触觉 converter 要求 HDF5 action 本身是 13D。

验证 HDF5/MP4 frame 对齐：

```bash
$PYTHON examples/tabero/verify_tabero_frame_alignment.py \
  --root "${STAGE_ROOT}" \
  --decode
```

期望结果：

```text
checked demos: 628
summary: {'all_equal': 628}
missing count: 0
```

检查转换结果：

```bash
for ROOT in "${ROOT_VISION}" "${ROOT_TACTILE}"; do
  echo "==== ${ROOT}"
  test -f "${ROOT}/meta/info.json"
  test -f "${ROOT}/meta/tasks.jsonl"
  $PYTHON - <<PY
import json
from pathlib import Path
root = Path("${ROOT}")
info = json.loads((root / "meta" / "info.json").read_text())
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("fps:", info.get("fps"))
print("features:", sorted(info.get("features", {}).keys()))
print("tasks_head:", (root / "meta" / "tasks.jsonl").read_text().splitlines()[:3])
PY
done
```

当前转换结果应为 628 episodes / 88104 frames / 20 tasks。

## 4. 计算 norm stats

这一步把训练所需的 `norm_stats.json` 写到四个 asset 目录。使用 `--low-dim-only`，避免为计算低维统计而解码所有视频。

```bash
$PYTHON scripts/compute_norm_stats.py \
  --config-name pi0_lora_vision_tabero \
  --repo-id "${REPO_VISION}" \
  --root "${ROOT_VISION}" \
  --assets-dir "${ASSET_PI0_VISION}" \
  --asset-id "${ASSET_ID_PI0_VISION}" \
  --low-dim-only \
  2>&1 | tee logs/tabero_training_manual/norm_pi0_vision.log

$PYTHON scripts/compute_norm_stats.py \
  --config-name pi05_lora_vision_tabero \
  --repo-id "${REPO_VISION}" \
  --root "${ROOT_VISION}" \
  --assets-dir "${ASSET_PI05_VISION}" \
  --asset-id "${ASSET_ID_PI05_VISION}" \
  --low-dim-only \
  2>&1 | tee logs/tabero_training_manual/norm_pi05_vision.log

$PYTHON scripts/compute_norm_stats.py \
  --config-name pi0_lora_tacall_tabero \
  --repo-id "${REPO_TACTILE}" \
  --root "${ROOT_TACTILE}" \
  --assets-dir "${ASSET_PI0_TACTILE}" \
  --asset-id "${ASSET_ID_PI0_TACTILE}" \
  --low-dim-only \
  2>&1 | tee logs/tabero_training_manual/norm_pi0_tactile.log

$PYTHON scripts/compute_norm_stats.py \
  --config-name pi05_lora_tacall_tabero \
  --repo-id "${REPO_TACTILE}" \
  --root "${ROOT_TACTILE}" \
  --assets-dir "${ASSET_PI05_TACTILE}" \
  --asset-id "${ASSET_ID_PI05_TACTILE}" \
  --low-dim-only \
  2>&1 | tee logs/tabero_training_manual/norm_pi05_tactile.log
```

检查输出：

```bash
find assets/${RUN_ID} -path '*norm_stats.json' -print | sort
```

期望至少有：

```text
assets/tabero_object_spatial_success_only_20260614/pi0_vision/tabero_vision_pi0_h50/norm_stats.json
assets/tabero_object_spatial_success_only_20260614/pi05_vision/tabero_vision_pi05_h50/norm_stats.json
assets/tabero_object_spatial_success_only_20260614/pi0_tactile/tabero_tactile_pi0_h50/norm_stats.json
assets/tabero_object_spatial_success_only_20260614/pi05_tactile/tabero_tactile_pi05_h50/norm_stats.json
```

## 5. 四种训练命令

建议一次跑两组：前四卡跑 vision，后四卡跑 tactile。pi0 和 pi05 如果都要正式跑，建议分批跑，避免 8 卡之外资源争抢。

### pi0 纯视觉，GPU 0-3

```bash
tmux new -s tabero_pi0_vision_train
```

tmux 里执行：

```bash
cd /data/mingxinwang/openpi-univtac

export PYTHON=/data/environment/miniconda3/envs/openpi/bin/python
export PYTHONPATH=/data/mingxinwang/openpi-univtac/src:/data/mingxinwang/openpi-univtac/packages/openpi-client/src:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/data/mingxinwang/openpi-univtac/.cache/lerobot
export HF_HOME=/data/mingxinwang/openpi-univtac/.cache/huggingface
export HF_DATASETS_CACHE=/data/mingxinwang/openpi-univtac/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/data/mingxinwang/openpi-univtac/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online

RUN_ID=tabero_object_spatial_success_only_20260614
REPO_VISION=local/tabero_vision_object_spatial_success_only_20260614
ROOT_VISION=/data/mingxinwang/openpi-univtac/.cache/lerobot/local/tabero_vision_object_spatial_success_only_20260614
PI0_BASE=/data/mingxinwang/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_base/params
ASSET_PI0_VISION=/data/mingxinwang/openpi-univtac/assets/${RUN_ID}/pi0_vision
ASSET_ID_PI0_VISION=tabero_vision_pi0_h50
EXP=tabero_object_spatial_success_only_pi0_vision_h50_bs256_4gpu_nw8_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
$PYTHON scripts/train.py pi0_lora_vision_tabero \
  --exp-name "${EXP}" \
  --data.repo-id "${REPO_VISION}" \
  --data.root "${ROOT_VISION}" \
  --data.assets.assets-dir "${ASSET_PI0_VISION}" \
  --data.assets.asset-id "${ASSET_ID_PI0_VISION}" \
  --weight-loader.params-path "${PI0_BASE}" \
  --fsdp-devices 4 \
  --batch-size 256 \
  --num-workers 8 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --log-interval 10 \
  --overwrite \
  2>&1 | tee logs/tabero_training_manual/${EXP}.log
```

### pi0 视觉 + 触觉，GPU 4-7

```bash
tmux new -s tabero_pi0_tactile_train
```

tmux 里执行：

```bash
cd /data/mingxinwang/openpi-univtac

export PYTHON=/data/environment/miniconda3/envs/openpi/bin/python
export PYTHONPATH=/data/mingxinwang/openpi-univtac/src:/data/mingxinwang/openpi-univtac/packages/openpi-client/src:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/data/mingxinwang/openpi-univtac/.cache/lerobot
export HF_HOME=/data/mingxinwang/openpi-univtac/.cache/huggingface
export HF_DATASETS_CACHE=/data/mingxinwang/openpi-univtac/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/data/mingxinwang/openpi-univtac/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online

RUN_ID=tabero_object_spatial_success_only_20260614
REPO_TACTILE=local/tabero_tactile_object_spatial_success_only_20260614
ROOT_TACTILE=/data/mingxinwang/openpi-univtac/.cache/lerobot/local/tabero_tactile_object_spatial_success_only_20260614
PI0_BASE=/data/mingxinwang/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_base/params
ASSET_PI0_TACTILE=/data/mingxinwang/openpi-univtac/assets/${RUN_ID}/pi0_tactile
ASSET_ID_PI0_TACTILE=tabero_tactile_pi0_h50
EXP=tabero_object_spatial_success_only_pi0_tactile_h50_bs256_4gpu_nw8_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=4,5,6,7 \
$PYTHON scripts/train.py pi0_lora_tacall_tabero \
  --exp-name "${EXP}" \
  --data.repo-id "${REPO_TACTILE}" \
  --data.root "${ROOT_TACTILE}" \
  --data.assets.assets-dir "${ASSET_PI0_TACTILE}" \
  --data.assets.asset-id "${ASSET_ID_PI0_TACTILE}" \
  --weight-loader.params-path "${PI0_BASE}" \
  --fsdp-devices 4 \
  --batch-size 256 \
  --num-workers 8 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --log-interval 10 \
  --overwrite \
  2>&1 | tee logs/tabero_training_manual/${EXP}.log
```

### pi05 纯视觉，GPU 0-3

```bash
tmux new -s tabero_pi05_vision_train
```

tmux 里执行：

```bash
cd /data/mingxinwang/openpi-univtac

export PYTHON=/data/environment/miniconda3/envs/openpi/bin/python
export PYTHONPATH=/data/mingxinwang/openpi-univtac/src:/data/mingxinwang/openpi-univtac/packages/openpi-client/src:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/data/mingxinwang/openpi-univtac/.cache/lerobot
export HF_HOME=/data/mingxinwang/openpi-univtac/.cache/huggingface
export HF_DATASETS_CACHE=/data/mingxinwang/openpi-univtac/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/data/mingxinwang/openpi-univtac/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online

RUN_ID=tabero_object_spatial_success_only_20260614
REPO_VISION=local/tabero_vision_object_spatial_success_only_20260614
ROOT_VISION=/data/mingxinwang/openpi-univtac/.cache/lerobot/local/tabero_vision_object_spatial_success_only_20260614
PI05_BASE=/data/mingxinwang/openpi-univtac/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
ASSET_PI05_VISION=/data/mingxinwang/openpi-univtac/assets/${RUN_ID}/pi05_vision
ASSET_ID_PI05_VISION=tabero_vision_pi05_h50
EXP=tabero_object_spatial_success_only_pi05_vision_h50_bs256_4gpu_nw8_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
$PYTHON scripts/train.py pi05_lora_vision_tabero \
  --exp-name "${EXP}" \
  --data.repo-id "${REPO_VISION}" \
  --data.root "${ROOT_VISION}" \
  --data.assets.assets-dir "${ASSET_PI05_VISION}" \
  --data.assets.asset-id "${ASSET_ID_PI05_VISION}" \
  --weight-loader.params-path "${PI05_BASE}" \
  --fsdp-devices 4 \
  --batch-size 256 \
  --num-workers 8 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --log-interval 10 \
  --overwrite \
  2>&1 | tee logs/tabero_training_manual/${EXP}.log
```

### pi05 视觉 + 触觉，GPU 4-7

```bash
tmux new -s tabero_pi05_tactile_train
```

tmux 里执行：

```bash
cd /data/mingxinwang/openpi-univtac

export PYTHON=/data/environment/miniconda3/envs/openpi/bin/python
export PYTHONPATH=/data/mingxinwang/openpi-univtac/src:/data/mingxinwang/openpi-univtac/packages/openpi-client/src:${PYTHONPATH:-}
export HF_LEROBOT_HOME=/data/mingxinwang/openpi-univtac/.cache/lerobot
export HF_HOME=/data/mingxinwang/openpi-univtac/.cache/huggingface
export HF_DATASETS_CACHE=/data/mingxinwang/openpi-univtac/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/data/mingxinwang/openpi-univtac/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online

RUN_ID=tabero_object_spatial_success_only_20260614
REPO_TACTILE=local/tabero_tactile_object_spatial_success_only_20260614
ROOT_TACTILE=/data/mingxinwang/openpi-univtac/.cache/lerobot/local/tabero_tactile_object_spatial_success_only_20260614
PI05_BASE=/data/mingxinwang/openpi-univtac/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
ASSET_PI05_TACTILE=/data/mingxinwang/openpi-univtac/assets/${RUN_ID}/pi05_tactile
ASSET_ID_PI05_TACTILE=tabero_tactile_pi05_h50
EXP=tabero_object_spatial_success_only_pi05_tactile_h50_bs256_4gpu_nw8_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=4,5,6,7 \
$PYTHON scripts/train.py pi05_lora_tacall_tabero \
  --exp-name "${EXP}" \
  --data.repo-id "${REPO_TACTILE}" \
  --data.root "${ROOT_TACTILE}" \
  --data.assets.assets-dir "${ASSET_PI05_TACTILE}" \
  --data.assets.asset-id "${ASSET_ID_PI05_TACTILE}" \
  --weight-loader.params-path "${PI05_BASE}" \
  --fsdp-devices 4 \
  --batch-size 256 \
  --num-workers 8 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --log-interval 10 \
  --overwrite \
  2>&1 | tee logs/tabero_training_manual/${EXP}.log
```

## 6. 监控训练

```bash
tmux ls | rg 'tabero_pi'
pgrep -af "scripts/train.py .*tabero"
nvidia-smi

rg -n "Step [0-9]+:|Progress on: [0-9]|View run" logs/tabero_training_manual/*.log | tail -40
rg -n "RESOURCE_EXHAUSTED|OutOfMemory|Traceback|Killed|CUDA_ERROR_OUT_OF_MEMORY|OOM|RuntimeError|ERROR" logs/tabero_training_manual/*.log
```

checkpoint 输出位置：

```bash
checkpoints/pi0_lora_vision_tabero/<EXP>/
checkpoints/pi05_lora_vision_tabero/<EXP>/
checkpoints/pi0_lora_tacall_tabero/<EXP>/
checkpoints/pi05_lora_tacall_tabero/<EXP>/
```

每 1000 step 会保存一次。用某个 checkpoint 起 server 或评测时，传 step 目录，例如：

```bash
/data/mingxinwang/openpi-univtac/checkpoints/pi0_lora_tacall_tabero/<EXP>/1000
```

## 7. 当前触觉模型证据边界

官方 Tabero 仓库公开的是 TacManip/Tabero 环境、HDF5+MP4 到 LeRobot 的转换、OpenPI inference client 和文档；完整 OpenPI model server 代码没有随 Tabero 仓库开源。

能从官方 ckpt 确认的内容：

- ckpt 是 pi0/OpenPI 风格参数树，内部 action 宽度是 32：`state_proj/kernel=(32,1024)`、`action_in_proj/kernel=(32,1024)`、`action_out_proj/kernel=(1024,32)`。
- ckpt 里存在 `tactile_prefix_encoder`、`tactile_suffix_encoder`、`tactile_suffix_to_prefix_proj` 三组触觉参数。
- `tactile_prefix_encoder` 的输入维度是 396，结构参数形状约束为 `396 -> 4096 -> 4096 -> 2048`。
- `tactile_suffix_encoder` 的输入维度是 48，结构参数形状约束为 `48 -> 2048 -> 1024`。
- `tactile_suffix_to_prefix_proj` 的形状是 `1024 -> 2048`。
- 本机 `src/openpi/models/pi0_tabero.py` 生成的 22 个 tactile 参数和官方 ckpt 的 22 个 tactile 参数 key/shape 完全一致。

不能只靠 ckpt 完全确认的内容：

- 触觉 encoder forward 里的激活函数、残差组合、缩放方式是否和作者训练源码逐行一致。
- tactile token 插入 prefix/suffix stream 的语义位置是否和作者未公开 server 代码完全一致。
- `tactile_marker_motion [9,198,2]` 中作者是否也只取最后一个 current marker frame 作为 396D prefix，还是训练源码里用了更复杂的历史处理。
- `tactile_image` mosaic 进入 OpenPI 图像通道的具体组织是否和作者训练源码完全一致；当前做法和官方数据/client字段一致，但源码级语义仍需要作者代码才能最终确认。

复查 ckpt 兼容性：

```bash
$PYTHON examples/tabero/inspect_tabero_ckpt_shapes.py \
  /data/mingxinwang/tabero_reports/checkpoints/pi0_lora_tacall_tabero_enc/49999
```

期望：

```text
ckpt_tactile_count: 22
local_tactile_count: 22
missing_in_local: 0
missing_in_ckpt: 0
shape_mismatch: 0
```

## 8. 数据转换脚本边界

完整转换命令在第 3 节。这里强调两个边界：

- 转换脚本会严格检查 HDF5 的 action/state/force/marker 与 MP4 帧数一致，不再静默裁剪或错位对齐。
- 如果只训练 `libero_object` 或只训练 `libero_spatial`，改第 3 节里的 `--task-suites` 和 `--task-subset-path` 即可；当前默认训练集是 object+spatial 合并版。
