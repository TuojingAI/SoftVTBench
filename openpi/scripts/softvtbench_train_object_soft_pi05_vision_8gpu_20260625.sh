#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/softvtbench_paths.sh"
softvtbench_require_openpi_python "${PYTHON}"
REPO="${REPO:-${OPENPI_CODE_DIR}}"
RAW_ROOT="${RAW_ROOT:?RAW_ROOT is required; point it at SoftVTBench_data/object-soft}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_lora_vision_softvtbench}"
# Paper Sec. 4.1: the vision-only baseline uses a binary open-close gripper so it
# never receives contact-calibrated continuous width bounds. Set GRIPPER_MODE=state
# to train it with the continuous width instead.
GRIPPER_MODE="${GRIPPER_MODE:-binary}"
export GRIPPER_MODE
BASE_PARAMS="${BASE_PARAMS:-${PI05_BASE:-${OPENPI_CACHE_ROOT}/openpi/openpi-assets/checkpoints/pi05_base/params}}"
RUN_ID="${RUN_ID:-object_soft_10assets_10tasks_50each_initjitter012_gripperjitter5_nl_20260625}"
REPO_VISION="${REPO_VISION:-local/object_soft_vision_10assets_10tasks_50each_20260625_targetnext}"
ASSET_ID_PI05_VISION="${ASSET_ID_PI05_VISION:-object_soft_vision_pi05_h50_targetnext_20260625}"
EXP="${EXP:-object_soft_10assets_10tasks_pi05_vision_h50_targetnext_bs256_8gpu_nw16_s500_20260625}"

STAGE_ROOT="${STAGE_ROOT:-${OPENPI_STAGE_ROOT}/${RUN_ID}}"
ROOT_VISION="${ROOT_VISION:-${OPENPI_CACHE_ROOT}/lerobot/${REPO_VISION}}"
ASSET_PI05_VISION="${ASSET_PI05_VISION:-${OPENPI_ASSETS_ROOT}/${RUN_ID}/pi05_vision}"
LOGDIR="${LOGDIR:-${OPENPI_LOG_ROOT}/${RUN_ID}}"
TASK_SUBSET="${TASK_SUBSET:-${OPENPI_LAUNCHER_ROOT}/configs/task_subset_softvtbench.json}"

FSDP_DEVICES="${FSDP_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-7000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_FLAG="--wandb-enabled"
if [[ "${WANDB_ENABLED}" == "0" || "${WANDB_ENABLED}" == "false" || "${WANDB_ENABLED}" == "False" ]]; then
  WANDB_FLAG="--no-wandb-enabled"
fi
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then OVERWRITE_ARGS=(--overwrite); fi

cd "${REPO}"
mkdir -p "${LOGDIR}" "${STAGE_ROOT}/replayed_demos" "${STAGE_ROOT}/video_datasets"

export PYTHONPATH="${REPO}/src:${REPO}/packages/openpi-client/src:${PYTHONPATH:-}"
export TRAIN_CONFIG
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_CACHE_ROOT}/lerobot}"
export HF_HOME="${HF_HOME:-${OPENPI_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${OPENPI_CACHE_ROOT}/huggingface/datasets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_CACHE_ROOT}/openpi}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "RAW_ROOT not found: ${RAW_ROOT}" >&2
  exit 2
fi

if [[ ! -d "${BASE_PARAMS}" ]]; then
  echo "BASE_PARAMS not found: ${BASE_PARAMS}" >&2
  exit 2
fi

echo "[1/6] Building stage symlinks from ${RAW_ROOT}"
find "${STAGE_ROOT}/replayed_demos" -mindepth 1 -maxdepth 1 -type l -delete
find "${STAGE_ROOT}/video_datasets" -mindepth 1 -maxdepth 1 -type l -delete

for TASK_ROOT in "${RAW_ROOT}/libero_object"/libero_object_task*; do
  [[ -d "${TASK_ROOT}" ]] || continue
  for H5 in "${TASK_ROOT}/replayed_demos/"*.hdf5; do
    [[ -e "${H5}" ]] || continue
    ln -sfn "${H5}" "${STAGE_ROOT}/replayed_demos/$(basename "${H5}")"
  done
  TASK_NAME="$(basename "${TASK_ROOT}")"
  VIDEO_TASK_DIR="${TASK_ROOT}/video_datasets/${TASK_NAME}"
  [[ -d "${VIDEO_TASK_DIR}" ]] || { echo "Missing video task dir: ${VIDEO_TASK_DIR}" >&2; exit 2; }
  ln -sfn "${VIDEO_TASK_DIR}" "${STAGE_ROOT}/video_datasets/${TASK_NAME}"
done

H5_COUNT="$(find -L "${STAGE_ROOT}/replayed_demos" -maxdepth 1 -name '*.hdf5' | wc -l)"
TASK_COUNT="$(find -L "${STAGE_ROOT}/video_datasets" -mindepth 1 -maxdepth 1 -type d | wc -l)"
MANIFEST_LINES="$(wc -l < "${RAW_ROOT}/manifest.jsonl")"
echo "stage_hdf5=${H5_COUNT} stage_tasks=${TASK_COUNT} manifest_lines=${MANIFEST_LINES}"
[[ "${H5_COUNT}" == "10" ]] || { echo "Expected 10 HDF5 files." >&2; exit 2; }
[[ "${TASK_COUNT}" == "10" ]] || { echo "Expected 10 task video dirs." >&2; exit 2; }
[[ "${MANIFEST_LINES}" == "500" ]] || { echo "Expected 500 manifest rows." >&2; exit 2; }

echo "[2/6] Verifying config disables DeltaActions for ${TRAIN_CONFIG}"
"${PYTHON}" - <<'PY'
import os
import openpi.training.config as config
cfg = config.get_config(os.environ["TRAIN_CONFIG"])
assert cfg.data.extra_delta_transform is False, cfg.data
print("extra_delta_transform=False")
PY

if [[ "${SKIP_CONVERT:-${SKIP_PREP:-0}}" == "1" ]]; then
  echo "[3/6] Reusing existing LeRobot dataset"
  test -f "${ROOT_VISION}/meta/info.json"
  "${PYTHON}" - <<PY
import json
from pathlib import Path

root = Path("${ROOT_VISION}")
info = json.loads((root / "meta/info.json").read_text())
print("total_episodes", info.get("total_episodes"))
print("total_frames", info.get("total_frames"))
print("features", sorted(info.get("features", {}).keys()))
assert info.get("total_episodes") == 500, info.get("total_episodes")
for key in ("image", "wrist_image", "state", "actions"):
    assert key in info.get("features", {}), key
PY
else
  echo "[3/6] Converting HDF5 to LeRobot vision dataset with absolute targetnext actions"
  "${PYTHON}" examples/softvtbench/convert_softvtbench_vision_data_to_lerobot.py \
    --data-root "${STAGE_ROOT}" \
    --repo-name "${REPO_VISION}" \
    --output-dir "${HF_LEROBOT_HOME}" \
    --task-suites libero_object \
    --task-subset-path "${TASK_SUBSET}" \
    --prompt-source hdf5_attr \
    --state-pose-format quat_gripper_pos \
    --action-source next_state \
    --drop-force-dims-from-13d \
    --gripper-target-mode "${GRIPPER_MODE}" \
    2>&1 | tee "${LOGDIR}/convert_vision_targetnext.log"
fi

echo "[4/6] Checking converted state is absolute 7D and actions[t] == state[t+1]"
"${PYTHON}" - <<PY
import os
from pathlib import Path
import h5py
import math
import numpy as np
import pandas as pd

def quat2axisangle(quats):
    out = []
    for quat in quats:
        q0 = float(np.clip(quat[0], -1.0, 1.0))
        den = math.sqrt(max(0.0, 1.0 - q0 * q0))
        if math.isclose(den, 0.0):
            out.append(np.zeros(3, dtype=np.float32))
        else:
            out.append((quat[1:] * 2.0 * math.acos(q0) / den).astype(np.float32))
    return np.asarray(out, dtype=np.float32)

stage = Path("${STAGE_ROOT}")
root = Path("${ROOT_VISION}")
h5 = sorted((stage / "replayed_demos").glob("libero_object_task0*.hdf5"))[0]
with h5py.File(h5, "r") as f:
    eef_pose = np.asarray(f["data/demo_0/obs/eef_pose"], dtype=np.float32)
    gripper = np.asarray(f["data/demo_0/obs/gripper_pos"], dtype=np.float32)
    gripper_scalar = gripper[:, 0:1] if gripper.ndim == 2 else gripper.reshape(-1, 1)
    expected_states = np.concatenate(
        [eef_pose[:, :3], quat2axisangle(eef_pose[:, 3:7]), gripper_scalar],
        axis=1,
    ).astype(np.float32)
    expected_actions = np.concatenate([expected_states[1:], expected_states[-1:]], axis=0)
parquet = sorted((root / "data").glob("**/*.parquet"))[0]
df = pd.read_parquet(parquet)
got_actions = np.stack(df["actions"].to_numpy()).astype(np.float32)[: len(expected_actions)]
got_states = np.stack(df["state"].to_numpy()).astype(np.float32)[: len(expected_states)]
action_max_err = float(np.max(np.abs(got_actions[:, :6] - expected_actions[:, :6])))
# (gripper channel excluded: it is a binary open/close target, not state[t+1])
state_max_err = float(np.max(np.abs(got_states - expected_states)))
arm_targetnext_max_err = float(np.max(np.abs(got_actions[:-1, :6] - got_states[1:, :6])))
gripper_values = np.unique(np.round(got_actions[:, 6], 6))
gripper_mode = os.environ.get("GRIPPER_MODE", "binary")
if gripper_mode == "binary":
    assert len(gripper_values) == 2, f"binary gripper must take exactly 2 values, got {gripper_values}"
    print("gripper_binary_values", gripper_values.tolist())
    targetnext_max_err = arm_targetnext_max_err
else:
    targetnext_max_err = float(np.max(np.abs(got_actions[:-1] - got_states[1:])))
print("arm_targetnext_max_err", arm_targetnext_max_err)
print("targetnext_action_max_err", action_max_err)
print("absolute_state_max_err", state_max_err)
print("action_t_minus_state_t_plus_1_max_err", targetnext_max_err)
assert action_max_err < 1e-6, action_max_err
assert state_max_err < 1e-6, state_max_err
assert targetnext_max_err < 1e-6, targetnext_max_err
PY

if [[ "${SOFTVTBENCH_STOP_AFTER_CONVERT:-0}" == "1" ]]; then
  echo "PHASE=convert: conversion and validation are complete."
  exit 0
fi

if [[ "${SKIP_NORM:-${SKIP_PREP:-0}}" == "1" ]]; then
  echo "[5/6] Reusing existing norm stats"
else
  echo "[5/6] Computing pi05 norm stats for targetnext absolute-action vision data"
  "${PYTHON}" scripts/compute_norm_stats.py \
    --config-name "${TRAIN_CONFIG}" \
    --repo-id "${REPO_VISION}" \
    --root "${ROOT_VISION}" \
    --assets-dir "${ASSET_PI05_VISION}" \
    --asset-id "${ASSET_ID_PI05_VISION}" \
    --low-dim-only \
    2>&1 | tee "${LOGDIR}/norm_pi05_vision_targetnext.log"
fi

test -f "${ASSET_PI05_VISION}/${ASSET_ID_PI05_VISION}/norm_stats.json"

if [[ "${PREP_ONLY:-0}" == "1" ]]; then
  echo "PREP_ONLY=1: data checks and norm stats are complete; skipping training."
  exit 0
fi

test -d "${BASE_PARAMS}"

# Guard against launching a formal run on fewer GPUs than the config assumes.
# The paper trains on 8 GPUs; any 8 will do, so this counts devices rather than
# matching a model name. Set REQUIRE_GPUS=0 to disable, MIN_GPUS to change the bar.
if [[ "${REQUIRE_GPUS:-${REQUIRE_A800:-1}}" == "1" ]]; then
  GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  GPU_COUNT="$(printf '%s\n' "${GPU_NAMES}" | sed '/^$/d' | wc -l)"
  echo "gpu_count=${GPU_COUNT} gpus=$(printf '%s' "${GPU_NAMES}" | paste -sd, -)"
  if [[ "${GPU_COUNT}" -lt "${MIN_GPUS:-${FSDP_DEVICES}}" ]]; then
    echo "Refusing to start training: need ${MIN_GPUS:-${FSDP_DEVICES}} GPUs, found ${GPU_COUNT} (set REQUIRE_GPUS=0 to override)." >&2
    exit 3
  fi
fi

echo "[6/6] Starting pi05 vision training on ${FSDP_DEVICES} GPUs"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(softvtbench_cuda_visible_devices "${FSDP_DEVICES}")}" \
"${PYTHON}" scripts/train.py "${TRAIN_CONFIG}" \
  --exp-name "${EXP}" \
  --data.repo-id "${REPO_VISION}" \
  --data.root "${ROOT_VISION}" \
  --data.assets.assets-dir "${ASSET_PI05_VISION}" \
  --data.assets.asset-id "${ASSET_ID_PI05_VISION}" \
  --weight-loader.params-path "${BASE_PARAMS}" \
  --fsdp-devices "${FSDP_DEVICES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  "${WANDB_FLAG}" \
  "${OVERWRITE_ARGS[@]}" \
  2>&1 | tee "${LOGDIR}/${EXP}.log"
