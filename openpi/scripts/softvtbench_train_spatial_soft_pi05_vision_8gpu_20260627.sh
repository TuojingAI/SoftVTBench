#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/softvtbench_paths.sh"
REPO="${REPO:-${OPENPI_CODE_DIR}}"
RAW_ROOT="${RAW_ROOT:?RAW_ROOT is required; point it at SoftVTBench_data/spatial-soft}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_lora_vision_softvtbench}"
# Paper Sec. 4.1: the vision-only baseline uses a binary open-close gripper so it
# never receives contact-calibrated continuous width bounds. Set GRIPPER_MODE=state
# to train it with the continuous width instead.
GRIPPER_MODE="${GRIPPER_MODE:-binary}"
export GRIPPER_MODE
BASE_PARAMS="${BASE_PARAMS:-${PI05_BASE:-${OPENPI_CACHE_ROOT}/openpi/openpi-assets/checkpoints/pi05_base/params}}"
RUN_ID="${RUN_ID:-spatial_soft_pastry005_10tasks_50each_gripperjitter_20260624}"
REPO_VISION="${REPO_VISION:-local/spatial_soft_pastry005_10tasks_50each_gripperjitter_vision_targetnext_20260627}"
ASSET_ID="${ASSET_ID:-spatial_soft_pastry005_pi05_vision_h50_targetnext_20260627}"
EXP="${EXP:-spatial_soft_pastry005_pi05_vision_h50_targetnext_7d_bs256_8gpu_nw16_wandb_20260627_a800_q2}"

STAGE_ROOT="${STAGE_ROOT:-${OPENPI_STAGE_ROOT}/${RUN_ID}_vision_targetnext}"
ROOT_VISION="${ROOT_VISION:-${OPENPI_CACHE_ROOT}/lerobot/${REPO_VISION}}"
ASSETS_DIR="${ASSETS_DIR:-${OPENPI_ASSETS_ROOT}/${RUN_ID}/pi05_vision}"
LOGDIR="${LOGDIR:-${OPENPI_LOG_ROOT}/${RUN_ID}_pi05_vision_targetnext}"
TASK_SUBSET="${TASK_SUBSET:-${OPENPI_LAUNCHER_ROOT}/configs/task_subset_softvtbench.json}"

FSDP_DEVICES="${FSDP_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"
# The paper trains for 7k steps (Sec. 4.1). Override to explore longer runs.
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
mkdir -p "${LOGDIR}"

export PYTHONPATH="${REPO}/src:${REPO}/packages/openpi-client/src:${PYTHONPATH:-}"
export TRAIN_CONFIG
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_CACHE_ROOT}/lerobot}"
export HF_HOME="${HF_HOME:-${OPENPI_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${OPENPI_CACHE_ROOT}/huggingface/datasets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_CACHE_ROOT}/openpi}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

softvtbench_require_openpi_python "${PYTHON}"
test -d "${RAW_ROOT}"
test -f "${TASK_SUBSET}"

echo "[1/6] Building a 10-task staging view from ${RAW_ROOT}"
mkdir -p "${STAGE_ROOT}/replayed_demos" "${STAGE_ROOT}/video_datasets"
find "${STAGE_ROOT}/replayed_demos" -mindepth 1 -maxdepth 1 -type l -delete
find "${STAGE_ROOT}/video_datasets" -mindepth 1 -maxdepth 1 -type l -delete

for TASK_ROOT in "${RAW_ROOT}/libero_spatial"/libero_spatial_task*; do
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
DEMO_COUNT="$(${PYTHON} - "${STAGE_ROOT}/replayed_demos" <<'PY'
from pathlib import Path
import h5py
import sys

count = 0
for path in sorted(Path(sys.argv[1]).glob("*.hdf5")):
    with h5py.File(path, "r") as f:
        count += len(f["data"])
print(count)
PY
)"
echo "stage_hdf5=${H5_COUNT} stage_tasks=${TASK_COUNT} demos=${DEMO_COUNT}"
[[ "${H5_COUNT}" == "10" ]] || { echo "Expected 10 HDF5 files." >&2; exit 2; }
[[ "${TASK_COUNT}" == "10" ]] || { echo "Expected 10 task video dirs." >&2; exit 2; }
[[ "${DEMO_COUNT}" == "500" ]] || { echo "Expected 500 demos." >&2; exit 2; }

echo "[2/6] Verifying ${TRAIN_CONFIG} uses no extra delta transform"
"${PYTHON}" - <<'PY'
import os
import openpi.training.config as config

cfg = config.get_config(os.environ["TRAIN_CONFIG"])
assert cfg.data.extra_delta_transform is False, cfg.data
print("extra_delta_transform=False")
PY

if [[ "${SKIP_CONVERT:-${SKIP_PREP:-0}}" == "1" ]]; then
  echo "[3/6] Reusing prepared pure-vision targetnext data"
  test -f "${ROOT_VISION}/meta/info.json"
else
  echo "[3/6] Converting spatial-soft to pure-vision 7D targetnext data"
  "${PYTHON}" examples/softvtbench/convert_softvtbench_vision_data_to_lerobot.py \
    --data-root "${STAGE_ROOT}" \
    --repo-name "${REPO_VISION}" \
    --output-dir "${HF_LEROBOT_HOME}" \
    --task-suites libero_spatial \
    --task-subset-path "${TASK_SUBSET}" \
    --prompt-source hdf5_attr \
    --state-pose-format quat_gripper_pos \
    --action-source next_state \
    --drop-force-dims-from-13d \
    --gripper-target-mode "${GRIPPER_MODE}" \
    2>&1 | tee "${LOGDIR}/convert_vision_targetnext.log"
fi

echo "[4/6] Checking 500 episodes, no tactile/force fields, and action[t] == state[t+1]"
"${PYTHON}" - <<PY
import os
import json
import math
from pathlib import Path

import h5py
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
info = json.loads((root / "meta/info.json").read_text())
features = info.get("features", {})
print("total_episodes", info.get("total_episodes"))
print("total_frames", info.get("total_frames"))
print("features", sorted(features))
assert info.get("total_episodes") == 500, info.get("total_episodes")
for key in ("image", "wrist_image", "state", "actions"):
    assert key in features, key
assert features["state"]["shape"] == [7], features["state"]
assert features["actions"]["shape"] == [7], features["actions"]
assert not any("tactile" in key.lower() or "force" in key.lower() for key in features), sorted(features)

h5_path = sorted((stage / "replayed_demos").glob("libero_spatial_task0*.hdf5"))[0]
with h5py.File(h5_path, "r") as f:
    demo = f["data/demo_0"]
    eef_pose = np.asarray(demo["obs/eef_pose"], dtype=np.float32)
    gripper = np.asarray(demo["obs/gripper_pos"], dtype=np.float32)
    gripper_scalar = gripper[:, 0:1] if gripper.ndim == 2 else gripper.reshape(-1, 1)
    expected_states = np.concatenate(
        [eef_pose[:, :3], quat2axisangle(eef_pose[:, 3:7]), gripper_scalar], axis=1
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

if [[ "${SKIP_NORM:-${SKIP_PREP:-0}}" != "1" ]]; then
  echo "[5/6] Computing pi05 pure-vision norm stats"
  "${PYTHON}" scripts/compute_norm_stats.py \
    --config-name "${TRAIN_CONFIG}" \
    --repo-id "${REPO_VISION}" \
    --root "${ROOT_VISION}" \
    --assets-dir "${ASSETS_DIR}" \
    --asset-id "${ASSET_ID}" \
    --low-dim-only \
    2>&1 | tee "${LOGDIR}/norm_pi05_vision_targetnext.log"
else
  echo "[5/6] Reusing existing norm stats"
fi

test -f "${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
"${PYTHON}" - "${ASSETS_DIR}/${ASSET_ID}/norm_stats.json" <<'PY'
import json
import sys

stats = json.load(open(sys.argv[1]))["norm_stats"]
assert sorted(stats) == ["actions", "state"], sorted(stats)
assert len(stats["actions"]["mean"]) == 7
assert len(stats["state"]["mean"]) == 7
print("norm_keys", sorted(stats))
PY

if [[ "${PREP_ONLY:-0}" == "1" ]]; then
  echo "PREP_ONLY=1: conversion, validation, and stats are complete."
  exit 0
fi

test -d "${BASE_PARAMS}"

if [[ "${REQUIRE_GPUS:-${REQUIRE_A800:-1}}" == "1" ]]; then
  GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  GPU_COUNT="$(printf '%s\n' "${GPU_NAMES}" | sed '/^$/d' | wc -l)"
  echo "local_gpu_count=${GPU_COUNT} gpus=$(printf '%s' "${GPU_NAMES}" | paste -sd, -)"
  [[ "${GPU_COUNT}" -ge "${MIN_GPUS:-${FSDP_DEVICES}}" ]] || \
    { echo "Refusing to start training: need ${MIN_GPUS:-${FSDP_DEVICES}} GPUs, found ${GPU_COUNT} (set REQUIRE_GPUS=0 to override)." >&2; exit 3; }
fi

echo "[6/6] Starting ${TRAIN_CONFIG} pure-vision training on ${FSDP_DEVICES} GPUs"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(softvtbench_cuda_visible_devices "${FSDP_DEVICES}")}" \
"${PYTHON}" scripts/train.py "${TRAIN_CONFIG}" \
  --exp-name "${EXP}" \
  --data.repo-id "${REPO_VISION}" \
  --data.root "${ROOT_VISION}" \
  --data.assets.assets-dir "${ASSETS_DIR}" \
  --data.assets.asset-id "${ASSET_ID}" \
  --weight-loader.params-path "${BASE_PARAMS}" \
  --fsdp-devices "${FSDP_DEVICES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  "${WANDB_FLAG}" \
  "${OVERWRITE_ARGS[@]}"
