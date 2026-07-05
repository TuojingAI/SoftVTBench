#!/usr/bin/env bash
# Train pi05 on the rigid LIBERO baselines (object-rigid / spatial-rigid).
#
# Unlike the soft datasets, the rigid folders already ship in the layout the
# converters expect (replayed_demos/ + video_datasets/), so no staging view is
# built. Demo counts are uneven across tasks by design (these are raw replay
# outputs, not a fixed 50-per-task quota), so counts are derived, not asserted.
#
# Usage:
#   SUITE=libero_object MODALITY=tactile \
#   RAW_ROOT=/path/to/SoftVTBench_data/object-rigid \
#   MODEL=pi05 BASE_PARAMS=/path/to/pi05_base/params \
#   OPENPI_PYTHON=/path/to/openpi-venv/bin/python \
#     bash openpi/scripts/train_rigid_pi05.sh
#
# Flags: PREP_ONLY=1  SKIP_CONVERT=1  SKIP_NORM=1  REQUIRE_A800=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/softvtbench_paths.sh"
REPO="${REPO:-${OPENPI_CODE_DIR}}"

SUITE="${SUITE:-libero_object}"
MODALITY="${MODALITY:-tactile}"
# Paper Sec. 4.1: the vision-only baseline uses a binary open-close gripper so it
# never receives contact-calibrated continuous width bounds; the tactile policy
# uses the continuous width. Only the vision branch reads this.
if [[ -z "${GRIPPER_MODE:-}" ]]; then
  if [[ "${MODALITY}" == "vision" ]]; then GRIPPER_MODE=binary; else GRIPPER_MODE=state; fi
fi
case "${SUITE}" in
  libero_object|libero_spatial) ;;
  *) echo "SUITE must be libero_object or libero_spatial, got: ${SUITE}" >&2; exit 2 ;;
esac
case "${MODALITY}" in
  tactile|vision) ;;
  *) echo "MODALITY must be tactile or vision, got: ${MODALITY}" >&2; exit 2 ;;
esac

SUITE_SHORT="${SUITE#libero_}"
RAW_ROOT="${RAW_ROOT:?RAW_ROOT is required, e.g. /path/to/SoftVTBench_data/${SUITE_SHORT}-rigid}"
RUN_ID="${RUN_ID:-${SUITE_SHORT}_rigid}"
MODEL="${MODEL:-pi05}"
if [[ "${MODEL}" != "pi05" ]]; then
  echo "SoftVTBench v1 supports MODEL=pi05 only; got: ${MODEL}" >&2
  exit 2
fi

if [[ "${MODALITY}" == "tactile" ]]; then
  TRAIN_CONFIG="${TRAIN_CONFIG:-${MODEL}_lora_tacall_softvtbench}"
else
  TRAIN_CONFIG="${TRAIN_CONFIG:-${MODEL}_lora_vision_softvtbench}"
fi

DATA_REPO="${DATA_REPO:-local/${RUN_ID}_${MODEL}_${MODALITY}_targetnext_7d}"
DATA_ROOT="${DATA_ROOT:-${OPENPI_CACHE_ROOT}/lerobot/${DATA_REPO}}"
ASSETS_DIR="${ASSETS_DIR:-${OPENPI_ASSETS_ROOT}/${RUN_ID}/${MODEL}_${MODALITY}}"
ASSET_ID="${ASSET_ID:-${RUN_ID}_${MODEL}_${MODALITY}_targetnext_7d}"
BASE_PARAMS="${BASE_PARAMS:-${PI05_BASE:-${OPENPI_CACHE_ROOT}/openpi/openpi-assets/checkpoints/${MODEL}_base/params}}"
EXP_NAME="${EXP_NAME:-${RUN_ID}_${MODEL}_${MODALITY}_targetnext_7d_bs256_8gpu}"

# Always pass the release-scoped subset explicitly. It covers all ten object and
# all ten spatial tasks, so converter fallback defaults cannot change a run.
TASK_SUBSET="${TASK_SUBSET:-${OPENPI_LAUNCHER_ROOT}/configs/task_subset_softvtbench.json}"
LOGDIR="${LOGDIR:-${OPENPI_LOG_ROOT}/${RUN_ID}_${MODEL}_${MODALITY}}"

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
mkdir -p "${LOGDIR}" "${ASSETS_DIR}"

export REPO PYTHON RAW_ROOT SUITE DATA_REPO DATA_ROOT ASSETS_DIR ASSET_ID TASK_SUBSET TRAIN_CONFIG
export PYTHONPATH="${REPO}/src:${REPO}/packages/openpi-client/src:${PYTHONPATH:-}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_CACHE_ROOT}/lerobot}"
export HF_HOME="${HF_HOME:-${OPENPI_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${OPENPI_CACHE_ROOT}/huggingface/datasets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_CACHE_ROOT}/openpi}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

softvtbench_require_openpi_python "${PYTHON}"
test -f "${TASK_SUBSET}"
test -d "${RAW_ROOT}/replayed_demos"
test -d "${RAW_ROOT}/video_datasets"

echo "[1/6] Inspect rigid dataset"
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

import h5py

raw = Path(os.environ["RAW_ROOT"])
suite = os.environ["SUITE"]

h5s = sorted((raw / "replayed_demos").glob("*.hdf5"))
task_dirs = sorted(d for d in (raw / "video_datasets").iterdir() if d.is_dir())
if len(h5s) != 10:
    raise SystemExit(f"expected 10 HDF5 files, found {len(h5s)}")
if len(task_dirs) != 10:
    raise SystemExit(f"expected 10 task video dirs, found {len(task_dirs)}")

total = 0
per_task = {}
for h5 in h5s:
    with h5py.File(h5, "r") as f:
        n = len([k for k in f["data"] if k.startswith("demo_")])
        d0 = f["data"]["demo_0"]
        actions = d0["actions"]
        if actions.shape[1] != 13:
            raise SystemExit(f"{h5.name}: expected 13D actions (recorder_type=7dpf), got {actions.shape[1]}D")
        if "gripper_marker_motion" not in d0["obs"]:
            raise SystemExit(f"{h5.name}: missing obs/gripper_marker_motion (tactile env required)")
    per_task[h5.name.split("_task")[1].split("_")[0]] = n
    total += n

cam = len(list(raw.glob("video_datasets/*/videos/*.mp4")))
tac = len(list(raw.glob("video_datasets/*/tactile_outputs/*.mp4")))
print(f"suite={suite} demos={total} per_task={per_task}")
print(f"camera_mp4={cam} tactile_mp4={tac}")
if cam != 2 * total:
    raise SystemExit(f"expected {2 * total} camera videos (2/demo), found {cam}")
if tac < 2 * total:
    raise SystemExit(f"expected >= {2 * total} tactile videos, found {tac}")
print("DEMOS", total)
PY

if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
  if [[ "${MODALITY}" == "vision" ]]; then
    echo "[2/6] Convert rigid -> LeRobot (pure vision, 7D targetnext)"
    "${PYTHON}" examples/softvtbench/convert_softvtbench_vision_data_to_lerobot.py \
      --data-root "${RAW_ROOT}" \
      --repo-name "${DATA_REPO}" \
      --output-dir "${HF_LEROBOT_HOME}" \
      --task-suites "${SUITE}" \
      --task-subset-path "${TASK_SUBSET}" \
      --prompt-source config \
      --state-pose-format quat_gripper_pos \
      --action-source next_state \
      --drop-force-dims-from-13d \
      --gripper-target-mode "${GRIPPER_MODE}" \
      2>&1 | tee "${LOGDIR}/convert_${MODALITY}.log"
  else
    echo "[2/6] Convert rigid -> LeRobot (vision+tactile, 7D targetnext, force dropped)"
    # The tactile converter has no --action-source flag, so wrap its trajectory
    # assembler to emit targetnext actions (action[t] = state[t+1]), matching the
    # soft pipeline. Task prompts come from examples/softvtbench/config/<suite>.json,
    # which is already correct for rigid (unlike soft, whose prompts live in the
    # HDF5 attrs).
    "${PYTHON}" - <<'PY' 2>&1 | tee "${LOGDIR}/convert_tactile.log"
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

repo = Path(os.environ["REPO"])
spec = importlib.util.spec_from_file_location(
    "softvtbench_tactile_converter",
    repo / "examples/softvtbench/convert_softvtbench_tactile_data_to_lerobot.py",
)
conv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = conv
assert spec.loader is not None
spec.loader.exec_module(conv)

cfg = conv.Config(
    task_suites=(os.environ["SUITE"],),
    data_root=Path(os.environ["RAW_ROOT"]),
    output_dir=Path(os.environ["HF_LEROBOT_HOME"]),
    repo_name=os.environ["DATA_REPO"],
    task_subset_path=Path(os.environ["TASK_SUBSET"]),
    tactile_history_len=8,
    marker_history_len=2,
    tactile_output_type="markers_rgb",
)

orig = conv.combine_traj_and_images_softvtbench


def targetnext_wrapper(config, trajectory_id, trajectory, suite_name, original_task_id, video_root=None):
    valid, actions, states, images_dict, tactile_images, tactile_marker_motion = orig(
        config, trajectory_id, trajectory, suite_name, original_task_id, video_root=video_root
    )
    if valid:
        actions = np.concatenate([states[1:], states[-1:]], axis=0).astype(np.float32)
    return valid, actions, states, images_dict, tactile_images, tactile_marker_motion


conv.combine_traj_and_images_softvtbench = targetnext_wrapper
conv.main(cfg)
PY
  fi
else
  echo "[2/6] SKIP_CONVERT=1: reusing ${DATA_ROOT}"
fi

echo "[3/6] Sanity check converted repo"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ["DATA_ROOT"])
info = json.loads((root / "meta/info.json").read_text())
features = info["features"]
print("repo", root)
print("episodes", info.get("total_episodes"), "frames", info.get("total_frames"), "tasks", info.get("total_tasks"))
print("features", sorted(features))

if info.get("total_tasks") != 10:
    raise SystemExit(f"expected 10 tasks, got {info.get('total_tasks')}")
for key in ("image", "wrist_image", "state", "actions"):
    if key not in features:
        raise SystemExit(f"missing feature {key}")
for key in features:
    if "force" in key.lower():
        raise SystemExit(f"force field leaked into dataset: {key}")
if features["actions"]["shape"] != [7]:
    raise SystemExit(f"actions must be 7D, got {features['actions']['shape']}")

max_err = 0.0
for parquet in sorted((root / "data").glob("**/*.parquet")):
    df = pd.read_parquet(parquet)
    states = np.stack(df["state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
    if len(states) > 1:
        max_err = max(max_err, float(np.max(np.abs(actions[:-1] - states[1:]))))
    max_err = max(max_err, float(np.max(np.abs(actions[-1] - states[-1]))))
print("targetnext_max_err", max_err)
if max_err >= 1e-6:
    raise SystemExit(f"actions are not targetnext: max_err={max_err}")
PY

if [[ "${SOFTVTBENCH_STOP_AFTER_CONVERT:-0}" == "1" ]]; then
  echo "PHASE=convert: conversion and validation are complete."
  exit 0
fi

if [[ "${SKIP_NORM:-0}" != "1" ]]; then
  echo "[4/6] Compute norm stats (${TRAIN_CONFIG})"
  "${PYTHON}" scripts/compute_norm_stats.py \
    --config-name "${TRAIN_CONFIG}" \
    --repo-id "${DATA_REPO}" \
    --root "${DATA_ROOT}" \
    --assets-dir "${ASSETS_DIR}" \
    --asset-id "${ASSET_ID}" \
    --low-dim-only \
    2>&1 | tee "${LOGDIR}/norm_${MODALITY}.log"
else
  echo "[4/6] SKIP_NORM=1: reusing ${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
fi

echo "[5/6] Verify norm stats"
test -f "${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
"${PYTHON}" - <<'PY'
import json
import os

import numpy as np

path = f"{os.environ['ASSETS_DIR']}/{os.environ['ASSET_ID']}/norm_stats.json"
stats = json.load(open(path))["norm_stats"]
print("norm_stats", path, "keys", sorted(stats))
if any("force" in key for key in stats):
    raise SystemExit(f"force key in norm stats: {sorted(stats)}")
if len(stats["actions"]["mean"]) != 7:
    raise SystemExit(f"actions must be 7D, got {len(stats['actions']['mean'])}")
if not np.all(np.asarray(stats["actions"]["std"], dtype=float) > 0):
    raise SystemExit("non-positive action std")
for key in sorted(stats):
    std = np.asarray(stats[key]["std"], dtype=float)
    print(key, "dim", len(stats[key]["mean"]), "std_min", float(std.min()), "std_max", float(std.max()))
PY

if [[ "${PREP_ONLY:-0}" == "1" ]]; then
  echo "PREP_ONLY=1: conversion and norm stats are complete; skipping training."
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

echo "[6/6] Start ${TRAIN_CONFIG} training on ${SUITE} rigid (${MODALITY})"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(softvtbench_cuda_visible_devices "${FSDP_DEVICES}")}" \
"${PYTHON}" scripts/train.py "${TRAIN_CONFIG}" \
  --exp-name "${EXP_NAME}" \
  --data.repo-id "${DATA_REPO}" \
  --data.root "${DATA_ROOT}" \
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
