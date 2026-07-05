#!/usr/bin/env bash
# Full spatial-soft vision+tactile pi05 pipeline: staging -> convert -> norm stats -> train.
#
# Previously this script started from an already-converted LeRobot dataset and
# pre-computed norm stats, and nothing in the repository produced them. Steps
# [1..4] below now build both from the raw replay-format dataset, mirroring
# train_object_soft_pi05_tactile_targetnext_7d_a800_20260626.sh.
#
# Usage:
#   RAW_ROOT=/path/to/SoftVTBench_data/spatial-soft \
#   BASE_PARAMS=/path/to/pi05_base/params \
#   OPENPI_PYTHON=/path/to/openpi-venv/bin/python \
#     bash openpi/scripts/train_spatial_soft_pastry005_pi05_tactile_7d_parquet_norm_a800_20260626.sh
#
# To reuse previously prepared artifacts: SKIP_CONVERT=1 SKIP_NORM=1
# Other flags: PREP_ONLY=1  REQUIRE_A800=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/softvtbench_paths.sh"
REPO="${REPO:-${OPENPI_CODE_DIR}}"
PYTHON="${PYTHON:-${OPENPI_PYTHON}}"

RAW_ROOT="${RAW_ROOT:?RAW_ROOT is required; point it at SoftVTBench_data/spatial-soft}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_lora_tacall_softvtbench}"
RUN_ID="${RUN_ID:-spatial_soft_pastry005_10tasks_50each_gripperjitter_20260624}"
DATA_REPO="${DATA_REPO:-local/spatial_soft_pastry005_10tasks_50each_gripperjitter_tactile_20260624}"
DATA_ROOT="${DATA_ROOT:-${OPENPI_CACHE_ROOT}/lerobot/${DATA_REPO}}"
ASSETS_DIR="${ASSETS_DIR:-${OPENPI_ASSETS_ROOT}/${RUN_ID}/pi05_tactile}"
ASSET_ID="${ASSET_ID:-spatial_soft_pastry005_pi05_tactile_h50_gripperjitter_7d_parquet_stable_20260626}"
BASE_PARAMS="${BASE_PARAMS:-${PI05_BASE:-${OPENPI_CACHE_ROOT}/openpi/openpi-assets/checkpoints/pi05_base/params}}"
EXP_NAME="${EXP_NAME:-spatial_soft_pastry005_pi05_tactile_h50_gripperjitter_7dparquetnorm_bs256_8gpu_20260626_a800_r3}"

STAGE_ROOT="${STAGE_ROOT:-${OPENPI_STAGE_ROOT}/${RUN_ID}_tactile}"
TASK_SUBSET="${TASK_SUBSET:-${OPENPI_LAUNCHER_ROOT}/configs/task_subset_softvtbench.json}"
LOGDIR="${LOGDIR:-${OPENPI_LOG_ROOT}/${RUN_ID}}"

FSDP_DEVICES="${FSDP_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
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
mkdir -p "${LOGDIR}" "${STAGE_ROOT}/replayed_demos" "${STAGE_ROOT}/video_datasets" "${ASSETS_DIR}"

export REPO PYTHON RAW_ROOT RUN_ID DATA_REPO DATA_ROOT ASSETS_DIR ASSET_ID STAGE_ROOT TASK_SUBSET TRAIN_CONFIG
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

if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
  test -d "${RAW_ROOT}"

  echo "[1/7] Build staging symlinks"
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
  CAMERA_MP4="$(find "${RAW_ROOT}" -path '*/video_datasets/*/videos/*.mp4' | wc -l)"
  TACTILE_MP4="$(find "${RAW_ROOT}" -path '*/video_datasets/*/tactile_outputs/*.mp4' | wc -l)"
  echo "stage_hdf5=${H5_COUNT} stage_tasks=${TASK_COUNT} camera_mp4=${CAMERA_MP4} tactile_mp4=${TACTILE_MP4}"
  [[ "${H5_COUNT}" == "10" ]] || { echo "Expected 10 HDF5 files." >&2; exit 2; }
  [[ "${TASK_COUNT}" == "10" ]] || { echo "Expected 10 task video dirs." >&2; exit 2; }
  [[ "${CAMERA_MP4}" == "1000" ]] || { echo "Expected 1000 camera videos." >&2; exit 2; }
  [[ "${TACTILE_MP4}" == "1000" ]] || { echo "Expected 1000 tactile videos." >&2; exit 2; }
  # NOTE: demo count is derived from the HDF5 files, not manifest.jsonl. The
  # spatial-soft manifest is split across manifest.jsonl (task0-4) and
  # manifest_task5_9_copy_*.jsonl (task5-9), so a line count would under-report.

  echo "[2/7] Verify sample camera/tactile videos"
  "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import cv2

raw = Path(os.environ["RAW_ROOT"])
samples = [
    next(raw.glob("libero_spatial/libero_spatial_task*/video_datasets/*/videos/*agentview_rgb.mp4")),
    next(raw.glob("libero_spatial/libero_spatial_task*/video_datasets/*/tactile_outputs/*gsmini_left_markers_rgb.mp4")),
]
for p in samples:
    cap = cv2.VideoCapture(str(p))
    ok = cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    cap.release()
    print("video_ok", p.name, "opened", ok, "frames", frames, "fps", fps, "first_frame", ret)
    assert ok and frames > 0 and ret
PY

  echo "[3/7] Convert tactile LeRobot repo with 7D targetnext actions and no force fields"
  "${PYTHON}" - <<'PY' 2>&1 | tee "${LOGDIR}/convert_tactile_targetnext_7d.log"
import importlib.util
import os
import re
import sys
from pathlib import Path

import h5py
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
    task_suites=("libero_spatial",),
    data_root=Path(os.environ["STAGE_ROOT"]),
    output_dir=Path(os.environ["HF_LEROBOT_HOME"]),
    repo_name=os.environ["DATA_REPO"],
    task_subset_path=Path(os.environ["TASK_SUBSET"]),
    tactile_history_len=8,
    marker_history_len=2,
    tactile_output_type="markers_rgb",
)

# Soft task prompts live in the HDF5 attrs, not in examples/softvtbench/config/libero_spatial.json.
task_configs = {8888: "valid"}
for h5_path in sorted(Path(cfg.hdf5_folder).glob("libero_spatial_task*_*.hdf5")):
    match = re.search(r"libero_spatial_task(\d+)_", h5_path.name)
    if match is None:
        continue
    task_id = int(match.group(1))
    with h5py.File(h5_path, "r") as f:
        demo_names = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        language = f["data"][demo_names[0]].attrs.get("language", "")
        if isinstance(language, bytes):
            language = language.decode("utf-8")
    if not language:
        raise RuntimeError(f"Missing language attr in {h5_path}")
    task_configs[task_id] = language
cfg.task_configs = task_configs
print("task_configs", task_configs)

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
else
  echo "[1-3/7] SKIP_CONVERT=1: reusing ${DATA_ROOT}"
fi

echo "[4/7] Sanity check converted tactile repo"
test -d "${DATA_ROOT}"
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
assert info.get("total_episodes") == 500, info.get("total_episodes")
assert info.get("total_tasks") == 10, info.get("total_tasks")
for key in ("image", "wrist_image", "tactile_image", "state", "actions", "tactile_marker_motion"):
    assert key in features, key
for key in features:
    assert "force" not in key.lower(), key
assert features["actions"]["shape"] == [7], features["actions"]["shape"]
assert features["tactile_marker_motion"]["shape"] == [3, 198, 2], features["tactile_marker_motion"]["shape"]

tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
print("task_texts", [t.get("task") for t in tasks])
assert len(tasks) == 10, len(tasks)
assert all("soft" in t.get("task", "") for t in tasks), tasks

max_err = 0.0
for parquet in sorted((root / "data").glob("**/*.parquet")):
    df = pd.read_parquet(parquet)
    states = np.stack(df["state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
    if len(states) > 1:
        max_err = max(max_err, float(np.max(np.abs(actions[:-1] - states[1:]))))
    max_err = max(max_err, float(np.max(np.abs(actions[-1] - states[-1]))))
print("targetnext_max_err", max_err)
assert max_err < 1e-6, max_err
PY

if [[ "${SOFTVTBENCH_STOP_AFTER_CONVERT:-0}" == "1" ]]; then
  echo "PHASE=convert: conversion and validation are complete."
  exit 0
fi

if [[ "${SKIP_NORM:-0}" != "1" ]]; then
  echo "[5/7] Compute pi05 tactile norm stats"
  "${PYTHON}" scripts/compute_norm_stats.py \
    --config-name "${TRAIN_CONFIG}" \
    --repo-id "${DATA_REPO}" \
    --root "${DATA_ROOT}" \
    --assets-dir "${ASSETS_DIR}" \
    --asset-id "${ASSET_ID}" \
    --low-dim-only \
    2>&1 | tee "${LOGDIR}/norm_pi05_tactile_targetnext_7d.log"
else
  echo "[5/7] SKIP_NORM=1: reusing ${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
fi

echo "[6/7] Verify norm stats"
test -f "${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
"${PYTHON}" - <<'PY'
import json
import os

import numpy as np

path = f"{os.environ['ASSETS_DIR']}/{os.environ['ASSET_ID']}/norm_stats.json"
stats = json.load(open(path))["norm_stats"]
print("norm_stats", path)
print("keys", sorted(stats))
assert sorted(stats) == ["actions", "state", "tactile_prefix", "tactile_suffix"], sorted(stats)
assert len(stats["actions"]["mean"]) == 7, len(stats["actions"]["mean"])
assert not any("force" in key for key in stats), sorted(stats)
action_std = np.asarray(stats["actions"]["std"], dtype=float)
assert action_std.shape == (7,), action_std.shape
assert np.all(action_std > 0), action_std
for key in ("actions", "state", "tactile_prefix", "tactile_suffix"):
    std = np.asarray(stats[key]["std"], dtype=float)
    print(key, "dim", len(stats[key]["mean"]), "std_min", float(std.min()), "std_max", float(std.max()))
# The previously prepared dataset had action_std[3] in (0.007, 0.010). Report rather
# than assert: a freshly converted dataset can land slightly outside that window.
if not (0.007 < action_std[3] < 0.010):
    print(f"WARNING: action_std[3]={action_std[3]:.6f} outside the historical (0.007, 0.010) window")
print("actions_std", action_std.tolist())
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

echo "[7/7] Starting ${EXP_NAME}"
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
