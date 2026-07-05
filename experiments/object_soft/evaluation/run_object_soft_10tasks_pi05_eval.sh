#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFTVTBENCH_ROOT="${SOFTVTBENCH_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${SOFTVTBENCH_ROOT}/openpi/scripts/softvtbench_paths.sh"

SOFTVTBENCH_DIR=${SOFTVTBENCH_DIR:-${SOFTVTBENCH_ROOT}/SoftVTBench}
OPENPI_DIR=${OPENPI_DIR:-${OPENPI_CODE_DIR}}
EXTERNAL_SERVER=${EXTERNAL_SERVER:-0}
SOFTVTBENCH_PYTHON=${SOFTVTBENCH_PYTHON:-${SOFTVTBENCH_PY:-}}
: "${SOFTVTBENCH_PYTHON:?set SOFTVTBENCH_PYTHON to the Isaac Sim/SoftVTBench interpreter}"
EVAL_UTIL_PYTHON=${EVAL_UTIL_PYTHON:-${SOFTVTBENCH_PYTHON}}
OPENPI_SERVER_PYTHON=${OPENPI_SERVER_PYTHON:-${OPENPI_PYTHON:-}}
"${SOFTVTBENCH_PYTHON}" -c 'import h5py, isaacsim, numpy, scipy' >/dev/null
if [[ "${EXTERNAL_SERVER}" != "1" ]]; then
  softvtbench_require_openpi_python "${OPENPI_SERVER_PYTHON}"
fi
COLLECTION_ROOT=${COLLECTION_ROOT:-${SOFTVTBENCH_DIR}/formal_collections/object_soft_10assets_10tasks_50each_initjitter012_gripperjitter5_natural_language_replay_format_20260625}
EMPTY_HDF5_DIR=${EMPTY_HDF5_DIR:-${SOFTVTBENCH_RESULTS_ROOT}/empty_hdf5_for_object_soft_10tasks_eval}
SAFETY_THRESHOLDS=${SAFETY_THRESHOLDS:-${SOFTVTBENCH_ROOT}/configs/safety_thresholds.json}
PHYSICS_CONFIG=${PHYSICS_CONFIG:-${SOFTVTBENCH_ROOT}/configs/simulation_physics_v1.json}

# Scene assets for closed-loop evaluation. Task configs + per-demo scene params ship
# in this repo; the USD library comes from the eval-assets bundle of the SoftVTBench
# dataset (see README "Downloading Assets & Datasets").
SOFTVT_EVAL_CONFIG_DIR=${SOFTVT_EVAL_CONFIG_DIR:-${SOFTVTBENCH_ROOT}/configs/object_soft}
SOFTVT_EVAL_USD_DIR=${SOFTVT_EVAL_USD_DIR:-${SOFTVTBENCH_ROOT}/eval-assets/USD}
export SOFTVT_EVAL_CONFIG_DIR SOFTVT_EVAL_USD_DIR

CONFIG=${CONFIG:?CONFIG is required, e.g. pi05_lora_vision_softvtbench or pi05_lora_tacall_softvtbench}
CKPT=${CKPT:-}
if [[ "${EXTERNAL_SERVER}" == "1" ]]; then
  EXP=${EXP:-external_server}
  STEP=${STEP:-external}
else
  [[ -n "${CKPT}" ]] || { echo "CKPT is required unless EXTERNAL_SERVER=1" >&2; exit 2; }
  [[ -d "${CKPT}" ]] || { echo "checkpoint directory does not exist: ${CKPT}" >&2; exit 2; }
  EXP=${EXP:-$(basename "$(dirname "${CKPT}")")}
  STEP=${STEP:-$(basename "${CKPT}")}
fi
[[ -f "${SAFETY_THRESHOLDS}" ]] || {
  echo "Missing formal compression-sweep thresholds: ${SAFETY_THRESHOLDS}" >&2
  echo "Generate them with experiments/common/calibrate_safety_thresholds.py." >&2
  exit 2
}
"${EVAL_UTIL_PYTHON}" - "${SAFETY_THRESHOLDS}" <<'PY_THRESHOLDS'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload.get("calibration", {}).get("method") == "compression_sweep", payload.get("calibration")
assert payload.get("metric_id") == "fem_rms_rigid_aligned_bbox_pct_v1", payload.get("metric_id")
assert payload.get("thresholds"), "empty thresholds"
PY_THRESHOLDS
VARIANT=${VARIANT:-object_soft_10tasks_pi05_eval}
MODE=${MODE:-tactile}  # tactile or vision_abs7d
CONTROL_MODE=${CONTROL_MODE:-binary}

PORT=${PORT:-8194}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
N="${N:-50}"
[[ "${N}" =~ ^[1-9][0-9]*$ ]] || { echo "N must be a positive integer; got: ${N}" >&2; exit 2; }
EVAL_INIT_STRATEGY=${EVAL_INIT_STRATEGY:-per_demo}
EVAL_DEMO_OFFSET=${EVAL_DEMO_OFFSET:-0}
TASKS_STR=${TASKS_STR:-"0 1 2 3 4 5 6 7 8 9"}
# The public dispatcher uses `object-soft`, while the Isaac task registry and
# config files use the legacy LIBERO suite identifier.
if [[ "${SUITE:-}" == "object-soft" ]]; then
  SUITE=libero_object
else
  SUITE=${SUITE:-libero_object}
fi
REPLAN_STEPS=${REPLAN_STEPS:-10}
MAX_INFERENCE_STEPS=${MAX_INFERENCE_STEPS:-70}
TACTILE_OUTPUT_TYPE=${TACTILE_OUTPUT_TYPE:-markers_rgb}
RUN_ROOT=${RUN_ROOT:-${SOFTVTBENCH_RESULTS_ROOT}/object_soft_10tasks_pi05_eval_20260627}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${RUN_ROOT}/${VARIANT}/${EXP}/${STEP}/replan_${REPLAN_STEPS}_n${N}/${RUN_ID}}
LOG_DIR=${OUT_ROOT}/logs
DEBUG_ROOT=${OUT_ROOT}/debug
ENV_DIR=${OUT_ROOT}/task_env
VIDEO_LOG=${LOG_DIR}/video_encode.log

mkdir -p "${LOG_DIR}" "${DEBUG_ROOT}" "${ENV_DIR}" "${EMPTY_HDF5_DIR}"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${OPENPI_CACHE_ROOT}/openpi}
export OPENPI_SAMPLE_NUM_STEPS=${OPENPI_SAMPLE_NUM_STEPS:-10}
export SOFTVTBENCH_SKIP_ISAAC_CLEANUP_ON_EXIT=1
export SOFTVTBENCH_MAX_CONSECUTIVE_FAILURES=${SOFTVTBENCH_MAX_CONSECUTIVE_FAILURES:-10}
export SOFTVTBENCH_GRIPPER_ACTION_MODE=${SOFTVTBENCH_GRIPPER_ACTION_MODE:-abs}
export SOFTVTBENCH_EVAL_GRIPPER_CONTROLLER=${SOFTVTBENCH_EVAL_GRIPPER_CONTROLLER:-policy_abs}
export SOFTVTBENCH_EVAL_GRIPPER_USE_DEMO_REFERENCES=${SOFTVTBENCH_EVAL_GRIPPER_USE_DEMO_REFERENCES:-0}
export SOFTVTBENCH_TACTILE_MOSAIC_LAYOUT=rows
# Isaac Sim native extensions. Resolve them from the active interpreter rather
# than a pinned conda path, and glob the versioned extension directories rather
# than pinning build hashes -- both differ across Isaac Sim installs.
if [[ -z "${ISAAC_EXTSCACHE:-}" ]]; then
  ISAAC_EXTSCACHE="$("${SOFTVTBENCH_PYTHON}" -c \
    'import isaacsim, pathlib; print(pathlib.Path(isaacsim.__file__).parent / "extscache")' 2>/dev/null || true)"
fi
export ISAAC_EXTSCACHE
_isaac_ext() {  # first match for a glob, empty if none
  compgen -G "${ISAAC_EXTSCACHE}/$1" 2>/dev/null | sort | head -1 || true
}
export WARP_EXT="${WARP_EXT:-$(_isaac_ext 'omni.warp.core-*')}"
export ISAAC_USD_LIB_DIR="${ISAAC_USD_LIB_DIR:-$(_isaac_ext 'omni.usd.libs-*')/bin}"
export ISAAC_USD_CORE_LIB_DIR="${ISAAC_USD_CORE_LIB_DIR:-$(_isaac_ext 'omni.usd.core-*')/bin}"
export ISAAC_USD_LAYERS_LIB_DIR="${ISAAC_USD_LAYERS_LIB_DIR:-$(_isaac_ext 'omni.kit.usd.layers-*')/bin}"
if [[ -z "${ISAAC_EXTSCACHE}" ]]; then
  echo "WARNING: could not locate the isaacsim package from ${SOFTVTBENCH_PYTHON};" >&2
  echo "         set ISAAC_EXTSCACHE explicitly if Isaac Sim fails to start." >&2
fi
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
export __GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME:-nvidia}
export __NV_PRIME_RENDER_OFFLOAD=${__NV_PRIME_RENDER_OFFLOAD:-1}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}
for lib_dir in "${ISAAC_USD_CORE_LIB_DIR}" "${ISAAC_USD_LIB_DIR}" "${ISAAC_USD_LAYERS_LIB_DIR}" /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
  if [[ -d "${lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
  fi
done
if [[ -n "${SOFTVTBENCH_CUDNN_COMPAT_DIR:-}" && -d "${SOFTVTBENCH_CUDNN_COMPAT_DIR}" ]]; then
  export LD_LIBRARY_PATH="${SOFTVTBENCH_CUDNN_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
fi
unset OPENPI_ADD_BYTES_KEY_ALIASES || true

cat > "${OUT_ROOT}/run.info" <<EOF_INFO
CONFIG=${CONFIG}
CKPT=${CKPT}
VARIANT=${VARIANT}
EXP=${EXP}
STEP=${STEP}
MODE=${MODE}
CONTROL_MODE=${CONTROL_MODE}
EXTERNAL_SERVER=${EXTERNAL_SERVER}
SERVER_HOST=${SERVER_HOST}
SUITE=${SUITE}
TASKS=${TASKS_STR}
N=${N}
EVAL_INIT_STRATEGY=${EVAL_INIT_STRATEGY}
EVAL_DEMO_OFFSET=${EVAL_DEMO_OFFSET}
REPLAN_STEPS=${REPLAN_STEPS}
MAX_INFERENCE_STEPS=${MAX_INFERENCE_STEPS}
OPENPI_SAMPLE_NUM_STEPS=${OPENPI_SAMPLE_NUM_STEPS}
TACTILE_OUTPUT_TYPE=${TACTILE_OUTPUT_TYPE}
COLLECTION_ROOT=${COLLECTION_ROOT}
OPENPI_SERVER_PYTHON=${OPENPI_SERVER_PYTHON}
SOFTVTBENCH_PYTHON=${SOFTVTBENCH_PYTHON}
EVAL_UTIL_PYTHON=${EVAL_UTIL_PYTHON}
PHYSICS_CONFIG=${PHYSICS_CONFIG}
SAFETY_THRESHOLDS=${SAFETY_THRESHOLDS}
SOFTVTBENCH_TACTILE_MOSAIC_LAYOUT=${SOFTVTBENCH_TACTILE_MOSAIC_LAYOUT}
SOFTVTBENCH_GRIPPER_ACTION_MODE=${SOFTVTBENCH_GRIPPER_ACTION_MODE}
SOFTVTBENCH_EVAL_GRIPPER_CONTROLLER=${SOFTVTBENCH_EVAL_GRIPPER_CONTROLLER}
STARTED_AT=$(date --iso-8601=seconds)
EOF_INFO

prepare_checkpoint_view() {
  local dst_asset_id=""
  local candidates=()
  case "${CONFIG}" in
    pi05_lora_vision_softvtbench)
      dst_asset_id="local/softvtbench_vision"
      candidates=("object_soft_vision_pi05_h50_targetnext_20260625" "local/softvtbench_vision")
      ;;
    pi05_lora_tacall_softvtbench)
      dst_asset_id="local/softvtbench_tactile"
      candidates=(
        "object_soft_10assets_pi05_tactile_h50_targetnext_7d"
        "object_soft_10assets_pi05_tactile_h50_targetnext_7d_parquet_stable_20260626"
        "local/softvtbench_tactile"
      )
      ;;
    *) echo "Unsupported pi05 evaluation config: ${CONFIG}" >&2; return 2 ;;
  esac
  # A user-trained checkpoint carries whatever asset id its training config
  # produced; let callers name it instead of relying on the built-in list.
  if [[ -n "${NORM_STATS_SOURCE_ASSET_ID:-}" ]]; then
    candidates=("${NORM_STATS_SOURCE_ASSET_ID}" "${candidates[@]}")
  fi
  POLICY_CKPT_DIR="${OUT_ROOT}/checkpoint_view"
  export POLICY_CKPT_DIR
  if [[ -e "${POLICY_CKPT_DIR}" ]]; then
    mv -- "${POLICY_CKPT_DIR}" "${POLICY_CKPT_DIR}.stale_$(date +%Y%m%d_%H%M%S)"
  fi
  softvtbench_make_checkpoint_view "${CKPT}" "${POLICY_CKPT_DIR}"
  echo "POLICY_CKPT_DIR=${POLICY_CKPT_DIR}" >> "${OUT_ROOT}/run.info"
  local src
  if ! src="$(softvtbench_alias_norm_stats "${CKPT}" "${POLICY_CKPT_DIR}" "${dst_asset_id}" "${candidates[@]}")"; then
    echo "ERROR: no compatible norm_stats in ${CKPT}/assets for ${dst_asset_id}." >&2
    echo "       Set NORM_STATS_SOURCE_ASSET_ID to your checkpoint's asset id." >&2
    return 1
  fi
  if [[ -n "${src}" ]]; then
    echo "NORM_STATS_VIEW=${src}->${dst_asset_id}" >> "${OUT_ROOT}/run.info"
  fi
}

physics_defaults() {
  "${EVAL_UTIL_PYTHON}" - "${PHYSICS_CONFIG}" <<'PY_PHYSICS'
import json
import sys

body = json.load(open(sys.argv[1], encoding="utf-8"))["deformable_body"]
print(
    body["simulation_hexahedral_resolution"],
    body["solver_position_iteration_count"],
    body["vertex_velocity_damping"],
    body["contact_offset"],
    body["rest_offset"],
    body["max_depenetration_velocity"],
)
PY_PHYSICS
}

make_hdf5_initial_state_staging() {
  local src_hdf5="$1"
  local out_dir="$2"
  local env_name="${3:-Isaac-Libero-Franka-IK-Camera-Tactile-v0}"
  mkdir -p "${out_dir}"
  "${EVAL_UTIL_PYTHON}" - "${src_hdf5}" "${out_dir}" "${env_name}" <<'PY'
import json
import sys
from pathlib import Path

import h5py

src = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
env_name = sys.argv[3]
dst = out_dir / src.name
if dst.exists():
    dst.unlink()

with h5py.File(src, "r") as fs, h5py.File(dst, "w") as fd:
    if "data" not in fs:
        raise SystemExit(f"missing /data group: {src}")
    gd = fd.create_group("data")
    gd.attrs["env_args"] = json.dumps({"env_name": env_name, "type": 2})
    if "total" in fs["data"].attrs:
        gd.attrs["total"] = fs["data"].attrs["total"]
    else:
        gd.attrs["total"] = 0
    for demo_key in fs["data"].keys():
        src_demo = fs["data"][demo_key]
        dst_demo = gd.create_group(demo_key)
        for attr_key, attr_val in src_demo.attrs.items():
            dst_demo.attrs[attr_key] = attr_val
        for key in src_demo.keys():
            if key != "initial_state":
                dst_demo[key] = h5py.ExternalLink(str(src), f"/data/{demo_key}/{key}")
                continue

            dst_init = dst_demo.create_group("initial_state")
            src_init = src_demo["initial_state"]
            for init_key in src_init.keys():
                if init_key != "deformable_object":
                    dst_init[init_key] = h5py.ExternalLink(str(src), f"/data/{demo_key}/initial_state/{init_key}")
                    continue

                dst_def = dst_init.create_group("deformable_object")
                src_def = src_init["deformable_object"]
                for asset_name in src_def.keys():
                    src_asset = src_def[asset_name]
                    dst_asset = dst_def.create_group(asset_name)
                    if "nodal_pos_w" not in src_asset:
                        raise SystemExit(f"{src}:{demo_key}:{asset_name} missing nodal_pos_w")
                    dst_asset["nodal_position"] = h5py.ExternalLink(
                        str(src), f"/data/{demo_key}/initial_state/deformable_object/{asset_name}/nodal_pos_w"
                    )
                    shape = src_asset["nodal_pos_w"].shape
                    dst_asset.create_dataset("nodal_velocity", data=__import__("numpy").zeros(shape, dtype="float32"))
print(dst)
PY
}

write_task_env() {
  local task_id="$1"
  local out_file="$2"
  local demo_index="${3:-0}"
  "${EVAL_UTIL_PYTHON}" - "${COLLECTION_ROOT}" "${task_id}" "${out_file}" "${demo_index}" <<'PY'
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

collection_root, task_id_raw, out_file, demo_index_raw = sys.argv[1:]
collection_root = Path(collection_root)
task_id = int(task_id_raw)
requested_demo_index = int(demo_index_raw)

# Scene definition and simulation assets are resolved from the public release,
# not from the original collection tree.
eval_config_dir = Path(os.environ["SOFTVT_EVAL_CONFIG_DIR"])
eval_usd_dir = Path(os.environ["SOFTVT_EVAL_USD_DIR"])
if not eval_config_dir.is_dir():
    raise SystemExit(f"SOFTVT_EVAL_CONFIG_DIR not a directory: {eval_config_dir}")
if not eval_usd_dir.is_dir():
    raise SystemExit(f"SOFTVT_EVAL_USD_DIR not a directory: {eval_usd_dir}")

manifest = collection_root / "manifest.jsonl"
if not manifest.exists():
    raise SystemExit(f"missing manifest: {manifest}")

records = []
for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    if int(rec.get("task_id", -1)) == task_id and rec.get("success") is not False:
        records.append(rec)
if not records:
    raise SystemExit(f"no success records for task{task_id} in {manifest}")
selected_demo_index = requested_demo_index % len(records)
record = records[selected_demo_index]

aligned_demo = str(record.get("episode") or record.get("aligned_demo") or f"demo_{selected_demo_index}")

# Resolve the replay HDF5 relative to collection_root so the dataset can live anywhere.
task_suite = str(record.get("task_suite") or "libero_object")
replay_dir = collection_root / task_suite / f"{task_suite}_task{task_id}" / "replayed_demos"
aligned_hdf5 = None
portable_hdf5 = Path(str(record.get("hdf5_path", "")))
if str(portable_hdf5) not in {"", "."} and not portable_hdf5.is_absolute() and (collection_root / portable_hdf5).exists():
    aligned_hdf5 = collection_root / portable_hdf5
manifest_name = Path(str(record.get("aligned_hdf5", ""))).name
if aligned_hdf5 is None and manifest_name and (replay_dir / manifest_name).exists():
    aligned_hdf5 = replay_dir / manifest_name
if aligned_hdf5 is None:
    candidates = sorted(replay_dir.glob("*.hdf5"))
    if not candidates:
        raise SystemExit(f"missing replay hdf5 under {replay_dir}")
    aligned_hdf5 = candidates[0]

with h5py.File(aligned_hdf5, "r") as f:
    demo = f["data"][aligned_demo]
    language = str(demo.attrs.get("language", record.get("language", "")))
    asset_name = str(demo.attrs.get("asset_name", "")).strip()
    arm0 = [float(x) for x in demo["obs/arm_joint_pos"][0].reshape(-1)[:7]]
    gp0 = demo["obs/gripper_pos"][0].reshape(-1)
    g0 = float(abs(gp0[0])) if len(gp0) else 0.04
    actions = np.asarray(demo["actions"]) if "actions" in demo else np.zeros((0, 0), dtype=np.float32)
    gripper_action = actions[:, 6].astype(np.float32) if actions.ndim == 2 and actions.shape[1] >= 7 else np.zeros((0,), dtype=np.float32)

if not asset_name:
    raise SystemExit(f"missing asset_name attr in {aligned_hdf5}:{aligned_demo}")
asset_base = asset_name[5:] if asset_name.startswith("soft_") else asset_name
soft_name = asset_name if asset_name.startswith("soft_") else f"soft_{asset_name}"

close_frame = ""
open_frame = ""
close_finger = ""
if gripper_action.size:
    closed = np.isfinite(gripper_action) & (gripper_action < 0.020)
    closed_idx = np.flatnonzero(closed)
    if closed_idx.size:
        close_frame = int(closed_idx[0])
        close_finger = float(np.median(gripper_action[closed]))
        open_idx = np.flatnonzero((np.arange(gripper_action.size) > close_frame) & (~closed))
        if open_idx.size:
            open_frame = int(open_idx[0])

extra_usd = eval_usd_dir / asset_base / f"{asset_base}.usd"
if not extra_usd.exists():
    matches = sorted(eval_usd_dir.glob(f"**/{asset_base}.usd"))
    if matches:
        extra_usd = matches[0]
if not extra_usd.exists():
    raise SystemExit(f"missing soft usd for {asset_base} under {eval_usd_dir}")

config_dir = Path(out_file).parent / f"config_task{task_id}_{aligned_demo}"
if config_dir.exists():
    shutil.rmtree(config_dir)
shutil.copytree(eval_config_dir, config_dir, symlinks=True)

cfg_path = config_dir / "libero_object.json"
data = json.loads(cfg_path.read_text())
selected_task = None
for task in data.get("tasks", []):
    if int(task.get("task_id", -1)) == task_id:
        selected_task = task
        task["language"] = language
        task["language_instruction"] = language
        for goal in task.get("goals", []):
            if str(goal.get("ref_obj", "")).startswith("soft_"):
                goal["ref_obj"] = soft_name
                goal["enable_force_threshold"] = "False"
        break
if selected_task is None:
    raise SystemExit(f"task{task_id} not found in {cfg_path}")
cfg_path.write_text(json.dumps(data, indent=2))

# Per-demo scene parameters captured at collection time. The deformable body's true
# initial state is restored from the HDF5 (world-frame nodal positions), so this pose
# only controls where the prim is spawned before reset_to() overwrites it.
region = (selected_task.get("regions", {}) or {}).get("floor_target_object_region", {}).get("pose_range", {})


def _mid(name, default):
    v = region.get(name)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (float(v[0]) + float(v[1])) / 2.0
    return float(default)


extra_asset_pos = ""
pos_source = ""
scene_params_path = eval_config_dir / "scene_params.json"
if scene_params_path.exists():
    scene_params = json.loads(scene_params_path.read_text())
    entry = scene_params.get(str(task_id), {}).get(aligned_demo, {})
    if entry.get("pos"):
        extra_asset_pos = str(entry["pos"]).strip()
        pos_source = f"scene_params.json:{task_id}/{aligned_demo}"
if not extra_asset_pos:
    extra_asset_pos = f"{_mid('x', 0):.8f} {_mid('y', 0):.8f} {_mid('z', 0.045):.8f}"
    pos_source = "config_floor_target_object_region"

q_xyzw = R.from_euler("xyz", [_mid("roll", 0.0), _mid("pitch", 0.0), _mid("yaw", 0.0)]).as_quat()
quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
extra_asset_rot = " ".join(f"{v:.8f}" for v in quat_wxyz)

hex_res = int(os.environ.get("SOFTVTBENCH_OBJECT_SOFT_HEX_RES", "6"))


def env_line(key, value):
    return f"export {key}={shlex.quote(str(value))}\n"


lines = {
    "TASK_ID_SOFT": task_id,
    "EVAL_DEMO_REQUESTED_INDEX_SOFT": requested_demo_index,
    "EVAL_DEMO_INDEX_SOFT": selected_demo_index,
    "EVAL_DEMO_KEY_SOFT": aligned_demo,
    "RUN_ROOT_SOFT": str(collection_root),
    "LIBERO_CONFIG_DIR_SOFT": str(config_dir),
    "LIBERO_CONFIG_DIR_ORIG": str(eval_config_dir),
    "LIBERO_ASSETS_DATA_DIR_SOFT": str(eval_usd_dir),
    "SOFTVTBENCH_EXTRA_ASSET_USD_SOFT": str(extra_usd),
    "SOFTVTBENCH_EXTRA_ASSET_NAME_SOFT": soft_name,
    "SOFTVTBENCH_EXTRA_ASSET_BASE_SOFT": asset_base,
    "SOFTVTBENCH_EXTRA_ASSET_POS_SOFT": extra_asset_pos,
    "SOFTVTBENCH_EXTRA_ASSET_POS_SOURCE_SOFT": pos_source,
    "SOFTVTBENCH_EXTRA_ASSET_ROT_SOFT": extra_asset_rot,
    "SOFTVTBENCH_EXTRA_HEX_RES_SOFT": hex_res,
    "SOFTVTBENCH_ROBOT_INIT_JOINT_POS_SOFT": " ".join(f"{x:.8f}" for x in (arm0 + [g0, g0])),
    "LANGUAGE_INSTRUCTION_SOFT": language,
    "GRIPPER_CLOSE_FRAME_REFERENCE": close_frame,
    "GRIPPER_OPEN_FRAME_REFERENCE": open_frame,
    "GRIPPER_CLOSE_FINGER_REFERENCE": close_finger,
    "SOURCE_HDF5_SOFT": str(record.get("source_hdf5", "")),
    "ALIGNED_HDF5_SOFT": str(aligned_hdf5),
    "SOURCE_LOG_SOFT": "",
}

with open(out_file, "w", encoding="utf-8") as f:
    for k, v in lines.items():
        f.write(env_line(k, v))
PY
}

cleanup() {
  if [[ "${EXTERNAL_SERVER}" == "1" ]]; then
    return 0
  fi
  if [[ -f "${OUT_ROOT}/server.pid" ]]; then
    local pid
    pid=$(cat "${OUT_ROOT}/server.pid" || true)
    if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      sleep 2
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

encode_one_exp() {
  local exp_dir=$1
  local videos_dir="${exp_dir}/videos"
  mkdir -p -- "${videos_dir}"
  if compgen -G "${exp_dir}/camera_rgb/frame_*_agentview.png" >/dev/null; then
    ffmpeg -nostdin -hide_banner -loglevel error -y -framerate 10 -pattern_type glob \
      -i "${exp_dir}/camera_rgb/frame_*_agentview.png" \
      -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p \
      "${videos_dir}/agentview.mp4" >> "${VIDEO_LOG}" 2>&1 || true
  fi
  if compgen -G "${exp_dir}/camera_rgb/frame_*_eye.png" >/dev/null; then
    ffmpeg -nostdin -hide_banner -loglevel error -y -framerate 10 -pattern_type glob \
      -i "${exp_dir}/camera_rgb/frame_*_eye.png" \
      -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p \
      "${videos_dir}/eye.mp4" >> "${VIDEO_LOG}" 2>&1 || true
  fi
  for tac in gsmini_left gsmini_right; do
    if compgen -G "${exp_dir}/tactile_markers_rgb/frame_*_${tac}_markers_rgb.png" >/dev/null; then
      ffmpeg -nostdin -hide_banner -loglevel error -y -framerate 10 -pattern_type glob \
        -i "${exp_dir}/tactile_markers_rgb/frame_*_${tac}_markers_rgb.png" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p \
        "${videos_dir}/${tac}_markers_rgb.mp4" >> "${VIDEO_LOG}" 2>&1 || true
    fi
  done
}

encode_task_videos() {
  local task_debug=$1
  while IFS= read -r exp_dir; do
    encode_one_exp "${exp_dir}"
  done < <(find "${task_debug}" -type d -name exp_* | sort)
}

if [[ "${EXTERNAL_SERVER}" != "1" ]]; then
  prepare_checkpoint_view
fi

cd "${OPENPI_DIR}"
export PYTHONPATH=${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src:${PYTHONPATH:-}
if [[ "${EXTERNAL_SERVER}" != "1" ]]; then
  "${OPENPI_SERVER_PYTHON}" scripts/serve_policy.py \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config "${CONFIG}" \
    --policy.dir "${POLICY_CKPT_DIR}" \
    > "${LOG_DIR}/server_${PORT}.log" 2>&1 &
  echo $! > "${OUT_ROOT}/server.pid"
  for i in $(seq 1 300); do
    if grep -Eq "server listening on|Listening on|Uvicorn running|Started server" "${LOG_DIR}/server_${PORT}.log"; then
      break
    fi
    if ! kill -0 "$(cat "${OUT_ROOT}/server.pid")" 2>/dev/null; then
      echo "server exited before ready" >&2
      tail -200 "${LOG_DIR}/server_${PORT}.log" >&2 || true
      exit 1
    fi
    [[ "${i}" == "300" ]] && { echo "server timeout" >&2; tail -200 "${LOG_DIR}/server_${PORT}.log" >&2 || true; exit 1; }
    sleep 1
  done
fi

cd "${SOFTVTBENCH_DIR}"
export PYTHONPATH=${WARP_EXT:+${WARP_EXT}:}${SOFTVTBENCH_DIR}/source/tac_manip:${OPENPI_DIR}/packages/openpi-client/src:${SOFTVTBENCH_DIR}:${PYTHONPATH:-}

ABS7D_ARGS=()
if [[ "${MODE}" == "vision_abs7d" ]]; then
  ABS7D_ARGS=(--abs7d)
fi

: > "${OUT_ROOT}/progress.tsv"
: > "${VIDEO_LOG}"
job_failed=0

run_task_client() {
  local task_id="$1"
  local demo_index="$2"
  local episode_label="$3"
  local run_n="$4"

  local task_tag=${SUITE}_task${task_id}
  local task_debug=${DEBUG_ROOT}/${task_tag}/${episode_label}
  local task_log=${LOG_DIR}/${task_tag}_${episode_label}.log
  local task_env=${ENV_DIR}/${task_tag}_${episode_label}.env
  write_task_env "${task_id}" "${task_env}" "${demo_index}"
  # shellcheck disable=SC1090
  source "${task_env}"

  unset SOFTVTBENCH_EVAL_GRIPPER_CLOSE_NORM
  unset SOFTVTBENCH_EVAL_GRIPPER_CLOSE_AFTER_FRAME
  unset SOFTVTBENCH_EVAL_GRIPPER_OPEN_AFTER_FRAME
  unset SOFTVTBENCH_EVAL_GRIPPER_CLOSE_FINGER

  export SOFTVTBENCH_ROBOT_INIT_JOINT_POS="${SOFTVTBENCH_ROBOT_INIT_JOINT_POS_SOFT}"
  read -r hex_res solver_iters vertex_damping contact_offset rest_offset max_depen < <(physics_defaults)
  local hdf5_replay_dir hdf5_env_name
  hdf5_replay_dir="${OUT_ROOT}/hdf5_initial_state/${task_tag}/${episode_label}"
  hdf5_env_name="Isaac-Libero-Franka-IK-Camera-Tactile-v0"
  if [[ "${CONTROL_MODE}" == "tactile" ]]; then
    hdf5_env_name="Isaac-Libero-Franka-Hybrid-Tactile-v0"
  elif [[ "${CONTROL_MODE}" == "hybrid" ]]; then
    hdf5_env_name="Isaac-Libero-Franka-Hybrid-ContactForce-v0"
  elif [[ "${CONTROL_MODE}" == "diffik" ]]; then
    hdf5_env_name="Isaac-Libero-Franka-IK-v0"
  elif [[ "${CONTROL_MODE}" == "osc" ]]; then
    hdf5_env_name="Isaac-Libero-Franka-OscPose-v0"
  fi
  make_hdf5_initial_state_staging "${ALIGNED_HDF5_SOFT}" "${hdf5_replay_dir}" "${hdf5_env_name}" >/dev/null

  export LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR_SOFT}"
  export LIBERO_ASSETS_DATA_DIR="${LIBERO_ASSETS_DATA_DIR_SOFT}"
  export HDF5_TRAJ_SOURCE_DIR="${hdf5_replay_dir}"
  export SOFTVTBENCH_EXTRA_ASSET_USD="${SOFTVTBENCH_EXTRA_ASSET_USD_SOFT}"
  export SOFTVTBENCH_EXTRA_ASSET_NAME="${SOFTVTBENCH_EXTRA_ASSET_NAME_SOFT}"
  export SOFTVTBENCH_EXTRA_ASSET_DEFORMABLE=1
  export SOFTVTBENCH_EXTRA_ASSET_POS="${SOFTVTBENCH_EXTRA_ASSET_POS_SOFT}"
  export SOFTVTBENCH_EXTRA_ASSET_ROT="${SOFTVTBENCH_EXTRA_ASSET_ROT_SOFT}"
  export SOFTVTBENCH_EXTRA_HEX_RES="${hex_res}"
  export SOFTVTBENCH_EXTRA_SOLVER_ITERS="${solver_iters}"
  export SOFTVTBENCH_EXTRA_VERTEX_DAMPING="${vertex_damping}"
  export SOFTVTBENCH_EXTRA_CONTACT_OFFSET="${contact_offset}"
  export SOFTVTBENCH_EXTRA_REST_OFFSET="${rest_offset}"
  export SOFTVTBENCH_EXTRA_MAX_DEPENETRATION_VELOCITY="${max_depen}"
  export SOFTVTBENCH_SKIP_DEFORMABLE_CONTACT_SENSORS=1

  [[ -d "${task_debug}" ]] && mv -- "${task_debug}" "${task_debug}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mkdir -p -- "${task_debug}"

  {
    echo "$(date --iso-8601=seconds) START ${task_tag} ${episode_label} mode=${MODE}"
    echo "eval_demo_index=${EVAL_DEMO_INDEX_SOFT}"
    echo "eval_demo_key=${EVAL_DEMO_KEY_SOFT}"
    echo "source_hdf5=${SOURCE_HDF5_SOFT}"
    echo "aligned_hdf5=${ALIGNED_HDF5_SOFT}"
    echo "hdf5_replay_dir=${HDF5_TRAJ_SOURCE_DIR}"
    echo "libero_config_dir=${LIBERO_CONFIG_DIR}"
    echo "asset_name=${SOFTVTBENCH_EXTRA_ASSET_NAME}"
    echo "asset_base=${SOFTVTBENCH_EXTRA_ASSET_BASE_SOFT}"
    echo "extra_usd=${SOFTVTBENCH_EXTRA_ASSET_USD}"
    echo "extra_pos=${SOFTVTBENCH_EXTRA_ASSET_POS}"
    echo "extra_pos_source=${SOFTVTBENCH_EXTRA_ASSET_POS_SOURCE_SOFT}"
    echo "extra_rot=${SOFTVTBENCH_EXTRA_ASSET_ROT}"
    echo "robot_init=${SOFTVTBENCH_ROBOT_INIT_JOINT_POS}"
    echo "language=${LANGUAGE_INSTRUCTION_SOFT}"
    echo "gripper_close_frame_reference=${GRIPPER_CLOSE_FRAME_REFERENCE}"
    echo "gripper_open_frame_reference=${GRIPPER_OPEN_FRAME_REFERENCE}"
    echo "gripper_close_finger_reference=${GRIPPER_CLOSE_FINGER_REFERENCE}"
    echo "eval_gripper_action_mode=${SOFTVTBENCH_GRIPPER_ACTION_MODE}"
    echo "eval_gripper_controller=${SOFTVTBENCH_EVAL_GRIPPER_CONTROLLER}"
  } | tee -a "${OUT_ROOT}/events.log"

  set +e
  "${SOFTVTBENCH_PYTHON}" -u benchmarks/openpi/openpi_inference_client.py \
    --server_host "${SERVER_HOST}" \
    --server_port "${PORT}" \
    --control_mode "${CONTROL_MODE}" \
    "${ABS7D_ARGS[@]}" \
    --task_suite "${SUITE}" \
    --task_id "${task_id}" \
    --task_config_path "${LIBERO_CONFIG_DIR}" \
    --language_instruction "${LANGUAGE_INSTRUCTION_SOFT}" \
    --tactile_output_type "${TACTILE_OUTPUT_TYPE}" \
    --num_total_experiments "${run_n}" \
    --num_success_steps 8 \
    --max_inference_steps "${MAX_INFERENCE_STEPS}" \
    --replan_steps "${REPLAN_STEPS}" \
    --num_steps_wait 5 \
    --hdf5_folder "${HDF5_TRAJ_SOURCE_DIR}" \
    --debug_mode 6 \
    --debug_path "${task_debug}" \
    --headless \
    > "${task_log}" 2>&1
  rc=$?
  set -e

  encode_task_videos "${task_debug}" || true

  "${EVAL_UTIL_PYTHON}" - <<PY >> "${OUT_ROOT}/progress.tsv"
from pathlib import Path
text = Path("${task_log}").read_text(errors="ignore") if Path("${task_log}").exists() else ""
succ = total = rate = None
gripper_lines = 0
for line in text.splitlines():
    s = line.strip()
    if s.startswith("Successful experiments:"):
        succ = int(s.split(":", 1)[1].strip())
    elif s.startswith("Total experiments:"):
        total = int(s.split(":", 1)[1].strip())
    elif s.startswith("Success rate:"):
        rate = float(s.split(":", 1)[1].strip().rstrip("%"))
    if "controller_gripper" in s or "policy_abs_joint_position" in s:
        gripper_lines += 1
print(f"${task_tag}\\t${episode_label}\\t${EVAL_DEMO_KEY_SOFT}\\t${SOFTVTBENCH_EXTRA_ASSET_NAME_SOFT}\\t{succ}\\t{total}\\t{rate}\\t${rc}\\t{gripper_lines}")
PY
  if [[ "${rc}" -ne 0 ]] || ! grep -Eq "^Total experiments:" "${task_log}"; then
    job_failed=1
  fi
  echo "$(date --iso-8601=seconds) END ${task_tag} ${episode_label} rc=${rc}" | tee -a "${OUT_ROOT}/events.log"
}

for task_id in ${TASKS_STR}; do
  if [[ "${EVAL_INIT_STRATEGY}" == "per_demo" ]]; then
    for episode_idx in $(seq 0 $((N - 1))); do
      demo_index=$((EVAL_DEMO_OFFSET + episode_idx))
      run_task_client "${task_id}" "${demo_index}" "episode_${episode_idx}" 1
    done
  else
    run_task_client "${task_id}" "${EVAL_DEMO_OFFSET}" "all" "${N}"
  fi
done

"${EVAL_UTIL_PYTHON}" - <<PY > "${OUT_ROOT}/summary.csv"
from pathlib import Path
from collections import defaultdict
root = Path("${OUT_ROOT}")
print("task,episode,demo_key,asset_name,success,total,success_rate,rc,gripper_debug_lines")
total_s = total_n = 0
by_task = defaultdict(lambda: [0, 0, []])
for line in (root / "progress.tsv").read_text().splitlines():
    if not line.strip():
        continue
    task, episode, demo_key, asset_name, succ, total, rate, rc, gripper_lines = line.split("\\t")
    s = 0 if succ == "None" else int(succ)
    n = 0 if total == "None" else int(total)
    r = 0.0 if rate == "None" else float(rate)
    total_s += s
    total_n += n
    by_task[task][0] += s
    by_task[task][1] += n
    if rc not in ("0", ""):
        by_task[task][2].append(rc)
    print(f"{task},{episode},{demo_key},{asset_name},{s},{n},{r:.2f},{rc},{gripper_lines}")
print(f"overall,,,,{total_s},{total_n},{(100 * total_s / total_n if total_n else 0):.2f},,")

with (root / "summary_by_task.csv").open("w", encoding="utf-8") as f:
    f.write("task,success,total,success_rate,nonzero_rc\\n")
    for task in sorted(by_task):
        s, n, rcs = by_task[task]
        f.write(f"{task},{s},{n},{(100 * s / n if n else 0):.2f},{'|'.join(rcs)}\\n")
    f.write(f"overall,{total_s},{total_n},{(100 * total_s / total_n if total_n else 0):.2f},\\n")
PY


# ---------------------------------------------------------------------------
# SoftVTBench metrics (paper Sec. 3.4):
#   D_peak = max_t obs/fem_deformation_rms                            (Eq. 3)
#   Safe Success = Goal Success AND (D_peak <= tau_o)
# SoftVTBench v1 reports Goal Success and Safe Success only.
# Point SAFETY_THRESHOLDS at a suite-specific calibration if you have one.
METRICS="${SOFTVTBENCH_ROOT}/experiments/common/softvtbench_metrics.py"

# Score only policy debug output. OUT_ROOT also contains HDF5 files staged for
# initial-state restoration, which the metric tool correctly treats as expert
# demonstrations when they are passed explicitly.
"${EVAL_UTIL_PYTHON}" "${METRICS}" "${DEBUG_ROOT}" \
  --thresholds "${SAFETY_THRESHOLDS}" \
  --output-dir "${OUT_ROOT}/metrics_online" \
  --strict \
  | tee "${OUT_ROOT}/metrics_online.log"

# Same metric over the released expert demonstrations, as a reference point.
"${EVAL_UTIL_PYTHON}" "${METRICS}" "${COLLECTION_ROOT}" \
  --thresholds "${SAFETY_THRESHOLDS}" \
  --output-dir "${OUT_ROOT}/metrics_reference" \
  | tee "${OUT_ROOT}/metrics_reference.log" || true

# Suite-specific diagnostic: per-episode gripper force / tactile control summary.
"${EVAL_UTIL_PYTHON}" "${SOFTVTBENCH_ROOT}/experiments/object_soft/evaluation/summarize_object_soft_eval_forces.py" \
  "${DEBUG_ROOT}" > "${OUT_ROOT}/eval_forces.log" 2>&1 || true

cat "${OUT_ROOT}/summary.csv"
echo "FINISHED_AT=$(date --iso-8601=seconds)" >> "${OUT_ROOT}/run.info"
if [[ "${job_failed}" -ne 0 ]]; then
  echo "JOB_STATUS=failed_incomplete_tasks" >> "${OUT_ROOT}/run.info"
  exit 1
fi
echo "JOB_STATUS=done" >> "${OUT_ROOT}/run.info"
touch "${OUT_ROOT}/DONE"
echo "OUT_ROOT=${OUT_ROOT}"
