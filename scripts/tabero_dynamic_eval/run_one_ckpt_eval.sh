#!/usr/bin/env bash
set -Eeuo pipefail

: "${TABERO_DIR:?Set TABERO_DIR to the simulator checkout on the host}"
: "${OPENPI_DIR:?Set OPENPI_DIR to the OpenPI checkout on the simulator host}"
: "${CONDA_SH:?Set CONDA_SH to conda.sh for the simulator conda installation}"
: "${DATA_DIR:?Set DATA_DIR to the Isaaclab_Libero dataset root on the simulator host}"
: "${WARP_EXT:?Set WARP_EXT to the simulator omni.warp.core extension path}"

CONFIG=${CONFIG:?CONFIG is required}
CKPT=${CKPT:?CKPT is required}
VARIANT=${VARIANT:?VARIANT is required}
EXP=${EXP:?EXP is required}
STEP=${STEP:?STEP is required}
MODE=${MODE:?MODE is required}  # tactile or vision_abs7d

PORT=${PORT:-8194}
N=${N:-10}
REPLAN_STEPS=${REPLAN_STEPS:-10}
TASKS_STR=${TASKS_STR:-"0 1 2 3 4 5 6 7 8 9"}
SUITES_STR=${SUITES_STR:-"libero_object libero_spatial"}
RUN_ROOT=${RUN_ROOT:-${TABERO_DIR}/evaluation_results/openpi_softtacworld_eval_20260616}

OUT_ROOT=${RUN_ROOT}/${VARIANT}/${EXP}/${STEP}/replan_${REPLAN_STEPS}_n${N}
LOG_DIR=${OUT_ROOT}/logs
DEBUG_ROOT=${OUT_ROOT}/debug
VIDEO_LOG=${LOG_DIR}/video_encode.log
mkdir -p "${LOG_DIR}" "${DEBUG_ROOT}"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME=${OPENPI_DIR}/.cache/openpi
export HDF5_TRAJ_SOURCE_DIR=${DATA_DIR}/assembled_hdf5
export TABERO_SKIP_ISAAC_CLEANUP_ON_EXIT=1
export WARP_EXT
export TABERO_MAX_CONSECUTIVE_FAILURES=${TABERO_MAX_CONSECUTIVE_FAILURES:-10}
unset OPENPI_ADD_BYTES_KEY_ALIASES || true

cat > "${OUT_ROOT}/run.info" <<EOF_INFO
CONFIG=${CONFIG}
CKPT=${CKPT}
VARIANT=${VARIANT}
EXP=${EXP}
STEP=${STEP}
MODE=${MODE}
NUM_TOTAL_EXPERIMENTS=${N}
ACTION_HORIZON=50
REPLAN_STEPS=${REPLAN_STEPS}
MAX_INFERENCE_STEPS=30
NUM_SUCCESS_STEPS=8
DEBUG_MODE=6
PROMPT_ADVERBS=disabled
SUITES=${SUITES_STR}
TASKS=${TASKS_STR}
STARTED_AT=$(date --iso-8601=seconds)
EOF_INFO

ensure_norm_stats_asset_id() {
  local src_asset_id=""
  local dst_asset_id=""

  case "${CONFIG}" in
    pi0_lora_vision_tabero)
      src_asset_id="tabero_vision_pi0_h50"
      dst_asset_id="NathanWu7/tabero_vision"
      ;;
    pi05_lora_vision_tabero)
      src_asset_id="tabero_vision_pi05_h50"
      dst_asset_id="NathanWu7/tabero_vision"
      ;;
    pi0_lora_tacall_tabero)
      src_asset_id="tabero_tactile_pi0_h50"
      dst_asset_id="NathanWu7/tabero"
      ;;
    pi05_lora_tacall_tabero)
      src_asset_id="tabero_tactile_pi05_h50"
      dst_asset_id="NathanWu7/tabero"
      ;;
    *)
      return 0
      ;;
  esac

  local src="${CKPT}/assets/${src_asset_id}/norm_stats.json"
  local dst="${CKPT}/assets/${dst_asset_id}/norm_stats.json"
  if [[ -f "${dst}" ]]; then
    return 0
  fi
  if [[ ! -f "${src}" ]]; then
    echo "missing source norm stats: ${src}" >&2
    return 1
  fi
  mkdir -p "$(dirname "${dst}")"
  cp -a "${src}" "${dst}"
  echo "NORM_STATS_COMPAT=${src_asset_id}->${dst_asset_id}" >> "${OUT_ROOT}/run.info"
}

cleanup() {
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
  : > "${VIDEO_LOG}"
  while IFS= read -r exp_dir; do
    encode_one_exp "${exp_dir}"
  done < <(find "${task_debug}" -type d -name 'exp_*' | sort)
}

ensure_norm_stats_asset_id

cd "${OPENPI_DIR}"
export PYTHONPATH=${OPENPI_DIR}/src:${PYTHONPATH:-}
"${OPENPI_DIR}/.venv/bin/python" scripts/serve_policy.py \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${CONFIG}" \
  --policy.dir "${CKPT}" \
  > "${LOG_DIR}/server_${PORT}.log" 2>&1 &
echo $! > "${OUT_ROOT}/server.pid"

for i in $(seq 1 300); do
  if grep -Eq "server listening on|Listening on|Uvicorn running|Started server" "${LOG_DIR}/server_${PORT}.log"; then
    break
  fi
  if ! kill -0 "$(cat "${OUT_ROOT}/server.pid")" 2>/dev/null; then
    echo "server exited before becoming ready" >&2
    tail -200 "${LOG_DIR}/server_${PORT}.log" >&2 || true
    exit 1
  fi
  if [[ "${i}" == "300" ]]; then
    echo "server timeout" >&2
    tail -200 "${LOG_DIR}/server_${PORT}.log" >&2 || true
    exit 1
  fi
  sleep 1
done

source "${CONDA_SH}"
conda activate tabero
cd "${TABERO_DIR}"
export PYTHONPATH=${WARP_EXT}:${TABERO_DIR}:${PYTHONPATH:-}

ABS7D_ARGS=()
if [[ "${MODE}" == "vision_abs7d" ]]; then
  ABS7D_ARGS=(--abs7d)
fi

touch "${OUT_ROOT}/progress.tsv"
job_failed=0
for suite in ${SUITES_STR}; do
  for task_id in ${TASKS_STR}; do
    task_tag=${suite}_task${task_id}
    task_debug=${DEBUG_ROOT}/${suite}/${task_tag}
    task_log=${LOG_DIR}/${task_tag}.log
    if awk -F '\t' -v task="${task_tag}" -v total="${N}" \
      '$1 == task && $3 == total && $5 == "0" { found = 1 } END { exit found ? 0 : 1 }' \
      "${OUT_ROOT}/progress.tsv"; then
      echo "$(date --iso-8601=seconds) SKIP ${task_tag} already complete" | tee -a "${OUT_ROOT}/events.log"
      continue
    fi

    if [[ -d "${task_debug}" ]]; then
      backup="${task_debug}.incomplete_$(date +%Y%m%d_%H%M%S)"
      mv -- "${task_debug}" "${backup}"
      echo "$(date --iso-8601=seconds) BACKUP_INCOMPLETE ${task_tag} ${backup}" | tee -a "${OUT_ROOT}/events.log"
    fi
    mkdir -p -- "${task_debug}"

    echo "$(date --iso-8601=seconds) START ${task_tag}" | tee -a "${OUT_ROOT}/events.log"
    set +e
    python -u benchmarks/openpi/openpi_inference_client.py \
      --server_host 127.0.0.1 \
      --server_port "${PORT}" \
      --control_mode tactile \
      "${ABS7D_ARGS[@]}" \
      --task_suite "${suite}" \
      --task_id "${task_id}" \
      --tactile_output_type tactile_rgb \
      --num_total_experiments "${N}" \
      --num_success_steps 8 \
      --max_inference_steps 30 \
      --replan_steps "${REPLAN_STEPS}" \
      --num_steps_wait 5 \
      --hdf5_folder "${DATA_DIR}/assembled_hdf5" \
      --debug_mode 6 \
      --debug_path "${task_debug}" \
      --headless \
      > "${task_log}" 2>&1
    rc=$?
    set -e

    encode_task_videos "${task_debug}" || true

    python - <<PY >> "${OUT_ROOT}/progress.tsv"
from pathlib import Path
log = Path("${task_log}")
text = log.read_text(errors="ignore") if log.exists() else ""
succ = total = rate = None
for line in text.splitlines():
    line = line.strip()
    if line.startswith("Successful experiments:"):
        succ = int(line.split(":", 1)[1].strip())
    elif line.startswith("Total experiments:"):
        total = int(line.split(":", 1)[1].strip())
    elif line.startswith("Success rate:"):
        rate = float(line.split(":", 1)[1].strip().rstrip("%"))
print(f"${task_tag}\t{succ}\t{total}\t{rate}\t${rc}")
PY
    if [[ "${rc}" -ne 0 ]] || ! grep -Eq "^Total experiments:[[:space:]]*${N}[[:space:]]*$" "${task_log}"; then
      job_failed=1
    fi
    echo "$(date --iso-8601=seconds) END ${task_tag} rc=${rc}" | tee -a "${OUT_ROOT}/events.log"
  done
done

python - <<PY > "${OUT_ROOT}/summary.csv"
from pathlib import Path
root = Path("${OUT_ROOT}")
print("task,success,total,success_rate,rc")
total_s = 0
total_n = 0
latest = {}
for line in (root / "progress.tsv").read_text().splitlines():
    if not line.strip():
        continue
    task, succ, total, rate, rc = line.split("\t")
    latest[task] = (succ, total, rate, rc)
for task, (succ, total, rate, rc) in latest.items():
    s = 0 if succ == "None" else int(succ)
    n = 0 if total == "None" else int(total)
    r = 0.0 if rate == "None" else float(rate)
    total_s += s
    total_n += n
    print(f"{task},{s},{n},{r:.2f},{rc}")
print(f"overall,{total_s},{total_n},{(100 * total_s / total_n if total_n else 0):.2f},")
PY

cat "${OUT_ROOT}/summary.csv"
echo "FINISHED_AT=$(date --iso-8601=seconds)" >> "${OUT_ROOT}/run.info"
if [[ "${job_failed}" -ne 0 ]]; then
  echo "JOB_STATUS=failed_incomplete_tasks" >> "${OUT_ROOT}/run.info"
  echo "One or more tasks failed or did not complete ${N} experiments; leaving DONE unset for scheduler retry." >&2
  exit 1
fi
echo "JOB_STATUS=done" >> "${OUT_ROOT}/run.info"
touch "${OUT_ROOT}/DONE"
echo "OUT_ROOT=${OUT_ROOT}"
