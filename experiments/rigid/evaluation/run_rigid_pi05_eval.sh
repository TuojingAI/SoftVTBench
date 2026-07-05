#!/usr/bin/env bash
# Closed-loop π0.5 evaluation for the downloaded rigid LIBERO baselines.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFTVTBENCH_ROOT="${SOFTVTBENCH_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${SOFTVTBENCH_ROOT}/openpi/scripts/softvtbench_paths.sh"

SOFTVTBENCH_DIR="${SOFTVTBENCH_DIR:-${SOFTVTBENCH_ROOT}/SoftVTBench}"
OPENPI_DIR="${OPENPI_DIR:-${OPENPI_CODE_DIR}}"
SOFTVTBENCH_PYTHON="${SOFTVTBENCH_PYTHON:-${SOFTVTBENCH_PY:-}}"
: "${SOFTVTBENCH_PYTHON:?set SOFTVTBENCH_PYTHON to the Isaac Sim/SoftVTBench interpreter}"
OPENPI_SERVER_PYTHON="${OPENPI_SERVER_PYTHON:-${OPENPI_PYTHON:-}}"
EXTERNAL_SERVER="${EXTERNAL_SERVER:-0}"

PUBLIC_SUITE="${SUITE:?SUITE is required: object-rigid or spatial-rigid}"
case "${PUBLIC_SUITE}" in
  object-rigid) TASK_SUITE=libero_object ;;
  spatial-rigid) TASK_SUITE=libero_spatial ;;
  *) echo "SUITE must be object-rigid or spatial-rigid; got: ${PUBLIC_SUITE}" >&2; exit 2 ;;
esac
MODALITY="${MODALITY:-tactile}"
case "${MODALITY}" in
  vision) CONFIG="${CONFIG:-pi05_lora_vision_softvtbench}" ;;
  tactile) CONFIG="${CONFIG:-pi05_lora_tacall_softvtbench}" ;;
  *) echo "MODALITY must be vision or tactile; got: ${MODALITY}" >&2; exit 2 ;;
esac

COLLECTION_ROOT="${COLLECTION_ROOT:?point COLLECTION_ROOT at the downloaded ${PUBLIC_SUITE} folder}"
HDF5_DIR="${HDF5_DIR:-${COLLECTION_ROOT}/replayed_demos}"
CONFIG_DIR="${CONFIG_DIR:-${SOFTVTBENCH_DIR}/benchmarks/datasets/libero/config}"
CKPT="${CKPT:-}"
PORT="${PORT:-8194}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
N="${N:-50}"
[[ "${N}" =~ ^[1-9][0-9]*$ ]] || { echo "N must be a positive integer; got: ${N}" >&2; exit 2; }
TASKS_STR="${TASKS_STR:-0 1 2 3 4 5 6 7 8 9}"
CONTROL_MODE="${CONTROL_MODE:-binary}"
OUT_ROOT="${OUT_ROOT:-${SOFTVTBENCH_RESULTS_ROOT}/${PUBLIC_SUITE}_${MODALITY}_pi05_eval/$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

[[ -d "${HDF5_DIR}" ]] || { echo "missing HDF5 directory: ${HDF5_DIR}" >&2; exit 2; }
[[ -f "${CONFIG_DIR}/${TASK_SUITE}.json" ]] || { echo "missing task config: ${CONFIG_DIR}/${TASK_SUITE}.json" >&2; exit 2; }
"${SOFTVTBENCH_PYTHON}" -c 'import h5py, isaacsim, numpy, scipy, tyro' >/dev/null

POLICY_CKPT_DIR=""
prepare_checkpoint_view() {
  [[ -n "${CKPT}" ]] || { echo "CKPT is required unless EXTERNAL_SERVER=1" >&2; return 2; }
  [[ -d "${CKPT}" ]] || { echo "checkpoint directory does not exist: ${CKPT}" >&2; return 2; }
  local dst_asset_id src=""
  if [[ "${MODALITY}" == "vision" ]]; then
    dst_asset_id="local/softvtbench_vision"
  else
    dst_asset_id="local/softvtbench_tactile"
  fi
  POLICY_CKPT_DIR="${OUT_ROOT}/checkpoint_view"
  softvtbench_make_checkpoint_view "${CKPT}" "${POLICY_CKPT_DIR}"
  src="$(softvtbench_alias_norm_stats \
    "${CKPT}" "${POLICY_CKPT_DIR}" "${dst_asset_id}" \
    "${PUBLIC_SUITE%-rigid}_rigid_pi05_${MODALITY}_targetnext_7d" \
    "${TASK_SUITE#libero_}_rigid_pi05_${MODALITY}_targetnext_7d" \
    "${dst_asset_id}" \
    "NathanWu7/softvtbench_vision" \
    "NathanWu7/softvtbench")"
}

cleanup() {
  if [[ -f "${OUT_ROOT}/server.pid" ]]; then
    local pid
    pid="$(cat "${OUT_ROOT}/server.pid")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

if [[ "${EXTERNAL_SERVER}" != 1 ]]; then
  softvtbench_require_openpi_python "${OPENPI_SERVER_PYTHON}"
  prepare_checkpoint_view
  cd "${OPENPI_DIR}"
  export PYTHONPATH="${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src:${PYTHONPATH:-}"
  "${OPENPI_SERVER_PYTHON}" scripts/serve_policy.py \
    --port "${PORT}" policy:checkpoint \
    --policy.config "${CONFIG}" \
    --policy.dir "${POLICY_CKPT_DIR}" \
    > "${LOG_DIR}/server_${PORT}.log" 2>&1 &
  echo $! > "${OUT_ROOT}/server.pid"
  for i in $(seq 1 300); do
    if grep -Eq 'server listening on|Listening on|Uvicorn running|Started server' "${LOG_DIR}/server_${PORT}.log"; then break; fi
    if ! kill -0 "$(cat "${OUT_ROOT}/server.pid")" 2>/dev/null; then
      echo "policy server exited before ready" >&2
      tail -200 "${LOG_DIR}/server_${PORT}.log" >&2 || true
      exit 1
    fi
    [[ "${i}" == 300 ]] && { echo "policy server timeout" >&2; exit 1; }
    sleep 1
  done
fi

cat > "${OUT_ROOT}/run.info" <<EOF
SUITE=${PUBLIC_SUITE}
TASK_SUITE=${TASK_SUITE}
MODALITY=${MODALITY}
CONFIG=${CONFIG}
CKPT=${CKPT}
POLICY_CKPT_DIR=${POLICY_CKPT_DIR}
COLLECTION_ROOT=${COLLECTION_ROOT}
HDF5_DIR=${HDF5_DIR}
CONTROL_MODE=${CONTROL_MODE}
N=${N}
TASKS=${TASKS_STR}
EXTERNAL_SERVER=${EXTERNAL_SERVER}
SERVER_HOST=${SERVER_HOST}
OPENPI_SERVER_PYTHON=${OPENPI_SERVER_PYTHON}
SOFTVTBENCH_PYTHON=${SOFTVTBENCH_PYTHON}
STARTED_AT=$(date --iso-8601=seconds)
EOF

cd "${SOFTVTBENCH_DIR}"
export PYTHONPATH="${SOFTVTBENCH_DIR}/source/tac_manip:${OPENPI_DIR}/packages/openpi-client/src:${SOFTVTBENCH_DIR}:${PYTHONPATH:-}"
read -r -a TASK_IDS <<< "${TASKS_STR}"
"${SOFTVTBENCH_PYTHON}" scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode "${CONTROL_MODE}" \
  --server_host "${SERVER_HOST}" \
  --server_port "${PORT}" \
  --num_total_experiments "${N}" \
  --task_suites "${TASK_SUITE}" \
  --task_ids "${TASK_IDS[@]}" \
  --hdf5_folder "${HDF5_DIR}" \
  --config_path "${CONFIG_DIR}" \
  --output_dir "${OUT_ROOT}" \
  --require_hdf5 \
  --headless \
  2>&1 | tee "${LOG_DIR}/evaluation.log"

echo "FINISHED_AT=$(date --iso-8601=seconds)" >> "${OUT_ROOT}/run.info"
echo "JOB_STATUS=done" >> "${OUT_ROOT}/run.info"
touch "${OUT_ROOT}/DONE"
echo "OUT_ROOT=${OUT_ROOT}"
