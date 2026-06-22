#!/usr/bin/env bash
set -Eeuo pipefail

MODE=${1:?Usage: scripts/prepare_tabero_dataset.sh <vision|tactile>}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
shopt -s nullglob

: "${RAW_ROOT:?Set RAW_ROOT or source configs/training/tabero_env.local}"
: "${STAGE_ROOT:?Set STAGE_ROOT or source configs/training/tabero_env.local}"
: "${TASK_SUBSET:?Set TASK_SUBSET or source configs/training/tabero_env.local}"
: "${HF_LEROBOT_HOME:?Set HF_LEROBOT_HOME or source configs/training/tabero_env.local}"

mkdir -p "${STAGE_ROOT}/replayed_demos" "${STAGE_ROOT}/video_datasets"

for suite in libero_object libero_spatial; do
  h5_files=("${RAW_ROOT}/${suite}/replayed_demos/"*.hdf5)
  task_dirs=("${RAW_ROOT}/${suite}/video_datasets/${suite}"_task*)
  if [[ "${#h5_files[@]}" -eq 0 ]]; then
    echo "No HDF5 files found for ${suite}: ${RAW_ROOT}/${suite}/replayed_demos/*.hdf5" >&2
    exit 1
  fi
  if [[ "${#task_dirs[@]}" -eq 0 ]]; then
    echo "No video task directories found for ${suite}: ${RAW_ROOT}/${suite}/video_datasets/${suite}_task*" >&2
    exit 1
  fi
  for h5 in "${RAW_ROOT}/${suite}/replayed_demos/"*.hdf5; do
    ln -sfn "${h5}" "${STAGE_ROOT}/replayed_demos/$(basename "${h5}")"
  done
  for task_dir in "${RAW_ROOT}/${suite}/video_datasets/${suite}"_task*; do
    ln -sfn "${task_dir}" "${STAGE_ROOT}/video_datasets/$(basename "${task_dir}")"
  done
done

case "${MODE}" in
  vision)
    : "${REPO_VISION:?Set REPO_VISION}"
    "${PYTHON}" "${REPO_ROOT}/examples/tabero/convert_tabero_vision_data_to_lerobot.py" \
      --data-root "${STAGE_ROOT}" \
      --repo-name "${REPO_VISION}" \
      --output-dir "${HF_LEROBOT_HOME}" \
      --task-suites libero_object libero_spatial \
      --task-subset-path "${TASK_SUBSET}"
    ;;
  tactile)
    : "${REPO_TACTILE:?Set REPO_TACTILE}"
    "${PYTHON}" "${REPO_ROOT}/examples/tabero/convert_tabero_tactile_data_to_lerobot.py" \
      --data-root "${STAGE_ROOT}" \
      --repo-name "${REPO_TACTILE}" \
      --output-dir "${HF_LEROBOT_HOME}" \
      --task-suites libero_object libero_spatial \
      --task-subset-path "${TASK_SUBSET}" \
      --tactile-output-type tactile_rgb \
      --force-history-len 8 \
      --marker-history-len 8
    ;;
  *)
    echo "Unsupported mode: ${MODE}. Expected vision or tactile." >&2
    exit 2
    ;;
esac
