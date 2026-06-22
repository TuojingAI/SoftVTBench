#!/usr/bin/env bash
set -Eeuo pipefail

MODEL=${1:?Usage: scripts/train_tabero_baseline.sh <pi0|pi05> <vision|tactile>}
MODE=${2:?Usage: scripts/train_tabero_baseline.sh <pi0|pi05> <vision|tactile>}

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
RUN_ID=${RUN_ID:-tabero_object_spatial_success_only_20260614}
FS_DDP=${FSDP_DEVICES:-4}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-10}
EXP_SUFFIX=${EXP_SUFFIX:-$(date +%Y%m%d_%H%M%S)}

case "${MODEL}_${MODE}" in
  pi0_vision)
    CONFIG=pi0_lora_vision_tabero
    REPO_ID=${REPO_VISION:?Set REPO_VISION}
    ROOT=${ROOT_VISION:?Set ROOT_VISION}
    ASSETS_DIR=${ASSET_PI0_VISION:?Set ASSET_PI0_VISION}
    ASSET_ID=${ASSET_ID_PI0_VISION:?Set ASSET_ID_PI0_VISION}
    BASE=${PI0_BASE:?Set PI0_BASE}
    EXP=${EXP:-${RUN_ID}_pi0_vision_h50_bs${BATCH_SIZE}_${FS_DDP}gpu_${EXP_SUFFIX}}
    ;;
  pi05_vision)
    CONFIG=pi05_lora_vision_tabero
    REPO_ID=${REPO_VISION:?Set REPO_VISION}
    ROOT=${ROOT_VISION:?Set ROOT_VISION}
    ASSETS_DIR=${ASSET_PI05_VISION:?Set ASSET_PI05_VISION}
    ASSET_ID=${ASSET_ID_PI05_VISION:?Set ASSET_ID_PI05_VISION}
    BASE=${PI05_BASE:?Set PI05_BASE}
    EXP=${EXP:-${RUN_ID}_pi05_vision_h50_bs${BATCH_SIZE}_${FS_DDP}gpu_${EXP_SUFFIX}}
    ;;
  pi0_tactile)
    CONFIG=pi0_lora_tacall_tabero
    REPO_ID=${REPO_TACTILE:?Set REPO_TACTILE}
    ROOT=${ROOT_TACTILE:?Set ROOT_TACTILE}
    ASSETS_DIR=${ASSET_PI0_TACTILE:?Set ASSET_PI0_TACTILE}
    ASSET_ID=${ASSET_ID_PI0_TACTILE:?Set ASSET_ID_PI0_TACTILE}
    BASE=${PI0_BASE:?Set PI0_BASE}
    EXP=${EXP:-${RUN_ID}_pi0_tactile_h50_bs${BATCH_SIZE}_${FS_DDP}gpu_${EXP_SUFFIX}}
    ;;
  pi05_tactile)
    CONFIG=pi05_lora_tacall_tabero
    REPO_ID=${REPO_TACTILE:?Set REPO_TACTILE}
    ROOT=${ROOT_TACTILE:?Set ROOT_TACTILE}
    ASSETS_DIR=${ASSET_PI05_TACTILE:?Set ASSET_PI05_TACTILE}
    ASSET_ID=${ASSET_ID_PI05_TACTILE:?Set ASSET_ID_PI05_TACTILE}
    BASE=${PI05_BASE:?Set PI05_BASE}
    EXP=${EXP:-${RUN_ID}_pi05_tactile_h50_bs${BATCH_SIZE}_${FS_DDP}gpu_${EXP_SUFFIX}}
    ;;
  *)
    echo "Unsupported pair: ${MODEL} ${MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${REPO_ROOT}/logs/tabero_training_manual"

"${PYTHON}" "${REPO_ROOT}/scripts/train.py" "${CONFIG}" \
  --exp-name "${EXP}" \
  --data.repo-id "${REPO_ID}" \
  --data.root "${ROOT}" \
  --data.assets.assets-dir "${ASSETS_DIR}" \
  --data.assets.asset-id "${ASSET_ID}" \
  --weight-loader.params-path "${BASE}" \
  --fsdp-devices "${FS_DDP}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --overwrite \
  2>&1 | tee "${REPO_ROOT}/logs/tabero_training_manual/${EXP}.log"

