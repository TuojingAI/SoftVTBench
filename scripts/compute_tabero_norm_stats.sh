#!/usr/bin/env bash
set -Eeuo pipefail

MODEL=${1:?Usage: scripts/compute_softtacworld_norm_stats.sh <pi0|pi05> <vision|tactile>}
MODE=${2:?Usage: scripts/compute_softtacworld_norm_stats.sh <pi0|pi05> <vision|tactile>}

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}

case "${MODEL}_${MODE}" in
  pi0_vision)
    CONFIG=pi0_lora_vision_tabero
    REPO_ID=${REPO_VISION:?Set REPO_VISION}
    ROOT=${ROOT_VISION:?Set ROOT_VISION}
    ASSETS_DIR=${ASSET_PI0_VISION:?Set ASSET_PI0_VISION}
    ASSET_ID=${ASSET_ID_PI0_VISION:?Set ASSET_ID_PI0_VISION}
    ;;
  pi05_vision)
    CONFIG=pi05_lora_vision_tabero
    REPO_ID=${REPO_VISION:?Set REPO_VISION}
    ROOT=${ROOT_VISION:?Set ROOT_VISION}
    ASSETS_DIR=${ASSET_PI05_VISION:?Set ASSET_PI05_VISION}
    ASSET_ID=${ASSET_ID_PI05_VISION:?Set ASSET_ID_PI05_VISION}
    ;;
  pi0_tactile)
    CONFIG=pi0_lora_tacall_tabero
    REPO_ID=${REPO_TACTILE:?Set REPO_TACTILE}
    ROOT=${ROOT_TACTILE:?Set ROOT_TACTILE}
    ASSETS_DIR=${ASSET_PI0_TACTILE:?Set ASSET_PI0_TACTILE}
    ASSET_ID=${ASSET_ID_PI0_TACTILE:?Set ASSET_ID_PI0_TACTILE}
    ;;
  pi05_tactile)
    CONFIG=pi05_lora_tacall_tabero
    REPO_ID=${REPO_TACTILE:?Set REPO_TACTILE}
    ROOT=${ROOT_TACTILE:?Set ROOT_TACTILE}
    ASSETS_DIR=${ASSET_PI05_TACTILE:?Set ASSET_PI05_TACTILE}
    ASSET_ID=${ASSET_ID_PI05_TACTILE:?Set ASSET_ID_PI05_TACTILE}
    ;;
  *)
    echo "Unsupported pair: ${MODEL} ${MODE}" >&2
    exit 2
    ;;
esac

"${PYTHON}" "${REPO_ROOT}/scripts/compute_norm_stats.py" \
  --config-name "${CONFIG}" \
  --repo-id "${REPO_ID}" \
  --root "${ROOT}" \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --low-dim-only
