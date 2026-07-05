#!/usr/bin/env bash
# Public SoftVTBench closed-loop evaluation entry point.
#
# Required:
#   SUITE=object-soft|spatial-soft|object-rigid|spatial-rigid
#   MODALITY=vision|tactile
#   COLLECTION_ROOT=/path/to/downloaded/suite
#   CKPT=/path/to/pi05/checkpoint/step  # unnecessary with EXTERNAL_SERVER=1
#   SOFTVTBENCH_PYTHON=/path/to/isaac-softvtbench/bin/python
#   OPENPI_PYTHON=/path/to/openpi/bin/python
#
# Soft suites additionally require eval-assets and compression-sweep thresholds:
#   SOFTVT_EVAL_USD_DIR=/path/to/SoftVTBench_data/eval-assets/USD
#   SAFETY_THRESHOLDS=/path/to/safety_thresholds.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/softvtbench_paths.sh"

SUITE="${SUITE:?SUITE is required: object-soft, spatial-soft, object-rigid, or spatial-rigid}"
MODALITY="${MODALITY:-tactile}"
MODEL="${MODEL:-pi05}"
COLLECTION_ROOT="${COLLECTION_ROOT:?COLLECTION_ROOT must point at the downloaded suite folder}"
N="${N:-50}"
if [[ ! "${N}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N must be a positive integer; got: ${N}" >&2
  exit 2
fi

if [[ "${MODEL}" != "pi05" ]]; then
  echo "SoftVTBench v1 supports MODEL=pi05 only; got: ${MODEL}" >&2
  exit 2
fi
case "${MODALITY}" in
  vision)
    export CONFIG="${CONFIG:-pi05_lora_vision_softvtbench}"
    export MODE="${MODE:-vision_abs7d}"
    ;;
  tactile)
    export CONFIG="${CONFIG:-pi05_lora_tacall_softvtbench}"
    export MODE="${MODE:-tactile}"
    ;;
  *) echo "MODALITY must be vision or tactile; got: ${MODALITY}" >&2; exit 2 ;;
esac
export MODEL MODALITY COLLECTION_ROOT N

case "${SUITE}" in
  object-soft)
    exec bash "${SOFTVTBENCH_ROOT}/experiments/object_soft/evaluation/run_object_soft_10tasks_pi05_eval.sh"
    ;;
  spatial-soft)
    exec bash "${SOFTVTBENCH_ROOT}/experiments/spatial_soft/evaluation/run_spatial_pastry005_pi05_eval.sh"
    ;;
  object-rigid|spatial-rigid)
    exec bash "${SOFTVTBENCH_ROOT}/experiments/rigid/evaluation/run_rigid_pi05_eval.sh"
    ;;
  *)
    echo "SUITE must be object-soft, spatial-soft, object-rigid, or spatial-rigid; got: ${SUITE}" >&2
    exit 2
    ;;
esac
