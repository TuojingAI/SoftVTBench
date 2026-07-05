#!/usr/bin/env bash
# Public SoftVTBench training entry point.
#
# Install OpenPI with the upstream instructions, then provide the three
# deployment-specific inputs below. All other paths default inside this repo.
#
#   OPENPI_PYTHON=/path/to/openpi-venv/bin/python \
#   RAW_ROOT=/path/to/SoftVTBench_data/object-soft \
#   SUITE=object-soft MODALITY=tactile \
#   bash openpi/scripts/train_softvtbench.sh
#
# Supported combinations:
#   SUITE=object-soft|spatial-soft|object-rigid|spatial-rigid
#   MODALITY=vision|tactile
#   PHASE=convert|stats|train|all (default: all)
#   MODEL=pi05 (the only public v1 model)
#
# BASE_PARAMS defaults to OpenPI's normal cache location. Set it explicitly if
# your upstream OpenPI installation keeps pi05_base elsewhere. It is only
# required by PHASE=train and PHASE=all.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/softvtbench_paths.sh"

SUITE="${SUITE:?SUITE is required: object-soft, spatial-soft, object-rigid, or spatial-rigid}"
MODEL="${MODEL:-pi05}"
MODALITY="${MODALITY:-tactile}"
PHASE="${PHASE:-all}"
# Paper Sec. 4.1: the vision-only baseline uses a binary open-close gripper; the
# visuo-tactile policy uses a continuous gripper-width action.
if [[ -z "${GRIPPER_MODE:-}" ]]; then
  if [[ "${MODALITY}" == "vision" ]]; then GRIPPER_MODE=binary; else GRIPPER_MODE=state; fi
fi
RAW_ROOT="${RAW_ROOT:?RAW_ROOT is required; point it at the downloaded SoftVTBench suite folder}"

if [[ "${MODEL}" != "pi05" ]]; then
  echo "SoftVTBench v1 supports MODEL=pi05 only; got: ${MODEL}" >&2
  exit 2
fi
case "${MODALITY}" in
  vision|tactile) ;;
  *) echo "MODALITY must be vision or tactile, got: ${MODALITY}" >&2; exit 2 ;;
esac

if [[ "${MODALITY}" == "vision" ]]; then
  TRAIN_CONFIG="${TRAIN_CONFIG:-${MODEL}_lora_vision_softvtbench}"
else
  TRAIN_CONFIG="${TRAIN_CONFIG:-${MODEL}_lora_tacall_softvtbench}"
fi
BASE_PARAMS="${BASE_PARAMS:-${OPENPI_CACHE_ROOT}/openpi/openpi-assets/checkpoints/${MODEL}_base/params}"

# The implementation launchers retain their legacy flags for direct internal
# use. The public interface exposes one explicit phase at a time.
case "${PHASE}" in
  all)
    export SOFTVTBENCH_STOP_AFTER_CONVERT=0
    ;;
  convert)
    export SKIP_CONVERT=0 SKIP_NORM=1 PREP_ONLY=1 SOFTVTBENCH_STOP_AFTER_CONVERT=1
    ;;
  stats)
    export SKIP_CONVERT=1 SKIP_NORM=0 PREP_ONLY=1 SOFTVTBENCH_STOP_AFTER_CONVERT=0
    ;;
  train)
    export SKIP_CONVERT=1 SKIP_NORM=1 PREP_ONLY=0 SOFTVTBENCH_STOP_AFTER_CONVERT=0
    ;;
  *)
    echo "PHASE must be convert, stats, train, or all; got: ${PHASE}" >&2
    exit 2
    ;;
esac
export MODEL MODALITY PHASE TRAIN_CONFIG BASE_PARAMS RAW_ROOT GRIPPER_MODE

case "${SUITE}" in
  object-soft)
    if [[ "${MODALITY}" == "vision" ]]; then
      export ASSET_PI05_VISION="${ASSET_PI05_VISION:-${OPENPI_ASSETS_ROOT}/object_soft_10assets_10tasks_50each_initjitter012_gripperjitter5_nl_20260625/${MODEL}_vision}"
      export ASSET_ID_PI05_VISION="${ASSET_ID_PI05_VISION:-object_soft_vision_${MODEL}_h50_targetnext_20260625}"
      export LOGDIR="${LOGDIR:-${OPENPI_LOG_ROOT}/object_soft_10assets_${MODEL}_vision_targetnext}"
      export EXP="${EXP:-object_soft_10assets_${MODEL}_vision_h50_targetnext_bs256_8gpu}"
      exec bash "${SCRIPT_DIR}/softvtbench_train_object_soft_pi05_vision_8gpu_20260625.sh"
    fi
    export DATA_REPO="${DATA_REPO:-local/object_soft_10assets_${MODEL}_tactile_targetnext_7d}"
    export ASSETS_DIR="${ASSETS_DIR:-${OPENPI_ASSETS_ROOT}/object_soft_10assets_10tasks_50each_initjitter012_gripperjitter5_nl_20260625/${MODEL}_tactile}"
    export ASSET_ID="${ASSET_ID:-object_soft_10assets_${MODEL}_tactile_h50_targetnext_7d}"
    export LOGDIR="${LOGDIR:-${OPENPI_LOG_ROOT}/object_soft_10assets_${MODEL}_tactile_targetnext}"
    export EXP_NAME="${EXP_NAME:-object_soft_10assets_${MODEL}_tactile_h50_targetnext_7d_bs256_8gpu}"
    exec bash "${SCRIPT_DIR}/train_object_soft_pi05_tactile_targetnext_7d_a800_20260626.sh"
    ;;
  spatial-soft)
    if [[ "${MODALITY}" == "vision" ]]; then
      export ASSETS_DIR="${ASSETS_DIR:-${OPENPI_ASSETS_ROOT}/spatial_soft_pastry005_10tasks_50each_gripperjitter_20260624/${MODEL}_vision}"
      export ASSET_ID="${ASSET_ID:-spatial_soft_pastry005_${MODEL}_vision_h50_targetnext_20260627}"
      export LOGDIR="${LOGDIR:-${OPENPI_LOG_ROOT}/spatial_soft_pastry005_10tasks_50each_gripperjitter_20260624_${MODEL}_vision_targetnext}"
      export EXP="${EXP:-spatial_soft_pastry005_${MODEL}_vision_h50_targetnext_7d_bs256_8gpu}"
      exec bash "${SCRIPT_DIR}/softvtbench_train_spatial_soft_pi05_vision_8gpu_20260627.sh"
    fi
    export DATA_REPO="${DATA_REPO:-local/spatial_soft_pastry005_${MODEL}_tactile_targetnext_7d}"
    export ASSETS_DIR="${ASSETS_DIR:-${OPENPI_ASSETS_ROOT}/spatial_soft_pastry005_10tasks_50each_gripperjitter_20260624/${MODEL}_tactile}"
    export ASSET_ID="${ASSET_ID:-spatial_soft_pastry005_${MODEL}_tactile_h50_targetnext_7d}"
    export LOGDIR="${LOGDIR:-${OPENPI_LOG_ROOT}/spatial_soft_pastry005_${MODEL}_tactile_targetnext}"
    export EXP_NAME="${EXP_NAME:-spatial_soft_pastry005_${MODEL}_tactile_h50_targetnext_7d_bs256_8gpu}"
    exec bash "${SCRIPT_DIR}/train_spatial_soft_pastry005_pi05_tactile_7d_parquet_norm_a800_20260626.sh"
    ;;
  object-rigid|spatial-rigid)
    export SUITE="libero_${SUITE%-rigid}"
    export DATA_REPO="${DATA_REPO:-local/${SUITE#libero_}_rigid_${MODEL}_${MODALITY}_targetnext_7d}"
    export ASSETS_DIR="${ASSETS_DIR:-${OPENPI_ASSETS_ROOT}/${SUITE#libero_}_rigid/${MODEL}_${MODALITY}}"
    export ASSET_ID="${ASSET_ID:-${SUITE#libero_}_rigid_${MODEL}_${MODALITY}_targetnext_7d}"
    export EXP_NAME="${EXP_NAME:-${SUITE#libero_}_rigid_${MODEL}_${MODALITY}_targetnext_7d_bs256_8gpu}"
    exec bash "${SCRIPT_DIR}/train_rigid_pi05.sh"
    ;;
  *)
    echo "SUITE must be object-soft, spatial-soft, object-rigid, or spatial-rigid; got: ${SUITE}" >&2
    exit 2
    ;;
esac
