#!/usr/bin/env bash
#
# 用法（必须 source 才能把环境变量设置到当前 shell）：
#   source scripts/tools/set_replay_env.sh <profile> [output_root|assembled_dir]
#
# profile 取值（示例）：
#   - inference | infer
#       仅用于推理/评测：使用仓库默认的 libero 软链接目录作为 assembled_hdf5，
#       不设置 replay 输出目录，并清除 REPLAYED_* / OUTPUT_REPLAYED_*，
#       避免沿用之前其它 profile 的路径。
#       例：source scripts/tools/set_replay_env.sh inference
#   - libero
#       为原版 LIBERO 设置一整套目录：
#       * HDF5_TRAJ_SOURCE_DIR      -> benchmarks/datasets/libero/assembled_hdf5
#       * OUTPUT_REPLAYED_DEMOS_DIR -> benchmarks/datasets/libero/replayed_demos/
#       * OUTPUT_REPLAYED_VIDEOS_DIR-> benchmarks/datasets/libero/video_datasets/
#       * REPLAYED_DEMOS_DIR        -> OUTPUT_REPLAYED_DEMOS_DIR
#       注意：默认目录通常是软链接到现成数据。若你想重新回放采集自己的数据，
#       请改用单独目录，避免覆盖默认数据。
#   - softvtbench_gentle
#   - softvtbench_force_gentle
#   - softvtbench_firm
#   - softvtbench_force_firm
#   - softvtbench_binary
#   - softvtbench_binary_gentle
#   - softvtbench_binary_firm
#
# output_root / assembled_dir（可选，视 profile 而定）：
#   - replay 类 profile：若提供，则覆盖默认输出根目录（用于把 replay 输出写到独立磁盘/目录）
#     例如：source scripts/tools/set_replay_env.sh softvtbench_binary /data/softvtbench_pi0_binary_inputs
#
# 会设置并导出：
#   - HDF5_TRAJ_SOURCE_DIR      : 原始 libero assembled_hdf5 数据源目录
#   - OUTPUT_REPLAYED_DEMOS_DIR : 回放再采集输出 HDF5 目录（replayed_demos）
#   - OUTPUT_REPLAYED_VIDEOS_DIR: 回放视频输出目录（video_datasets）
#   - REPLAYED_DEMOS_DIR        : 读取回放后数据的目录（默认等于 OUTPUT_REPLAYED_DEMOS_DIR）
#   - USE_SOFTVTBENCH_TASKS          : 启用 softvtbench_tasks.json 子集（replay_demos_with_camera.py 自动识别）
#

# 只保存/恢复本脚本会改动的选项，避免 RETURN 时 eval 一长串 `set +o ...`
#（否则 verbose/xtrace 下容易刷屏，也可能让终端记录里充满 set +o 行）
_set_replay_env_restore_opts() {
  if [[ "${_SET_REPLAY_ENV_HAD_ERRExit:-0}" -eq 1 ]]; then set -e; else set +e; fi
  if [[ "${_SET_REPLAY_ENV_HAD_NOUNSET:-0}" -eq 1 ]]; then set -u; else set +u; fi
  if [[ "${_SET_REPLAY_ENV_HAD_PIPEFAIL:-0}" -eq 1 ]]; then set -o pipefail; else set +o pipefail; fi
}
[[ -o errexit ]] && _SET_REPLAY_ENV_HAD_ERRExit=1 || _SET_REPLAY_ENV_HAD_ERRExit=0
[[ -o nounset ]] && _SET_REPLAY_ENV_HAD_NOUNSET=1 || _SET_REPLAY_ENV_HAD_NOUNSET=0
[[ -o pipefail ]] && _SET_REPLAY_ENV_HAD_PIPEFAIL=1 || _SET_REPLAY_ENV_HAD_PIPEFAIL=0
trap '_set_replay_env_restore_opts' RETURN
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[set_replay_env] 请使用 source 执行，以便环境变量写入当前 shell："
  echo "  source scripts/tools/set_replay_env.sh <profile>"
  exit 2
fi

PROFILE="${1:-}"
if [[ -z "${PROFILE}" ]]; then
  echo "[set_replay_env] 缺少 profile 参数。示例：inference | libero | softvtbench_gentle | softvtbench_force_firm"
  return 2
fi

# 自动推导仓库根目录（scripts/tools -> repo_root）
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
_DEFAULT_LIBERO_ROOT="${_REPO_ROOT}/benchmarks/datasets/libero"
_DEFAULT_HDF5_SOURCE="${_DEFAULT_LIBERO_ROOT}/assembled_hdf5"
_DEFAULT_ASSETS_DATA="${_REPO_ROOT}/source/tac_manip/tac_manip/assets/data"

_print_default_data_hints() {
  echo "  期望默认数据软链接："
  echo "    benchmarks/datasets/libero -> Isaaclab_Libero"
  echo "    source/tac_manip/tac_manip/assets/data -> Tactile_manipulation_dataset"
}

# inference / infer：只配置标准 assembled_hdf5，不设 replay 输出路径（适合直接跑推理/评测）
if [[ "${PROFILE}" == "inference" || "${PROFILE}" == "infer" ]]; then
  export HDF5_TRAJ_SOURCE_DIR="${_DEFAULT_HDF5_SOURCE}"
  unset OUTPUT_REPLAYED_DEMOS_DIR OUTPUT_REPLAYED_VIDEOS_DIR REPLAYED_DEMOS_DIR 2>/dev/null || true
  export USE_SOFTVTBENCH_TASKS="${USE_SOFTVTBENCH_TASKS:-0}"
  echo "[set_replay_env] 已设置（inference：仅 assembled，已清除 replay 相关环境变量）："
  echo "  PROFILE=${PROFILE}"
  echo "  REPO_ROOT=${_REPO_ROOT}"
  echo "  HDF5_TRAJ_SOURCE_DIR=${HDF5_TRAJ_SOURCE_DIR}"
  echo "  OUTPUT_REPLAYED_DEMOS_DIR=<unset>"
  echo "  OUTPUT_REPLAYED_VIDEOS_DIR=<unset>"
  echo "  REPLAYED_DEMOS_DIR=<unset>"
  echo "  USE_SOFTVTBENCH_TASKS=${USE_SOFTVTBENCH_TASKS}"
  if [[ ! -d "${HDF5_TRAJ_SOURCE_DIR}" ]]; then
    echo "[set_replay_env] ⚠️ 未找到默认 assembled_hdf5: ${HDF5_TRAJ_SOURCE_DIR}"
    _print_default_data_hints
  fi
  if [[ ! -d "${_DEFAULT_ASSETS_DATA}" ]]; then
    echo "[set_replay_env] ⚠️ 未找到默认 assets/data: ${_DEFAULT_ASSETS_DATA}"
    _print_default_data_hints
  fi
  return 0
fi

# 数据源（replay 类 profile 共用）：允许用户在 source 之前自行 export 覆盖
if [[ -z "${HDF5_TRAJ_SOURCE_DIR:-}" ]]; then
  export HDF5_TRAJ_SOURCE_DIR="${_DEFAULT_HDF5_SOURCE}"
fi

DATASET_KIND=""
FORCE_MODE=""
CUSTOM_ROOT="${2:-}"
PROFILE_KIND=""

case "${PROFILE}" in
  libero)
    DATASET_KIND="libero"; FORCE_MODE=""; PROFILE_KIND="libero" ;;
  softvtbench_gentle) DATASET_KIND="softvtbench"; FORCE_MODE="gentle_force" ;;
  softvtbench_force_gentle) DATASET_KIND="softvtbench_force"; FORCE_MODE="gentle_force" ;;
  softvtbench_firm) DATASET_KIND="softvtbench"; FORCE_MODE="firm_force" ;;
  softvtbench_force_firm) DATASET_KIND="softvtbench_force"; FORCE_MODE="firm_force" ;;
  softvtbench_binary) DATASET_KIND="softvtbench_binary"; FORCE_MODE="" ;;
  softvtbench_binary_gentle) DATASET_KIND="softvtbench_binary"; FORCE_MODE="gentle_force" ;;
  softvtbench_binary_firm) DATASET_KIND="softvtbench_binary"; FORCE_MODE="firm_force" ;;
  *)
    echo "[set_replay_env] 未知 profile: ${PROFILE}"
    echo "  允许值：inference | infer | libero | softvtbench_gentle | softvtbench_force_gentle | softvtbench_firm | softvtbench_force_firm | softvtbench_binary | softvtbench_binary_gentle | softvtbench_binary_firm"
    return 2
    ;;
esac

if [[ -n "${CUSTOM_ROOT}" ]]; then
  ROOT="$(cd "${CUSTOM_ROOT}" 2>/dev/null && pwd || true)"
  if [[ -z "${ROOT}" ]]; then
    echo "[set_replay_env] output_root 无法解析为有效目录：${CUSTOM_ROOT}"
    echo "  你可以先 mkdir -p，然后再 source。"
    return 2
  fi
else
  if [[ "${PROFILE_KIND}" == "libero" ]]; then
    ROOT="${_DEFAULT_LIBERO_ROOT}"
  elif [[ "${DATASET_KIND}" == "softvtbench_binary" ]]; then
    if [[ -n "${FORCE_MODE}" ]]; then
      ROOT="${_REPO_ROOT}/benchmarks/datasets/softvtbench_binary/${FORCE_MODE}"
    else
      ROOT="${_REPO_ROOT}/benchmarks/datasets/softvtbench_binary"
    fi
  else
    ROOT="${_REPO_ROOT}/benchmarks/datasets/${DATASET_KIND}/${FORCE_MODE}"
  fi
fi

#
# IMPORTANT:
# replay_utils.py uses string concatenation for `root_dir_prefix` (expects a trailing slash),
# e.g. base = f"{root_dir_prefix}{task_suite}_task{task_id}".
# Therefore OUTPUT_REPLAYED_VIDEOS_DIR MUST end with '/' to avoid paths like:
#   video_datasetslibero_goal_task0/...
#
export OUTPUT_REPLAYED_DEMOS_DIR="${ROOT}/replayed_demos/"
export OUTPUT_REPLAYED_VIDEOS_DIR="${ROOT}/video_datasets/"

# 回放后的数据读取目录（评估/推理脚本读取）
export REPLAYED_DEMOS_DIR="${OUTPUT_REPLAYED_DEMOS_DIR}"

# 启用 SoftVTBench 子集（softvtbench_tasks.json）
if [[ "${PROFILE_KIND}" == "libero" || "${PROFILE}" == "inference" || "${PROFILE}" == "infer" ]]; then
  export USE_SOFTVTBENCH_TASKS=0
else
  export USE_SOFTVTBENCH_TASKS="${USE_SOFTVTBENCH_TASKS:-0}"
fi

echo "[set_replay_env] 已设置："
echo "  PROFILE=${PROFILE}"
echo "  REPO_ROOT=${_REPO_ROOT}"
echo "  HDF5_TRAJ_SOURCE_DIR=${HDF5_TRAJ_SOURCE_DIR}"
echo "  OUTPUT_REPLAYED_DEMOS_DIR=${OUTPUT_REPLAYED_DEMOS_DIR}"
echo "  OUTPUT_REPLAYED_VIDEOS_DIR=${OUTPUT_REPLAYED_VIDEOS_DIR}"
echo "  REPLAYED_DEMOS_DIR=${REPLAYED_DEMOS_DIR}"
echo "  USE_SOFTVTBENCH_TASKS=${USE_SOFTVTBENCH_TASKS}"

if [[ ! -d "${HDF5_TRAJ_SOURCE_DIR}" ]]; then
  echo "[set_replay_env] ⚠️ 未找到 assembled_hdf5: ${HDF5_TRAJ_SOURCE_DIR}"
  _print_default_data_hints
fi
if [[ ! -d "${_DEFAULT_ASSETS_DATA}" ]]; then
  echo "[set_replay_env] ⚠️ 未找到默认 assets/data: ${_DEFAULT_ASSETS_DATA}"
  _print_default_data_hints
fi

if [[ "${PROFILE_KIND}" == "libero" ]]; then
  echo
  echo "[set_replay_env] 提醒："
  echo "  - 如果你只是直接使用现成的 LIBERO 数据训练/评测，则不需要 replay 再采集。"
  echo "  - 当前 libero profile 默认指向 benchmarks/datasets/libero（通常是软链接后的默认数据目录）。"
  echo "  - 如果你想重新回放采集自己的 LIBERO 数据，请不要直接复用这个默认目录。"
  echo "  - 请改为手动 export 到单独目录，例如："
  echo "      export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/my_libero_replay/replayed_demos"
  echo "      export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/my_libero_replay/video_datasets"
  echo "      export REPLAYED_DEMOS_DIR=\"\$OUTPUT_REPLAYED_DEMOS_DIR\""
fi


