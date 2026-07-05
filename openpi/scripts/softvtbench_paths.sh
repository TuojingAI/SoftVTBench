#!/usr/bin/env bash
# Shared SoftVTBench path resolver for OpenPI launchers.
# Source this file from scripts instead of hard-coding external reference repos.

_softvtbench_paths_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOFTVTBENCH_ROOT="${SOFTVTBENCH_ROOT:-$(cd "${_softvtbench_paths_dir}/../.." && pwd)}"
export OPENPI_LAUNCHER_ROOT="${OPENPI_LAUNCHER_ROOT:-${SOFTVTBENCH_ROOT}/openpi}"
export OPENPI_CODE_DIR="${OPENPI_CODE_DIR:-${OPENPI_LAUNCHER_ROOT}/upstream}"
export SOFTVTBENCH_DIR="${SOFTVTBENCH_DIR:-${SOFTVTBENCH_ROOT}/SoftVTBench}"
export SOFTVTBENCH_RESULTS_ROOT="${SOFTVTBENCH_RESULTS_ROOT:-${SOFTVTBENCH_ROOT}/results}"
export OPENPI_CACHE_ROOT="${OPENPI_CACHE_ROOT:-${OPENPI_LAUNCHER_ROOT}/.cache}"
export OPENPI_ASSETS_ROOT="${OPENPI_ASSETS_ROOT:-${OPENPI_LAUNCHER_ROOT}/assets}"
export OPENPI_LOG_ROOT="${OPENPI_LOG_ROOT:-${OPENPI_LAUNCHER_ROOT}/logs}"
export OPENPI_STAGE_ROOT="${OPENPI_STAGE_ROOT:-${OPENPI_LAUNCHER_ROOT}/data_softvtbench/.stage}"

export SOFTVTBENCH_OPENPI_PYTHON_SOURCE="explicit OPENPI_PYTHON"
if [[ -z "${OPENPI_PYTHON:-}" ]]; then
  if [[ -x "${OPENPI_LAUNCHER_ROOT}/.venv/bin/python" ]]; then
    export OPENPI_PYTHON="${OPENPI_LAUNCHER_ROOT}/.venv/bin/python"
    export SOFTVTBENCH_OPENPI_PYTHON_SOURCE="openpi/.venv"
  elif [[ -x "${OPENPI_CODE_DIR}/.venv/bin/python" ]]; then
    export OPENPI_PYTHON="${OPENPI_CODE_DIR}/.venv/bin/python"
    export SOFTVTBENCH_OPENPI_PYTHON_SOURCE="openpi/upstream/.venv"
  else
    export OPENPI_PYTHON=""
    export SOFTVTBENCH_OPENPI_PYTHON_SOURCE="not configured"
  fi
fi
export PYTHON="${PYTHON:-${OPENPI_PYTHON:-}}"

# OpenPI training and serve_policy need JAX. The Isaac Sim environment does not
# have it, so the fallback above would otherwise surface as a confusing
# ModuleNotFoundError deep inside train.py. Callers that need JAX must run this
# guard first.
softvtbench_require_openpi_python() {
  local py="${1:-${OPENPI_PYTHON}}"
  if [[ -z "${py}" ]]; then
    {
      echo "ERROR: OPENPI_PYTHON is not set and neither openpi/.venv nor openpi/upstream/.venv was found."
      echo "       Install OpenPI using its upstream instructions, then set:"
      echo "  export OPENPI_PYTHON=/path/to/openpi-venv/bin/python"
    } >&2
    return 1
  fi
  if [[ ! -x "${py}" ]]; then
    echo "ERROR: OpenPI python is not executable: ${py}" >&2
    return 1
  fi
  if ! "${py}" -c 'import jax' >/dev/null 2>&1; then
    {
      echo "ERROR: ${py} cannot import jax."
      echo "       resolved from: ${SOFTVTBENCH_OPENPI_PYTHON_SOURCE}"
      echo
      echo "OpenPI training and serve_policy require a JAX/Flax environment."
      echo "Install or activate one using the upstream OpenPI instructions, then point"
      echo "OPENPI_PYTHON at its Python executable:"
      echo
      echo "  export OPENPI_PYTHON=/path/to/openpi-venv/bin/python"
    } >&2
    return 1
  fi
}

softvtbench_cuda_visible_devices() {
  local count="${1:?device count is required}"
  if [[ ! "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid GPU count: ${count}" >&2
    return 2
  fi
  local result="" index
  for ((index = 0; index < count; index++)); do
    if [[ -n "${result}" ]]; then result+=","; fi
    result+="${index}"
  done
  printf '%s' "${result}"
}

# Build a read-only checkpoint view without ever adding files to the source
# checkpoint. Directories below assets/ are recreated as real directories and
# every file is symlinked individually, so adding a norm-stat alias cannot
# accidentally traverse a directory symlink back into the original checkpoint.
softvtbench_make_checkpoint_view() {
  local checkpoint="${1:?checkpoint is required}"
  local view="${2:?view directory is required}"
  [[ -d "${checkpoint}" ]] || { echo "checkpoint directory does not exist: ${checkpoint}" >&2; return 2; }
  [[ ! -e "${view}" ]] || { echo "checkpoint view already exists: ${view}" >&2; return 2; }
  mkdir -p "${view}/assets"

  local entry name directory relative target
  for entry in "${checkpoint}"/*; do
    [[ -e "${entry}" ]] || continue
    name="$(basename "${entry}")"
    [[ "${name}" == assets ]] && continue
    ln -s "$(realpath "${entry}")" "${view}/${name}"
  done
  if [[ -d "${checkpoint}/assets" ]]; then
    while IFS= read -r -d '' directory; do
      relative="${directory#"${checkpoint}/assets"}"
      mkdir -p "${view}/assets${relative}"
    done < <(find "${checkpoint}/assets" -type d -print0)
    while IFS= read -r -d '' entry; do
      relative="${entry#"${checkpoint}/assets/"}"
      target="${view}/assets/${relative}"
      [[ -e "${entry}" ]] || { echo "broken checkpoint asset symlink: ${entry}" >&2; return 2; }
      ln -s "$(realpath "${entry}")" "${target}"
    done < <(find "${checkpoint}/assets" \( -type f -o -type l \) -print0)
  fi
}

# Add the asset ID expected by the OpenPI config to a generated checkpoint
# view. Prints the selected source asset ID, or nothing if the destination was
# already present.
softvtbench_alias_norm_stats() {
  local checkpoint="${1:?checkpoint is required}"
  local view="${2:?view directory is required}"
  local destination_asset_id="${3:?destination asset id is required}"
  shift 3
  local destination="${view}/assets/${destination_asset_id}/norm_stats.json"
  [[ -f "${destination}" ]] && return 0
  local candidate
  for candidate in "$@"; do
    if [[ -f "${checkpoint}/assets/${candidate}/norm_stats.json" ]]; then
      mkdir -p "$(dirname "${destination}")"
      ln -s "$(realpath "${checkpoint}/assets/${candidate}/norm_stats.json")" "${destination}"
      printf '%s' "${candidate}"
      return 0
    fi
  done
  echo "missing compatible norm_stats under ${checkpoint}/assets" >&2
  find "${checkpoint}/assets" -maxdepth 3 -type f -name norm_stats.json -print >&2 || true
  return 1
}
