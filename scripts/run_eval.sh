#!/usr/bin/env bash
# Launch wrapper: prepare the Isaac native-extension/USD/Vulkan environment, then run runner.py.
# The environment-setup section is transcribed verbatim from the formal runner (identical for object_soft/rigid).
# Usage: bash run_eval.sh --suite object_rigid --policy object_rigid/pi05_vo_b --episodes 1 --out DIR ...
set -euo pipefail

SOFTVTBENCH_PYTHON="${SOFTVTBENCH_PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
export SOFTVTBENCH_ROOT="${SOFTVTBENCH_ROOT:-${REPO_ROOT}}"
export SOFTVT_CONFIG_DIR="${SOFTVT_CONFIG_DIR:-${REPO_ROOT}/config}"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"
if [[ -z "${SOFTVT_RELEASE_COMMIT:-}" ]]; then
  if git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
    export SOFTVT_RELEASE_COMMIT
    SOFTVT_RELEASE_COMMIT="$(git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}" rev-parse HEAD)"
    export SOFTVT_RELEASE_DIRTY
    if [[ -n "$(git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}" status --porcelain \
                   -- src config scripts source)" ]]; then
      SOFTVT_RELEASE_DIRTY=1
    else
      SOFTVT_RELEASE_DIRTY=0
    fi
  else
    export SOFTVT_RELEASE_COMMIT=UNVERSIONED
    export SOFTVT_RELEASE_DIRTY=1
  fi
fi
# Shared filesystem + multi-node: disable bytecode caching (NFS mtime caching can let stale .pyc shadow newer source)
export PYTHONDONTWRITEBYTECODE=1

if [[ -z "${ISAAC_EXTSCACHE:-}" ]]; then
  ISAAC_EXTSCACHE="$("${SOFTVTBENCH_PYTHON}" -c \
    'import isaacsim, pathlib; print(pathlib.Path(isaacsim.__file__).parent / "extscache")' 2>/dev/null || true)"
fi
export ISAAC_EXTSCACHE
_isaac_ext() { compgen -G "${ISAAC_EXTSCACHE}/$1" 2>/dev/null | sort | head -1 || true; }
export WARP_EXT="${WARP_EXT:-$(_isaac_ext 'omni.warp.core-*')}"
export ISAAC_USD_LIB_DIR="${ISAAC_USD_LIB_DIR:-$(_isaac_ext 'omni.usd.libs-*')/bin}"
export ISAAC_USD_CORE_LIB_DIR="${ISAAC_USD_CORE_LIB_DIR:-$(_isaac_ext 'omni.usd.core-*')/bin}"
export ISAAC_USD_LAYERS_LIB_DIR="${ISAAC_USD_LAYERS_LIB_DIR:-$(_isaac_ext 'omni.kit.usd.layers-*')/bin}"
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
export __GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME:-nvidia}
export __NV_PRIME_RENDER_OFFLOAD=${__NV_PRIME_RENDER_OFFLOAD:-1}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}
for lib_dir in "${ISAAC_USD_CORE_LIB_DIR}" "${ISAAC_USD_LIB_DIR}" "${ISAAC_USD_LAYERS_LIB_DIR}" \
               /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
  [[ -d "${lib_dir}" ]] && export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
done
# Isaac's bundled omni.warp.core must take precedence over pip warp
export PYTHONPATH="${WARP_EXT:+${WARP_EXT}:}${PYTHONPATH:-}"
# Optional custom headless Kit experience. Isaac Sim's default is used when
# this variable is unset.
[[ -n "${SOFTVTBENCH_APP_EXPERIENCE:-}" ]] && export SOFTVTBENCH_APP_EXPERIENCE

exec "${SOFTVTBENCH_PYTHON}" -u -m softvtbench.evaluation.runner "$@"
