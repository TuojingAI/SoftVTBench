#!/usr/bin/env bash
# =============================================================================
# SoftVTBench — two-environment installer
#
# Installs BOTH required environments in one run:
#   1. softvtbench-eval    (conda, Python 3.10)  Isaac Sim 4.5 + Isaac Lab 2.1.1
#                                                + tac_manip + openpi-client
#   2. softvtbench-openpi  (uv,    Python 3.11)  openpi training + policy server
#                                                (lives at openpi/upstream/.venv)
#
# The two environments MUST be separate: Isaac Sim needs Python 3.10 / numpy<2,
# while the openpi training stack pins conflicting versions on Python 3.11.
# A single environment cannot satisfy both — this is by design, not a workaround.
#
# Usage:
#   tools/install_environments.sh                    # install both (default)
#   tools/install_environments.sh --eval-only        # only the simulator/eval env
#   tools/install_environments.sh --openpi-only      # only the training env
#   tools/install_environments.sh --use-conda-lock   # exact linux-64 conda base
#   tools/install_environments.sh --skip-isaaclab    # reuse an existing Isaac Lab
#   tools/install_environments.sh --allow-broken-deps# force --no-deps (see below)
#   tools/install_environments.sh -h | --help
#
# Overridable via environment variables (defaults shown):
#   EVAL_ENV_NAME          softvtbench-eval
#   ISAACLAB_ROOT          $HOME/isaaclab-softvtbench
#   ISAACLAB_COMMIT        90b79bb2d44feb8d833f260f2bf37da3487180ba
#   OPENPI_PY_VERSION      3.11
#
# KNOWN ISSUE (ENV-01): the published requirements.txt does not currently
# pip-resolve (numpy==1.26.4 vs rerun-sdk>=numpy2; lerobot vs transformers/
# huggingface-hub). Until that is fixed, the eval-env dependency step stops with
# a clear message. --allow-broken-deps forces a --no-deps install for AUDITING
# ONLY; the result is not a valid, reproducible release environment.
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
if [[ -t 1 ]]; then C_INFO=$'\033[1;34m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'; C_ERR=$'\033[1;31m'; C_OFF=$'\033[0m'; else C_INFO=; C_OK=; C_WARN=; C_ERR=; C_OFF=; fi
log()  { printf '%s[install]%s %s\n' "$C_INFO" "$C_OFF" "$*"; }
ok()   { printf '%s[  ok  ]%s %s\n' "$C_OK"   "$C_OFF" "$*"; }
warn() { printf '%s[ warn ]%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die()  { printf '%s[ fail ]%s %s\n' "$C_ERR"  "$C_OFF" "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
DO_EVAL=1; DO_OPENPI=1; USE_CONDA_LOCK=0; SKIP_ISAACLAB=0; ALLOW_BROKEN_DEPS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-only)         DO_OPENPI=0 ;;
    --openpi-only)       DO_EVAL=0 ;;
    --use-conda-lock)    USE_CONDA_LOCK=1 ;;
    --skip-isaaclab)     SKIP_ISAACLAB=1 ;;
    --allow-broken-deps) ALLOW_BROKEN_DEPS=1 ;;
    -h|--help)           sed -n '2,45p' "$0"; exit 0 ;;
    *)                   die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

# -----------------------------------------------------------------------------
# Locate the repository root (this script lives in <root>/tools/)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFTVTBENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[[ -f "$SOFTVTBENCH_ROOT/requirements.txt" && -d "$SOFTVTBENCH_ROOT/openpi/upstream" ]] \
  || die "could not locate SoftVTBench root (expected requirements.txt and openpi/upstream next to tools/)"
cd "$SOFTVTBENCH_ROOT"
export SOFTVTBENCH_ROOT
log "repository root: $SOFTVTBENCH_ROOT"

EVAL_ENV_NAME="${EVAL_ENV_NAME:-softvtbench-eval}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/isaaclab-softvtbench}"
ISAACLAB_COMMIT="${ISAACLAB_COMMIT:-90b79bb2d44feb8d833f260f2bf37da3487180ba}"
OPENPI_PY_VERSION="${OPENPI_PY_VERSION:-3.11}"

# -----------------------------------------------------------------------------
# Preflight checks
# -----------------------------------------------------------------------------
preflight() {
  log "preflight checks"
  command -v git >/dev/null || die "git not found"
  if [[ "$DO_EVAL" == 1 ]]; then
    command -v conda >/dev/null || die "conda not found — install Miniconda/Miniforge first"
  fi
  if command -v nvidia-smi >/dev/null; then
    ok "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  else
    warn "nvidia-smi not found — Isaac Sim and CUDA training need an NVIDIA GPU"
  fi
  command -v ffmpeg >/dev/null \
    && ok "ffmpeg present ($(ffmpeg -version 2>/dev/null | head -1))" \
    || warn "ffmpeg not found — closed-loop evaluation needs it to encode rollout videos (install via system/conda)"
}

# -----------------------------------------------------------------------------
# Environment 1: simulator + evaluation  (conda, Python 3.10)
# -----------------------------------------------------------------------------
install_eval_env() {
  log "=== [1/2] simulator + evaluation env: $EVAL_ENV_NAME ==="

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if conda env list | awk '{print $1}' | grep -qx "$EVAL_ENV_NAME"; then
    log "conda env '$EVAL_ENV_NAME' already exists — reusing"
  elif [[ "$USE_CONDA_LOCK" == 1 ]]; then
    log "creating '$EVAL_ENV_NAME' from conda-linux-64.lock (exact linux-64 base)"
    conda create -y -n "$EVAL_ENV_NAME" --file conda-linux-64.lock
  else
    log "creating '$EVAL_ENV_NAME' from environment.yml"
    conda env create -f environment.yml -n "$EVAL_ENV_NAME"
  fi

  conda activate "$EVAL_ENV_NAME"
  python -m pip install --upgrade pip
  ok "activated $EVAL_ENV_NAME (python $(python -V 2>&1 | awk '{print $2}'))"

  # --- Isaac Sim 4.5 + remaining pinned simulator dependencies ---------------
  log "installing simulator dependencies from requirements.txt"
  local pip_args=(-r requirements.txt
    --extra-index-url https://pypi.nvidia.com
    --extra-index-url https://download.pytorch.org/whl/cu128
    --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html)

  if python -m pip install "${pip_args[@]}"; then
    ok "requirements.txt installed"
  elif [[ "$ALLOW_BROKEN_DEPS" == 1 ]]; then
    warn "requirements.txt did not resolve (known ENV-01)."
    warn "retrying with --no-deps — AUDIT ONLY, not a valid release environment."
    python -m pip install --no-deps "${pip_args[@]}"
  else
    die "requirements.txt failed to resolve.

This is the known ENV-01 dependency conflict, not a network error:
  - numpy==1.26.4    vs  rerun-sdk==0.26.2 (needs numpy>=2)
  - lerobot==0.4.4   vs  transformers==5.8.0 / huggingface-hub

Fix upstream first (split simulator direct deps into a solver-clean lock), or
re-run with --allow-broken-deps to force a --no-deps install for auditing only."
  fi

  # --- Isaac Lab 2.1.1 (pinned commit) ---------------------------------------
  if [[ "$SKIP_ISAACLAB" == 1 ]]; then
    log "skipping Isaac Lab clone/install (--skip-isaaclab); ISAACLAB_ROOT=$ISAACLAB_ROOT"
  else
    if [[ ! -d "$ISAACLAB_ROOT/.git" ]]; then
      log "cloning Isaac Lab into $ISAACLAB_ROOT"
      git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
    fi
    log "checking out Isaac Lab commit $ISAACLAB_COMMIT"
    git -C "$ISAACLAB_ROOT" fetch --quiet origin || true
    git -C "$ISAACLAB_ROOT" checkout "$ISAACLAB_COMMIT"
    local pkg
    for pkg in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks; do
      log "pip install -e Isaac Lab / $pkg"
      python -m pip install -e "$ISAACLAB_ROOT/source/$pkg"
    done
  fi

  # --- SoftVTBench extension + evaluation-side policy client -----------------
  log "installing tac_manip and openpi-client (editable)"
  python -m pip install -e "$SOFTVTBENCH_ROOT/SoftVTBench/source/tac_manip"
  python -m pip install -e "$SOFTVTBENCH_ROOT/openpi/upstream/packages/openpi-client"

  ok "simulator + evaluation env ready"
}

# -----------------------------------------------------------------------------
# Environment 2: openpi training + policy server  (uv, Python 3.11)
# -----------------------------------------------------------------------------
install_openpi_env() {
  log "=== [2/2] openpi training + policy-server env (uv) ==="
  if ! command -v uv >/dev/null; then
    log "uv not found — installing it"
    python3 -m pip install --user --upgrade uv || python3 -m pip install --upgrade uv
    command -v uv >/dev/null || die "uv still not on PATH after install (check ~/.local/bin)"
  fi
  ok "uv: $(uv --version)"

  ( cd "$SOFTVTBENCH_ROOT/openpi/upstream"
    log "uv python install $OPENPI_PY_VERSION"
    uv python install "$OPENPI_PY_VERSION"
    log "uv sync --frozen --python $OPENPI_PY_VERSION (authoritative uv.lock)"
    uv sync --frozen --python "$OPENPI_PY_VERSION" )

  ok "openpi env ready at openpi/upstream/.venv"
}

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
verify() {
  log "=== verification ==="
  if [[ "$DO_EVAL" == 1 ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$EVAL_ENV_NAME"
    log "eval env import + GPU check"
    env -u PYTHONPATH PYTHONNOUSERSITE=1 python - <<'PY' || warn "eval env import failed (see ENV-01 above if deps were forced)"
import torch, isaacsim, isaaclab, tac_manip
print("  torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("  isaacsim / isaaclab / tac_manip import OK")
PY
    env -u PYTHONPATH PYTHONNOUSERSITE=1 python -m pip check \
      && ok "eval env: pip check clean" \
      || warn "eval env: pip check reports conflicts (expected until ENV-01 is fixed)"
  fi
  if [[ "$DO_OPENPI" == 1 ]]; then
    log "openpi env import + JAX device check"
    "$SOFTVTBENCH_ROOT/openpi/upstream/.venv/bin/python" - <<'PY' || warn "openpi env import failed"
import jax, flax, openpi
print("  jax devices:", jax.devices())
print("  jax / flax / openpi import OK")
PY
  fi
}

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
preflight
[[ "$DO_EVAL"   == 1 ]] && install_eval_env
[[ "$DO_OPENPI" == 1 ]] && install_openpi_env
verify

cat <<EOF

$(ok "done.")
Next steps:
  1. Accept the NVIDIA Isaac Sim EULA once, then export:  OMNI_KIT_ACCEPT_EULA=YES
  2. Export the two interpreters for the eval/train scripts:
       export SOFTVTBENCH_PYTHON="\$(conda run -n $EVAL_ENV_NAME command -v python)"
       export OPENPI_PYTHON="$SOFTVTBENCH_ROOT/openpi/upstream/.venv/bin/python"
  3. Download data/assets/base checkpoint (see README "Data and Asset Downloads").
  4. Preflight before use:
       "\$SOFTVTBENCH_PYTHON" tools/doctor.py --mode train --suite object-soft \\
         --data-root "\$SOFTVTBENCH_DATA/object-soft" --openpi-python "\$OPENPI_PYTHON"
EOF
