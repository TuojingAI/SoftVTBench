#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

: "${CONFIG:?Set CONFIG, e.g. pi0_lora_tacall_tabero}"
: "${CKPT:?Set CKPT to a checkpoint step directory}"
: "${MODE:?Set MODE to tactile or vision_abs7d}"

VARIANT=${VARIANT:-${CONFIG}}
EXP=${EXP:-$(basename "$(dirname "${CKPT}")")}
STEP=${STEP:-$(basename "${CKPT}")}

export CONFIG CKPT VARIANT EXP STEP MODE

bash "${REPO_ROOT}/scripts/tabero_dynamic_eval/run_one_ckpt_eval.sh"
