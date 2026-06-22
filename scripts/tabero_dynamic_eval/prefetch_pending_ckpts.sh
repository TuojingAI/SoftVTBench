#!/usr/bin/env bash
set -Eeuo pipefail
REMOTE_HOST=root@124.174.13.117
REMOTE_PORT=44998
LOG=/data/mingxinwang/openpi-univtac/logs/tabero_dynamic_eval_20260616/prefetch.log
mkdir -p "$(dirname "$LOG")"

copy_one() {
  local label=$1
  local local_ckpt=$2
  local remote_ckpt=$3
  local marker="${remote_ckpt}/.copy_complete"
  local parent
  parent=$(dirname "${remote_ckpt}")
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local tmp="${remote_ckpt}.tmp.prefetch_${stamp}"
  local incomplete="${remote_ckpt}.incomplete.prefetch_${stamp}"

  echo "$(date --iso-8601=seconds) prefetch start ${label}" | tee -a "$LOG"
  if ssh -p "$REMOTE_PORT" "$REMOTE_HOST" "test -f '$marker'"; then
    echo "$(date --iso-8601=seconds) prefetch skip ${label}; marker exists" | tee -a "$LOG"
    return 0
  fi
  ssh -p "$REMOTE_PORT" "$REMOTE_HOST" "mkdir -p '$parent' && if [ -f '$marker' ]; then exit 0; fi && if [ -d '$remote_ckpt' ]; then mv '$remote_ckpt' '$incomplete'; fi && mkdir -p '$tmp'"
  tar -C "$local_ckpt" -cf - . | ssh -p "$REMOTE_PORT" "$REMOTE_HOST" "tar -C '$tmp' -xf - && mv '$tmp' '$remote_ckpt' && touch '$marker'"
  echo "$(date --iso-8601=seconds) prefetch done ${label}" | tee -a "$LOG"
}

copy_one tactile:4000 \
  /data/mingxinwang/openpi-univtac/checkpoints/pi0_lora_tacall_tabero/tabero_object_spatial_success_only_pi0_tactile_h50_bs256_4gpu_nw8_20260615/4000 \
  '/vepfs-C区/visuotactile/checkpoints/openpi_runs/pi0_lora_tacall_tabero/tabero_object_spatial_success_only_pi0_tactile_h50_bs256_4gpu_nw8_20260615/4000'

copy_one vision:8000 \
  /data/mingxinwang/openpi-univtac/checkpoints/pi0_lora_vision_tabero/tabero_object_spatial_success_only_pi0_vision_h50_bs256_4gpu_nw8_20260615/8000 \
  '/vepfs-C区/visuotactile/checkpoints/openpi_runs/pi0_lora_vision_tabero/tabero_object_spatial_success_only_pi0_vision_h50_bs256_4gpu_nw8_20260615/8000'
