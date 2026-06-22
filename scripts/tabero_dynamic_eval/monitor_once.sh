#!/usr/bin/env bash
set -u
LOCAL_ROOT=/data/mingxinwang/openpi-univtac
LOG_DIR=${LOCAL_ROOT}/logs/tabero_dynamic_eval_20260616
REMOTE='root@124.174.13.117'
PORT=44998
REMOTE_BASE='/vepfs-C区/visuotactile/Tabero/evaluation_results/openpi_dynamic_eval_20260616'

echo "=== $(date '+%F %T %Z') local ==="
echo "-- scheduler/train tmux --"
tmux ls 2>/dev/null | rg 'tabero_dynamic_eval_scheduler|tabero_pi0' || true
echo "-- queue --"
cat "${LOG_DIR}/queue_state.tsv" 2>/dev/null || true
echo "-- training steps --"
tail -n 6 "${LOG_DIR}/training_status.tsv" 2>/dev/null || true
echo "-- ckpts --"
printf 'vision: '; find "${LOCAL_ROOT}/checkpoints/pi0_lora_vision_tabero/tabero_object_spatial_success_only_pi0_vision_h50_bs256_4gpu_nw8_20260615" -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null | sed 's#.*/##' | sort -n | tr '\n' ' '; echo
printf 'tactile: '; find "${LOCAL_ROOT}/checkpoints/pi0_lora_tacall_tabero/tabero_object_spatial_success_only_pi0_tactile_h50_bs256_4gpu_nw8_20260615" -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null | sed 's#.*/##' | sort -n | tr '\n' ' '; echo

echo "=== remote ==="
ssh -p "${PORT}" "${REMOTE}" "BASE='${REMOTE_BASE}';
  echo '-- tmux --'; tmux ls 2>/dev/null || true;
  echo '-- processes --'; ps -ef | grep -E 'openpi_inference_client|serve_policy' | grep -v grep || true;
  echo '-- recent progress --';
  find \"\$BASE\" -path '*/replan_*_n10/progress.tsv' -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -n 6 | while read -r _ p; do echo \"--- \$p\"; tail -n 25 \"\$p\"; done;
  echo '-- recent events --';
  find \"\$BASE\" -path '*/replan_*_n10/events.log' -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -n 3 | while read -r _ p; do echo \"--- \$p\"; tail -n 20 \"\$p\"; done;
  echo '-- mp4 counts --';
  find \"\$BASE\" -type d -name 'replan_*_n10' 2>/dev/null | sort | while read -r d; do c=\$(find \"\$d/debug\" -name '*.mp4' 2>/dev/null | wc -l); echo \"\$c \$d\"; done | tail -n 10;
  echo '-- space --'; df -h '/vepfs-C区' | tail -n 1;
"
