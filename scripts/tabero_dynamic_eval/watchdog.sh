#!/usr/bin/env bash
set -u
ROOT=/data/mingxinwang/openpi-univtac
LOG_DIR=${ROOT}/logs/tabero_dynamic_eval_20260616
MONITOR=${ROOT}/scripts/tabero_dynamic_eval/monitor_once.sh
SCHED=${ROOT}/scripts/tabero_dynamic_eval/scheduler.py
SCHED_TMUX=tabero_dynamic_eval_scheduler
WATCH_LOG=${LOG_DIR}/watchdog.log
SCHED_OUT=${LOG_DIR}/tmux_scheduler.out
mkdir -p "${LOG_DIR}"
while true; do
  {
    echo "===== WATCHDOG $(date '+%F %T %Z') ====="
    if ! tmux has-session -t "${SCHED_TMUX}" 2>/dev/null; then
      echo "scheduler missing; restarting"
      tmux new-session -d -s "${SCHED_TMUX}" "cd /data/mingxinwang && python3 ${SCHED} >> ${SCHED_OUT} 2>&1"
    else
      echo "scheduler alive"
    fi
    "${MONITOR}" || true
  } >> "${WATCH_LOG}" 2>&1
  sleep 300
done
