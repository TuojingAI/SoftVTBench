#!/usr/bin/env bash
# Unattended work queue: runs the stages listed in the queue file sequentially.
# Line format: suite policy name port episodes offset [conditions_file]
# (tasks are always the full set 0-9).
# A conditions file in column 7 runs ID plus multiple OOD conditions on one
# server+env. Any eval_stage.sh already running on the machine is allowed to
# finish first so its progress is not wasted.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="${SCRIPT_DIR}"
Q="${SOFTVT_QUEUE_FILE:-${PWD}/queue.txt}"
RESULT_ROOT="${SOFTVT_RESULT_ROOT:-${PWD}/results}"
MACHINE_ID="${SOFTVT_MACHINE_ID:-$(hostname)}"
RAW_ROOT="${RESULT_ROOT}/raw/${MACHINE_ID}"
LOG_ROOT="${RESULT_ROOT}/logs/${MACHINE_ID}"
SUMMARY_ROOT="${RESULT_ROOT}/summaries"
LOG="${LOG_ROOT}/queue_progress.log"
mkdir -p "${RAW_ROOT}" "${LOG_ROOT}" "${SUMMARY_ROOT}"

# Formal queues may only consume a committed, clean evaluation release.
# Ad-hoc smoke tests can still invoke eval_stage.sh directly; their receipts
# explicitly record dirty/unversioned state and therefore remain non-formal.
# Set SOFTVT_REQUIRE_CLEAN_RELEASE=0 before launching to run from an
# extracted (unversioned) copy of this release.
export SOFTVT_REQUIRE_CLEAN_RELEASE="${SOFTVT_REQUIRE_CLEAN_RELEASE:-1}"

echo "$(date +%H:%M:%S) QUEUE_START pid=$$" >> "${LOG}"
while pgrep -f '[e]val_stage.sh' >/dev/null 2>&1; do sleep 60; done

while read -r suite policy name port episodes offset conds; do
  [[ -z "${suite}" || "${suite}" == \#* ]] && continue
  STAGE_OUT="${RAW_ROOT}/${name}"
  if [[ -e "${STAGE_OUT}" ]]; then
    echo "$(date +%H:%M:%S) REFUSE ${name}: output already exists at ${STAGE_OUT}" >> "${LOG}"
    continue
  fi
  mkdir -p "${STAGE_OUT}"
  echo "$(date +%H:%M:%S) START ${name}" >> "${LOG}"
  bash "${CODE}/eval_stage.sh" "${suite}" "${policy}" "${episodes}" \
    "${STAGE_OUT}" "${port}" "0 1 2 3 4 5 6 7 8 9" "${offset:-0}" "${conds:-}" \
    > "${STAGE_OUT}/stage.log" 2>&1
  rc=$?
  s=$(grep -o '"goal_success_rate": [0-9.]*' "${STAGE_OUT}/stage_summary.json" 2>/dev/null | head -1)
  if [[ -f "${STAGE_OUT}/stage_summary.json" ]]; then
    cp "${STAGE_OUT}/stage_summary.json" "${SUMMARY_ROOT}/${MACHINE_ID}__${name}.json"
  fi
  echo "$(date +%H:%M:%S) END ${name} rc=${rc} ${s:-nosummary}" >> "${LOG}"
  # eval_stage owns and reclaims only the PIDs it started; do not use broad
  # process-name cleanup on shared machines.
  sleep 5
done < "${Q}"
echo "$(date +%H:%M:%S) QUEUE_DONE" >> "${LOG}"
touch "${LOG_ROOT}/QUEUE_DONE"
