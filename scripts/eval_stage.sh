#!/usr/bin/env bash
# One stage = one (suite x policy). The outer loop spawns one process per task
# (workaround for Isaac crashing on App close): each task gets one App launch and
# runs all of its episodes in-process; the pi05 server is started once per stage.
# Usage: eval_stage.sh SUITE POLICY_ID EPISODES OUT_ROOT [PORT] [TASKS] [DEMO_OFFSET]
#                      [OOD_JSON|CONDITIONS_FILE] [LEVEL]
# Arg 8 as a .json = a single OOD condition (legacy usage); as a conditions file
# (one `<label> <json|ID> [level]` per line) = one server + one env runs ID and
# multiple OOD conditions back to back, avoiding the fixed restart cost per condition.
set -uo pipefail

SUITE="${1:?suite}"; POLICY="${2:?policy id}"; EPISODES="${3:?episodes per task}"
OUT="${4:?out root}"; PORT="${5:-9021}"; TASKS="${6:-0 1 2 3 4 5 6 7 8 9}"; OFFSET="${7:-0}"
N_COND=1
if [[ -n "${8:-}" ]]; then
  if [[ "${8}" == *.json ]]; then          # legacy usage: single condition -> synthesize a conditions file in place
    CONDITIONS="${OUT}/conditions.txt"
    mkdir -p "${OUT}"
    echo "$(basename "${8}" .json)_L${9:-file} ${8} ${9:-}" > "${CONDITIONS}"
  else
    CONDITIONS="${8}"
  fi
  export SOFTVT_CONDITIONS="${CONDITIONS}"
  N_COND=$(grep -cvE '^\s*(#|$)' "${CONDITIONS}")
  echo "[stage] conditions(${N_COND}): $(grep -vE '^\s*(#|$)' "${CONDITIONS}" | awk '{printf "%s ", $1}')"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
CONFIG_DIR="${REPO_ROOT}/config"
MODELS_ROOT="${SOFTVT_MODELS_ROOT:?set SOFTVT_MODELS_ROOT to the SoftVTBench-Models checkout}"
POLICY_FILE="${SOFTVT_POLICY_CONFIG:-${MODELS_ROOT}/configs/policies.yaml}"
MODEL_PATHS_FILE="${SOFTVT_MODEL_PATHS_CONFIG:-${MODELS_ROOT}/configs/paths.yaml}"
export SOFTVTBENCH_ROOT="${SOFTVTBENCH_ROOT:-${REPO_ROOT}}"
export SOFTVT_CONFIG_DIR="${CONFIG_DIR}"
export SOFTVT_POLICY_CONFIG="${POLICY_FILE}"
SOFTVTBENCH_PYTHON="${SOFTVTBENCH_PYTHON:-python}"
ACT_PYTHON="${ACT_PYTHON:-python}"
DP_PYTHON="${DP_PYTHON:-python}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-python}"
mkdir -p "${OUT}"
# Every exit path reclaims the server/workers started by this stage (no orphaned ports)
trap 'kill $(cat "${OUT}/ckpt_view.pid" 2>/dev/null) 2>/dev/null; kill $(cat "${OUT}/worker.pids" 2>/dev/null) 2>/dev/null' EXIT

# Fingerprint every benchmark runtime/config file plus the model manifest. Sorting
# paths makes the value independent of filesystem enumeration order.
CODE_FPR=$(
  find "${SRC_DIR}" "${CONFIG_DIR}" "${REPO_ROOT}/scripts" -type f -print0 \
    | sort -z | xargs -0 sha256sum
  sha256sum "${POLICY_FILE}" "${MODEL_PATHS_FILE}"
  )
CODE_FPR=$(printf '%s' "${CODE_FPR}" | sha256sum | cut -c1-16)
export SOFTVT_CODE_FINGERPRINT="${CODE_FPR}"
if git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  export SOFTVT_RELEASE_COMMIT
  SOFTVT_RELEASE_COMMIT="$(git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}" rev-parse HEAD)"
  if [[ -n "$(git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}" status --porcelain \
                 -- src config scripts source)" ]]; then
    export SOFTVT_RELEASE_DIRTY=1
  else
    export SOFTVT_RELEASE_DIRTY=0
  fi
else
  export SOFTVT_RELEASE_COMMIT=UNVERSIONED
  export SOFTVT_RELEASE_DIRTY=1
fi
if git -c "safe.directory=${MODELS_ROOT}" -C "${MODELS_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SOFTVT_MODELS_COMMIT="$(git -c "safe.directory=${MODELS_ROOT}" -C "${MODELS_ROOT}" rev-parse HEAD)"
  SOFTVT_MODELS_DIRTY=0
  [[ -n "$(git -c "safe.directory=${MODELS_ROOT}" -C "${MODELS_ROOT}" status --porcelain -- src configs scripts backends)" ]] \
    && SOFTVT_MODELS_DIRTY=1
else
  SOFTVT_MODELS_COMMIT=UNVERSIONED
  SOFTVT_MODELS_DIRTY=1
fi
export SOFTVT_MODELS_COMMIT SOFTVT_MODELS_DIRTY
if [[ "${SOFTVT_REQUIRE_CLEAN_RELEASE:-0}" == 1 ]] &&
   [[ "${SOFTVT_RELEASE_COMMIT}" == UNVERSIONED || "${SOFTVT_RELEASE_DIRTY}" != 0 ||
      "${SOFTVT_MODELS_COMMIT}" == UNVERSIONED || "${SOFTVT_MODELS_DIRTY}" != 0 ]]; then
  echo "[stage] refusing formal run: benchmark=${SOFTVT_RELEASE_COMMIT}/${SOFTVT_RELEASE_DIRTY} models=${SOFTVT_MODELS_COMMIT}/${SOFTVT_MODELS_DIRTY}" >&2
  exit 2
fi
echo "[stage] code+config fingerprint=${CODE_FPR}" | tee "${OUT}/fingerprint.txt"
echo "[stage] release=${SOFTVT_RELEASE_COMMIT} dirty=${SOFTVT_RELEASE_DIRTY}" | tee -a "${OUT}/fingerprint.txt"
echo "[stage] models=${SOFTVT_MODELS_COMMIT} dirty=${SOFTVT_MODELS_DIRTY}" | tee -a "${OUT}/fingerprint.txt"

# Start the pi05 server or act/dp worker(s) according to the backend
SERVER_ARGS=()
BACKEND="$("${SOFTVTBENCH_PYTHON}" - "${POLICY_FILE}" "${POLICY}" <<'PY'
import os, sys, yaml
manifest, pid = sys.argv[1], sys.argv[2]
text = open(manifest).read()
policies = yaml.safe_load(os.path.expandvars(text))["policies"]
print(next(p for p in policies if p["id"] == pid)["backend"])
PY
)"

# PARALLEL=N: run N task processes in parallel within one stage.
# pi05/replay: N tasks share one server (no cross-request state).
# act/dp: workers hold per-episode buffers, so sharing would cross-contaminate ->
#         each parallel slot gets its own worker; slots are assigned via an
#         atomic mkdir lock (task completion order is nondeterministic, so
#         index-modulo assignment would collide on ports).
#
# Known failure mode: with PARALLEL=2 the first wave of two Isaac instances can
# initialize concurrently and one may deadlock in teardown (log stops at
# "unloading all plugins", zero results, process never exits), leaving the final
# wait blocked forever. -> Default is 1 (one Isaac per GPU).
# To re-enable N>1, also set STAGGER (staggered startup) and verify no deadlock.
PARALLEL=${PARALLEL:-1}
[[ "${PARALLEL}" -ge 1 ]] || PARALLEL=1
STAGGER=${STAGGER:-0}                 # seconds between adjacent task launches; recommend >= 120 when N>1
# Hard per-task timeout: a hung task must not consume the whole stage
# (killed on timeout; surfaced by the fail-closed validation below)
TASK_TIMEOUT=${TASK_TIMEOUT:-3600}

if [[ "${BACKEND}" == openpi ]]; then
  read -r CKPT MODALITY OPENPI_CFG <<<"$("${SOFTVTBENCH_PYTHON}" - "${POLICY_FILE}" "${POLICY}" <<'PY'
import os, sys, yaml
manifest, pid = sys.argv[1], sys.argv[2]
pols = yaml.safe_load(os.path.expandvars(open(manifest).read()))["policies"]
p = next(p for p in pols if p["id"] == pid)
mod = "tactile" if p["modality"] == "vt" else "vision"
# openpi_config must be passed down if it differs from the modality-derived default (see comment in start_pi05_server.sh)
want = p.get("openpi_config") or ""
default = "pi05_lora_tacall_softvtbench" if mod == "tactile" else "pi05_lora_vision_softvtbench"
print(p["checkpoint"], mod, "" if (not want or want == default) else want)
PY
)"
  OPENPI_CONFIG_OVERRIDE="${OPENPI_CFG:-}" \
  bash "${MODELS_ROOT}/scripts/start_pi05_server.sh" "${CKPT}" "${MODALITY}" "${PORT}" "${OUT}/ckpt_view" "${OUT}/server.log" \
    || { echo "server start failed" >&2; exit 1; }
  SERVER_ARGS=(--server-port "${PORT}")
elif [[ "${BACKEND}" == act || "${BACKEND}" == diffusion || "${BACKEND}" == fastwam ]]; then
  KWARGS="$("${SOFTVTBENCH_PYTHON}" - "${POLICY_FILE}" "${MODEL_PATHS_FILE}" "${POLICY}" <<'PY'
import json, os, sys, yaml
manifest, paths_file, pid = sys.argv[1], sys.argv[2], sys.argv[3]
load = lambda path: yaml.safe_load(os.path.expandvars(open(path).read()))
paths = load(paths_file)
p = next(p for p in load(manifest)["policies"] if p["id"] == pid)
kw = {"modality": p["modality"]}
if p["backend"] == "act":
    kw.update(ckpt_dir=p["ckpt_dir"], act_repo=paths["act_repo"], train_config=p["train_config"])
    for k in ("execution", "replan_steps"):
        if p.get(k) is not None:
            kw[k] = p[k]
elif p["backend"] == "diffusion":
    kw.update(ckpt_path=p["ckpt_path"], dp_repo=paths["dp_repo"])
    for k in ("execution", "replan_steps"):
        if p.get(k) is not None:
            kw[k] = p[k]
else:  # fastwam
    kw.update(ckpt_path=p["ckpt_path"], stats_path=p["stats_path"],
              text_cache_dir=p["text_cache_dir"],
              fastwam_repo=paths["fastwam_repo"], vae_path=paths["fastwam_vae"])
    if p.get("marker_stats_path"):
        kw["marker_stats_path"] = p["marker_stats_path"]
    for k in ("replan_steps", "num_inference_steps"):
        if p.get(k) is not None:
            kw[k] = p[k]
print(json.dumps(kw, ensure_ascii=False))
PY
)"
  WORKER_PY="${ACT_PYTHON}"
  [[ "${BACKEND}" == diffusion ]] && WORKER_PY="${DP_PYTHON}"
  [[ "${BACKEND}" == fastwam ]] && WORKER_PY="${FASTWAM_PYTHON}"
  # fastwam loads a 12-14GB ckpt + VAE; extend the ready wait to 20 minutes
  READY_TRIES=150
  [[ "${BACKEND}" == fastwam ]] && READY_TRIES=600
  # One independent worker per parallel slot (port PORT+slot); episode buffers are not shared
  : > "${OUT}/worker.pids"
  for s in $(seq 0 $((PARALLEL - 1))); do
    wp=$((PORT + s))
    nohup env PYTHONPATH="${MODELS_ROOT}/src:${REPO_ROOT}/src:${PYTHONPATH:-}" \
      "${WORKER_PY}" -m softvtbench_models.worker --backend "${BACKEND}" \
      --port "${wp}" --kwargs "${KWARGS}" > "${OUT}/worker${s}.log" 2>&1 &
    echo $! >> "${OUT}/worker.pids"
  done
  for s in $(seq 0 $((PARALLEL - 1))); do
    pid=$(sed -n "$((s + 1))p" "${OUT}/worker.pids")
    for i in $(seq 1 "${READY_TRIES}"); do
      grep -q 'ready' "${OUT}/worker${s}.log" && break
      kill -0 "${pid}" 2>/dev/null || { echo "worker${s} died"; tail -20 "${OUT}/worker${s}.log"; exit 1; }
      sleep 2
      [[ ${i} == ${READY_TRIES} ]] && { echo "worker${s} timeout"; exit 1; }
    done
  done
  echo "[worker] ready (${BACKEND}) x${PARALLEL} ports ${PORT}..$((PORT + PARALLEL - 1))"
  SERVER_ARGS=(--worker-port "${PORT}")   # placeholder; run_task overrides it per slot
fi

# Slot pool: mkdir is atomic, first-come-first-served; release with rmdir.
rm -rf "${OUT}"/.slot* 2>/dev/null
acquire_slot() {
  local s
  while true; do
    for s in $(seq 0 $((PARALLEL - 1))); do
      if mkdir "${OUT}/.slot${s}" 2>/dev/null; then echo "${s}"; return; fi
    done
    sleep 1
  done
}
run_task() {
  local t="$1" slot args
  slot="$(acquire_slot)"
  if [[ "${BACKEND}" == act || "${BACKEND}" == diffusion || "${BACKEND}" == fastwam ]]; then
    args=(--worker-port "$((PORT + slot))")     # worker dedicated to this slot
  else
    args=("${SERVER_ARGS[@]}")                  # pi05/replay share one server
  fi
  echo "=== [stage] ${SUITE}/${POLICY} task${t} (slot${slot}) ==="
  timeout -k 30 "${TASK_TIMEOUT}" \
    bash "${SCRIPT_DIR}/run_eval.sh" --suite "${SUITE}" --policy "${POLICY}" \
      --tasks "${t}" --episodes "${EPISODES}" --demo-offset "${OFFSET}" \
      --out "${OUT}/task${t}" "${args[@]}" \
      > "${OUT}/task${t}.log" 2>&1
  rc=$?
  [[ ${rc} -eq 124 || ${rc} -eq 137 ]] && echo "!! task${t} killed after ${TASK_TIMEOUT}s timeout (Isaac hang?)"
  rmdir "${OUT}/.slot${slot}" 2>/dev/null
  grep -E '\[runner\] task[0-9]+/demo' "${OUT}/task${t}.log" | tail -$(( EPISODES * N_COND ))
}

# Wait only on run_task PIDs -- a bare `wait` would also wait on the resident
# workers (background jobs of the same shell) and hang forever after all tasks finish.
TASK_PIDS=()
alive_tasks() { local n=0 p; for p in "${TASK_PIDS[@]}"; do kill -0 "${p}" 2>/dev/null && n=$((n + 1)); done; echo "${n}"; }
for t in ${TASKS}; do
  run_task "${t}" &
  TASK_PIDS+=("$!")
  [[ "${STAGGER}" -gt 0 ]] && sleep "${STAGGER}"   # stagger Isaac initialization to avoid the concurrent-startup deadlock
  while [[ "$(alive_tasks)" -ge "${PARALLEL}" ]]; do sleep 2; done
done
for p in "${TASK_PIDS[@]}"; do wait "${p}" 2>/dev/null; done

fail=0
WANT=$(( EPISODES * N_COND ))   # results per task = episodes x conditions
for t in ${TASKS}; do
  lines=$(wc -l < "${OUT}/task${t}/results.jsonl" 2>/dev/null || echo 0)
  if [[ "${lines}" -ne "${WANT}" ]]; then
    echo "!! task${t}: expected ${WANT} results (${EPISODES}ep x ${N_COND}cond), got ${lines} -- FAILED"
    fail=1
  fi
done

# Aggregate all tasks' results.jsonl (fail-closed: total-count and uniqueness validation)
N_TASKS=$(echo ${TASKS} | wc -w)
"${SOFTVTBENCH_PYTHON}" - "${OUT}" "${N_TASKS}" "${EPISODES}" "${N_COND}" <<'PY' || fail=1
import hashlib, json, sys
from pathlib import Path
root, n_tasks, eps, n_cond = (Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
rows = []
receipt_keys = []
result_files = sorted(root.glob("task*/results.jsonl"))
for result_file in result_files:
    task_rows = [json.loads(line) for line in open(result_file)]
    receipt_dir = result_file.parent / "episode_receipts"
    receipt_files = sorted(receipt_dir.glob("*.json"))
    assert len(receipt_files) == len(task_rows), (
        f"{result_file.parent}: {len(task_rows)} results but {len(receipt_files)} episode receipts"
    )
    referenced = set()
    for row in task_rows:
        rel = row.get("episode_receipt")
        want_sha = row.get("episode_receipt_sha256")
        assert rel and want_sha, f"result lacks episode receipt identity: {row}"
        receipt_path = result_file.parent / rel
        assert receipt_path.is_file(), f"missing episode receipt: {receipt_path}"
        payload = receipt_path.read_bytes()
        got_sha = hashlib.sha256(payload).hexdigest()
        assert got_sha == want_sha, f"episode receipt digest mismatch: {receipt_path}"
        rec = json.loads(payload)
        assert (rec.get("contract") or {}).get("passed") is True, (
            f"episode receipt contract not passed: {receipt_path}"
        )
        result_key = (row.get("condition", "id"), row["task_id"], row["episode"])
        receipt_key = (rec.get("condition"), rec.get("task_id"), rec.get("episode"))
        assert receipt_key == result_key, (
            f"receipt/result identity mismatch: {receipt_key} != {result_key}"
        )
        referenced.add(receipt_path.resolve())
        receipt_keys.append(receipt_key)
    assert referenced == {p.resolve() for p in receipt_files}, (
        f"{result_file.parent}: unreferenced or multiply referenced episode receipts"
    )
    rows.extend(task_rows)
# (condition, task, episode) must be unique; total = tasks x episodes x conditions
keys = [(r.get("condition", "id"), r["task_id"], r["episode"]) for r in rows]
assert len(rows) == n_tasks * eps * n_cond, \
    f"expected {n_tasks*eps*n_cond} rows ({n_tasks}task x {eps}ep x {n_cond}cond), got {len(rows)}"
assert len(set(keys)) == len(keys), "duplicate (condition, task, episode) rows"
assert receipt_keys == keys, "episode receipt order/identity differs from results"

def pct(v, q):
    if not v: return None
    v = sorted(v); i = (len(v) - 1) * q / 100.0
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (i - lo), 3)

def block(sel):
    # Safety criterion not yet finalized (see metrics.py notes): report only goal success + deformation distribution
    by = {}
    for r in sel:
        t = by.setdefault(r["task_id"], [0, 0, []])   # goal, n, peaks
        t[0] += int(r["success"]); t[1] += 1
        if r.get("d_peak") is not None:
            t[2].append(float(r["d_peak"]))
    tot_goal = sum(v[0] for v in by.values()); tot_n = sum(v[1] for v in by.values())
    peaks = [p for v in by.values() for p in v[2]]
    return {"per_task": {k: {"goal": f"{v[0]}/{v[1]}",
                             "d_peak": (f"{min(v[2]):.2f}-{max(v[2]):.2f}" if v[2] else None)}
                         for k, v in sorted(by.items())},
            "goal_success": tot_goal, "episodes": tot_n,
            "goal_success_rate": round(tot_goal / tot_n, 4) if tot_n else None,
            "d_peak_min": pct(peaks, 0), "d_peak_median": pct(peaks, 50),
            "d_peak_p95": pct(peaks, 95), "d_peak_max": pct(peaks, 100),
            "d_peak_n": len(peaks)}

conds = sorted({r.get("condition", "id") for r in rows})
summary = block(rows)
summary["by_condition"] = {c: block([r for r in rows if r.get("condition", "id") == c]) for c in conds}
(root / "stage_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"conditions": {c: {"goal": v["goal_success_rate"],
                                    "d_peak_median": v["d_peak_median"], "d_peak_max": v["d_peak_max"]}
                                 for c, v in summary["by_condition"].items()},
                  "goal_success_rate": summary["goal_success_rate"],
                  "d_peak_median": summary["d_peak_median"]}, indent=2))
PY

# Teardown: stop server / workers
[[ -f "${OUT}/ckpt_view.pid" ]] && kill "$(cat "${OUT}/ckpt_view.pid")" 2>/dev/null
[[ -f "${OUT}/worker.pids" ]] && kill $(cat "${OUT}/worker.pids") 2>/dev/null
exit ${fail}
