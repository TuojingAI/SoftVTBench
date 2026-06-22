#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


LOCAL_ROOT = Path("/data/mingxinwang/openpi-univtac")
RUN_ID = "tabero_object_spatial_success_only_20260615"
LOCAL_RUN_DIR = LOCAL_ROOT / "logs" / "tabero_dynamic_eval_20260616"
REMOTE_HOST = "root@124.174.13.117"
REMOTE_PORT = "44998"
REMOTE_RUN_ROOT = "/vepfs-C区/visuotactile/Tabero/evaluation_results/openpi_dynamic_eval_20260616"
REMOTE_SCRIPT = f"{REMOTE_RUN_ROOT}/scripts/run_one_ckpt_eval_v2.sh"
REMOTE_CKPT_ROOT = "/vepfs-C区/visuotactile/checkpoints/openpi_runs"
REMOTE_OFFICIAL_TACTILE_CKPT = (
    "/vepfs-C区/visuotactile/checkpoints/pi0_lora_tacall_tabero_enc/checkpoints/"
    "pi0_lora_tacall_tabero/pi0_lora_tacall_tabero_new/49999"
)
DEFAULT_PORT = 8194
N = 10
TASKS_STR = "0 1 2 3 4 5 6 7 8 9"
SUITES_STR = "libero_object libero_spatial"
SCAN_INTERVAL_SECONDS = 60
MIN_CKPT_AGE_SECONDS = 120
MAX_ATTEMPTS = 3
STALE_JOB_SECONDS = 45 * 60
INITIAL_MAIN_MAX_STEP = {
    "tactile": 10000,
    "vision": 10000,
}
REPLAN30_MIN_BEFORE_NEW_MAIN = 3


@dataclass(frozen=True)
class Variant:
    name: str
    config: str
    mode: str
    exp: str
    local_dir: Path
    kind: str


VARIANTS = [
    Variant(
        name="vision",
        config="pi0_lora_vision_tabero",
        mode="vision_abs7d",
        exp="tabero_object_spatial_success_only_pi0_vision_h50_bs256_4gpu_nw8_20260615",
        local_dir=LOCAL_ROOT
        / "checkpoints"
        / "pi0_lora_vision_tabero"
        / "tabero_object_spatial_success_only_pi0_vision_h50_bs256_4gpu_nw8_20260615",
        kind="vision",
    ),
    Variant(
        name="tactile",
        config="pi0_lora_tacall_tabero",
        mode="tactile",
        exp="tabero_object_spatial_success_only_pi0_tactile_h50_bs256_4gpu_nw8_20260615",
        local_dir=LOCAL_ROOT
        / "checkpoints"
        / "pi0_lora_tacall_tabero"
        / "tabero_object_spatial_success_only_pi0_tactile_h50_bs256_4gpu_nw8_20260615",
        kind="tactile",
    ),
]

OFFICIAL_TACTILE_EXP = "official_release_pi0_lora_tacall_tabero_49999_noadverb_20260616"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def q(s: str | os.PathLike[str]) -> str:
    return shlex.quote(str(s))


def run_local(args: list[str], *, check: bool = True, stdin=None, stdout=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, check=check, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE)


def run_ssh(cmd: str, *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-p", REMOTE_PORT, REMOTE_HOST, cmd],
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


def log(message: str) -> None:
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{now()} {message}"
    print(line, flush=True)
    with (LOCAL_RUN_DIR / "scheduler.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict[str, Any]:
    path = LOCAL_RUN_DIR / "state.json"
    if not path.exists():
        return {"jobs": {}, "running": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LOCAL_RUN_DIR / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(LOCAL_RUN_DIR / "state.json")


def latest_log_step(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    with log_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 2_000_000))
        text = f.read().decode(errors="ignore")
    matches = re.findall(r"Step\s+(\d+):", text)
    return int(matches[-1]) if matches else None


def write_training_status() -> None:
    status_path = LOCAL_RUN_DIR / "training_status.tsv"
    vision_log = LOCAL_ROOT / "logs/tabero_training_20260615/pi0_vision_4gpu_nw8.log"
    tactile_log = LOCAL_ROOT / "logs/tabero_training_20260615/pi0_tactile_4gpu_nw8.log"
    rows = [
        ("vision", latest_log_step(vision_log)),
        ("tactile", latest_log_step(tactile_log)),
    ]
    with status_path.open("a", encoding="utf-8") as f:
        for name, step in rows:
            f.write(f"{now()}\t{name}\t{step if step is not None else 'NA'}\n")


def step_is_desired(variant: Variant, step: int) -> bool:
    if variant.kind == "vision":
        return step in {6000, 8000, 10000}
    if variant.kind == "tactile":
        return step >= 3000 and step % 1000 == 0
    return False


def is_ckpt_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    metadata = path / "_CHECKPOINT_METADATA"
    if not metadata.is_file():
        return False
    if time.time() - metadata.stat().st_mtime < MIN_CKPT_AGE_SECONDS:
        return False
    required = [path / "params", path / "train_state", path / "assets"]
    if not all(p.is_dir() for p in required):
        return False
    for p in required:
        try:
            next(p.rglob("*"))
        except StopIteration:
            return False
    return True


def discover_jobs(state: dict[str, Any]) -> None:
    jobs = state.setdefault("jobs", {})
    for variant in VARIANTS:
        if not variant.local_dir.exists():
            continue
        for child in sorted(variant.local_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if not child.name.isdigit():
                continue
            step = int(child.name)
            if not step_is_desired(variant, step):
                continue
            if not is_ckpt_ready(child):
                continue
            add_job(jobs, variant, step, 10, "main")
            main_key = job_key(variant.name, step, 10)
            if jobs.get(main_key, {}).get("status") == "done":
                add_job(jobs, variant, step, 30, "replan30")


def ensure_static_remote_jobs(state: dict[str, Any]) -> None:
    jobs = state.setdefault("jobs", {})
    key = job_key("official_tactile", 49999, 10)
    if key in jobs:
        return
    out_root = posixpath.join(
        REMOTE_RUN_ROOT,
        "official_tactile",
        OFFICIAL_TACTILE_EXP,
        "49999",
        f"replan_10_n{N}",
    )
    jobs[key] = {
        "key": key,
        "variant": "official_tactile",
        "config": "pi0_lora_tacall_tabero",
        "mode": "tactile",
        "exp": OFFICIAL_TACTILE_EXP,
        "step": 49999,
        "replan_steps": 10,
        "tier": "official",
        "local_ckpt": "REMOTE_ONLY",
        "remote_ckpt": REMOTE_OFFICIAL_TACTILE_CKPT,
        "remote_only": True,
        "out_root": out_root,
        "status": "pending",
        "attempts": 0,
        "first_seen": now(),
        "synced_at": "remote_only",
    }
    log(f"queued {key} remote_ckpt={REMOTE_OFFICIAL_TACTILE_CKPT}")


def add_job(jobs: dict[str, Any], variant: Variant, step: int, replan: int, tier: str) -> None:
    key = job_key(variant.name, step, replan)
    if key in jobs:
        return
    local_ckpt = variant.local_dir / str(step)
    remote_ckpt = posixpath.join(REMOTE_CKPT_ROOT, variant.config, variant.exp, str(step))
    out_root = posixpath.join(
        REMOTE_RUN_ROOT,
        variant.name,
        variant.exp,
        str(step),
        f"replan_{replan}_n{N}",
    )
    jobs[key] = {
        "key": key,
        "variant": variant.name,
        "config": variant.config,
        "mode": variant.mode,
        "exp": variant.exp,
        "step": step,
        "replan_steps": replan,
        "tier": tier,
        "local_ckpt": str(local_ckpt),
        "remote_ckpt": remote_ckpt,
        "out_root": out_root,
        "status": "pending",
        "attempts": 0,
        "first_seen": now(),
    }
    log(f"queued {key} local={local_ckpt}")


def job_key(variant_name: str, step: int, replan: int) -> str:
    return f"{variant_name}:{step}:r{replan}"


def deploy_remote_script() -> None:
    local_script = LOCAL_ROOT / "scripts/tabero_dynamic_eval/run_one_ckpt_eval.sh"
    run_ssh(f"mkdir -p {q(posixpath.dirname(REMOTE_SCRIPT))}")
    subprocess.run(
        ["scp", "-P", REMOTE_PORT, str(local_script), f"{REMOTE_HOST}:{REMOTE_SCRIPT}"],
        check=True,
        text=True,
    )
    run_ssh(f"chmod +x {q(REMOTE_SCRIPT)}")
    log(f"deployed remote worker to {REMOTE_SCRIPT}")


def remote_file_exists(path: str) -> bool:
    return run_ssh(f"test -f {q(path)}", check=False).returncode == 0


def remote_session_exists(session: str) -> bool:
    return run_ssh(f"tmux has-session -t {q(session)}", check=False).returncode == 0


def local_session_exists(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], check=False).returncode == 0


def expected_task_tags() -> list[str]:
    tags: list[str] = []
    for suite in SUITES_STR.split():
        for task_id in TASKS_STR.split():
            tags.append(f"{suite}_task{task_id}")
    return tags


def remote_progress_complete(out_root: str) -> tuple[bool, str]:
    progress = posixpath.join(out_root, "progress.tsv")
    proc = run_ssh(f"cat {q(progress)}", check=False)
    if proc.returncode != 0:
        return False, "missing progress.tsv"

    latest: dict[str, tuple[str, str, str, str]] = {}
    for line in proc.stdout.splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 5:
            continue
        task, succ, total, rate, rc = fields
        latest[task] = (succ, total, rate, rc)

    missing: list[str] = []
    bad: list[str] = []
    for task in expected_task_tags():
        row = latest.get(task)
        if row is None:
            missing.append(task)
            continue
        _succ, total, _rate, rc = row
        if total != str(N) or rc != "0":
            bad.append(f"{task}:total={total},rc={rc}")

    if missing or bad:
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing[:8]) + ("..." if len(missing) > 8 else ""))
        if bad:
            detail.append("bad=" + ",".join(bad[:8]) + ("..." if len(bad) > 8 else ""))
        return False, "; ".join(detail)
    return True, "complete"


def remote_has_partial_progress(out_root: str) -> bool:
    progress = posixpath.join(out_root, "progress.tsv")
    proc = run_ssh(f"test -s {q(progress)}", check=False)
    return proc.returncode == 0


def remote_activity_age_seconds(out_root: str) -> int | None:
    cmd = f"""python3 - {q(out_root)} <<'PY'
import os
import sys
import time

root = sys.argv[1]
latest = 0.0
for rel in ("events.log", "progress.tsv", "tmux_stdout.log"):
    path = os.path.join(root, rel)
    if os.path.exists(path):
        latest = max(latest, os.path.getmtime(path))
logs = os.path.join(root, "logs")
if os.path.isdir(logs):
    for name in os.listdir(logs):
        path = os.path.join(logs, name)
        if os.path.isfile(path) and name.endswith(".log"):
            latest = max(latest, os.path.getmtime(path))
print(int(time.time() - latest) if latest else 999999)
PY"""
    proc = run_ssh(cmd, check=False, timeout=30)
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def cleanup_remote_eval_processes() -> None:
    run_ssh(
        "pkill -f 'openpi_inference_client.py .*--server_port 8194' 2>/dev/null || true; "
        "pkill -f 'scripts/serve_policy.py --port 8194' 2>/dev/null || true",
        check=False,
        timeout=30,
    )


def sync_ckpt(job: dict[str, Any]) -> None:
    if job.get("remote_only"):
        job["synced_at"] = job.get("synced_at") or "remote_only"
        return
    marker = posixpath.join(job["remote_ckpt"], ".copy_complete")
    if remote_file_exists(marker):
        job["synced_at"] = job.get("synced_at") or now()
        return

    local_ckpt = Path(job["local_ckpt"])
    remote_ckpt = job["remote_ckpt"]
    parent = posixpath.dirname(remote_ckpt)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp = f"{remote_ckpt}.tmp.{stamp}"
    incomplete = f"{remote_ckpt}.incomplete.{stamp}"

    log(f"sync start {job['key']} -> {remote_ckpt}")
    prep = (
        f"mkdir -p {q(parent)} && "
        f"if [ -f {q(marker)} ]; then exit 0; fi && "
        f"if [ -d {q(remote_ckpt)} ]; then mv {q(remote_ckpt)} {q(incomplete)}; fi && "
        f"mkdir -p {q(tmp)}"
    )
    run_ssh(prep)

    tar_proc = subprocess.Popen(
        ["tar", "-C", str(local_ckpt), "-cf", "-", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    extract_cmd = f"tar -C {q(tmp)} -xf - && mv {q(tmp)} {q(remote_ckpt)} && touch {q(marker)}"
    ssh_proc = subprocess.Popen(
        ["ssh", "-p", REMOTE_PORT, REMOTE_HOST, extract_cmd],
        stdin=tar_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert tar_proc.stdout is not None
    tar_proc.stdout.close()
    ssh_out, ssh_err = ssh_proc.communicate()
    tar_err = tar_proc.stderr.read() if tar_proc.stderr else b""
    tar_rc = tar_proc.wait()
    if tar_rc != 0 or ssh_proc.returncode != 0:
        raise RuntimeError(
            f"sync failed for {job['key']}: tar_rc={tar_rc} ssh_rc={ssh_proc.returncode} "
            f"tar_err={tar_err.decode(errors='ignore')} ssh_err={ssh_err.decode(errors='ignore')}"
        )

    job["synced_at"] = now()
    log(f"sync done {job['key']}")


def sync_session_name(job: dict[str, Any]) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", job["key"])
    return f"tabero_sync_{safe}"


def refresh_sync_status(job: dict[str, Any]) -> bool:
    if job.get("remote_only"):
        job["synced_at"] = job.get("synced_at") or "remote_only"
        job.pop("sync_session", None)
        job.pop("sync_started_at", None)
        return True
    marker = posixpath.join(job["remote_ckpt"], ".copy_complete")
    if remote_file_exists(marker):
        if not job.get("synced_at"):
            job["synced_at"] = now()
            log(f"sync done {job['key']}")
        job.pop("sync_session", None)
        job.pop("sync_started_at", None)
        return True

    session = job.get("sync_session")
    if session and not local_session_exists(session):
        job.pop("sync_session", None)
        job["sync_failed_at"] = now()
        log(f"sync failed {job['key']}; tmux session ended without marker")
    return False


def start_sync_ckpt_background(job: dict[str, Any]) -> None:
    if job.get("remote_only"):
        job["synced_at"] = job.get("synced_at") or "remote_only"
        return
    if refresh_sync_status(job):
        return

    session = sync_session_name(job)
    if local_session_exists(session):
        job["sync_session"] = session
        return

    local_ckpt = Path(job["local_ckpt"])
    remote_ckpt = job["remote_ckpt"]
    marker = posixpath.join(remote_ckpt, ".copy_complete")
    parent = posixpath.dirname(remote_ckpt)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp = f"{remote_ckpt}.tmp.{stamp}"
    incomplete = f"{remote_ckpt}.incomplete.{stamp}"
    log_path = LOCAL_RUN_DIR / f"sync_{job['key'].replace(':', '_')}.log"

    prep = (
        f"mkdir -p {q(parent)} && "
        f"if [ -f {q(marker)} ]; then exit 0; fi && "
        f"if [ -d {q(remote_ckpt)} ]; then mv {q(remote_ckpt)} {q(incomplete)}; fi && "
        f"mkdir -p {q(tmp)}"
    )
    run_ssh(prep)

    shell_cmd = (
        f"set -o pipefail; "
        f"tar -C {q(local_ckpt)} -cf - . | "
        f"ssh -p {q(REMOTE_PORT)} {q(REMOTE_HOST)} "
        f"{q(f'tar -C {q(tmp)} -xf - && mv {q(tmp)} {q(remote_ckpt)} && touch {q(marker)}')}"
    )
    tmux_cmd = f"{shell_cmd} >> {q(log_path)} 2>&1"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "bash", "-lc", tmux_cmd],
        check=True,
        text=True,
    )
    job["sync_session"] = session
    job["sync_started_at"] = now()
    log(f"sync background start {job['key']} session={session}")


def session_name(job: dict[str, Any]) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", job["key"])
    return f"tabero_eval_{safe}"


def start_remote_eval(job: dict[str, Any]) -> None:
    refresh_sync_status(job)
    sync_ckpt(job)
    cleanup_remote_eval_processes()
    session = session_name(job)
    env = {
        "RUN_ROOT": REMOTE_RUN_ROOT,
        "CONFIG": job["config"],
        "CKPT": job["remote_ckpt"],
        "VARIANT": job["variant"],
        "EXP": job["exp"],
        "STEP": str(job["step"]),
        "MODE": job["mode"],
        "N": str(N),
        "REPLAN_STEPS": str(job["replan_steps"]),
        "TASKS_STR": TASKS_STR,
        "SUITES_STR": SUITES_STR,
        "PORT": str(DEFAULT_PORT),
    }
    out_root = job["out_root"]
    env_prefix = " ".join(f"{k}={q(v)}" for k, v in env.items())
    inner = (
        f"mkdir -p {q(out_root)} && "
        f"rm -f {q(posixpath.join(out_root, 'DONE'))} && "
        f"{env_prefix} bash {q(REMOTE_SCRIPT)} > {q(posixpath.join(out_root, 'tmux_stdout.log'))} 2>&1"
    )
    cmd = (
        f"tmux has-session -t {q(session)} 2>/dev/null || "
        f"tmux new-session -d -s {q(session)} {q('bash -lc ' + q(inner))}"
    )
    run_ssh(cmd)
    job["status"] = "running"
    job["attempts"] = int(job.get("attempts", 0)) + 1
    job["started_at"] = now()
    job["session"] = session
    job.pop("finished_at", None)
    job.pop("failed_at", None)
    job.pop("failure_reason", None)
    log(f"started remote eval {job['key']} session={session}")


def poll_running_job(state: dict[str, Any]) -> None:
    key = state.get("running")
    if not key:
        return
    job = state["jobs"].get(key)
    if not job:
        state["running"] = None
        return
    done_marker = posixpath.join(job["out_root"], "DONE")
    if remote_file_exists(done_marker):
        complete, reason = remote_progress_complete(job["out_root"])
        if not complete:
            run_ssh(f"rm -f {q(done_marker)}", check=False)
            job["status"] = "failed"
            job["failed_at"] = now()
            job["failure_reason"] = reason
            state["running"] = None
            log(f"failed {key}; DONE present but progress incomplete: {reason}")
            return
        job["status"] = "done"
        job["finished_at"] = now()
        state["running"] = None
        log(f"done {key}")
        return

    session = job.get("session") or session_name(job)
    if run_ssh(f"tmux has-session -t {q(session)}", check=False).returncode == 0:
        age = remote_activity_age_seconds(job["out_root"])
        if age is not None and age > STALE_JOB_SECONDS:
            run_ssh(f"tmux kill-session -t {q(session)} 2>/dev/null || true", check=False)
            cleanup_remote_eval_processes()
            job["status"] = "failed"
            job["failed_at"] = now()
            job["failure_reason"] = f"stale for {age}s"
            state["running"] = None
            log(f"failed {key}; stale for {age}s")
            return
        return

    job["status"] = "failed"
    job["failed_at"] = now()
    state["running"] = None
    log(f"failed {key}; tmux session ended without DONE")


def maybe_add_replan30_for_completed_main(state: dict[str, Any]) -> None:
    jobs = state.setdefault("jobs", {})
    by_variant = {v.name: v for v in VARIANTS}
    for job in list(jobs.values()):
        if job.get("tier") != "main" or job.get("status") != "done":
            continue
        variant = by_variant[job["variant"]]
        add_job(jobs, variant, int(job["step"]), 30, "replan30")


def select_next_job(state: dict[str, Any]) -> dict[str, Any] | None:
    jobs = list(state.get("jobs", {}).values())
    main_scores: dict[str, tuple[float, int, int]] = {}

    def priority(job: dict[str, Any]) -> tuple[str, int, str]:
        variant_rank = 0 if job["variant"] == "tactile" else 1
        return job["first_seen"], int(job["step"]), str(variant_rank)

    def eligible(job: dict[str, Any], tier: str) -> bool:
        if job.get("tier") != tier:
            return False
        if job.get("status") not in {"pending", "failed"}:
            return False
        if int(job.get("attempts", 0)) >= MAX_ATTEMPTS and not remote_has_partial_progress(job["out_root"]):
            return False
        return True

    def is_initial_main(job: dict[str, Any]) -> bool:
        max_step = INITIAL_MAIN_MAX_STEP.get(job["variant"])
        return max_step is not None and int(job["step"]) <= max_step

    def main_score(job: dict[str, Any]) -> tuple[float, int, int]:
        key = job["key"]
        if key in main_scores:
            return main_scores[key]

        latest: dict[str, tuple[int, int]] = {}
        progress = posixpath.join(job["out_root"], "progress.tsv")
        proc = run_ssh(f"cat {q(progress)}", check=False, timeout=30)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 5:
                    continue
                task, succ, total, _rate, _rc = fields
                try:
                    latest[task] = (int(succ), int(total))
                except ValueError:
                    continue

        successes = sum(succ for succ, _total in latest.values())
        total = sum(total for _succ, total in latest.values())
        score = (successes / total if total else -1.0, successes, total)
        main_scores[key] = score
        return score

    def completed_main_job(variant: str, step: int) -> dict[str, Any] | None:
        job = state.get("jobs", {}).get(job_key(variant, step, 10))
        if job and job.get("tier") == "main" and job.get("status") == "done":
            return job
        return None

    best_main_step_by_variant: dict[str, int] = {}
    best_main_score_by_variant: dict[str, tuple[float, int, int, int]] = {}
    for job in jobs:
        if job.get("tier") != "main" or job.get("status") != "done":
            continue
        variant = str(job["variant"])
        step = int(job["step"])
        rate, successes, total = main_score(job)
        rank = (rate, successes, total, step)
        if rank > best_main_score_by_variant.get(variant, (-1.0, -1, -1, -1)):
            best_main_score_by_variant[variant] = rank
            best_main_step_by_variant[variant] = step

    def replan30_priority(job: dict[str, Any]) -> tuple[int, int, float, int, str]:
        variant = str(job["variant"])
        step = int(job["step"])
        variant_rank = 0 if variant == "tactile" else 1
        main_job = completed_main_job(variant, step)
        rate = main_score(main_job)[0] if main_job else -1.0
        is_variant_best = best_main_step_by_variant.get(variant) == step
        # First run the best completed r10 ckpt for each modality, then continue by r10 success rate.
        return (0 if is_variant_best else 1, variant_rank if is_variant_best else 0, -rate, -step, job["first_seen"])

    def completed_replan30_count() -> int:
        return sum(
            1
            for job in jobs
            if job.get("tier") == "replan30" and job.get("status") == "done"
        )

    main = [j for j in jobs if eligible(j, "main")]
    initial_main = [j for j in main if is_initial_main(j)]
    if initial_main:
        return sorted(initial_main, key=priority)[0]

    official = [j for j in jobs if eligible(j, "official")]
    if official:
        return sorted(official, key=priority)[0]

    alt = [j for j in jobs if eligible(j, "replan30")]
    if alt:
        if completed_replan30_count() < REPLAN30_MIN_BEFORE_NEW_MAIN:
            return sorted(alt, key=replan30_priority)[0]
        if not main:
            return sorted(alt, key=replan30_priority)[0]

    if main:
        return sorted(main, key=priority)[0]

    if alt:
        return sorted(alt, key=replan30_priority)[0]
    return None


def select_prefetch_job(state: dict[str, Any]) -> dict[str, Any] | None:
    jobs = list(state.get("jobs", {}).values())

    def priority(job: dict[str, Any]) -> tuple[str, int, str]:
        variant_rank = 0 if job["variant"] == "tactile" else 1
        return job["first_seen"], int(job["step"]), str(variant_rank)

    def eligible(job: dict[str, Any], tier: str) -> bool:
        if job.get("tier") != tier:
            return False
        if job.get("status") not in {"pending", "failed"}:
            return False
        if job.get("synced_at"):
            return False
        if int(job.get("attempts", 0)) >= MAX_ATTEMPTS and not remote_has_partial_progress(job["out_root"]):
            return False
        return True

    for tier in ("main", "official", "replan30"):
        candidates = [j for j in jobs if eligible(j, tier)]
        if candidates:
            return sorted(candidates, key=priority)[0]
    return None


def write_queue_tsv(state: dict[str, Any]) -> None:
    path = LOCAL_RUN_DIR / "queue_state.tsv"
    with path.open("w", encoding="utf-8") as f:
        f.write("key\tstatus\ttier\tvariant\tstep\treplan\tattempts\tout_root\n")
        for job in sorted(state.get("jobs", {}).values(), key=lambda j: (j["tier"], j["first_seen"])):
            f.write(
                "\t".join(
                    [
                        job["key"],
                        job["status"],
                        job["tier"],
                        job["variant"],
                        str(job["step"]),
                        str(job["replan_steps"]),
                        str(job.get("attempts", 0)),
                        job["out_root"],
                    ]
                )
                + "\n"
            )


def main() -> None:
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    log("scheduler starting")
    deploy_remote_script()
    state = load_state()

    while True:
        try:
            write_training_status()
            poll_running_job(state)
            discover_jobs(state)
            ensure_static_remote_jobs(state)
            maybe_add_replan30_for_completed_main(state)
            if not state.get("running"):
                job = select_next_job(state)
                if job:
                    start_remote_eval(job)
                    state["running"] = job["key"]
            else:
                job = select_prefetch_job(state)
                if job:
                    log(f"prefetch pending ckpt {job['key']}")
                    start_sync_ckpt_background(job)
            write_queue_tsv(state)
            save_state(state)
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
            save_state(state)
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
