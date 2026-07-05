#!/usr/bin/env python3
"""SoftVTBench v1 metrics: Goal Success and Safe Success.

Implements the protocol of Section 3.4 of the SoftVTBench paper, identically for
the object-soft and spatial-soft suites.

    D(t)    = obs/fem_deformation_rms
              object-size-normalised FEM-RMS nodal displacement after removing
              global rigid-body motion, as a percentage of the reference
              bounding-box diagonal.

    D_peak  = max_t D(t)                                                 (Eq. 3)

    Safe Success = Goal Success  AND  (D_peak <= tau_o)

    Goal Success Rate   = (1/N) sum_i GoalSuccess^(i)                    (Eq. 5)
    Safe Success Rate = (1/N) sum_i SafeSuccess^(i)

`tau_o` is the calibrated, object-specific compression-sweep threshold from
`configs/safety_thresholds.json` (see calibrate_safety_thresholds.py). SoftVTBench
v1 reports no separate NoDrop metric; the public metrics are Goal Success and
Safe Success only.

For the rigid control suites the deformation term is inactive; pass --rigid to
score Safe Success as Goal Success alone for diagnostics.

Two input shapes are accepted, both discovered by recursive glob:

    *.jsonl   policy_action_debug.jsonl written by the evaluation client
              (one record per logged frame; online policy rollouts)
    *.hdf5    replayed_demos/*.hdf5 from the released dataset
              (expert demonstrations; the reference distribution)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .deformation import METRIC_ID, deformation_series
except ImportError:  # direct script execution
    from deformation import METRIC_ID, deformation_series

RMS_KEYS = ("obs/fem_deformation_rms", "soft_extras/fem_deformation_rms")


@dataclass
class Episode:
    source: str          # "jsonl" | "hdf5"
    path: str
    task: str
    episode: str
    asset: str
    goal_success: bool | None
    d_peak: float | None
    tau: float | None
    frames: int
    reason: str

    @property
    def safety_known(self) -> bool:
        if self.goal_success is None:
            return False
        if self.tau is None or self.d_peak is None:
            return False
        return True

    @property
    def deform_ok(self) -> bool | None:
        if self.d_peak is None or self.tau is None:
            return None
        return self.d_peak <= self.tau

    def safe_success(self, rigid: bool) -> bool | None:
        if self.goal_success is None:
            return None
        if not self.goal_success:
            return False
        if rigid:
            return True
        if self.deform_ok is None:
            return None
        return bool(self.deform_ok)


# --------------------------------------------------------------------- loading
def _finite_max(values: Iterable[Any]) -> float | None:
    out = [float(v) for v in values
           if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
    return max(out) if out else None


def _task_from_path(path: Path) -> str:
    for part in path.parts[::-1]:
        m = re.search(r"(libero_\w+?_task\d+)", part)
        if m:
            return m.group(1)
    return path.parent.name


def _episode_from_jsonl(path: Path, taus: dict[str, float]) -> Episode | None:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return None
    if not rows:
        return None
    # Collection manifests are JSONL too; eval debug rows carry frame/exp_idx/task_id.
    if not any("frame" in r and "exp_idx" in r and "task_id" in r for r in rows):
        return None

    metric_ids = {
        str(r["fem_deformation_metric_id"])
        for r in rows
        if r.get("fem_deformation_known") and r.get("fem_deformation_metric_id")
    }
    metric_ok = metric_ids == {METRIC_ID}
    d_peak = _finite_max(r.get("fem_deformation_rms") for r in rows) if metric_ok else None
    assets = [str(r.get("fem_deformation_asset")) for r in rows if r.get("fem_deformation_asset")]
    asset = assets[-1] if assets else ""

    successes = [r.get("success_now") for r in rows if r.get("success_now") is not None]
    goal = bool(successes[-1]) if successes else None

    reasons = []
    if goal is None:
        reasons.append("missing_success_now")
    if d_peak is None:
        reasons.append("missing_fem_deformation_rms")
    if not metric_ids:
        reasons.append("missing_fem_deformation_metric_id")
    elif not metric_ok:
        reasons.append(f"metric_id_mismatch:{sorted(metric_ids)}")
    if asset and asset not in taus:
        reasons.append(f"missing_tau:{asset}")
    if not asset:
        reasons.append("missing_asset")

    return Episode(
        source="jsonl", path=str(path),
        task=str(rows[-1].get("task_id", _task_from_path(path))),
        episode=str(rows[-1].get("exp_idx", path.stem)),
        asset=asset, goal_success=goal, d_peak=d_peak,
        tau=taus.get(asset), frames=len(rows),
        reason=";".join(reasons),
    )


def _nodal_deformation_series(demo: Any, asset: str) -> np.ndarray | None:
    if "states/deformable_object" not in demo:
        return None
    group = demo["states/deformable_object"]
    if asset and asset in group:
        asset_group = group[asset]
    else:
        names = sorted(group.keys())
        if not names:
            return None
        asset_group = group[names[0]]
    if "nodal_pos_w" not in asset_group:
        return None
    nodes = np.asarray(asset_group["nodal_pos_w"], dtype=np.float64)
    if nodes.ndim != 3 or nodes.shape[0] == 0:
        return None
    rms, _ = deformation_series(nodes[0], nodes)
    return rms


def _episodes_from_hdf5(path: Path, taus: dict[str, float]):
    import h5py
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return
        for demo_key in f["data"]:
            demo = f["data"][demo_key]
            asset = str(demo.attrs.get("asset_name", "")).strip()
            # Recompute from FEM nodes so legacy raw-displacement summary fields
            # cannot be mixed with the v1 normalized online metric.
            series = _nodal_deformation_series(demo, asset)
            success = demo.attrs.get("success")
            goal = None if success is None else bool(success)
            reasons = []
            if goal is None:
                reasons.append("missing_success")
            if series is None:
                reasons.append("missing_fem_deformation_rms")
            if asset and asset not in taus:
                reasons.append(f"missing_tau:{asset}")
            if not asset:
                reasons.append("missing_asset")
            yield Episode(
                source="hdf5", path=str(path), task=_task_from_path(path), episode=demo_key,
                asset=asset, goal_success=goal,
                d_peak=None if series is None else float(series.max()),
                tau=taus.get(asset), frames=0 if series is None else int(series.size),
                reason=";".join(reasons),
            )


def collect(root: Path, taus: dict[str, float]) -> list[Episode]:
    episodes: list[Episode] = []
    for p in sorted(root.rglob("*.jsonl")):
        ep = _episode_from_jsonl(p, taus)
        if ep is not None:
            episodes.append(ep)
    for p in sorted(root.rglob("*.hdf5")):
        episodes.extend(_episodes_from_hdf5(p, taus))
    return episodes


# ------------------------------------------------------------------ summarising
def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else 100.0 * num / den


def summarize(eps: list[Episode], rigid: bool) -> dict:
    n = len(eps)
    goal_known = [e for e in eps if e.goal_success is not None]
    n_goal = sum(1 for e in goal_known if e.goal_success)

    safe_vals = [e.safe_success(rigid) for e in eps]
    n_unknown = sum(1 for v in safe_vals if v is None)
    n_safe = sum(1 for v in safe_vals if v is True)

    gsr = _rate(n_goal, len(goal_known))
    # Episodes with unknown safety are counted as not safety-successful: missing
    # evidence is not evidence of safety. n_safe_unknown is reported alongside.
    ssr = _rate(n_safe, n)

    by_asset: dict[str, dict] = {}
    for e in eps:
        b = by_asset.setdefault(e.asset or "?", {"n": 0, "goal": 0, "safe": 0, "d_peaks": [], "tau": e.tau})
        b["n"] += 1
        if e.goal_success:
            b["goal"] += 1
        if e.safe_success(rigid) is True:
            b["safe"] += 1
        if e.d_peak is not None:
            b["d_peaks"].append(e.d_peak)
    for b in by_asset.values():
        dp = b.pop("d_peaks")
        b["d_peak_p50"] = round(float(np.percentile(dp, 50)), 4) if dp else None
        b["d_peak_max"] = round(float(max(dp)), 4) if dp else None
        b["goal_success_rate"] = _rate(b["goal"], b["n"])
        b["safe_success_rate"] = _rate(b["safe"], b["n"])

    return {
        "n_episodes": n,
        "n_goal_known": len(goal_known),
        "n_safe_unknown": n_unknown,
        "goal_success_rate": gsr,
        "safe_success_rate": ssr,
        "goal_safe_gap": None if (gsr is None or ssr is None) else round(gsr - ssr, 2),
        "rigid_mode": rigid,
        "deformation_metric_id": METRIC_ID,
        "by_asset": by_asset,
    }


def write_outputs(eps: list[Episode], summary: dict, out_dir: Path, rigid: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "task", "episode", "asset", "goal_success",
                    "d_peak", "tau", "deform_ok", "safe_success", "frames", "reason"])
        for e in eps:
            w.writerow([e.source, e.task, e.episode, e.asset, e.goal_success,
                        "" if e.d_peak is None else f"{e.d_peak:.6f}",
                        "" if e.tau is None else f"{e.tau:.6f}",
                        e.deform_ok, e.safe_success(rigid), e.frames, e.reason])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="directory to scan for *.jsonl (online) and *.hdf5 (reference)")
    ap.add_argument("--thresholds", type=Path, required=True, help="configs/safety_thresholds.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rigid", action="store_true",
                    help="rigid control suite: deformation term inactive, Safe Success = Goal Success")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any episode lacks Goal Success or safety evidence")
    args = ap.parse_args()

    threshold_payload = json.loads(args.thresholds.read_text())
    calibration = threshold_payload.get("calibration", {})
    if calibration.get("method") != "compression_sweep":
        raise SystemExit(
            f"{args.thresholds} is not a SoftVTBench v1 compression-sweep threshold file"
        )
    if threshold_payload.get("metric_id") != METRIC_ID:
        raise SystemExit(
            f"{args.thresholds} metric_id must be {METRIC_ID!r}, "
            f"got {threshold_payload.get('metric_id')!r}"
        )
    taus_raw = threshold_payload["thresholds"]
    taus = {k: float(v["tau"] if isinstance(v, dict) else v) for k, v in taus_raw.items()}

    eps = collect(args.root, taus)
    if not eps:
        raise SystemExit(f"no episodes found under {args.root}")

    summary = summarize(eps, args.rigid)
    write_outputs(eps, summary, args.output_dir, args.rigid)

    if args.strict and (
        summary["n_goal_known"] != summary["n_episodes"]
        or summary["n_safe_unknown"] != 0
    ):
        raise SystemExit(
            "strict metric validation failed: "
            f"goal_known={summary['n_goal_known']}/{summary['n_episodes']}, "
            f"safe_unknown={summary['n_safe_unknown']}"
        )

    gsr, ssr = summary["goal_success_rate"], summary["safe_success_rate"]
    print(f"root                 {args.root}")
    print(f"episodes             {summary['n_episodes']}")
    print(f"Goal Success Rate    {'n/a' if gsr is None else f'{gsr:.1f}%'}")
    print(f"Safe Success Rate    {'n/a' if ssr is None else f'{ssr:.1f}%'}"
          f"{'  (rigid: = goal)' if args.rigid else ''}")
    print(f"Goal-Safe Gap        {summary['goal_safe_gap']}")
    if summary["n_safe_unknown"]:
        print(f"WARNING: {summary['n_safe_unknown']} episode(s) lack the evidence to score safety; "
              f"they are counted as not safe-successful.")
    print()
    print(f"{'asset':<24}{'n':>5}{'tau':>8}{'Goal%':>8}{'Safe%':>8}{'D_peak p50':>12}{'D_peak max':>12}")
    print("-" * 77)
    for asset, b in sorted(summary["by_asset"].items()):
        tau = "" if b["tau"] is None else f"{b['tau']:.3f}"
        print(f"{asset:<24}{b['n']:>5}{tau:>8}"
              f"{b['goal_success_rate']:>8.1f}{b['safe_success_rate']:>8.1f}"
              f"{(b['d_peak_p50'] if b['d_peak_p50'] is not None else float('nan')):>12.4f}"
              f"{(b['d_peak_max'] if b['d_peak_max'] is not None else float('nan')):>12.4f}")
    print(f"\nwrote {args.output_dir}/episodes.csv and summary.json")


if __name__ == "__main__":
    main()
