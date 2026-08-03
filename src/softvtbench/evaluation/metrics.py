"""FEM deformation metric and result aggregation.

Deformation convention = fem_rms_rigid_aligned_bbox_pct_v1: RMS nodal displacement after
Kabsch rigid-body removal / rest bounding-box diagonal x 100. Line-for-line the same
algorithm as the collection-side soft_collection_expert._fem_kabsch_summary_pct (reference
configuration is likewise the episode's frame 0), so evaluation values are directly
comparable to obs/fem_kabsch_rms_pct in the HDF5 data.

**Safe Success is intentionally NOT computed here** (decided 7/26):
  - the tau values in safety_thresholds.json are 3-10x smaller than measured deformation;
    all 2000/2000 collected episodes exceed them
  - D_safe in CARDS_LOCKED is a **contact-patch uniaxial strain**, not the same quantity
    as this convention (global RMS); the measured ratio fluctuates between 1.0-3.4 and
    cannot be applied directly
We record the full per-episode deformation distribution and will backfill the criterion
once the convention is settled.
"""
from __future__ import annotations

METRIC_ID = "fem_rms_rigid_aligned_bbox_pct_v1"


def deformation_stats(series: list[float]) -> dict:
    """Per-episode deformation statistics: peak plus mean/final/percentiles needed for over-threshold counting."""
    if not series:
        return {"d_peak": None, "d_mean": None, "d_final": None, "d_p95": None, "d_n": 0}
    import numpy as np
    a = np.asarray(series, dtype=np.float64)
    return {"d_peak": float(a.max()), "d_mean": float(a.mean()),
            "d_final": float(a[-1]), "d_p95": float(np.percentile(a, 95)), "d_n": int(a.size)}


def get_nodal_pos(env, asset_name: str, idx: int = 0):
    """World-frame nodal positions (N,3) tensor of the deformable asset; None if unavailable."""
    import torch
    scene = getattr(env, "scene", env)
    try:
        obj = scene[asset_name]
        nodes = getattr(getattr(obj, "data", None), "nodal_pos_w", None)
        if nodes is None:
            return None
        if isinstance(nodes, torch.Tensor):
            return nodes[idx].detach().float().clone()
        return torch.as_tensor(nodes[idx], dtype=torch.float32).clone()
    except Exception:
        return None


def fem_deformation_pct(reference, current) -> float | None:
    """RMS displacement after Kabsch translation/rotation removal / reference bbox diagonal x 100. Transcribed from the client."""
    import torch
    if reference is None or current is None or tuple(reference.shape) != tuple(current.shape):
        return None
    try:
        reference = reference.to(device=current.device, dtype=current.dtype)
        ref_c = reference - reference.mean(dim=0, keepdim=True)
        cur_c = current - current.mean(dim=0, keepdim=True)
        cov = cur_c.transpose(0, 1) @ ref_c
        u, _, vh = torch.linalg.svd(cov, full_matrices=False)
        corr = torch.eye(3, device=current.device, dtype=current.dtype)
        corr[-1, -1] = torch.where(torch.linalg.det(u @ vh) < 0,
                                   current.new_tensor(-1.0), current.new_tensor(1.0))
        aligned = cur_c @ (u @ corr @ vh)
        disp = torch.linalg.vector_norm(aligned - ref_c, dim=-1)
        ref_diag = torch.linalg.vector_norm(
            reference.max(dim=0).values - reference.min(dim=0).values).clamp_min(1.0e-8)
        return float((100.0 * torch.sqrt(torch.mean(disp * disp)) / ref_diag).detach().cpu())
    except Exception:
        return None


def summarize(results: list[dict]) -> dict:
    """Aggregate goal success + deformation distribution per task (threshold TBD; report percentiles)."""
    import numpy as np
    by_task: dict[int, dict] = {}
    for r in results:
        t = by_task.setdefault(r["task_id"], {"n": 0, "goal": 0, "peaks": []})
        t["n"] += 1
        t["goal"] += int(r["success"])
        if r.get("d_peak") is not None:
            t["peaks"].append(float(r["d_peak"]))

    def block(peaks: list[float]) -> dict:
        if not peaks:
            return {}
        a = np.asarray(peaks)
        return {"d_peak_min": round(float(a.min()), 3),
                "d_peak_median": round(float(np.median(a)), 3),
                "d_peak_p95": round(float(np.percentile(a, 95)), 3),
                "d_peak_max": round(float(a.max()), 3), "d_peak_n": int(a.size)}

    per_task = {k: {"n": v["n"], "goal": v["goal"], "goal_rate": v["goal"] / v["n"], **block(v["peaks"])}
                for k, v in sorted(by_task.items())}
    total_n = sum(v["n"] for v in by_task.values())
    total_goal = sum(v["goal"] for v in by_task.values())
    all_peaks = [p for v in by_task.values() for p in v["peaks"]]
    return {
        "metric_id": METRIC_ID,
        "per_task": per_task,
        "episodes": total_n,
        "goal_success": total_goal,
        "goal_success_rate": (total_goal / total_n) if total_n else 0.0,
        "deformation": block(all_peaks),
    }
