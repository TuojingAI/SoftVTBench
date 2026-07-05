#!/usr/bin/env python3
# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""Raw data retention analysis for SoftVTBench / SoftVTBench-force replayed_demos datasets.

This script estimates "raw data retention" by counting how many successful episodes
were retained in `replayed_demos/*.hdf5` (each `/data/demo_*` is treated as one success),
then comparing against an expected target count.

Data import convention (strict, the ONLY supported way):
- This script locates datasets via a fixed relative index from the repo root:

    <REPO_ROOT>/benchmarks/datasets/{softvtbench|softvtbench_force}/{firm_force|gentle_force}/replayed_demos

Supported datasets (strict):
- SoftVTBench:       path must contain 'softvtbench' (but NOT 'softvtbench_force')
- SoftVTBench-force: path must contain 'softvtbench_force'
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import h5py
import numpy as np
import tyro

# Ensure project root is in sys.path so `import benchmarks.*` works when running as a script.
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.metrics import (
    compute_contact_force_metrics_from_13d,
    compute_contact_force_metrics_from_lr_forces,
)

DatasetKind = Literal['softvtbench', 'softvtbench_force']


def _infer_dataset_kind(replayed_demos_dir: Path) -> DatasetKind:
    p = str(replayed_demos_dir.resolve())
    if 'softvtbench_force' in p:
        return 'softvtbench_force'
    if 'softvtbench' in p:
        return 'softvtbench'
    raise ValueError(
        '该脚本仅兼容 softvtbench / softvtbench_force 数据集：'
        f'replayed_demos_dir={replayed_demos_dir} 不包含 softvtbench/softvtbench_force 关键字。'
    )


@dataclass
class Config:
    """Raw data retention analysis config."""

    # 数据集类型：如果 repo 下同时存在 softvtbench 和 softvtbench_force，则必须显式指定其一。
    dataset_kind: DatasetKind | None = None

    # 需要统计的 task_suites（SoftVTBench / SoftVTBench-force 均沿用 LIBERO suite 命名）
    task_suites: tuple[str, ...] = ('libero_10', 'libero_spatial', 'libero_goal', 'libero_object')

    # 每个 HDF5 文件理论期望采集的 episode 数（例如 50 条）
    expected_episodes_per_file: int = 50

    # 是否额外统计力学指标（与 run_task_evaluations 的 squeeze/ap 指标定义一致）：
    # - squeeze_mean / squeeze_max
    # - external_norm_mean / external_norm_max  (即 applied force 的模长统计)
    #
    # 说明：
    # - 将在每个 task 的 replayed_demos HDF5 中遍历所有 demo_*（成功 episode），
    #   逐 demo 计算指标，然后在 task 级别做 mean 聚合。
    compute_force_metrics: bool = True

    # 输出目录（默认写到 <REPO_ROOT>/benchmarks/datasets/<dataset_kind>/evaluation_results/）
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Resolve fixed relative paths and validate dataset selection."""
        repo_root = Path(__file__).resolve().parents[2]
        datasets_root = repo_root / 'benchmarks' / 'datasets'
        if not datasets_root.exists():
            raise FileNotFoundError(f'未找到 datasets 根目录: {datasets_root}')

        softvtbench_exists = (datasets_root / 'softvtbench').is_dir()
        softvtbench_force_exists = (datasets_root / 'softvtbench_force').is_dir()

        if self.dataset_kind is None:
            if softvtbench_exists and not softvtbench_force_exists:
                self.dataset_kind = 'softvtbench'
            elif softvtbench_force_exists and not softvtbench_exists:
                self.dataset_kind = 'softvtbench_force'
            elif softvtbench_exists and softvtbench_force_exists:
                raise ValueError(
                    '检测到同时存在 softvtbench 和 softvtbench_force，请通过 --dataset-kind 显式指定其一。'
                )
            else:
                raise FileNotFoundError(
                    f'在 {datasets_root} 下未找到 softvtbench 或 softvtbench_force 目录。'
                )

        if self.output_dir is None or str(self.output_dir) in ('.', ''):
            self.output_dir = datasets_root / str(self.dataset_kind) / 'evaluation_results'
        self.output_dir = self.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)


def get_all_task_hdf5_files(replayed_demos_dir: Path, task_suites: Tuple[str, ...]) -> Dict[str, List[str]]:
    """Aggregate all HDF5 files by (suite_name, original_task_id)."""
    hdf5_dir = replayed_demos_dir
    task_files: Dict[str, List[str]] = {}

    for suite_name in task_suites:
        pattern = f'{suite_name}_task*_*demo.hdf5'
        for path in hdf5_dir.glob(pattern):
            # libero_goal_task1_xxx_demo.hdf5 -> task_id = 1
            filename = path.name
            task_id_str = filename.split('_task')[1].split('_')[0]
            original_task_id = int(task_id_str)
            key = f'{suite_name}_task{original_task_id}'
            task_files.setdefault(key, []).append(str(path))

    return task_files


def _infer_demo_steps(demo_group: h5py.Group) -> int | None:
    """Infer trajectory length (steps) from a demo group.

    Policy:
    - Prefer demo_group['actions'].shape[0]
    - Fallback: find the first dataset under demo_group['obs'] (possibly nested) and use shape[0]
    """
    if 'actions' in demo_group:
        ds = demo_group['actions']
        if isinstance(ds, h5py.Dataset) and ds.shape and len(ds.shape) >= 1:
            return int(ds.shape[0])

    if 'obs' not in demo_group:
        return None
    obs = demo_group['obs']

    def _walk(g: h5py.Group) -> int | None:
        for k in g.keys():
            item = g[k]
            if isinstance(item, h5py.Dataset):
                if item.shape and len(item.shape) >= 1:
                    return int(item.shape[0])
            elif isinstance(item, h5py.Group):
                out = _walk(item)
                if out is not None:
                    return out
        return None

    return _walk(obs) if isinstance(obs, h5py.Group) else None


def _extract_lr_forces_from_demo(demo_group: h5py.Group) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to extract left/right 3D forces from a demo group.

    Priority:
    1) obs/gripper_net_force (shape typically (T,1,2,3) or (T,2,3))
    2) actions (shape (T,13)) -> split by metrics helper
    """
    # 1) obs/gripper_net_force
    try:
        if "obs" in demo_group and isinstance(demo_group["obs"], h5py.Group):
            obs = demo_group["obs"]
            if "gripper_net_force" in obs and isinstance(obs["gripper_net_force"], h5py.Dataset):
                gnf = np.asarray(obs["gripper_net_force"])
                # Expected: (T,1,2,3) or (T,2,3)
                if gnf.ndim == 4 and gnf.shape[1] == 1 and gnf.shape[2:] == (2, 3):
                    gnf = np.squeeze(gnf, axis=1)  # (T,2,3)
                if gnf.ndim == 3 and gnf.shape[1:] == (2, 3):
                    fL = gnf[:, 0, :].astype(np.float32)
                    fR = gnf[:, 1, :].astype(np.float32)
                    return fL, fR
    except Exception:
        pass

    # 2) actions (T,13)
    try:
        if "actions" in demo_group and isinstance(demo_group["actions"], h5py.Dataset):
            actions = np.asarray(demo_group["actions"])
            if actions.ndim == 2 and actions.shape[1] == 13:
                # metrics helper will validate shape; we reuse it by calling compute_* which splits internally.
                # Here we just return None and let caller use compute_contact_force_metrics_from_13d.
                return None
    except Exception:
        pass

    return None


def _compute_demo_force_metrics(demo_group: h5py.Group) -> dict | None:
    """Compute squeeze/applied force metrics for a single demo.

    Returns a dict with:
      squeeze_mean, squeeze_max, applied_mean, applied_max
    or None if metrics cannot be computed.
    """
    # Prefer obs/gripper_net_force if present (works for both softvtbench and softvtbench_force replays).
    lr = _extract_lr_forces_from_demo(demo_group)
    try:
        if lr is not None:
            fL, fR = lr
            m = compute_contact_force_metrics_from_lr_forces(fL, fR)
            return {
                "squeeze_mean": float(m.squeeze_mean),
                "squeeze_max": float(m.squeeze_max),
                "applied_mean": float(m.external_norm_mean),
                "applied_max": float(m.external_norm_max),
            }
    except Exception:
        pass

    # Fallback: actions (T,13)
    try:
        if "actions" in demo_group and isinstance(demo_group["actions"], h5py.Dataset):
            actions = np.asarray(demo_group["actions"])
            if actions.ndim == 2 and actions.shape[1] == 13:
                m = compute_contact_force_metrics_from_13d(actions)
                return {
                    "squeeze_mean": float(m.squeeze_mean),
                    "squeeze_max": float(m.squeeze_max),
                    "applied_mean": float(m.external_norm_mean),
                    "applied_max": float(m.external_norm_max),
                }
    except Exception:
        pass

    return None


def analyze_raw_data_retention(
    replayed_demos_dir: Path,
    task_suites: Tuple[str, ...],
    expected_episodes_per_file: int,
    *,
    label: str | None = None,
    compute_force_metrics: bool = True,
) -> Dict[str, dict]:
    """Compute per-task and overall retention statistics."""
    replayed_demos_dir = replayed_demos_dir.expanduser().resolve()
    if not replayed_demos_dir.exists():
        raise FileNotFoundError(f'replayed_demos_dir 不存在: {replayed_demos_dir}')
    if not replayed_demos_dir.is_dir():
        raise NotADirectoryError(f'replayed_demos_dir 不是目录: {replayed_demos_dir}')

    dataset_kind = _infer_dataset_kind(replayed_demos_dir)
    task_files = get_all_task_hdf5_files(replayed_demos_dir, task_suites)

    results: Dict[str, dict] = {}

    print('\n========== Raw Data Retention (replayed_demos) ==========')
    if label:
        print(f'Category: {label}')
    print(f'Dataset kind: {dataset_kind}')
    print(f'Replayed demos dir: {replayed_demos_dir}')
    print(f'Expected episodes per file: {expected_episodes_per_file}')
    print('========================================================\n')

    for key in sorted(task_files.keys()):
        suite_name, task_token = key.split('_task')
        task_id = int(task_token)
        files = task_files[key]

        successes = 0
        max_steps_success = 0
        # force metrics aggregates (mean over demos)
        force_demo_count = 0
        squeeze_mean_sum = 0.0
        squeeze_max_sum = 0.0
        applied_mean_sum = 0.0
        applied_max_sum = 0.0
        for hdf5_path in files:
            with h5py.File(hdf5_path, 'r') as f:
                data_group = f['data']
                demos = [k for k in data_group.keys() if k.startswith('demo_')]
                successes += len(demos)
                for demo_id in demos:
                    demo_group = data_group[demo_id]
                    steps = _infer_demo_steps(demo_group)
                    if steps is not None:
                        max_steps_success = max(max_steps_success, int(steps))
                    if compute_force_metrics:
                        m = _compute_demo_force_metrics(demo_group)
                        if m is not None:
                            force_demo_count += 1
                            squeeze_mean_sum += float(m["squeeze_mean"])
                            squeeze_max_sum += float(m["squeeze_max"])
                            applied_mean_sum += float(m["applied_mean"])
                            applied_max_sum += float(m["applied_max"])

        expected = expected_episodes_per_file * max(len(files), 1)
        ratio = successes / expected if expected > 0 else 0.0

        results[key] = {
            'dataset_kind': dataset_kind,
            'category': label,
            'task_suite': suite_name,
            'task_id': task_id,
            'num_files': len(files),
            'successes': successes,
            'expected': expected,
            'retention_ratio': ratio,
            'max_steps_success': max_steps_success,
            # Optional force metrics (mean over demos)
            'force_demo_count': force_demo_count,
            'squeeze_mean': (squeeze_mean_sum / force_demo_count) if force_demo_count > 0 else None,
            'squeeze_max': (squeeze_max_sum / force_demo_count) if force_demo_count > 0 else None,
            'applied_mean': (applied_mean_sum / force_demo_count) if force_demo_count > 0 else None,
            'applied_max': (applied_max_sum / force_demo_count) if force_demo_count > 0 else None,
        }

        line = (
            f'{key:24s} | files={len(files):2d} | '
            f'retained={successes:3d}/{expected:3d} | ratio={ratio:5.3f} | '
            f'max_steps={max_steps_success:4d}'
        )
        if compute_force_metrics:
            if force_demo_count > 0:
                line += (
                    f' | sq_mean={squeeze_mean_sum/force_demo_count:6.3f}'
                    f' ap_mean={applied_mean_sum/force_demo_count:6.3f}'
                    f' sq_max={squeeze_max_sum/force_demo_count:6.3f}'
                    f' ap_max={applied_max_sum/force_demo_count:6.3f}'
                    f' (n={force_demo_count})'
                )
            else:
                line += ' | sq_mean=N/A ap_mean=N/A sq_max=N/A ap_max=N/A'
        print(line)

    total_successes = sum(r['successes'] for r in results.values())
    total_expected = sum(r['expected'] for r in results.values())
    overall_ratio = total_successes / total_expected if total_expected > 0 else 0.0
    overall_max_steps = max((r.get('max_steps_success', 0) for r in results.values()), default=0)

    print('\n---------------- Overall ----------------')
    print(f'Total retained={total_successes}/{total_expected} (ratio={overall_ratio:5.3f})')
    print(f'Max steps (success only)={overall_max_steps}')
    print('-----------------------------------------\n')

    return {
        'dataset_kind': dataset_kind,
        'category': label,
        'replayed_demos_dir': str(replayed_demos_dir),
        'per_task': results,
        'overall': {
            'total_successes': total_successes,
            'total_expected': total_expected,
            'overall_retention_ratio': overall_ratio,
            'overall_max_steps_success': overall_max_steps,
        },
    }


def main(cfg: Config) -> None:  # noqa: C901
    """CLI entrypoint."""
    repo_root = Path(__file__).resolve().parents[2]
    datasets_root = repo_root / 'benchmarks' / 'datasets'
    dataset_root = datasets_root / str(cfg.dataset_kind)

    # 统计 firm_force 与 gentle_force 两类（若某类目录不存在则自动跳过）
    all_category_dirs: list[tuple[str, Path]] = [
        ('firm', dataset_root / 'firm_force' / 'replayed_demos'),
        ('gentle', dataset_root / 'gentle_force' / 'replayed_demos'),
    ]
    category_dirs: list[tuple[str, Path]] = []
    for label, replayed_dir in all_category_dirs:
        if replayed_dir.is_dir():
            category_dirs.append((label, replayed_dir))
        else:
            print(f'[WARN] 找不到 {label} replayed_demos_dir，已跳过: {replayed_dir}')

    if not category_dirs:
        raise FileNotFoundError(
            f'在 {dataset_root} 下未找到任何 replayed_demos 目录（firm_force/gentle_force）。'
        )

    results_by_label: dict[str, dict] = {}
    for label, replayed_dir in category_dirs:
        results = analyze_raw_data_retention(
            replayed_dir,
            cfg.task_suites,
            cfg.expected_episodes_per_file,
            label=label,
            compute_force_metrics=cfg.compute_force_metrics,
        )
        results_by_label[label] = results

        dataset_kind: str = results['dataset_kind']
        json_name = f'raw_data_retention_{label}_{dataset_kind}.json'
        txt_name = f'raw_data_retention_{label}_{dataset_kind}.txt'

        # Write per-category outputs
        json_path = cfg.output_dir / json_name
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        txt_path = cfg.output_dir / txt_name
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('Raw Data Retention (replayed_demos)\n')
            if results.get('category'):
                f.write(f'Category: {results["category"]}\n')
            f.write(f'Dataset kind: {results["dataset_kind"]}\n')
            f.write(f'Replayed demos dir: {results["replayed_demos_dir"]}\n')
            f.write(f'Expected episodes per file: {cfg.expected_episodes_per_file}\n\n')
            if cfg.compute_force_metrics:
                f.write('Force metrics (mean over demos; contact-only steps; max = top5% mean):\n')
                f.write('  sq_mean=avg squeeze, ap_mean=avg applied force norm, sq_max=top5% squeeze, ap_max=top5% applied\n\n')

            for key in sorted(results['per_task'].keys()):
                r = results['per_task'][key]
                line = (
                    f'{key:24s} | files={r["num_files"]:2d} | '
                    f'retained={r["successes"]:3d}/{r["expected"]:3d} | '
                    f'ratio={r["retention_ratio"]:5.3f} | '
                    f'max_steps={r.get("max_steps_success", 0):4d}'
                )
                if cfg.compute_force_metrics:
                    if r.get("force_demo_count", 0) > 0:
                        line += (
                            f' | sq_mean={r.get("squeeze_mean"):6.3f}'
                            f' ap_mean={r.get("applied_mean"):6.3f}'
                            f' sq_max={r.get("squeeze_max"):6.3f}'
                            f' ap_max={r.get("applied_max"):6.3f}'
                            f' (n={int(r.get("force_demo_count", 0))})'
                        )
                    else:
                        line += ' | sq_mean=N/A ap_mean=N/A sq_max=N/A ap_max=N/A'
                f.write(line + "\n")

            overall = results['overall']
            f.write('\nOverall:\n')
            f.write(
                f'  total_retained={overall["total_successes"]}/{overall["total_expected"]} '
                f'(ratio={overall["overall_retention_ratio"]:5.3f})\n'
            )
            f.write(f'  max_steps_success={overall.get("overall_max_steps_success", 0)}\n')

        print(f'✔ [{label}] 留存率分析已写入 JSON: {json_path}')
        print(f'✔ [{label}] 留存率分析已写入 TXT:  {txt_path}')

    # --- Cross-category comparisons (firm vs gentle) ---
    if 'firm' not in results_by_label or 'gentle' not in results_by_label:
        return

    firm = results_by_label['firm']
    gentle = results_by_label['gentle']
    firm_tasks: dict[str, dict] = firm.get('per_task', {})
    gentle_tasks: dict[str, dict] = gentle.get('per_task', {})

    common_keys = sorted(set(firm_tasks.keys()) & set(gentle_tasks.keys()))
    only_firm = sorted(set(firm_tasks.keys()) - set(gentle_tasks.keys()))
    only_gentle = sorted(set(gentle_tasks.keys()) - set(firm_tasks.keys()))

    # NOTE: 用户需求：原版筛选 "firm & gentle 都 >= 0.5" 改为 "都 >= 0.2"
    BOTH_RETENTION_THRESHOLD = 0.2
    # NOTE: 用户需求：在 retention 阈值基础上，再要求 task-level squeeze_mean > 5 才计入列表
    SQUEEZE_MEAN_THRESHOLD = 5.0

    both_ge_threshold: list[dict] = []
    cross_50_and_gap: list[dict] = []
    gentle_dominant: list[dict] = []
    firm_dominant: list[dict] = []
    max_gap: dict | None = None
    max_abs_gap = -1.0

    for key in common_keys:
        rf = float(firm_tasks[key].get('retention_ratio', 0.0))
        rg = float(gentle_tasks[key].get('retention_ratio', 0.0))
        sf = firm_tasks[key].get('squeeze_mean')
        sg = gentle_tasks[key].get('squeeze_mean')
        sf_val = float(sf) if sf is not None else None
        sg_val = float(sg) if sg is not None else None
        diff = rg - rf
        absdiff = abs(diff)

        # 规则：(retention_firm >= t) & (retention_gentle >= t) & (sq_mean_firm > 2) & (sq_mean_gentle > 2)
        if (
            rf >= BOTH_RETENTION_THRESHOLD
            and rg >= BOTH_RETENTION_THRESHOLD
            and (sf_val is not None and sf_val > SQUEEZE_MEAN_THRESHOLD)
            and (sg_val is not None and sg_val > SQUEEZE_MEAN_THRESHOLD)
        ):
            both_ge_threshold.append(
                {
                    'task': key,
                    'firm': rf,
                    'gentle': rg,
                    'diff(gentle-firm)': diff,
                    'sq_mean_firm': sf_val,
                    'sq_mean_gentle': sg_val,
                }
            )

        # One side >= 0.5, the other side <= 0.5, and abs gap >= 0.1
        if ((rf >= 0.5 and rg <= 0.5) or (rg >= 0.5 and rf <= 0.5)) and absdiff >= 0.1:
            cross_50_and_gap.append({'task': key, 'firm': rf, 'gentle': rg, 'diff(gentle-firm)': diff, 'abs_diff': absdiff})

        if diff >= 0.1:
            gentle_dominant.append({'task': key, 'firm': rf, 'gentle': rg, 'diff(gentle-firm)': diff})
        elif diff <= -0.1:
            firm_dominant.append({'task': key, 'firm': rf, 'gentle': rg, 'diff(gentle-firm)': diff})

        if absdiff > max_abs_gap:
            max_abs_gap = absdiff
            max_gap = {'task': key, 'firm': rf, 'gentle': rg, 'diff(gentle-firm)': diff, 'abs_diff': absdiff}

    # Prefer more informative ordering
    both_ge_threshold.sort(key=lambda x: min(x['firm'], x['gentle']), reverse=True)
    cross_50_and_gap.sort(key=lambda x: x['abs_diff'], reverse=True)
    gentle_dominant.sort(key=lambda x: x['diff(gentle-firm)'], reverse=True)
    firm_dominant.sort(key=lambda x: x['diff(gentle-firm)'])

    compare = {
        'thresholds': {
            'both_gt': BOTH_RETENTION_THRESHOLD,
            'sq_mean_gt': SQUEEZE_MEAN_THRESHOLD,
            'dominant_gap': 0.1,
        },
        'common_task_count': len(common_keys),
        'only_in_firm': only_firm,
        'only_in_gentle': only_gentle,
        # Backward compatible key name (historical): keep it but now uses BOTH_RETENTION_THRESHOLD.
        'both_retention_gt_0_5': both_ge_threshold,
        'cross_0_5_and_gap_ge_0_1': cross_50_and_gap,
        'max_gap_task': max_gap,
        'gentle_dominant_tasks': gentle_dominant,
        'firm_dominant_tasks': firm_dominant,
    }

    dataset_kind = str(cfg.dataset_kind)
    print('\n========== Cross-category Compare (firm vs gentle) ==========')
    print(f'Dataset: {dataset_kind}')
    print(f'Common tasks: {compare["common_task_count"]}')
    if only_firm:
        print(f'[WARN] Only in firm: {len(only_firm)}')
    if only_gentle:
        print(f'[WARN] Only in gentle: {len(only_gentle)}')

    print(
        f'\n(1) firm & gentle retention >= {BOTH_RETENTION_THRESHOLD} '
        f'AND sq_mean > {SQUEEZE_MEAN_THRESHOLD}:'
    )
    if both_ge_threshold:
        for item in both_ge_threshold:
            print(
                f'  {item["task"]:24s} | firm={item["firm"]:.3f} | gentle={item["gentle"]:.3f} | '
                f'diff={item["diff(gentle-firm)"]:+.3f} | '
                f'sq_mean(firm)={item.get("sq_mean_firm", float("nan")):.3f} '
                f'sq_mean(gentle)={item.get("sq_mean_gentle", float("nan")):.3f}'
            )
    else:
        print('  (none)')

    print('\n(2) max retention gap task:')
    if max_gap is not None:
        print(
            f'  {max_gap["task"]:24s} | firm={max_gap["firm"]:.3f} | gentle={max_gap["gentle"]:.3f} | '
            f'diff={max_gap["diff(gentle-firm)"]:+.3f} | abs={max_gap["abs_diff"]:.3f}'
        )
    else:
        print('  (none)')

    print('\n(3) cross 0.5 threshold & gap >= 0.1:')
    if cross_50_and_gap:
        for item in cross_50_and_gap:
            print(
                f'  {item["task"]:24s} | firm={item["firm"]:.3f} | gentle={item["gentle"]:.3f} | '
                f'diff={item["diff(gentle-firm)"]:+.3f} | abs={item["abs_diff"]:.3f}'
            )
    else:
        print('  (none)')

    print('\n(4) dominant tasks (gap >= 0.1):')
    print(f'  gentle dominant: {len(gentle_dominant)}')
    for item in gentle_dominant:
        print(
            f'    {item["task"]:24s} | firm={item["firm"]:.3f} | gentle={item["gentle"]:.3f} | '
            f'diff={item["diff(gentle-firm)"]:+.3f}'
        )
    print(f'  firm dominant:   {len(firm_dominant)}')
    for item in firm_dominant:
        print(
            f'    {item["task"]:24s} | firm={item["firm"]:.3f} | gentle={item["gentle"]:.3f} | '
            f'diff={item["diff(gentle-firm)"]:+.3f}'
        )
    print('============================================================\n')

    compare_results = {
        'dataset_kind': dataset_kind,
        'compare': compare,
    }

    compare_json = cfg.output_dir / f'raw_data_retention_compare_{dataset_kind}.json'
    with open(compare_json, 'w', encoding='utf-8') as f:
        json.dump(compare_results, f, indent=2, ensure_ascii=False)

    compare_txt = cfg.output_dir / f'raw_data_retention_compare_{dataset_kind}.txt'
    with open(compare_txt, 'w', encoding='utf-8') as f:
        f.write('Cross-category Compare (firm vs gentle)\n')
        f.write(f'Dataset: {dataset_kind}\n')
        f.write(f'Common tasks: {compare["common_task_count"]}\n')
        f.write(f'Only in firm: {len(only_firm)}\n')
        f.write(f'Only in gentle: {len(only_gentle)}\n')
        f.write(
            f'\n(1) firm & gentle retention >= {BOTH_RETENTION_THRESHOLD} '
            f'AND sq_mean > {SQUEEZE_MEAN_THRESHOLD}:\n'
        )
        if both_ge_threshold:
            for item in both_ge_threshold:
                f.write(
                    f'  {item["task"]:24s} | firm={item["firm"]:.3f} | gentle={item["gentle"]:.3f} | '
                    f'diff={item["diff(gentle-firm)"]:+.3f} | '
                    f'sq_mean(firm)={item.get("sq_mean_firm", float("nan")):.3f} '
                    f'sq_mean(gentle)={item.get("sq_mean_gentle", float("nan")):.3f}\n'
                )
        else:
            f.write('  (none)\n')
        f.write('\n(2) max retention gap task:\n')
        if max_gap is not None:
            f.write(
                f'  {max_gap["task"]:24s} | firm={max_gap["firm"]:.3f} | gentle={max_gap["gentle"]:.3f} | '
                f'diff={max_gap["diff(gentle-firm)"]:+.3f} | abs={max_gap["abs_diff"]:.3f}\n'
            )
        else:
            f.write('  (none)\n')

        f.write('\n(3) cross 0.5 threshold & gap >= 0.1:\n')
        if cross_50_and_gap:
            for item in cross_50_and_gap:
                f.write(
                    f'  {item["task"]:24s} | firm={item["firm"]:.3f} | gentle={item["gentle"]:.3f} | '
                    f'diff={item["diff(gentle-firm)"]:+.3f} | abs={item["abs_diff"]:.3f}\n'
                )
        else:
            f.write('  (none)\n')

        f.write('\n(4) dominant tasks (gap >= 0.1):\n')
        f.write(f'  gentle dominant: {len(gentle_dominant)}\n')
        for item in gentle_dominant:
            f.write(
                f'    {item["task"]:24s} | firm={item["firm"]:.3f} | gentle={item["gentle"]:.3f} | '
                f'diff={item["diff(gentle-firm)"]:+.3f}\n'
            )
        f.write(f'  firm dominant:   {len(firm_dominant)}\n')
        for item in firm_dominant:
            f.write(
                f'    {item["task"]:24s} | firm={item["firm"]:.3f} | gentle={item["gentle"]:.3f} | '
                f'diff={item["diff(gentle-firm)"]:+.3f}\n'
            )

    print(f'✔ [compare] 对比结果已写入 JSON: {compare_json}')
    print(f'✔ [compare] 对比结果已写入 TXT:  {compare_txt}')


if __name__ == '__main__':
    main(tyro.cli(Config))
