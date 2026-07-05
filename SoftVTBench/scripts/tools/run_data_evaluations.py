#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate replay data quality by testing success rates across multiple episodes.

This script systematically replays demonstration data and measures success rates
to validate data quality and environment consistency.

Usage:
    # Evaluate all task suites (default, Hybrid env)
    python scripts/tools/run_data_evaluations.py
    
    # Evaluate specific task suites
    python scripts/tools/run_data_evaluations.py --task_suites libero_goal libero_object
    
    # Evaluate specific tasks
    python scripts/tools/run_data_evaluations.py --task_suites libero_goal --task_ids 0 1 2
    
    # Use task space environment (IK)
    python scripts/tools/run_data_evaluations.py --task Isaac-Libero-Franka-IK-v0
    
    # Use task space environment (OSC)
    python scripts/tools/run_data_evaluations.py --task Isaac-Libero-Franka-OscPose-v0
    
    # Use Hybrid + tactile environment explicitly
    python scripts/tools/run_data_evaluations.py --task Isaac-Libero-Franka-Hybrid-Tactile-v0
    
    # Available environments:
    # - Joint Space: Isaac-Libero-Franka-Replay-Camera-v0
    # - Task Space: Isaac-Libero-Franka-IK-v0, Isaac-Libero-Franka-OscPose-v0
    # - Hybrid: Isaac-Libero-Franka-Hybrid-ContactForce-v0, Isaac-Libero-Franka-Hybrid-Tactile-v0
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import tyro

# 确保项目根目录在 sys.path 中，使得在不同运行方式下都能导入 `scripts.tools`。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Task suite to task count mapping（原版 Libero：每个 suite 默认 10 个任务）
TASK_SUITE_CONFIGS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
}


def _load_softvtbench_task_subset(workspace_root: Path) -> dict[str, list[int]]:
    """Load SoftVTBench task subset mapping from JSON."""
    path = workspace_root / "benchmarks" / "datasets" / "softvtbench" / "config" / "softvtbench_tasks.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        out: dict[str, list[int]] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, list):
                    out[k] = [int(x) for x in v]
        return out
    except Exception:
        return {}


def get_task_name_and_instruction(task_suite: str, task_id: int, config_path: Path) -> tuple[str, str]:
    """
    Get task name and language instruction from config file.
    
    Args:
        task_suite: Task suite name (e.g., "libero_goal")
        task_id: Task ID (0-indexed)
        config_path: Path to config directory
        
    Returns:
        Tuple of (task_name, language_instruction)
    """
    config_file = config_path / f"{task_suite}.json"
    
    if not config_file.exists():
        return f"task_{task_id}", "N/A"
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        tasks = config.get("tasks", [])
        if task_id < len(tasks):
            task_info = tasks[task_id]
            return task_info.get("name", f"task_{task_id}"), task_info.get("language", "N/A")
        
        return f"task_{task_id}", "N/A"
    except Exception as e:
        print(f"Warning: Failed to load config from {config_file}: {e}")
        return f"task_{task_id}", "N/A"


@dataclass
class DataReplayEvalConfig:
    """Configuration for data replay evaluation."""
    
    # Task selection
    task_suites: tuple[str, ...] = ()  # Specific task suites to evaluate (empty = all suites)
    task_ids: tuple[int, ...] = ()  # Specific task IDs to evaluate (empty = all IDs in selected suites)
    
    # Environment selection
    # Available environments:
    # - Joint Space: Isaac-Libero-Franka-Replay-Camera-v0
    # - Task Space: Isaac-Libero-Franka-IK-v0, Isaac-Libero-Franka-OscPose-v0
    # - Hybrid: Isaac-Libero-Franka-Hybrid-ContactForce-v0, Isaac-Libero-Franka-Hybrid-Tactile-v0
    # 默认留空：由 control_mode 自动选择 Hybrid-ContactForce / Hybrid-Tactile 环境：
    #   * diffik  -> Isaac-Libero-Franka-IK-v0
    #   * osc     -> Isaac-Libero-Franka-OscPose-v0
    #   * hybrid  -> Isaac-Libero-Franka-Hybrid-ContactForce-v0
    #   * tactile -> Isaac-Libero-Franka-Hybrid-Tactile-v0
    # 如需自定义环境，可通过 CLI 显式传入 `--task xxx`。
    task: str = ""
    
    # Data source
    # NOTE:
    # Data import is exclusively controlled via the environment variable:
    #   REPLAYED_DEMOS_DIR=<dir containing {task_suite}_task{task_id}_*_demo.hdf5>
    # The script will select the matching HDF5 for each task and pass it to the replay script
    # via --dataset_file (explicit file path, no implicit env-based selection in child process).
    
    # Evaluation parameters
    max_episodes: int = 50  # Maximum number of episodes to evaluate per task
    num_envs: int = 1  # Number of parallel environments
    
    # Output configuration
    output_dir: Path = Path("./evaluation_results")
    output_format: str = "both"  # Output format: "json", "txt", or "both"
    
    # Simulation parameters
    headless: bool = True
    validate_states: bool = False  # Validate state consistency
    
    # Task configuration
    task_suite: Optional[str] = None  # For compatibility with setup_task_objects
    task_id: Optional[int] = None  # For compatibility with setup_task_objects
    config_path: Optional[Path] = None  # Path to task config files
    
    # 控制模式关键字（与 run_task_evaluations.py / OpenPI 保持一致）：
    # - "diffik"  -> Isaac-Libero-Franka-IK-v0
    # - "osc"     -> Isaac-Libero-Franka-OscPose-v0
    # - "hybrid"  -> Isaac-Libero-Franka-Hybrid-ContactForce-v0
    # - "tactile" -> Isaac-Libero-Franka-Hybrid-Tactile-v0
    # 仅在 task 为空字符串时生效；若显式传入 --task，则以 task 为准。
    control_mode: str = "tactile"
    # Script path: by default use the lightweight pure-replay script
    # (replay_demos_with_camera.py can still be used by overriding this field)
    replay_script: str = "scripts/tools/replay_demos.py"


def run_single_replay_evaluation(
    config: DataReplayEvalConfig,
    task_suite: str,
    task_id: int,
    workspace_root: Path,
) -> dict:
    """Run a single replay evaluation and return results."""
    
    # Get task information
    config_base_path = config.config_path or (workspace_root / "benchmarks/datasets/libero/config")
    task_name, language_instruction = get_task_name_and_instruction(task_suite, task_id, config_base_path)
    
    # Auto-detect space type from environment name
    is_task_space_env = "IK" in config.task or "Osc" in config.task
    space_type = "Task Space" if is_task_space_env else "Joint Space"

    # Strict: resolve per-task HDF5 from REPLAYED_DEMOS_DIR using the shared resolver.
    from common.replay_utils import resolve_hdf5_from_replayed_demos_dir

    demos_dir_str = os.environ.get('REPLAYED_DEMOS_DIR', '').strip()
    if not demos_dir_str:
        return {
            'task_suite': task_suite,
            'task_id': task_id,
            'task_name': task_name,
            'language_instruction': language_instruction,
            'data_source': None,
            'error': 'Missing env var: REPLAYED_DEMOS_DIR',
        }

    demos_dir = Path(demos_dir_str).expanduser().resolve()
    lower = str(demos_dir).lower()
    dataset_kind = (
        'softvtbench_force'
        if 'softvtbench_force' in lower
        else 'softvtbench'
        if 'softvtbench' in lower
        else 'libero'
        if 'libero' in lower
        else 'unknown'
    )

    try:
        custom_dataset_file = Path(resolve_hdf5_from_replayed_demos_dir(task_suite, task_id))
    except Exception as e:
        return {
            'task_suite': task_suite,
            'task_id': task_id,
            'task_name': task_name,
            'language_instruction': language_instruction,
            'data_source': None,
            'error': str(e),
        }

    data_source = f'{dataset_kind}:{custom_dataset_file.name}'

    print(f"\n{'='*80}")
    print(f"Evaluating: {task_suite} - Task {task_id} ({task_name})")
    print(f"Environment: {config.task} ({space_type})")
    print(f"Data Source: {data_source}")
    print(f"{'='*80}")
    
    # Build command
    cmd = [
        sys.executable,
        config.replay_script,
        "--task", config.task,
        "--task_suite", task_suite,
        "--task_id", str(task_id),
        "--dataset_file", str(custom_dataset_file),
        "--num_envs", str(config.num_envs),
        "--enable_cameras",
    ]
    
    # Always use replayed_demos (task-space) for this evaluation path (provided via env in parent process).
    
    # Add headless flag
    if config.headless:
        cmd.append("--headless")
    else:
        cmd.append("--no-headless")
    
    # Add state validation flag
    if config.validate_states and config.num_envs == 1:
        cmd.append("--validate_states")
    
    # Print command (verbose mode could be added later)
    # print(f"Running command: {' '.join(cmd)}")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
    
    # Run the evaluation
    start_time = time.time()
    try:
        # Real-time output with parsing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=workspace_root,
            env=env,
        )
        
        # Variables to capture metrics
        total_episodes = 0
        successful_episodes = 0
        failed_episodes = 0
        failed_demo_ids = []
        output_lines = []
        avg_squeeze_pred = None
        avg_squeeze_meas = None
        # Hybrid 力学 metrics（task 级别，由 replay_demos* 打印的汇总行提供）
        task_squeeze_max_mean = None
        task_squeeze_mean_mean = None
        task_app_max_mean = None
        task_app_mean_mean = None
        task_squeeze_max_meas_mean = None
        task_ap_max_meas_mean = None
        task_ap_mean_meas_mean = None
        
        # Read output line by line (output will be printed in real-time)
        import re
        keyval_pattern = re.compile(
            r"([a-zA-Z0-9_]+)\s*=\s*([+-]?(?:\d+\.?\d*|\d*\.?\d+))(?:[eE][+-]?\d+)?"
        )
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')  # Real-time output
                output_lines.append(line)
                
                # Parse total replayed episodes from "Finished replaying X episode(s)."
                if "Finished replaying" in line and "episode" in line:
                    try:
                        match = re.search(r'Finished replaying (\d+) episode', line)
                        if match:
                            total_episodes = int(match.group(1))
                    except (ValueError, AttributeError):
                        pass
                
                # Also count successful replays from "Successfully replayed X episode(s) out of Y demos."
                if "Successfully replayed" in line and "out of" in line:
                    try:
                        match = re.search(r'Successfully replayed (\d+) episode', line)
                        if match:
                            successful_episodes = int(match.group(1))
                    except (ValueError, AttributeError):
                        pass
                
                # Capture failed demo IDs from "Failed demo IDs written to failure.jsonl: task_name: [id1, id2, ...]"
                if "Failed demo IDs written to" in line:
                    try:
                        # Extract list after the colon following task name
                        # Format: "Failed demo IDs written to failure.jsonl: libero_goal_task0: [0, 1, 2, ...]"
                        if ":" in line:
                            parts = line.split(":")
                            if len(parts) >= 3:
                                ids_part = parts[-1].strip()
                                if ids_part.startswith("[") and ids_part.endswith("]"):
                                    ids_str = ids_part.strip("[]")
                                    if ids_str:
                                        failed_demo_ids = [int(x.strip()) for x in ids_str.split(",")]
                    except (ValueError, IndexError):
                        pass

                # Parse Hybrid task-level metrics from统一的 Hybrid-Metrics 行
                if "[Hybrid-Metrics] Task contact_metrics" in line:
                    try:
                        # 提取 "Task contact_metrics" 之后到 "over" 之前的部分
                        fragments = line.split("Task contact_metrics", 1)[1]
                        if "over" in fragments:
                            fragments = fragments.split("over", 1)[0]
                        for m in keyval_pattern.finditer(fragments):
                            key, val = m.group(1), m.group(2)
                            try:
                                v = float(val)
                            except ValueError:
                                continue
                            if key == "squeeze_avg_pred":
                                avg_squeeze_pred = v
                            elif key == "squeeze_avg_meas":
                                avg_squeeze_meas = v
                            elif key == "squeeze_max_mean":
                                task_squeeze_max_mean = v
                            elif key == "squeeze_mean_mean":
                                task_squeeze_mean_mean = v
                            elif key == "app_max_mean":
                                task_app_max_mean = v
                            elif key == "app_mean_mean":
                                task_app_mean_mean = v
                            elif key == "squeeze_max_meas_mean":
                                task_squeeze_max_meas_mean = v
                            elif key == "ap_max_meas_mean":
                                task_ap_max_meas_mean = v
                            elif key == "ap_mean_meas_mean":
                                task_ap_mean_meas_mean = v
                    except Exception:
                        pass
        
        # Wait for process to complete
        return_code = process.wait()
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        # If we captured "Successfully replayed X", use that
        # Otherwise calculate: Success = Total replayed - Failed
        if total_episodes > 0:
            failed_episodes = len(failed_demo_ids)
            if successful_episodes == 0:  # Not captured from "Successfully replayed"
                successful_episodes = total_episodes - failed_episodes
            success_rate = successful_episodes / total_episodes
        else:
            success_rate = 0.0
            failed_episodes = 0
            successful_episodes = 0
        
        # Prepare result
        result = {
            "task_suite": task_suite,
            "task_id": task_id,
            "task_name": task_name,
            "language_instruction": language_instruction,
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "success_rate": success_rate,
            "failed_demo_ids": failed_demo_ids,
            "elapsed_time_seconds": elapsed_time,
            "return_code": return_code,
            "status": "completed" if return_code == 0 else "failed",
            # Hybrid 下可选：成功 episode 的平均挤压力（可能为 None）
            "avg_squeeze_pred": avg_squeeze_pred,
            "avg_squeeze_meas": avg_squeeze_meas,
            # Hybrid 下可选：task 级别的挤压力 / 加持力 metrics（可能为 None）
            "task_squeeze_max_mean": task_squeeze_max_mean,
            "task_squeeze_mean_mean": task_squeeze_mean_mean,
            "task_app_max_mean": task_app_max_mean,
            "task_app_mean_mean": task_app_mean_mean,
            "task_squeeze_max_meas_mean": task_squeeze_max_meas_mean,
            "task_ap_max_meas_mean": task_ap_max_meas_mean,
            "task_ap_mean_meas_mean": task_ap_mean_meas_mean,
        }
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Summary: {task_suite} Task {task_id} ({task_name})")
        print(f"{'='*80}")
        print(f"Episodes: {successful_episodes}/{total_episodes} successful ({success_rate:.2%})")
        if failed_demo_ids:
            print(f"Failed IDs: {failed_demo_ids}")
        print(f"Time: {elapsed_time:.2f}s")
        print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Error during replay evaluation: {e}")
        return {
            "task_suite": task_suite,
            "task_id": task_id,
            "task_name": task_name,
            "language_instruction": language_instruction,
            "error": str(e),
            "elapsed_time_seconds": elapsed_time,
            "status": "error",
        }


def save_results(results: list[dict], config: DataReplayEvalConfig):
    """Save evaluation results to file(s)."""
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demos_dir_str = os.environ.get("REPLAYED_DEMOS_DIR", "").strip()
    lower = demos_dir_str.lower()
    data_type = "softvtbench_force" if "softvtbench_force" in lower else "softvtbench" if "softvtbench" in lower else "libero" if "libero" in lower else "unknown"
    base_filename = f"replay_eval_{data_type}_{timestamp}"
    
    # Save JSON format
    if config.output_format in ["json", "both"]:
        json_path = config.output_dir / f"{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "config": {
                    "data_type": data_type,
                    "REPLAYED_DEMOS_DIR": demos_dir_str,
                    "task_env": config.task,
                    "num_envs": config.num_envs,
                    "max_episodes": config.max_episodes,
                },
                "results": results,
                "summary": generate_summary(results),
            }, f, indent=2)
        print(f"✅ Results saved to: {json_path}")
    
    # Save text format
    if config.output_format in ["txt", "both"]:
        txt_path = config.output_dir / f"{base_filename}.txt"
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Data Replay Evaluation Results\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Data Type: {data_type}\n")
            f.write(f"Environment: {config.task}\n")
            f.write("="*80 + "\n\n")
            
            # Write individual results
            for result in results:
                f.write(f"\nTask Suite: {result['task_suite']}\n")
                f.write(f"Task ID: {result['task_id']}\n")
                f.write(f"Task Name: {result.get('task_name', 'N/A')}\n")
                f.write(f"Language: {result.get('language_instruction', 'N/A')}\n")
                f.write(f"-" * 80 + "\n")
                
                if result.get('status') == 'error':
                    f.write(f"Status: ERROR\n")
                    f.write(f"Error: {result.get('error', 'Unknown')}\n")
                else:
                    f.write(f"Total Episodes: {result.get('total_episodes', 0)}\n")
                    f.write(f"Successful: {result.get('successful_episodes', 0)}\n")
                    f.write(f"Failed: {result.get('failed_episodes', 0)}\n")
                    f.write(f"Success Rate: {result.get('success_rate', 0):.2%}\n")
                    if result.get('failed_demo_ids'):
                        f.write(f"Failed Demo IDs: {result['failed_demo_ids']}\n")
                    if result.get('avg_squeeze_pred') is not None and result.get('avg_squeeze_meas') is not None:
                        f.write(
                            "Avg squeeze (success only): "
                            f"pred={result['avg_squeeze_pred']:.4f}, "
                            f"meas={result['avg_squeeze_meas']:.4f}\n"
                        )
                    # 详细 Hybrid 力学指标（与 run_task_evaluations.py 输出形式保持一致）
                    tsq_max = result.get("task_squeeze_max_mean")
                    tapp_max = result.get("task_app_max_mean")
                    tapp_mean = result.get("task_app_mean_mean")
                    tsq_max_meas = result.get("task_squeeze_max_meas_mean")
                    tap_max_meas = result.get("task_ap_max_meas_mean")
                    tap_mean_meas = result.get("task_ap_mean_meas_mean")

                    has_any_hf_metric = any(
                        v is not None
                        for v in [
                            result.get("avg_squeeze_pred"),
                            result.get("avg_squeeze_meas"),
                            tsq_max,
                            tsq_max_meas,
                            tapp_mean,
                            tap_mean_meas,
                            tapp_max,
                            tap_max_meas,
                        ]
                    )
                    if has_any_hf_metric:
                        f.write("Hybrid metrics (success only, task-level mean over demos):\n")
                        # 1) episode 级平均挤压力
                        avg_squeeze_pred = result.get("avg_squeeze_pred")
                        avg_squeeze_meas = result.get("avg_squeeze_meas")
                        if avg_squeeze_pred is not None or avg_squeeze_meas is not None:
                            f.write("  ")
                            if avg_squeeze_pred is not None:
                                f.write(f"squeeze_avg_pred={avg_squeeze_pred:.4f} ")
                            else:
                                f.write("squeeze_avg_pred=N/A ")
                            if avg_squeeze_meas is not None:
                                f.write(f"squeeze_avg_meas={avg_squeeze_meas:.4f}")
                            else:
                                f.write("squeeze_avg_meas=N/A")
                            f.write("\n")

                        # 2) 预测 / 实测 Top5% 最大挤压力
                        if tsq_max is not None or tsq_max_meas is not None:
                            f.write("  ")
                            if tsq_max is not None:
                                f.write(f"squeeze_max_pred={tsq_max:.4f} ")
                            else:
                                f.write("squeeze_max_pred=N/A ")
                            if tsq_max_meas is not None:
                                f.write(f"squeeze_max_meas={tsq_max_meas:.4f}")
                            else:
                                f.write("squeeze_max_meas=N/A")
                            f.write("\n")

                        # 3) 预测 / 实测加持力平均值
                        if tapp_mean is not None or tap_mean_meas is not None:
                            f.write("  ")
                            if tapp_mean is not None:
                                f.write(f"ap_avg_pred={tapp_mean:.4f} ")
                            else:
                                f.write("ap_avg_pred=N/A ")
                            if tap_mean_meas is not None:
                                f.write(f"ap_avg_meas={tap_mean_meas:.4f}")
                            else:
                                f.write("ap_avg_meas=N/A")
                            f.write("\n")

                        # 4) 预测 / 实测 Top5% 最大加持力
                        if tapp_max is not None or tap_max_meas is not None:
                            f.write("  ")
                            if tapp_max is not None:
                                f.write(f"ap_max_pred={tapp_max:.4f} ")
                            else:
                                f.write("ap_max_pred=N/A ")
                            if tap_max_meas is not None:
                                f.write(f"ap_max_meas={tap_max_meas:.4f}")
                            else:
                                f.write("ap_max_meas=N/A")
                            f.write("\n")
                f.write(f"Time: {result.get('elapsed_time_seconds', 0):.2f}s\n")
                f.write("\n")
            
            # Write summary
            summary = generate_summary(results)
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total Tasks Evaluated: {summary['total_tasks']}\n")
            f.write(f"Total Episodes: {summary['total_episodes']}\n")
            f.write(f"Successful Episodes: {summary['successful_episodes']}\n")
            f.write(f"Failed Episodes: {summary['failed_episodes']}\n")
            f.write(f"Overall Success Rate: {summary['overall_success_rate']:.2%}\n")
            f.write(f"Total Time: {summary['total_time']:.2f}s\n")
        
        print(f"✅ Results saved to: {txt_path}")


def generate_summary(results: list[dict]) -> dict:
    """Generate summary statistics from results."""
    
    total_tasks = len(results)
    total_episodes = sum(r.get('total_episodes', 0) for r in results)
    successful_episodes = sum(r.get('successful_episodes', 0) for r in results)
    failed_episodes = sum(r.get('failed_episodes', 0) for r in results)
    total_time = sum(r.get('elapsed_time_seconds', 0) for r in results)
    
    overall_success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
    
    return {
        "total_tasks": total_tasks,
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "failed_episodes": failed_episodes,
        "overall_success_rate": overall_success_rate,
        "total_time": total_time,
    }


def print_metrics_ascii_table(results: list[dict]):
    """按统一模板打印 Hybrid metrics 表（支持单 task 或多 task）."""
    if not results:
        return

    rows: list[list[str]] = []
    for result in results:
        # 仅跳过显式 error 的结果，其余（包括 partial/failed 但有 episode 的）都展示出来
        if result.get("status") == "error":
            continue

        task_suite = result.get("task_suite", "")
        task_id = result.get("task_id", -1)
        success_rate = result.get("success_rate")  # 这里是 0~1 的比例
        avg_squeeze_pred = result.get("avg_squeeze_pred")
        avg_squeeze_meas = result.get("avg_squeeze_meas")
        # 统一读取 task 级别的 8 个 Hybrid 力学指标（与 run_task_evaluations 中命名保持一致）
        task_squeeze_max_mean = result.get("task_squeeze_max_mean")
        task_squeeze_max_meas_mean = result.get("task_squeeze_max_meas_mean")
        task_ap_mean_mean = result.get("task_app_mean_mean")
        task_ap_mean_meas_mean = result.get("task_ap_mean_meas_mean")
        task_ap_max_mean = result.get("task_app_max_mean")
        task_ap_max_meas_mean = result.get("task_ap_max_meas_mean")

        # 任务标签：如 libero_goal -> goal 0
        if isinstance(task_suite, str) and task_suite.startswith("libero_"):
            suite_short = task_suite.split("libero_")[1]
        else:
            suite_short = task_suite
        task_label = f"{suite_short} {task_id}"

        def _fmt(value, digits: int) -> str:
            if value is None:
                return "N/A"
            return f"{value:.{digits}f}"

        rows.append(
            [
                task_label,
                _fmt(success_rate, 2),
                _fmt(avg_squeeze_pred, 4),
                _fmt(avg_squeeze_meas, 4),
                _fmt(task_squeeze_max_mean, 4),
                _fmt(task_squeeze_max_meas_mean, 4),
                _fmt(task_ap_mean_mean, 4),
                _fmt(task_ap_mean_meas_mean, 4),
                _fmt(task_ap_max_mean, 4),
                _fmt(task_ap_max_meas_mean, 4),
            ]
        )

    if not rows:
        return

    headers = [
        "task id",
        "success_rate",
        "squeeze_avg_pred",
        "squeeze_avg_meas",
        "squeeze_max_pred",
        "squeeze_max_meas",
        "ap_avg_pred",
        "ap_avg_meas",
        "ap_max_pred",
        "ap_max_meas",
    ]

    # 计算每列宽度
    cols = list(zip(headers, *rows))
    col_widths = [max(len(str(cell)) for cell in col) + 2 for col in cols]  # 两侧各空 1 格

    def _hline() -> str:
        return "+" + "+".join("-" * w for w in col_widths) + "+"

    def _format_row(cells: list[str]) -> str:
        padded = []
        for cell, width in zip(cells, col_widths):
            cell_str = str(cell)
            space = width - len(cell_str)
            left = space // 2
            right = space - left
            padded.append(" " * left + cell_str + " " * right)
        return "|" + "|".join(padded) + "|"

    print("\n=== metric ===\n")
    print(_hline())
    print(_format_row(headers))
    print(_hline())
    for row in rows:
        print(_format_row(row))
    print(_hline())


def _aggregate_task_results(per_pass_results: list[dict]) -> dict:
    """将同一 task 多次 replay 的结果合并成一次统一统计."""
    if not per_pass_results:
        return {}

    base = per_pass_results[0].copy()

    total_episodes = sum(r.get("total_episodes", 0) or 0 for r in per_pass_results)
    successful_episodes = sum(r.get("successful_episodes", 0) or 0 for r in per_pass_results)
    failed_episodes = sum(r.get("failed_episodes", 0) or 0 for r in per_pass_results)

    if total_episodes > 0:
        success_rate = successful_episodes / total_episodes
    else:
        success_rate = 0.0

    # 失败 demo 的 ID：做去重并排序，侧重告诉用户「哪些 demo 曾经失败过」
    failed_demo_ids: list[int] = []
    for r in per_pass_results:
        failed_demo_ids.extend(r.get("failed_demo_ids", []) or [])
    failed_demo_ids = sorted(set(failed_demo_ids))

    elapsed_time = sum(r.get("elapsed_time_seconds", 0.0) or 0.0 for r in per_pass_results)

    # 聚合后整体状态：只要有 episode，认为是 completed（哪怕子进程在退出时有非 0 返回码）
    return_codes = [r.get("return_code", 0) for r in per_pass_results if "return_code" in r]
    if total_episodes > 0:
        status = "completed"
    else:
        status = "failed" if return_codes else "error"
    return_code = 0 if not return_codes else max(return_codes)

    def _avg_metric(key: str):
        vals = [r.get(key) for r in per_pass_results if r.get(key) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    base.update(
        {
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "success_rate": success_rate,
            "failed_demo_ids": failed_demo_ids,
            "elapsed_time_seconds": elapsed_time,
            "return_code": return_code,
            "status": status,
            # Hybrid：同一 task 多次 replay 的 task 级别平均（简单算数平均）
            "avg_squeeze_pred": _avg_metric("avg_squeeze_pred"),
            "avg_squeeze_meas": _avg_metric("avg_squeeze_meas"),
            "task_squeeze_max_mean": _avg_metric("task_squeeze_max_mean"),
            "task_squeeze_mean_mean": _avg_metric("task_squeeze_mean_mean"),
            "task_app_max_mean": _avg_metric("task_app_max_mean"),
            "task_app_mean_mean": _avg_metric("task_app_mean_mean"),
            "task_squeeze_max_meas_mean": _avg_metric("task_squeeze_max_meas_mean"),
            "task_ap_max_meas_mean": _avg_metric("task_ap_max_meas_mean"),
            "task_ap_mean_meas_mean": _avg_metric("task_ap_mean_meas_mean"),
        }
    )

    # 额外记录一共重复了多少轮（方便 debug）
    base["num_replay_passes"] = len(per_pass_results)
    return base


def main():
    """Main function to run replay evaluations."""
    
    # Parse configuration
    config = tyro.cli(DataReplayEvalConfig)

    # 根据 control_mode 与 task 关键字做环境选择（与 OpenPI / run_task_evaluations 保持一致）
    if not config.task:
        mode = config.control_mode

        if mode == "diffik":
            config.task = "Isaac-Libero-Franka-IK-v0"
        elif mode == "osc":
            config.task = "Isaac-Libero-Franka-OscPose-v0"
        elif mode == "hybrid":
            config.task = "Isaac-Libero-Franka-Hybrid-ContactForce-v0"
        elif mode == "tactile":
            config.task = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
        else:
            raise ValueError(
                f"Unsupported control_mode '{mode}'. "
                "Expected one of: 'diffik', 'osc', 'hybrid', 'tactile', or specify --task explicitly."
            )
    # Get workspace root
    workspace_root = Path(__file__).parent.parent.parent.resolve()
    print(f"Workspace root: {workspace_root}")
    
    # Determine which task suites to evaluate
    if config.task_suites:
        task_suites_to_eval = list(config.task_suites)
    else:
        task_suites_to_eval = list(TASK_SUITE_CONFIGS.keys())
    
    # Auto-detect space type from environment name
    is_task_space_env = "IK" in config.task or "Osc" in config.task
    space_type = "Task Space" if is_task_space_env else "Joint Space"
    
    demos_dir_str = os.environ.get("REPLAYED_DEMOS_DIR", "").strip()
    demos_dir = Path(demos_dir_str).expanduser().resolve() if demos_dir_str else None
    if demos_dir is None:
        raise RuntimeError("Missing env var: REPLAYED_DEMOS_DIR")
    lower = str(demos_dir).lower()
    dataset_kind = (
        "softvtbench_force" if "softvtbench_force" in lower else "softvtbench" if "softvtbench" in lower else "libero" if "libero" in lower else "unknown"
    )
    
    print(f"\n{'='*80}")
    print(f"Data Replay Evaluation")
    print(f"{'='*80}")
    print(f"Task Suites: {task_suites_to_eval}")
    print(f"Environment: {config.task} ({space_type})")
    print(f"Data Source: {dataset_kind} (REPLAYED_DEMOS_DIR={demos_dir})")
    print(f"Max Episodes per Task: {config.max_episodes}")
    print(f"{'='*80}\n")
    
    # Collect all results
    all_results = []
    
    # Iterate through task suites
    for task_suite in task_suites_to_eval:
        if task_suite not in TASK_SUITE_CONFIGS:
            print(f"⚠️  Warning: Unknown task suite '{task_suite}', skipping...")
            continue
        
        # Determine which task IDs to evaluate
        total_tasks = TASK_SUITE_CONFIGS[task_suite]

        # 1) 基础任务集合：
        # - SoftVTBench 数据集：默认使用 softvtbench_tasks.json 的白名单子集
        # - 其它数据集：默认评估该 suite 的全部任务
        subset_map = _load_softvtbench_task_subset(workspace_root)
        if dataset_kind in ("softvtbench", "softvtbench_force") and task_suite in subset_map:
            base_task_ids = [tid for tid in subset_map[task_suite] if 0 <= tid < total_tasks]
        else:
            base_task_ids = list(range(total_tasks))

        # 2) 可选：再与 CLI 传入的 task_ids 取交集
        if config.task_ids:
            task_ids_to_eval = [tid for tid in base_task_ids if tid in config.task_ids]
        else:
            task_ids_to_eval = base_task_ids
        
        print(f"\n{'#'*80}")
        print(f"Evaluating Task Suite: {task_suite}")
        print(f"Task IDs: {task_ids_to_eval}")
        print(f"{'#'*80}")
        
        # Evaluate each task（单次回放；不再强制累积到 max_episodes 条）
        for task_id in task_ids_to_eval:
            single_result = run_single_replay_evaluation(
                config=config,
                task_suite=task_suite,
                task_id=task_id,
                workspace_root=workspace_root,
            )
            all_results.append(single_result)

            # 每个 task 评估完成后打印一次仅包含当前 task 的 metrics 表
            print_metrics_ascii_table([single_result])

            # Small delay between tasks to ensure clean separation
            time.sleep(2)
    
    # Save results
    if all_results:
        save_results(all_results, config)
        
        # Print final summary
        summary = generate_summary(all_results)
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tasks Evaluated: {summary['total_tasks']}")
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Successful Episodes: {summary['successful_episodes']}")
        print(f"Failed Episodes: {summary['failed_episodes']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.2%}")
        print(f"Total Time: {summary['total_time']:.2f}s")
        print(f"{'='*80}\n")

        # 额外打印一个包含所有 task 的 metrics 总览表
        print_metrics_ascii_table(all_results)
    else:
        print("\n⚠️  No results to save.")


if __name__ == "__main__":
    main()

