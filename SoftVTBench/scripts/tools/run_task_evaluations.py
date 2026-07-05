# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run policy inference evaluation on Libero task suites and record success rates."""

import json
import os
import re
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import tyro

# Keep progress visible when stdout is redirected to a log file.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# 确保项目根目录在 sys.path 中，使得在不同运行方式下都能导入 `scripts.tools`。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_softvtbench_task_subset(workspace_root: Path) -> dict[str, list[int]]:
    """Load SoftVTBench task subset mapping from JSON.

    Expected path:
      benchmarks/datasets/softvtbench/config/softvtbench_tasks.json
    """
    path = workspace_root / "benchmarks" / "datasets" / "softvtbench" / "config" / "softvtbench_tasks.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Normalize values to list[int]
        out: dict[str, list[int]] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, list):
                    out[k] = [int(x) for x in v]
        return out
    except Exception:
        return {}


def get_task_suites_and_tasks() -> dict[str, list[int]]:
    """Get all available task suites and their task IDs."""
    return {
        "libero_10": list(range(10)),
        "libero_spatial": list(range(10)),
        "libero_object": list(range(10)),
        "libero_goal": list(range(10)),
    }


def get_task_name_and_instruction(task_suite: str, task_id: int, config_base_path: Path) -> tuple[str, str]:
    """Get task name and language instruction from config file."""
    config_path = config_base_path / f"{task_suite}.json"
    if not config_path.exists():
        return f"task_{task_id}", f"Task {task_id}"

    with open(config_path) as f:
        config = json.load(f)

    for task in config["tasks"]:
        if task["task_id"] == task_id:
            return task["task_name"], task["language_instruction"]

    return f"task_{task_id}", f"Task {task_id}"


def _should_auto_use_softvtbench_tasks(hdf5_folder: Optional[Path], replayed_demos_dir: str) -> bool:
    """Infer SoftVTBench subset mode from dataset directory names, not parent checkout names."""
    for raw_path in (hdf5_folder, replayed_demos_dir):
        if not raw_path:
            continue
        parts = [part.lower() for part in Path(raw_path).parts]
        if "softvtbench_force" in parts:
            return True
        for idx, part in enumerate(parts[:-1]):
            if part == "datasets" and parts[idx + 1] in {"softvtbench", "softvtbench_force"}:
                return True
    return False


@dataclass
class EvaluationConfig:
    """Configuration for running task evaluations."""
    
    policy_model: str = "openpi"
    control_mode: str = "diffik"
    # OpenPI tactile 消融：透传到 openpi client 的 `--abs7d`
    # （tactile 观测/模型分支不变，但动作按绝对 7D 执行，补 0 到 13D，并关闭 pos_kp/squeeze_kp 修正）
    abs7d: bool = False
    # 任务环境 ID：
    # - 默认留空：由 `benchmarks/openpi/openpi_inference_client.py` 根据 control_mode 自动选择：
    #   * diffik  -> Isaac-Libero-Franka-IK-v0
    #   * osc     -> Isaac-Libero-Franka-OscPose-v0
    #   * hybrid  -> Isaac-Libero-Franka-Hybrid-ContactForce-v0      (13D 力–位混合控制 + contact forces)
    #   * tactile -> Isaac-Libero-Franka-Hybrid-Tactile-v0           (13D 力–位混合控制 + 触觉)
    # - 如需自定义环境，可通过 CLI 显式传入 `--task xxx`。
    task: str = ""
    
    server_host: str = "127.0.1.1"
    server_port: int = 8000
    
    num_total_experiments: int = 50
    num_success_steps: int = 8
    max_inference_steps: int = 80
    replan_steps: int = 10
    
    camera_names: tuple[str, ...] = ("agentview_cam", "eye_in_hand_cam")
    target_image_size: tuple[int, int, int] = (224, 224, 3)
    num_steps_wait: int = 5
    
    task_suites: tuple[str, ...] = ()
    task_ids: tuple[int, ...] = ()
    
    hdf5_folder: Optional[Path] = None
    config_path: Optional[Path] = None
    
    output_dir: Path = Path("./evaluation_results")
    output_format: str = "both"
    
    headless: bool = True
    visualize: bool = False
    debug_mode: int = 0
    # Optional: override OpenPI client's debug output root (e.g., for debug_mode=6 captures).
    # If empty, OpenPI client uses its own default.
    debug_path: str = ""
    seed: int = 11
    
    openpi_script: str = "benchmarks/openpi/openpi_inference_client.py"
    gr00t_script: str = "benchmarks/gr00t/gr00t_inference_client.py"

    # 是否使用 SoftVTBench 任务子集（True: 只评估 SoftVTBench 固定列表中的任务；False: 使用原版 Libero 全任务）
    use_softvtbench_tasks: bool = False

    # SoftVTBench-style prompt rewrite (adverb augmentation). Passed through to OpenPI client.
    prompt_adverb: str = ""
    # 与 convert_all_libero_to_softvtbench.py 保持一致：strong_adverbs=("firmly","tightly"), soft_adverbs=("gently","softly")
    # 这里作为评估侧的默认候选集合（OpenPI 侧会确定性选择其中一个并决定 prefix/suffix 风格）。
    prompt_adverbs: tuple[str, ...] = ("firmly", "tightly", "gently", "softly")
    prompt_seed: int = 0

    # If True, only evaluate tasks that have matching HDF5 files under hdf5_folder
    # (pattern: <suite>_task<id>_*_demo.hdf5). Useful when you must rely on assembled HDF5 for scene setup.
    require_hdf5: bool = False


def build_command(config: EvaluationConfig, task_suite: str, task_id: int) -> list[str]:
    """Build command line arguments for subprocess."""
    python_script = config.openpi_script if config.policy_model == "openpi" else config.gr00t_script
    # Per-suite inference step policy (user request):
    # - libero_10:      max_inference_steps = 50
    # - libero_goal/spatial/object: max_inference_steps = 30
    if task_suite == "libero_10":
        max_inference_steps = 50
    elif task_suite in ("libero_goal", "libero_spatial", "libero_object"):
        max_inference_steps = 30
    else:
        max_inference_steps = config.max_inference_steps
    
    cmd = [
        sys.executable, python_script,
        "--server_host", config.server_host,
        "--server_port", str(config.server_port),
        "--num_total_experiments", str(config.num_total_experiments),
        "--num_success_steps", str(config.num_success_steps),
        "--max_inference_steps", str(max_inference_steps),
        "--task_suite", task_suite,
        "--task_id", str(task_id),
        "--seed", str(config.seed),
    ]

    if config.policy_model == "openpi":
        cmd.extend([
            "--control_mode", config.control_mode,
            "--replan_steps", str(config.replan_steps),
            "--target_image_size"] + [str(x) for x in config.target_image_size] + [
            "--num_steps_wait", str(config.num_steps_wait),
        ])
        # The OpenPI client currently exposes camera_names as a single tyro STR even
        # though its default contains both required cameras. Avoid passing this flag
        # here so formal evaluations use the client-side default without parse errors.
        if config.abs7d:
            cmd.append("--abs7d")
        if config.task:
            cmd.extend(["--task", config.task])
        # Optional prompt rewrite knobs
        # Always pass prompt_seed explicitly (including 0) so prompt behavior is stable and traceable.
        cmd.extend(["--prompt_seed", str(config.prompt_seed)])
        if config.prompt_adverb:
            cmd.extend(["--prompt_adverb", str(config.prompt_adverb)])
        if config.prompt_adverbs:
            cmd.append("--prompt_adverbs")
            cmd.extend([str(x) for x in config.prompt_adverbs])

    if config.headless and not config.visualize:
        cmd.append("--headless")
    
    if config.debug_mode > 0:
        cmd.extend(["--debug_mode", str(config.debug_mode)])
        if config.debug_path:
            cmd.extend(["--debug_path", str(config.debug_path)])
    
    if config.hdf5_folder:
        cmd.extend(["--hdf5_folder", str(config.hdf5_folder)])

    return cmd


_AVG_PATTERN = re.compile(
    r"squeeze_pred=([+-]?(?:\d+\.?\d*|\d*\.?\d+))(?:[eE][+-]?\d+)?\s*,\s*"
    r"squeeze_meas=([+-]?(?:\d+\.?\d*|\d*\.?\d+))(?:[eE][+-]?\d+)?"
)
_KEYVAL_PATTERN = re.compile(
    r"([a-zA-Z0-9_]+)\s*=\s*([+-]?(?:\d+\.?\d*|\d*\.?\d+))(?:[eE][+-]?\d+)?"
)


def parse_success_metrics(
    output_lines: list[str],
) -> tuple[
    Optional[float],
    Optional[int],
    Optional[int],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """从 OpenPI 子进程 stdout 中解析成功率和 Hybrid 相关力学指标."""

    success_rate: Optional[float] = None
    successful_experiments: Optional[int] = None
    total_experiments: Optional[int] = None

    # 9 个 metrics，按字典暂存，最后统一取出
    metrics: dict[str, float] = {}

    for line in output_lines:
        if "Success rate:" in line:
            # Success rate: 68.00%
            with suppress(Exception):
                success_rate = float(line.split("Success rate:", 1)[1].strip().replace("%", ""))
            continue

        if "Successful experiments:" in line:
            with suppress(Exception):
                successful_experiments = int(line.split("Successful experiments:", 1)[1].strip())
            continue

        if "Total experiments:" in line:
            with suppress(Exception):
                total_experiments = int(line.split("Total experiments:", 1)[1].strip())
            continue

        if "[Hybrid] Task avg squeeze_pred=" in line:
            # 例子：
            # [Hybrid] Task avg squeeze_pred=0.1234, squeeze_meas=0.5678 over 10 successes
            match = _AVG_PATTERN.search(line)
            if match:
                with suppress(Exception):
                    metrics["squeeze_avg_pred"] = float(match.group(1))
                with suppress(Exception):
                    metrics["squeeze_avg_meas"] = float(match.group(2))
            continue

        if "[Hybrid-Metrics] Task contact_metrics" in line:
            # 例子：
            # [Hybrid-Metrics] Task contact_metrics squeeze_max_mean=0.1, app_max_mean=0.2, ...
            for m in _KEYVAL_PATTERN.finditer(line):
                key, val = m.group(1), m.group(2)
                with suppress(Exception):
                    metrics[key] = float(val)

    return (
        success_rate,
        successful_experiments,
        total_experiments,
        metrics.get("squeeze_avg_pred"),
        metrics.get("squeeze_avg_meas"),
        metrics.get("squeeze_max_mean"),
        metrics.get("app_max_mean"),
        metrics.get("app_mean_mean"),
        metrics.get("squeeze_max_meas_mean"),
        metrics.get("ap_max_meas_mean"),
        metrics.get("ap_mean_meas_mean"),
    )


def run_single_evaluation(
    config: EvaluationConfig,
    task_suite: str,
    task_id: int,
    workspace_root: Path,
) -> dict:
    """Run a single task evaluation and return results."""
    config_base_path = config.config_path or (workspace_root / "benchmarks/datasets/libero/config")
    task_name, language_instruction = get_task_name_and_instruction(task_suite, task_id, config_base_path)

    print(f"\n{'='*80}")
    print(f"Running evaluation for {task_suite} - Task {task_id}")
    print(f"Task Name: {task_name}")
    print(f"Language Instruction: {language_instruction}")
    print(f"{'='*80}")

    # Keep the effective max_inference_steps in results for traceability.
    if task_suite == "libero_10":
        effective_max_inference_steps = 50
    elif task_suite in ("libero_goal", "libero_spatial", "libero_object"):
        effective_max_inference_steps = 30
    else:
        effective_max_inference_steps = config.max_inference_steps
    cmd = build_command(config, task_suite, task_id)
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if "USE_RELATIVE_MODE" not in env:
        env["USE_RELATIVE_MODE"] = "False"

    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=workspace_root,
            env=env,
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line)
                # 屏蔽 OpenPI 内部的部分统计行，只在本脚本里做表格展示
                stripped = line.strip()
                if (
                    stripped.startswith("[Hybrid-Metrics] Task contact_metrics")
                    or stripped.startswith("[Hybrid] Task avg squeeze_pred=")
                    or stripped.startswith("Evaluation Results:")
                    or stripped.startswith("Total experiments:")
                    or stripped.startswith("Successful experiments:")
                    or stripped.startswith("Success rate:")
                ):
                    continue
                print(line, end='')
        
        process.wait(timeout=3600)
        end_time = time.time()

        (
            success_rate,
            successful_experiments,
            total_experiments,
            avg_squeeze_pred,
            avg_squeeze_meas,
            task_squeeze_max_mean,
            task_app_max_mean,
            task_app_mean_mean,
            task_squeeze_max_meas_mean,
            task_ap_max_meas_mean,
            task_ap_mean_meas_mean,
        ) = parse_success_metrics(output_lines)
        
        # Some subprocesses may print the final evaluation stats successfully but still exit with
        # a non-zero return code (e.g., teardown issues when closing IsaacSim). For reporting,
        # treat the task as "completed" if we could reliably parse success stats for the full
        # requested number of experiments.
        parsed_eval = (
            (success_rate is not None)
            and (successful_experiments is not None)
            and (total_experiments is not None)
        )
        status = "completed" if parsed_eval else ("completed" if process.returncode == 0 else "failed")
        
        print(f"\n{'='*80}")
        print(f"TASK COMPLETED: {task_suite} - Task {task_id}")
        print(f"{'='*80}")
        if success_rate is not None:
            print(f"✓ Success Rate: {success_rate:.2f}% ({successful_experiments}/{total_experiments} experiments)")
        else:
            print(f"✗ Failed to extract success rate (return code: {process.returncode})")
        print(f"Execution Time: {end_time - start_time:.1f}s")
        print(f"{'='*80}\n")
        
        return {
            "task_suite": task_suite,
            "task_id": task_id,
            "task_name": task_name,
            "language_instruction": language_instruction,
            "success_rate": success_rate,
            "successful_experiments": successful_experiments,
            "total_experiments": total_experiments,
            "max_inference_steps": effective_max_inference_steps,
            "avg_squeeze_pred": avg_squeeze_pred,
            "avg_squeeze_meas": avg_squeeze_meas,
            "task_squeeze_max_mean": task_squeeze_max_mean,
            "task_app_max_mean": task_app_max_mean,
            "task_app_mean_mean": task_app_mean_mean,
            "task_squeeze_max_meas_mean": task_squeeze_max_meas_mean,
            "task_ap_max_meas_mean": task_ap_max_meas_mean,
            "task_ap_mean_meas_mean": task_ap_mean_meas_mean,
            "return_code": process.returncode,
            "execution_time": end_time - start_time,
            "status": status,
        }

    except subprocess.TimeoutExpired:
        print(f"\n{'='*80}")
        print(f"✗ TASK TIMEOUT: {task_suite} - Task {task_id}")
        print(f"{'='*80}\n")
        return {
            "task_suite": task_suite,
            "task_id": task_id,
            "task_name": task_name,
            "language_instruction": language_instruction,
            "success_rate": None,
            "successful_experiments": None,
            "total_experiments": None,
            "max_inference_steps": effective_max_inference_steps,
            "return_code": -1,
            "execution_time": 3600,
            "status": "timeout",
        }
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ TASK ERROR: {task_suite} - Task {task_id}")
        print(f"Error: {e}")
        print(f"{'='*80}\n")
        return {
            "task_suite": task_suite,
            "task_id": task_id,
            "task_name": task_name,
            "language_instruction": language_instruction,
            "success_rate": None,
            "successful_experiments": None,
            "total_experiments": None,
            "max_inference_steps": effective_max_inference_steps,
            "return_code": -1,
            "execution_time": 0,
            "status": "error",
            "error": str(e),
        }


def save_success_rates_json(results: list[dict], output_file: Path, config: EvaluationConfig):
    """Save success rates to a JSON file."""
    if not results:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "policy_model": config.policy_model,
            "control_mode": config.control_mode if config.policy_model == "openpi" else "N/A",
            "abs7d": bool(config.abs7d) if config.policy_model == "openpi" else False,
            "task_environment": config.task if config.task else "auto",
            "num_total_experiments": config.num_total_experiments,
            "num_success_steps": config.num_success_steps,
            # NOTE: max_inference_steps is enforced per task_suite when evaluating LIBERO suites:
            #   - libero_10: 50
            #   - libero_goal/libero_spatial/libero_object: 30
            # Other suites fall back to config.max_inference_steps.
            "max_inference_steps_policy": {
                "libero_10": 50,
                "libero_goal": 30,
                "libero_spatial": 30,
                "libero_object": 30,
                "default": config.max_inference_steps,
            },
            "timestamp": datetime.now().isoformat(),
        },
        "results": {}
    }

    for result in results:
        task_key = f"{result['task_suite']}_task{result['task_id']}"
        output_data["results"][task_key] = {
            "task_suite": result["task_suite"],
            "task_id": result["task_id"],
            "task_name": result["task_name"],
            "language_instruction": result["language_instruction"],
            "success_rate": result["success_rate"],
            "successful_experiments": result["successful_experiments"],
            "total_experiments": result["total_experiments"],
            "execution_time": result["execution_time"],
            "status": result["status"],
            # Hybrid 下可选：成功 experiment 的平均挤压力（可能为 None）
            "avg_squeeze_pred": result.get("avg_squeeze_pred"),
            "avg_squeeze_meas": result.get("avg_squeeze_meas"),
            # Hybrid 下可选：task 级别的挤压力 / 加持力 metrics（可能为 None）
            "task_squeeze_max_mean": result.get("task_squeeze_max_mean"),
            "task_app_max_mean": result.get("task_app_max_mean"),
            "task_app_mean_mean": result.get("task_app_mean_mean"),
            # 新增：实测 Top5% 最大挤压力 / 加持力 & 实测加持力平均值
            "task_squeeze_max_meas_mean": result.get("task_squeeze_max_meas_mean"),
            "task_ap_max_meas_mean": result.get("task_ap_max_meas_mean"),
            "task_ap_mean_meas_mean": result.get("task_ap_mean_meas_mean"),
        }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Success rates saved to: {output_file}")


def save_success_rates_txt(results: list[dict], output_file: Path, config: EvaluationConfig):  # noqa: C901
    """Save success rates to a TXT file."""
    if not results:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("POLICY EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Policy Model: {config.policy_model}\n")
        if config.policy_model == "openpi":
            f.write(f"  Control Mode: {config.control_mode}\n")
            f.write(f"  abs7d: {config.abs7d}\n")
        f.write(f"  Experiments per task: {config.num_total_experiments}\n")
        f.write(f"  Success steps required: {config.num_success_steps}\n")
        f.write(f"  Max inference steps: {config.max_inference_steps}\n")
        f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "=" * 60 + "\n\n")

        suites = {}
        for result in results:
            suite = result["task_suite"]
            if suite not in suites:
                suites[suite] = []
            suites[suite].append(result)

        suite_averages = {}
        for suite, suite_results in suites.items():
            valid_results = [r for r in suite_results if r["status"] == "completed" and r["success_rate"] is not None]
            suite_averages[suite] = sum(r["success_rate"] for r in valid_results) / len(valid_results) if valid_results else None

        for suite, suite_results in sorted(suites.items()):
            f.write(f"{suite.upper()}:\n")
            f.write("-" * 40 + "\n")
            suite_results.sort(key=lambda x: x["task_id"])

            for result in suite_results:
                task_id = result["task_id"]
                task_name = result["task_name"]
                success_rate = result["success_rate"]
                successful = result["successful_experiments"]
                total = result["total_experiments"]
                status = result["status"]
                exec_time = result["execution_time"]
                avg_squeeze_pred = result.get("avg_squeeze_pred")
                avg_squeeze_meas = result.get("avg_squeeze_meas")
                task_squeeze_max_mean = result.get("task_squeeze_max_mean")
                task_app_max_mean = result.get("task_app_max_mean")
                task_app_mean_mean = result.get("task_app_mean_mean")
                task_squeeze_max_meas_mean = result.get("task_squeeze_max_meas_mean")
                task_ap_max_meas_mean = result.get("task_ap_max_meas_mean")
                task_ap_mean_meas_mean = result.get("task_ap_mean_meas_mean")

                if success_rate is not None:
                    f.write(f"  Task {task_id} ({task_name}): {success_rate:.2f}% ({successful}/{total}) [{exec_time:.1f}s]\n")
                    # 9 个 Hybrid metric（与终端表头保持一致的命名）
                    # - squeeze_avg_pred / squeeze_avg_meas
                    # - squeeze_max_pred / squeeze_max_meas
                    # - ap_avg_pred / ap_avg_meas
                    # - ap_max_pred / ap_max_meas
                    has_any_hf_metric = any(
                        v is not None
                        for v in [
                            avg_squeeze_pred,
                            avg_squeeze_meas,
                            task_squeeze_max_mean,
                            task_squeeze_max_meas_mean,
                            task_app_mean_mean,
                            task_ap_mean_meas_mean,
                            task_app_max_mean,
                            task_ap_max_meas_mean,
                        ]
                    )
                    if has_any_hf_metric:
                        f.write("    Hybrid metrics (success only, task-level mean over experiments):\n")
                        if avg_squeeze_pred is not None or avg_squeeze_meas is not None:
                            f.write("      ")
                            if avg_squeeze_pred is not None:
                                f.write(f"squeeze_avg_pred={avg_squeeze_pred:.4f} ")
                            else:
                                f.write("squeeze_avg_pred=N/A ")
                            if avg_squeeze_meas is not None:
                                f.write(f"squeeze_avg_meas={avg_squeeze_meas:.4f}")
                            else:
                                f.write("squeeze_avg_meas=N/A")
                            f.write("\n")

                        if (
                            task_squeeze_max_mean is not None
                            or task_squeeze_max_meas_mean is not None
                        ):
                            f.write("      ")
                            if task_squeeze_max_mean is not None:
                                f.write(f"squeeze_max_pred={task_squeeze_max_mean:.4f} ")
                            else:
                                f.write("squeeze_max_pred=N/A ")
                            if task_squeeze_max_meas_mean is not None:
                                f.write(f"squeeze_max_meas={task_squeeze_max_meas_mean:.4f}")
                            else:
                                f.write("squeeze_max_meas=N/A")
                            f.write("\n")

                        if task_app_mean_mean is not None or task_ap_mean_meas_mean is not None:
                            f.write("      ")
                            if task_app_mean_mean is not None:
                                f.write(f"ap_avg_pred={task_app_mean_mean:.4f} ")
                            else:
                                f.write("ap_avg_pred=N/A ")
                            if task_ap_mean_meas_mean is not None:
                                f.write(f"ap_avg_meas={task_ap_mean_meas_mean:.4f}")
                            else:
                                f.write("ap_avg_meas=N/A")
                            f.write("\n")

                        if task_app_max_mean is not None or task_ap_max_meas_mean is not None:
                            f.write("      ")
                            if task_app_max_mean is not None:
                                f.write(f"ap_max_pred={task_app_max_mean:.4f} ")
                            else:
                                f.write("ap_max_pred=N/A ")
                            if task_ap_max_meas_mean is not None:
                                f.write(f"ap_max_meas={task_ap_max_meas_mean:.4f}")
                            else:
                                f.write("ap_max_meas=N/A")
                            f.write("\n")
                else:
                    f.write(f"  Task {task_id} ({task_name}): N/A ({status})\n")

            avg_sr = suite_averages[suite]
            if avg_sr is not None:
                completed_count = len([r for r in suite_results if r["status"] == "completed"])
                f.write(f"\n  Suite Average: {avg_sr:.2f}% ({completed_count}/{len(suite_results)} tasks completed)\n")
            else:
                f.write("\n  Suite Average: N/A (no completed tasks)\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 60 + "\n")

        for suite, avg_sr in sorted(suite_averages.items()):
            f.write(f"  {suite}: {avg_sr:.2f}%\n" if avg_sr is not None else f"  {suite}: N/A\n")

        valid_averages = [sr for sr in suite_averages.values() if sr is not None]
        if valid_averages:
            overall_avg = sum(valid_averages) / len(valid_averages)
            f.write(f"\n  Overall Average: {overall_avg:.2f}% ({len(valid_averages)}/{len(suite_averages)} suites)\n")
        else:
            f.write("\n  Overall Average: N/A\n")

        f.write("=" * 60 + "\n")

        # 在 TXT 文件最后附上「所有任务」的大表（与终端 print_metrics_ascii_table 一致）
        # 构造行数据
        rows: list[list[str]] = []
        for result in results:
            if result.get("status") != "completed":
                continue

            task_suite = result.get("task_suite", "")
            task_id = result.get("task_id", -1)
            success_rate_val = result.get("success_rate")  # 百分比
            avg_squeeze_pred = result.get("avg_squeeze_pred")
            avg_squeeze_meas = result.get("avg_squeeze_meas")
            task_squeeze_max_mean = result.get("task_squeeze_max_mean")
            task_squeeze_max_meas_mean = result.get("task_squeeze_max_meas_mean")
            # NOTE: result dict uses "task_app_mean_mean" (double 'p') as the canonical key.
            # Older code mistakenly looked up "task_ap_mean_mean", which caused ap_avg_pred to be N/A
            # in the final TXT metrics table (even though per-task Hybrid metrics were present).
            task_ap_mean_mean = result.get("task_app_mean_mean")
            task_ap_mean_meas_mean = result.get("task_ap_mean_meas_mean")
            task_app_max_mean = result.get("task_app_max_mean")
            task_ap_max_meas_mean = result.get("task_ap_max_meas_mean")

            if isinstance(task_suite, str) and task_suite.startswith("libero_"):
                suite_short = task_suite.split("libero_")[1]
            else:
                suite_short = task_suite
            task_label = f"{suite_short} {task_id}"

            def _fmt(value: Optional[float], digits: int) -> str:
                if value is None:
                    return "N/A"
                return f"{value:.{digits}f}"

            # success_rate 以 0.x 形式展示
            if success_rate_val is not None:
                sr_display = success_rate_val / 100.0
            else:
                sr_display = None

            rows.append(
                [
                    task_label,
                    _fmt(sr_display, 2),
                    _fmt(avg_squeeze_pred, 4),
                    _fmt(avg_squeeze_meas, 4),
                    _fmt(task_squeeze_max_mean, 4),
                    _fmt(task_squeeze_max_meas_mean, 4),
                    _fmt(task_ap_mean_mean, 4),
                    _fmt(task_ap_mean_meas_mean, 4),
                    _fmt(task_app_max_mean, 4),
                    _fmt(task_ap_max_meas_mean, 4),
                ]
            )

        if rows:
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

            cols = list(zip(headers, *rows))
            col_widths = [max(len(str(cell)) for cell in col) + 2 for col in cols]

            def _hline() -> str:
                return "+" + "+".join("-" * w for w in col_widths) + "+"

            def _format_row(cells: list[str]) -> str:
                padded: list[str] = []
                for cell, width in zip(cells, col_widths):
                    cell_str = str(cell)
                    space = width - len(cell_str)
                    left = space // 2
                    right = space - left
                    padded.append(" " * left + cell_str + " " * right)
                return "|" + "|".join(padded) + "|"

            f.write("\n=== metric (all tasks) ===\n\n")
            f.write(_hline() + "\n")
            f.write(_format_row(headers) + "\n")
            f.write(_hline() + "\n")
            for row in rows:
                f.write(_format_row(row) + "\n")
            f.write(_hline() + "\n")

    print(f"Success rates saved to: {output_file}")


def print_summary(results: list[dict]):
    """Print a summary to console."""
    if not results:
        return

    suites = {}
    for result in results:
        suite = result["task_suite"]
        if suite not in suites:
            suites[suite] = []
        suites[suite].append(result)

    completed_results = [r for r in results if r["status"] == "completed" and r["success_rate"] is not None]

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {len(results)}")
    print(f"Completed tasks: {len(completed_results)}")
    print(f"Failed tasks: {len([r for r in results if r['status'] == 'failed'])}")
    print(f"Timeout tasks: {len([r for r in results if r['status'] == 'timeout'])}")

    if completed_results:
        success_rates = [r["success_rate"] for r in completed_results]
        avg_success_rate = sum(success_rates) / len(success_rates)
        min_success_rate = min(success_rates)
        max_success_rate = max(success_rates)
        print(f"\nOverall average success rate: {avg_success_rate:.2f}%")
        print(f"Success rate range: {min_success_rate:.2f}% - {max_success_rate:.2f}%")

    print(f"\n{'='*60}")
    print("SUITE AVERAGES")
    print(f"{'='*60}")

    for suite, suite_results in sorted(suites.items()):
        valid_results = [r for r in suite_results if r["status"] == "completed" and r["success_rate"] is not None]
        if valid_results:
            suite_avg = sum(r["success_rate"] for r in valid_results) / len(valid_results)
            print(f"{suite}: {suite_avg:.2f}% ({len(valid_results)}/{len(suite_results)} tasks)")
        else:
            print(f"{suite}: N/A ({len(suite_results)} tasks, none completed)")

    print(f"{'='*60}\n")


def print_metrics_ascii_table(results: list[dict]):
    """打印符合用户模板的整体 metrics 表."""
    if not results:
        return

    # 收集已完成任务的核心指标
    rows: list[list[str]] = []
    for result in results:
        if result.get("status") != "completed":
            continue

        task_suite = result.get("task_suite", "")
        task_id = result.get("task_id", -1)
        success_rate = result.get("success_rate")  # 单位：百分比
        avg_squeeze_pred = result.get("avg_squeeze_pred")
        avg_squeeze_meas = result.get("avg_squeeze_meas")
        task_squeeze_max_mean = result.get("task_squeeze_max_mean")
        task_squeeze_max_meas_mean = result.get("task_squeeze_max_meas_mean")
        # 注意：结果字典中统一使用 "task_app_mean_mean" / "task_app_max_mean" 作为 key
        # 这里读取后赋值给 task_ap_* 变量，仅用于表格展示字段名（ap_*）
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

        def _fmt(value: Optional[float], digits: int) -> str:
            if value is None:
                return "N/A"
            return f"{value:.{digits}f}"

        # 成功率按 0.x 形式展示：success_rate 本身是百分比
        if success_rate is not None:
            sr_display = success_rate / 100.0
        else:
            sr_display = None

        rows.append(
            [
                task_label,
                _fmt(sr_display, 2),
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

    # 计算每一列宽度
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


def _fmt_float(value: Optional[float], precision: int = 3) -> str:
    """Format float value for pretty table output."""
    if value is None:
        return "   N/A   "
    return f"{value:.{precision}f}".rjust(9)


def print_live_force_table_header():
    """Print table header for per-task Hybrid metrics."""
    header = [
        "Task",
        "Succ(%)",
        "SqPred",
        "SqMeas",
        "SqMaxPred",
        "SqMaxMeas",
        "ApPred",
        "ApMeas",
        "ApMaxPred",
        "ApMaxMeas",
    ]
    line = " | ".join(h.center(12) for h in header)
    print("\n" + "=" * len(line))
    print(line)
    print("-" * len(line))


def print_live_force_table_row(result: dict):
    """Print one row of Hybrid metrics for a single task.

    列约定（从左到右）：
    - Task: 任务名（形如 "goal task 0"）
    - Succ(%): 成功率
    - SqPred / SqMeas: 预测 / 实测 挤压力平均值（来自 [Hybrid] Task avg ...）
    - SqMaxPred: 预测最大挤压力（单条 demo 内 Top5% 帧均值，再在 task 级取平均）
    - SqMaxMeas: 实测最大挤压力（同样的 Top5% 规则）
    - ApPred: 预测加持力平均值（task 级别）
    - ApMeas: 实测加持力平均值（task 级别）
    - ApMaxPred: 预测最大加持力（单条 demo 内 Top5% 帧均值，再在 task 级取平均）
    - ApMaxMeas: 实测最大加持力（同样的 Top5% 规则）
    """
    task_suite = result.get("task_suite", "")
    task_id = result.get("task_id", -1)

    # 任务标签：优先使用 Libero 风格的简写（例如 "goal task 0"）
    if isinstance(task_suite, str) and task_suite.startswith("libero_"):
        suite_short = task_suite.split("libero_")[1]
    else:
        suite_short = task_suite
    task_label = f"{suite_short} task {task_id}"

    success_rate = result.get("success_rate")
    avg_squeeze_pred = result.get("avg_squeeze_pred")
    avg_squeeze_meas = result.get("avg_squeeze_meas")
    task_squeeze_max_mean = result.get("task_squeeze_max_mean")
    task_squeeze_max_meas_mean = result.get("task_squeeze_max_meas_mean")
    task_ap_mean_mean = result.get("task_app_mean_mean")
    task_ap_mean_meas_mean = result.get("task_ap_mean_meas_mean")
    task_ap_max_mean = result.get("task_app_max_mean")
    task_ap_max_meas_mean = result.get("task_ap_max_meas_mean")

    cols = [
        task_label.ljust(12),
        _fmt_float(success_rate, precision=2),
        _fmt_float(avg_squeeze_pred, precision=4),
        _fmt_float(avg_squeeze_meas, precision=4),
        _fmt_float(task_squeeze_max_mean, precision=4),
        _fmt_float(task_squeeze_max_meas_mean, precision=4),
        _fmt_float(task_ap_mean_mean, precision=4),
        _fmt_float(task_ap_mean_meas_mean, precision=4),
        _fmt_float(task_ap_max_mean, precision=4),
        _fmt_float(task_ap_max_meas_mean, precision=4),
    ]

    print(" | ".join(cols))


def main():
    """Main entry point."""
    config = tyro.cli(EvaluationConfig)

    workspace_root = Path(__file__).parent.parent.parent.resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 如果用户没有显式传 --hdf5_folder，则尝试从 set_replay_env.sh 导出的 HDF5_TRAJ_SOURCE_DIR 读取
    # （该路径指向 libero assembled_hdf5，用于场景 setup / initial_state reset）。
    if config.hdf5_folder is None:
        env_hdf5 = os.environ.get("HDF5_TRAJ_SOURCE_DIR", "").strip()
        if env_hdf5:
            config.hdf5_folder = Path(env_hdf5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.policy_model == "openpi":
        model_name = f"{config.policy_model}_{config.control_mode}"
        if config.abs7d:
            model_name += "_abs7d"
    else:
        model_name = config.policy_model

    all_task_suites = get_task_suites_and_tasks()

    # 1) 先按 task_suites 过滤 suite
    if config.task_suites:
        task_suites = {k: v for k, v in all_task_suites.items() if k in config.task_suites}
        if not task_suites:
            print(f"Error: No valid task suites found in {config.task_suites}")
            print(f"Available task suites: {list(all_task_suites.keys())}")
            return
    else:
        task_suites = all_task_suites
    # 2) SoftVTBench 子集（默认：只要数据路径是 softvtbench/softvtbench_force，就启用；也可用 --use_softvtbench_tasks 强制启用）
    subset_map = _load_softvtbench_task_subset(workspace_root)
    auto_softvtbench = False
    with suppress(Exception):
        auto_softvtbench = _should_auto_use_softvtbench_tasks(
            config.hdf5_folder,
            os.environ.get("REPLAYED_DEMOS_DIR", ""),
        )

    if config.use_softvtbench_tasks or auto_softvtbench:
        if not subset_map:
            print("⚠️  [run_task_evaluations] SoftVTBench subset enabled but softvtbench_tasks.json not found or invalid.")
        for suite, tasks in list(task_suites.items()):
            if suite in subset_map:
                task_suites[suite] = [tid for tid in subset_map[suite] if tid in tasks]
            else:
                task_suites[suite] = tasks

    # 3) 可选：再与 CLI 传入的 task_ids 取交集
    if config.task_ids:
        for suite in task_suites:
            task_suites[suite] = [tid for tid in task_suites[suite] if tid in config.task_ids]

    total_task_count = sum(len(tasks) for tasks in task_suites.values())
    print(f"\n{'='*60}")
    print(f"Starting evaluation with {config.policy_model.upper()}")
    if config.policy_model == "openpi":
        print(f"Control mode: {config.control_mode}")
        if config.abs7d:
            print("abs7d: True")
    print(f"{'='*60}")
    print(f"Will evaluate {total_task_count} tasks across {len(task_suites)} task suites")
    print(f"Task suites: {list(task_suites.keys())}")
    print(f"Experiments per task: {config.num_total_experiments}")
    print(f"{'='*60}\n")

    # 4) 可选：按 assembled HDF5 实际存在的任务文件过滤（避免无 HDF5 时退化为默认 reset）
    if config.require_hdf5:
        if config.hdf5_folder is None:
            print("Error: --require_hdf5 requires --hdf5_folder to be set.")
            return
        hdf5_root = Path(config.hdf5_folder)
        if not hdf5_root.exists():
            print(f"Error: hdf5_folder does not exist: {hdf5_root}")
            return
        for suite in list(task_suites.keys()):
            keep: list[int] = []
            for tid in task_suites[suite]:
                pattern = f"{suite}_task{tid}_*_demo.hdf5"
                if list(hdf5_root.glob(pattern)):
                    keep.append(tid)
            task_suites[suite] = keep
        total_task_count = sum(len(tasks) for tasks in task_suites.values())
        print(f"[require_hdf5] After filtering: {total_task_count} tasks\n")

    results = []
    completed_tasks = 0

    for task_suite, task_ids in sorted(task_suites.items()):
        for task_id in task_ids:
            result = run_single_evaluation(config, task_suite, task_id, workspace_root)
            results.append(result)
            completed_tasks += 1

            # 每个 task 完成后打印一次只包含当前 task 的小表
            print_metrics_ascii_table([result])

            print(f"\nProgress: {completed_tasks}/{total_task_count} tasks completed")

    if config.output_format in ["json", "both"]:
        json_file = config.output_dir / f"success_rates_{model_name}_{timestamp}.json"
        save_success_rates_json(results, json_file, config)

    if config.output_format in ["txt", "both"]:
        txt_file = config.output_dir / f"success_rates_{model_name}_{timestamp}.txt"
        save_success_rates_txt(results, txt_file, config)

    print_summary(results)

    # 所有 task 完成后，再额外打印一次「包含全部任务」的 metrics 总览表
    print_metrics_ascii_table(results)


if __name__ == "__main__":
    main()
