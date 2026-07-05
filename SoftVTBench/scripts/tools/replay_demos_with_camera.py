# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to replay demonstrations with Isaac Lab environments.

Always record demos with Actions in absolute eef_pose space, prepare for mimic workflow.

Record videos for each camera view, for both successful and failed demos.

Record demo_ids for failed demos in a file of "failure.json".
"""

"""Launch Isaac Sim Simulator first."""
import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# 确保项目根目录在 sys.path 中，便于在 Isaac/Kit 改变工作目录后仍能导入 `benchmarks.*` 等顶层包。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Force to use the specified task.",
)
parser.add_argument(
    "--demo_id",
    type=int,
    default=None,
    help=(
        "Replay a single demo/episode id. "
        "Prefer using (task_suite, task_id, demo_id) instead of specifying a single dataset file path."
    ),
)
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--dump_data", action="store_true", default=False, help="Whether to dump dataset file during demo replay."
)

parser.add_argument(
    "--output_file",
    type=str,
    default=None,
    help="File path to export recorded demos with absolute ee_pose as actions.",
)
parser.add_argument(
    "--task_suite",
    type=str,
    nargs="+",
    default=None,
    help="Task suite(s) to generate dataset for (space-separated), e.g. libero_goal libero_10 libero_spatial libero_object.",
)
parser.add_argument(
    "--task_id",
    type=int,
    nargs="+",
    default=None,
    help=(
        "Task ID to generate dataset for, select from: [416, 425, 443, 461, 470, 480] for xhumanoid, [0-90] for libero."
    ),
)
parser.add_argument(
    "--camera_view_list",
    type=str,
    nargs="+",
    default=None,
    help=(
        "A list of camera views to record videos (e.g., agentview eye_in_hand). "
        "If omitted and --video is set, defaults will be auto-filled."
    ),
)
parser.add_argument(
    "--tactile_sensor_list",
    type=str,
    nargs="+",
    default=None,
    help=(
        "A list of tactile sensors to record videos (e.g., gsmini_left gsmini_right). "
        "If omitted and --video is set in a *Tactile* env, defaults will be auto-filled."
    ),
)
parser.add_argument(
    "--tactile_output_type",
    type=str,
    nargs="+",
    default="markers_rgb",
    help=(
        "Tactile output(s) to record. Use one or more of "
        "[markers_rgb, tactile_rgb, height_map, marker_motion], or 'all'. "
        "RGB/depth-like outputs are exported as videos; tactile_rgb, height_map, "
        "and marker_motion are also exported as raw .npz arrays."
    ),
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during demo replay.")
parser.add_argument("--save_depth", action="store_true", default=False, help="Save depth images during demo replay.")
parser.add_argument(
    "--output_failure_record_file",
    type=str,
    default="failure.jsonl",
    help="File path to the failure record file.",
)
parser.add_argument(
    "--recorder_type",
    type=str,
    default=None,
    choices=["7d2", "8d2", "8dp", "7dp", "7dpf", "7dp2"],
    help=(
        "Select recorder type:\n"
        "  7d2:  Axis-angle (3) + Position (3) + Binary Gripper (1)         = 7D\n"
        "  8d2:  Quaternion (4) + Position (3) + Binary Gripper (1)         = 8D\n"
        "  8dp:  Quaternion (4) + Position (3) + Abs Gripper (1)            = 8D\n"
        "  7dp:  Axis-angle (3) + Position (3) + Abs Gripper (1)            = 7D\n"
        "  7dpf: Axis-angle (3) + Position (3) + Abs Gripper (1) + Force(6) = 13D\n"
        "  7dp2: 7dp saved as actions PLUS command-gripper 7D saved as actions_7d2\n"
    ),
)
parser.add_argument(
    "--viz_force_position",
    action="store_true",
    default=False,
    help="Enable online visualization for hybrid force-position control (Hybrid env only).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Normalize: treat single task_id as scalar int to keep downstream logic unchanged.
if isinstance(args_cli.task_id, list) and len(args_cli.task_id) == 1:
    args_cli.task_id = int(args_cli.task_id[0])

# Normalize optional lists to avoid None downstream
if args_cli.camera_view_list is None:
    args_cli.camera_view_list = []
if args_cli.tactile_sensor_list is None:
    args_cli.tactile_sensor_list = []

if isinstance(args_cli.tactile_output_type, str):
    args_cli.tactile_output_type = [args_cli.tactile_output_type]
if "all" in args_cli.tactile_output_type:
    args_cli.tactile_output_type = ["markers_rgb", "tactile_rgb", "height_map", "marker_motion"]

# 默认启用相机以便录制视频/图像
args_cli.enable_cameras = True
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    # Auto-fill defaults for UX:
    # - If neither camera views nor tactile sensors are provided, record the default camera views.
    # - If the task is tactile and tactile sensors are not provided, record default GelSight sensors.
    is_tactile_task = bool(args_cli.task) and ("Tactile" in args_cli.task)

    if len(args_cli.camera_view_list) == 0 and len(args_cli.tactile_sensor_list) == 0:
        # NOTE: replay_utils uses env.scene.sensors[f"{view}_cam"], so view names should be like "agentview"/"eye_in_hand".
        args_cli.camera_view_list = ["agentview", "eye_in_hand"]

    if is_tactile_task and len(args_cli.tactile_sensor_list) == 0:
        args_cli.tactile_sensor_list = ["gsmini_left", "gsmini_right"]

# 兼容旧版行为：如果只指定了 --task_suite 而没有指定 --task_id，则自动遍历该 suite 的所有任务。
# 注意：这里通过子进程逐个 task_id 调用自身脚本，避免在单进程里反复重建 Isaac/Kit 环境导致不稳定。
if args_cli.task_suite is not None and (
    len(args_cli.task_suite) > 1
    or args_cli.task_id is None
    or (isinstance(args_cli.task_id, list) and len(args_cli.task_id) > 0)
):
    import json
    import subprocess

    def _load_softvtbench_task_subset() -> dict[str, list[int]]:
        """Load SoftVTBench task subset mapping from JSON (best-effort)."""
        path = _PROJECT_ROOT / "benchmarks" / "datasets" / "softvtbench" / "config" / "softvtbench_tasks.json"
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                out: dict[str, list[int]] = {}
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, list):
                        out[k] = [int(x) for x in v]
                return out
        except Exception:
            pass
        return {}

    def _auto_use_softvtbench_subset() -> bool:
        """Enable SoftVTBench subset only when explicitly requested via env var."""
        flag = os.environ.get("USE_SOFTVTBENCH_TASKS", "").strip().lower()
        return flag in ("1", "true", "yes", "y", "on")

    _SOFTVTBENCH_SUBSET_MAP = _load_softvtbench_task_subset() if _auto_use_softvtbench_subset() else {}

    def _infer_task_ids_for_suite(task_suite: str) -> list[int]:
        if task_suite.startswith("libero"):
            cfg_path = _PROJECT_ROOT / "benchmarks" / "datasets" / "libero" / "config" / f"{task_suite}.json"
            if not cfg_path.exists():
                raise FileNotFoundError(f"Cannot find suite config: {cfg_path}")
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            tasks = cfg.get("tasks", [])
            # Prefer explicit task_id field when present.
            ids = []
            for t in tasks:
                try:
                    ids.append(int(t.get("task_id")))
                except Exception:
                    pass
            if ids:
                all_ids = sorted(set(ids))
            else:
                all_ids = list(range(len(tasks)))

            # SoftVTBench / SoftVTBench-force: optionally filter by softvtbench_tasks.json whitelist.
            if _SOFTVTBENCH_SUBSET_MAP and task_suite in _SOFTVTBENCH_SUBSET_MAP:
                allow = set(_SOFTVTBENCH_SUBSET_MAP.get(task_suite, []))
                return [tid for tid in all_ids if tid in allow]
            return all_ids
        raise ValueError(
            f"--task_id is required for task_suite='{task_suite}'. "
            "Auto-enumeration currently supports libero_* suites via benchmarks/datasets/libero/config/<suite>.json."
        )

    suites = list(args_cli.task_suite)

    base_args = list(sys.argv[1:])

    def _remove_flag_and_values_until_next_flag(argv: list[str], flag: str) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(argv):
            if argv[i] == flag:
                i += 1
                while i < len(argv) and not argv[i].startswith("-"):
                    i += 1
                continue
            out.append(argv[i])
            i += 1
        return out

    def _remove_flag_and_one_value(argv: list[str], flag: str) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(argv):
            if argv[i] == flag:
                i += 2
                continue
            out.append(argv[i])
            i += 1
        return out

    base_args = _remove_flag_and_values_until_next_flag(base_args, "--task_suite")
    base_args = _remove_flag_and_values_until_next_flag(base_args, "--task_id")

    failed: list[tuple[str, int]] = []
    for suite in suites:
        if isinstance(args_cli.task_id, list) and len(args_cli.task_id) > 0:
            task_ids = [int(x) for x in args_cli.task_id]
            print(f"[replay_demos_with_camera] Will run suite='{suite}' with task_ids={task_ids}")
        elif args_cli.task_id is None:
            task_ids = _infer_task_ids_for_suite(suite)
            print(f"[replay_demos_with_camera] --task_id not set; will run all tasks in suite '{suite}': {task_ids}")
        else:
            task_ids = [int(args_cli.task_id)]
            print(f"[replay_demos_with_camera] Will run suite='{suite}' with task_id={task_ids[0]}")

        for tid in task_ids:
            cmd = (
                [sys.executable, str(Path(__file__).resolve())]
                + base_args
                + ["--task_suite", suite, "--task_id", str(tid)]
            )
            print(f"\n[replay_demos_with_camera] Running suite='{suite}' task_id={tid}: {' '.join(cmd)}")
            r = subprocess.run(cmd, env=os.environ.copy())
            if r.returncode != 0:
                # Isaac Sim / Replicator 在退出阶段偶发 pybind11::error_already_set（常见为 SIGABRT=-6）。
                # 该崩溃通常发生在 episode 已全部回放完成之后；这里将其视为“已完成但退出异常”，不计为任务失败。
                if r.returncode == -6:
                    print(
                        f"[replay_demos_with_camera] ⚠️ exit=-6 (shutdown crash) for suite='{suite}' task_id={tid}; treat as completed, continue..."
                    )
                    continue
                failed.append((suite, tid))
                print(f"[replay_demos_with_camera] ⚠️ failed suite='{suite}' task_id={tid} (exit={r.returncode}), continue...")
    if failed:
        raise SystemExit(f"[replay_demos_with_camera] Some runs failed: {failed}")
    raise SystemExit(0)

# Normalize to a single suite for the rest of the script (single-run mode).
if args_cli.task_suite is not None and isinstance(args_cli.task_suite, list):
    args_cli.task_suite = args_cli.task_suite[0]

# args_cli.headless = True

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401


# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""
Add configs for dataset generation for various task_suite and task_id,
- supported task_suites: [xhumanoid, libero, etc.]
"""
if args_cli.task_suite is not None:
    from tac_manip.utils.task_configs import setup_task_objects
    from common.replay_utils import resolve_input_hdf5

    setup_task_objects(args_cli.task_suite, args_cli.task_id)
    if args_cli.task_suite.startswith("libero"):
        # 记录模式（dump_data=True）：始终使用 assembled_hdf5 作为「数据源」
        # assembled_hdf5 被视为所有数据的根源，replayed_demos 仅作为回放后重新录制的结果
        if args_cli.dump_data:
            dataset_file, desc = resolve_input_hdf5(
                task=args_cli.task,
                task_suite=args_cli.task_suite,
                task_id=args_cli.task_id,
                prefer='assembled',
            )
            args_cli.dataset_file = dataset_file
            print(f'📊 {desc}')

            # 输出文件（replayed_demos）：
            # - 用户指定 --output_file 最优先
            # - 否则必须设置 OUTPUT_REPLAYED_DEMOS_DIR，并沿用 assembled 的完整文件名（避免丢失 scene/指令信息）
            if args_cli.output_file is None:
                out_dir = os.environ.get("OUTPUT_REPLAYED_DEMOS_DIR", "").strip()
                if not out_dir:
                    raise ValueError(
                        "Missing required env var for recording output: OUTPUT_REPLAYED_DEMOS_DIR\n"
                        "Please set:\n"
                        "  export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos"
                    )
                out_dir_path = Path(out_dir).expanduser().resolve()
                out_dir_path.mkdir(parents=True, exist_ok=True)
                src_name = Path(dataset_file).name
                if src_name.endswith("_demo.hdf5"):
                    filename = src_name[: -len("_demo.hdf5")] + "_replayed_demo.hdf5"
                elif src_name.endswith(".hdf5"):
                    filename = src_name[: -len(".hdf5")] + "_replayed_demo.hdf5"
                else:
                    filename = src_name + "_replayed_demo.hdf5"
                args_cli.output_file = str(out_dir_path / filename)
            print(f"💾 Output file path: {args_cli.output_file}")
        else:
            # 非记录模式（严格）：只允许从 REPLAYED_DEMOS_DIR 按 (task_suite, task_id) 解析输入
            dataset_file, desc = resolve_input_hdf5(
                task=args_cli.task,
                task_suite=args_cli.task_suite,
                task_id=args_cli.task_id,
                prefer='replayed',
            )
            args_cli.dataset_file = dataset_file
            print(f'📊 {desc}')
    else:
        # Non-libero/custom suites: resolve input HDF5 by (task_suite, task_id) from REPLAYED_DEMOS_DIR.
        # This supports the same UX as Libero for "re-collect with Isaac" workflows.
        dataset_file, desc = resolve_input_hdf5(
            task=args_cli.task,
            task_suite=args_cli.task_suite,
            task_id=args_cli.task_id,
            prefer='replayed',
        )
        args_cli.dataset_file = dataset_file
        print(f'📊 {desc}')

        # 输出文件：
        # - 用户指定 --output_file 最优先
        # - 否则优先使用 OUTPUT_REPLAYED_DEMOS_DIR 作为输出目录，并沿用输入 HDF5 的文件名
        # - 不允许回落到默认路径（严格）
        if args_cli.output_file is None:
            out_dir = os.environ.get("OUTPUT_REPLAYED_DEMOS_DIR", "").strip()
            if out_dir:
                out_dir_path = Path(out_dir).expanduser().resolve()
                out_dir_path.mkdir(parents=True, exist_ok=True)
                src_name = None
                if getattr(args_cli, "dataset_file", None):
                    try:
                        src_name = Path(args_cli.dataset_file).name
                    except Exception:
                        src_name = None
                filename = src_name or "replay_demos.hdf5"
                args_cli.output_file = str(out_dir_path / filename)
            else:
                raise ValueError(
                    "Missing required env var for recording output: OUTPUT_REPLAYED_DEMOS_DIR\n"
                    "Please set:\n"
                    "  export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos\n"
                    "Or pass --output_file explicitly."
                )

        print(f"💾 Output file path: {args_cli.output_file}")
else:
    # Strict mode: require task_suite/task_id to resolve input HDF5 for replay/recording.
    # This avoids silently selecting a wrong data source via env fallbacks.
    raise ValueError(
        "Please specify --task_suite and --task_id for replay_demos_with_camera.py "
        "(--demo_id is optional)."
    )

# Recording note:
# If you want to include finger/gelpad forces in recorded actions (13D), you must use 7dpf recorder.
# For tactile/contact-force replay envs, using 7dp/8dp/7d2/8d2 will record *no* Force(6) in actions.
if args_cli.dump_data and args_cli.task:
    if (
        ("Replay-Camera-ContactForce" in args_cli.task)
        or ("Replay-Camera-Tactile" in args_cli.task)
        or ("Hybrid-" in args_cli.task)
    ):
        if args_cli.recorder_type in ("7dp", "8dp", "7d2", "8d2", "7dp2") or args_cli.recorder_type is None:
            print(
                "⚠️  [Recorder] You are recording in a force-capable env, but recorder_type is not set to 7dpf.\n"
                "    To include current Force(6) (left/right 3D) in recorded actions, use: --recorder_type 7dpf."
            )

"""Rest everything follows."""

import contextlib

import gymnasium as gym
import torch
try:
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
except ImportError:
    from isaaclab.devices import Se3Keyboard
    Se3KeyboardCfg = None
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from benchmarks.common.metrics import compute_contact_force_metrics_from_lr_forces

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from common.replay_utils import (
    compare_states,
    get_episode_map,
    display_action_info,
    process_failed_demo_videos,
    process_failed_tactile_videos,
    process_successful_demo_videos,
    process_successful_tactile_videos,
    save_camera_images,
    save_tactile_images,
    setup_replay_output_directories,
    write_failure_jsonl,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tac_manip.core.recorders import (
    AbsEEFPoseAbsGripperActionStateRecorderManagerCfg,
    AbsEEFPoseAxisAngleAbsGripperActionStateRecorderManagerCfg,
    AbsEEFPoseAxisAngleAbsGripperWithForceActionStateRecorderManagerCfg,
    AbsEEFPoseAxisAngleBinaryGripperActionStateRecorderManagerCfg,
    AbsEEFPoseAxisAngleDualGripperActionStateRecorderManagerCfg,
    AbsEEFPoseBinaryGripperActionStateRecorderManagerCfg,
)

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def _print_goal_diagnostics(env, goals, env_id: int) -> None:
    """Print final goal metrics for replay failures without changing success logic."""
    if not goals:
        return
    try:
        print(f"[GoalDiag] env={env_id} final goal diagnostics:")
        for i, goal in enumerate(goals):
            if "relationship" not in goal:
                print(f"[GoalDiag] goal[{i}] unsupported goal type keys={list(goal.keys())}")
                continue

            obj_name = goal["ref_obj"]
            target_name = goal["target"]
            obj = env.scene[obj_name]
            target = env.scene[target_name]
            pos_diff = obj.data.root_pos_w - target.data.root_pos_w

            if "orientation" in goal:
                orientation = goal["orientation"]
                if orientation == "right":
                    pos_diff = pos_diff + torch.tensor([0.0, -0.05, 0.0], device=env.device)
                elif orientation == "left":
                    pos_diff = pos_diff + torch.tensor([0.0, 0.05, 0.0], device=env.device)
                elif orientation == "front":
                    pos_diff = pos_diff + torch.tensor([-0.35, 0.0, 0.0], device=env.device)

            xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)[env_id].item()
            height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)[env_id].item()
            xy_threshold = float(goal["xy_threshold"])
            height_threshold = float(goal["height_threshold"])
            height_diff = float(goal["height_diff"])
            xy_ok = xy_dist < xy_threshold
            height_ok = (height_dist - height_diff) < height_threshold

            contact_name = f"contact_{target_name}_{obj_name}"
            contact_msg = "disabled"
            if goal.get("enable_force_threshold") == "True":
                if contact_name in env.scene.keys() and env.scene[contact_name] is not None:
                    contact_force = env.scene[contact_name].data.force_matrix_w
                    contact_norm = torch.linalg.vector_norm(contact_force.squeeze(2).squeeze(1), dim=1)[env_id].item()
                    force_threshold = float(goal["force_threshold"])
                    contact_msg = f"{contact_norm:.6f} < {force_threshold:.6f} => {contact_norm > force_threshold}"
                else:
                    contact_msg = f"missing sensor {contact_name}"

            print(
                "[GoalDiag] "
                f"goal[{i}] obj={obj_name} target={target_name} "
                f"xy={xy_dist:.6f}<{xy_threshold:.6f}=>{xy_ok} "
                f"height_adj={(height_dist - height_diff):.6f}<{height_threshold:.6f}=>{height_ok} "
                f"contact={contact_msg}"
            )
    except Exception as exc:
        print(f"[GoalDiag] failed to compute diagnostics: {exc}")


def _print_tactile_diagnostics(env, frame_index: int, sensor_names: list[str]) -> None:
    """Print TacEx/GelSight internal buffers for debugging dead tactile outputs."""
    if os.environ.get("SOFTVTBENCH_TACTILE_DIAG", "").strip().lower() not in ("1", "true", "yes", "y", "on"):
        return
    every = int(os.environ.get("SOFTVTBENCH_TACTILE_DIAG_EVERY", "10"))
    if every > 0 and frame_index % every != 0:
        return
    for sensor_name in sensor_names:
        try:
            if sensor_name not in env.scene.sensors:
                continue
            sensor = env.scene.sensors[sensor_name]
            out = sensor.data.output
            height_map = out.get("height_map", None)
            tactile_rgb = out.get("tactile_rgb", None)
            marker_motion = out.get("marker_motion", None)
            indentation = getattr(sensor, "indentation_depth", None)

            parts = [f"[TactileDiag] frame={frame_index} sensor={sensor_name}"]
            if height_map is not None:
                hm = height_map.detach()
                parts.append(
                    f"height_mm[min={hm.min().item():.4f}, max={hm.max().item():.4f}, mean={hm.mean().item():.4f}]"
                )
            else:
                parts.append("height_map=None")
            if indentation is not None:
                ind = indentation.detach()
                parts.append(
                    f"indent_mm[min={ind.min().item():.6f}, max={ind.max().item():.6f}, mean={ind.mean().item():.6f}]"
                )
            else:
                parts.append("indent=None")
            if tactile_rgb is not None:
                rgb = tactile_rgb.detach().float()
                parts.append(f"rgb[std={rgb.std().item():.4f}, min={rgb.min().item():.1f}, max={rgb.max().item():.1f}]")
            else:
                parts.append("tactile_rgb=None")
            if marker_motion is not None:
                mm = marker_motion.detach()
                disp = mm[:, 1, :, :] - mm[:, 0, :, :]
                mag = torch.linalg.vector_norm(disp, dim=-1)
                parts.append(f"marker_px[mean={mag.mean().item():.6f}, max={mag.max().item():.6f}]")
            else:
                parts.append("marker_motion=None")
            print(" ".join(parts))
        except Exception as exc:
            print(f"[TactileDiag] sensor={sensor_name} failed: {exc}")


def main():  # noqa: C901
    """Replay episodes loaded from a file."""
    global is_paused

    # Load dataset
    if not getattr(args_cli, "dataset_file", None):
        raise ValueError(
            "Input HDF5 is not set.\n"
            "Strict mode requires:\n"
            "  - For replay (non-dump): export REPLAYED_DEMOS_DIR=... and provide --task_suite/--task_id\n"
            "  - For recording (dump_data): export HDF5_TRAJ_SOURCE_DIR=... and export OUTPUT_REPLAYED_DEMOS_DIR=...\n"
        )
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    # Get actual episode indices from episode_map (may be sparse, not necessarily 0,1,2,...)
    episode_map = get_episode_map(dataset_file_handler.get_episode_names())
    
    episode_indices_to_replay = [args_cli.demo_id] if args_cli.demo_id is not None else []
    if len(episode_indices_to_replay) == 0:
        # Get actual episode indices from the dataset instead of using range
        episode_map = get_episode_map(dataset_file_handler.get_episode_names())
        episode_indices_to_replay = sorted(episode_map.keys())
        print(f"Found {len(episode_indices_to_replay)} episodes in dataset:")
        print(f"Episode indices: {episode_indices_to_replay}")
        if len(episode_indices_to_replay) != episode_count:
            print(f"Warning: Duplicate episodes found in the dataset: {episode_indices_to_replay}")
    else:
        episode_map = get_episode_map(dataset_file_handler.get_episode_names())
        print(f"Selected demo_id to replay: {episode_indices_to_replay[0]}")
        # Validate that selected episodes exist in the dataset
        available_episodes = set(episode_map.keys())
        invalid_episodes = [ep for ep in episode_indices_to_replay if ep not in available_episodes]
        if invalid_episodes:
            print(f"Warning: The following selected episodes do not exist in the dataset: {invalid_episodes}")
            print(f"Available episodes: {sorted(available_episodes)}")
            episode_indices_to_replay = [ep for ep in episode_indices_to_replay if ep in available_episodes]

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    # 视频与触觉输出的根目录前缀（严格）：
    # - 如果启用 --video，则必须设置 OUTPUT_REPLAYED_VIDEOS_DIR
    replay_videos_root = ""
    if args_cli.video:
        replay_videos_root = os.environ.get("OUTPUT_REPLAYED_VIDEOS_DIR", "").strip()
        if not replay_videos_root:
            raise ValueError(
                "Missing required env var for video output: OUTPUT_REPLAYED_VIDEOS_DIR\n"
                "Please set:\n"
                "  export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets"
    )

    if args_cli.video:
        video_save_dir, tactile_outputs_save_dir = setup_replay_output_directories(
            True,
            args_cli.tactile_sensor_list,
            args_cli.task_suite,
            args_cli.task_id,
            args_cli.task,
            root_dir_prefix=replay_videos_root,
        )

    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    if args_cli.dump_data:
        # get directory path and file name (without extension) from cli arguments
        if args_cli.output_file:
            output_dir = os.path.dirname(args_cli.output_file)
            output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
        else:
            raise ValueError(
                "Missing output_file in dump_data mode.\n"
                "Strict mode requires either:\n"
                "  - pass --output_file\n"
                "  - or export OUTPUT_REPLAYED_DEMOS_DIR (the script will auto-derive a filename)"
            )

        # create directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.env_name = args_cli.task

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        print(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.dump_data:
        # Select recorder based on type
        if args_cli.recorder_type == "7d2":
            # 7D: Axis-angle (6D) + Binary Gripper (1D)
            env_cfg.recorders = AbsEEFPoseAxisAngleBinaryGripperActionStateRecorderManagerCfg()
        elif args_cli.recorder_type == "7dp":
            # 7D: Axis-angle (6D) + Abs Gripper (1D)
            env_cfg.recorders = AbsEEFPoseAxisAngleAbsGripperActionStateRecorderManagerCfg()
        elif args_cli.recorder_type == "7dp2":
            # Dual: actions = 7dp (abs gripper pos); actions_7d2 = command gripper
            env_cfg.recorders = AbsEEFPoseAxisAngleDualGripperActionStateRecorderManagerCfg()
        elif args_cli.recorder_type == "8d2":
            # 8D: Quaternion (7D) + Binary Gripper (1D) - Default legacy custom recorder
            env_cfg.recorders = AbsEEFPoseBinaryGripperActionStateRecorderManagerCfg()
        elif args_cli.recorder_type == "8dp":
            # 8D: Quaternion (7D) + Abs Gripper (1D)
            env_cfg.recorders = AbsEEFPoseAbsGripperActionStateRecorderManagerCfg()
        elif args_cli.recorder_type == "7dpf":
            # 13D: Axis-angle (6D) + Abs Gripper (1D) + Force (6D)
            env_cfg.recorders = AbsEEFPoseAxisAngleAbsGripperWithForceActionStateRecorderManagerCfg()
        else:
            # Default
            env_cfg.recorders = ActionStateRecorderManagerCfg()

        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.export_in_record_pre_reset = False

    # -- Setting physics option. -- #
    env_cfg.sim.physx.enable_ccd = True

    # create environment from loaded config
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # 可选：为 Hybrid 环境开启力–位混合调试可视化
    force_viz = None
    if args_cli.viz_force_position and "Isaac-Libero-Franka-Hybrid-" in args_cli.task:
        # 延迟导入 matplotlib 相关依赖，只有真的需要可视化时才导入
        from common.force_position_debug_viz import ForcePositionDebugVisualizer

        force_viz = ForcePositionDebugVisualizer()

    # 夹爪挤压力的 per-episode / per-task 统计（无论是否打开可视化，都统计）
    fsq_pred_sum = 0.0
    fsq_meas_sum = 0.0
    fsq_count = 0
    succ_squeeze_pred_sum = 0.0
    succ_squeeze_meas_sum = 0.0
    succ_squeeze_count = 0

    def _update_squeeze_stats(debug: dict):
        """在每个 step 更新本 episode 的挤压力统计."""
        nonlocal fsq_pred_sum, fsq_meas_sum, fsq_count
        if not debug:
            return
        f_pred = debug.get("f_sq_pred", None)
        f_meas = debug.get("f_sq_meas", None)
        if f_pred is None or f_meas is None:
            return
        try:
            fsq_pred_sum += float(f_pred)
            fsq_meas_sum += float(f_meas)
            fsq_count += 1
        except Exception:
            pass

    # 7dpf：逐步缓存左右指力，用于在每条成功 demo 后打印「平均夹持力/平均外力」
    # 说明：7dpf 录制到 actions 的 Force(6) 就来自 policy obs 的 gripper_net_force（见 tac_manip/core/recorders）。
    force_l_hist: dict[int, list[torch.Tensor]] = {}
    force_r_hist: dict[int, list[torch.Tensor]] = {}

    def _append_force_histories() -> None:
        """Append current finger forces (if available) into per-env histories."""
        if args_cli.recorder_type != '7dpf':
            return
        try:
            obs = env.obs_buf.get('policy', {})
            gnf = obs.get('gripper_net_force', None)
            if gnf is None:
                return
            # Shape: (N, history_len, 2, 3) -> take most recent index 0
            cur = gnf[:, 0, :, :]  # (N, 2, 3)
            for env_id in range(int(cur.shape[0])):
                force_l_hist.setdefault(env_id, []).append(cur[env_id, 0, :].detach().cpu())
                force_r_hist.setdefault(env_id, []).append(cur[env_id, 1, :].detach().cpu())
        except Exception:
            # force stats should never break replay/recording
            return

    if Se3KeyboardCfg is not None:
        teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    else:
        teleop_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

        # For abs eef_space actions with quaternions, we need to initialize quaternion part to identity
        # Check if this is an eef_space action with quaternions
        action_dim = env.action_space.shape[-1]
        is_eef_space = "IK" in args_cli.task or "RmpFlow" in args_cli.task
        if is_eef_space and action_dim >= 8:  # Common Abs eef_space dimensions
            # Initialize quaternion part (indices 3:7) to identity quaternion [1, 0, 0, 0]
            idle_action[..., 3] = 1.0  # w component
            idle_action[..., 4:7] = 0.0  # x, y, z components

    # reset before starting
    env.reset()
    teleop_interface.reset()

    # simulate environment -- run everything in inference mode
    # episode_map already created earlier (line 229)
    replayed_episode_count = 0
    recorded_episode_count = 0

    # Track current episode indices for each environment
    current_episode_indices = [None] * num_envs

    # Track failed demo IDs
    failed_demo_ids = []

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            episode_ended = [False] * num_envs
            while has_next_action:
                # initialize actions with idle action so those without next action will not move
                actions = idle_action

                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    if env_next_action is None:
                        next_episode_index = None
                        if episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            episode_ended[env_id] = False
                        else:
                            next_episode_index = None

                        # check if the episode is successful after the whole episode_data is evaluated, and episode not ended, which means we have next_episode to play for this env
                        if success_term is not None and current_episode_indices[env_id] is not None:
                            if (
                                bool(success_term.func(env, **success_term.params)[env_id])
                                and not episode_ended[env_id]
                            ):
                                recorded_episode_count += 1
                                plural_trailing_s = "s" if recorded_episode_count > 1 else ""
                                episode_ended[env_id] = True
                                if args_cli.dump_data:
                                    print(f"📝 Dumping data for env {env_id} (demo_id: {current_episode_indices[env_id]})...")

                                    # 标记 episode 成功并导出到 HDF5（replayed_demos）
                                    env.recorder_manager.set_success_to_episodes(
                                        [env_id], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                                    )
                                    env.recorder_manager.export_episodes([env_id])
                                    env.recorder_manager.reset([env_id])

                                    # 计算并打印轨迹 HDF5 文件路径（与 REPLAYED_DEMOS_PATH / --output_file 一致）
                                    if args_cli.output_file:
                                        replay_hdf5_path = args_cli.output_file
                                    else:
                                        raise ValueError("Internal error: output_file must be set in dump_data mode.")

                                    print(
                                        f"💾 Replayed demos HDF5 saved at: {replay_hdf5_path} "
                                        f"(total dumped {recorded_episode_count} episode{plural_trailing_s})."
                                    )

                                    # 7dpf：在每条成功 demo 后打印平均夹持力/平均外力（基于 gripper_net_force）
                                    if args_cli.recorder_type == '7dpf':
                                        try:
                                            fL_list = force_l_hist.get(env_id, [])
                                            fR_list = force_r_hist.get(env_id, [])
                                            if fL_list and fR_list and len(fL_list) == len(fR_list):
                                                fL = torch.stack(fL_list, dim=0)  # (T, 3)
                                                fR = torch.stack(fR_list, dim=0)  # (T, 3)
                                                m = compute_contact_force_metrics_from_lr_forces(fL, fR)
                                                print(
                                                    f"[7dpf-Forces] Episode #{current_episode_indices[env_id]} "
                                                    f"squeeze_mean={m.squeeze_mean:.4f}, "
                                                    f"external_mean={m.external_norm_mean:.4f}"
                                                )
                                        except Exception:
                                            pass
                                        finally:
                                            force_l_hist[env_id] = []
                                            force_r_hist[env_id] = []
                                else:
                                    print(
                                        f"Successfully replayed {recorded_episode_count} episode{plural_trailing_s} out"
                                        f" of {replayed_episode_count} demos."
                                    )
                                    if args_cli.recorder_type == '7dpf':
                                        force_l_hist[env_id] = []
                                        force_r_hist[env_id] = []
                                # 仅在成功 episode 时，输出整集挤压力均值（env0）并累积到 task 级别
                                if env_id == 0 and current_episode_indices[env_id] is not None:
                                    if fsq_count > 0:
                                        avg_pred = fsq_pred_sum / fsq_count
                                        avg_meas = fsq_meas_sum / fsq_count
                                        succ_squeeze_pred_sum += avg_pred
                                        succ_squeeze_meas_sum += avg_meas
                                        succ_squeeze_count += 1
                                        print(
                                            f"[Hybrid] Episode #{current_episode_indices[env_id]} "
                                            f"avg squeeze_pred={avg_pred:.4f}, squeeze_meas={avg_meas:.4f}"
                                        )

                                    # 重置本 episode 统计
                                    fsq_pred_sum = fsq_meas_sum = 0.0
                                    fsq_count = 0
                                # 成功 episode 也重置可视化
                                if force_viz is not None and env_id == 0:
                                    try:
                                        force_viz.reset()
                                    except Exception:
                                        pass
                                # if video is enabled, encode image sequence to video to save space
                                if args_cli.video:
                                    process_successful_demo_videos(
                                        current_episode_indices[env_id],
                                        video_save_dir,
                                        True,
                                        args_cli.camera_view_list,
                                        save_depth=args_cli.save_depth,
                                        task_suite=args_cli.task_suite,
                                        task_id=args_cli.task_id,
                                        task=args_cli.task,
                                        root_dir_prefix=replay_videos_root,
                                    )
                                    process_successful_tactile_videos(
                                        current_episode_indices[env_id],
                                        tactile_outputs_save_dir,
                                        True,
                                        args_cli.tactile_sensor_list,
                                        tactile_output_type=args_cli.tactile_output_type,
                                        task_suite=args_cli.task_suite,
                                        task_id=args_cli.task_id,
                                        task=args_cli.task,
                                        root_dir_prefix=replay_videos_root,
                                    )
                            else:
                                # if not successful, add to failed demo IDs list
                                if current_episode_indices[env_id] is not None:
                                    failed_demo_ids.append(current_episode_indices[env_id])
                                    if args_cli.dump_data:
                                        print(f"⚠️  Replay failed for demo {current_episode_indices[env_id]}, skipping data dump.")
                                        _print_goal_diagnostics(
                                            env,
                                            getattr(success_term, "params", {}).get("goals", []) if success_term is not None else [],
                                            env_id,
                                        )
                                # 失败 episode：只重置 per-episode 统计与可视化，不打印均值
                                if env_id == 0:
                                    fsq_pred_sum = fsq_meas_sum = 0.0
                                    fsq_count = 0
                                    if force_viz is not None:
                                        try:
                                            force_viz.reset()
                                        except Exception:
                                            pass

                                # 失败时也清空 7dpf 力缓存
                                if args_cli.recorder_type == '7dpf':
                                    force_l_hist[env_id] = []
                                    force_r_hist[env_id] = []

                                # if not successful, rename the generated_images folder from demo_id to failed_id
                                if args_cli.video and not first_loop and current_episode_indices[env_id] is not None:
                                    process_failed_demo_videos(
                                        current_episode_indices[env_id],
                                        video_save_dir,
                                        True,
                                        args_cli.camera_view_list,
                                        save_depth=args_cli.save_depth,
                                        task_suite=args_cli.task_suite,
                                        task_id=args_cli.task_id,
                                        task=args_cli.task,
                                        root_dir_prefix=replay_videos_root,
                                    )
                                    process_failed_tactile_videos(
                                        current_episode_indices[env_id],
                                        tactile_outputs_save_dir,
                                        True,
                                        args_cli.tactile_sensor_list,
                                        tactile_output_type=args_cli.tactile_output_type,
                                        task_suite=args_cli.task_suite,
                                        task_id=args_cli.task_id,
                                        task=args_cli.task,
                                        root_dir_prefix=replay_videos_root,
                                    )

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            current_episode_indices[env_id] = next_episode_index
                            print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_map[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data

                            # Display action information based on environment type and data source
                            display_action_info(
                                episode_data,
                                args_cli.task,
                                args_cli.dataset_file,
                                env.device,
                            )

                            # 每个新 episode 开始时重置力–位可视化的时间轴与曲线
                            if force_viz is not None and env_id == 0:
                                try:
                                    force_viz.reset()
                                except Exception:
                                    pass

                            # check if initial_state exists
                            if "initial_state" in episode_data.data:
                                # Set initial state for the new episode
                                initial_state = episode_data.get_initial_state()
                                env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)

                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True

                    actions[env_id] = env_next_action

                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue

                if has_next_action:  # Apply actions only if get value from hdf5 file
                    # -- save images during replay -- #
                    if args_cli.video:
                        for env_id in range(num_envs):
                            if current_episode_indices[env_id] is not None:
                                frame_index = env_episode_data_map[env_id].next_action_index - 1
                                # tactile frames
                                save_tactile_images(
                                    env,
                                    env_id,
                                    current_episode_indices[env_id],
                                    frame_index,
                                    True,
                                    args_cli.tactile_sensor_list,
                                    tactile_output_type=args_cli.tactile_output_type,
                                    task_suite=args_cli.task_suite,
                                    task_id=args_cli.task_id,
                                    task=args_cli.task,
                                    root_dir_prefix=replay_videos_root,
                                )
                                # camera frames
                                save_camera_images(
                                    env,
                                    env_id,
                                    current_episode_indices[env_id],
                                    frame_index,
                                    True,
                                    args_cli.camera_view_list,
                                    save_depth=args_cli.save_depth,
                                    task_suite=args_cli.task_suite,
                                    task_id=args_cli.task_id,
                                    task=args_cli.task,
                                    root_dir_prefix=replay_videos_root,
                                )

                    # apply actions
                    env.step(actions)
                    try:
                        current_frame_index = max(
                            0,
                            max(
                                [
                                    env_episode_data_map[env_id].next_action_index - 1
                                    for env_id in range(num_envs)
                                    if current_episode_indices[env_id] is not None
                                ]
                                or [0]
                            ),
                        )
                        _print_tactile_diagnostics(env, current_frame_index, args_cli.tactile_sensor_list)
                    except Exception:
                        pass

                    # 更新力–位混合调试可视化 / 统计挤压力（仅 Hybrid 的 arm_action 提供 debug_info）
                    try:
                        term = env.action_manager.get_term("arm_action")
                        debug = getattr(term, "debug_info", None)
                    except Exception:
                        debug = None

                    if debug:
                        _update_squeeze_stats(debug)
                        if force_viz is not None and num_envs == 1:
                            try:
                                force_viz.update(debug)
                            except Exception:
                                # 调试可视化不应影响正常重放流程
                                pass

                    # 7dpf：缓存当前 step 的左右指力（用于 episode 结束后打印均值）
                    _append_force_histories()

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1 :4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)

            break

    # Write failed demo IDs to failure.json file
    if failed_demo_ids:
        if args_cli.task_suite is not None:
            key = f"{args_cli.task_suite}_task{args_cli.task_id}"
        else:
            key = args_cli.task
        write_failure_jsonl(args_cli.output_failure_record_file, key, failed_demo_ids)

    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")

    # 输出 task 级（所有成功 episode）的平均挤压力，供 run_data_evaluations 解析使用
    if succ_squeeze_count > 0:
        task_avg_pred = succ_squeeze_pred_sum / succ_squeeze_count
        task_avg_meas = succ_squeeze_meas_sum / succ_squeeze_count
        print(
            f"[Hybrid] Task avg squeeze_pred={task_avg_pred:.4f}, "
            f"squeeze_meas={task_avg_meas:.4f} over {succ_squeeze_count} successes"
        )
    # IsaacSim/Replicator 在退出清理阶段偶发崩溃（tiled_camera.__del__ -> weakref 失效）。
    # 若你只关心“数据已落盘”，可用该开关直接绕过清理阶段，避免 core dump。
    if os.environ.get("ISAAC_FAST_EXIT", "").strip().lower() in ("1", "true", "yes", "y", "on"):
        os._exit(0)

    try:
        env.close()
    except Exception:
        pass

    if force_viz is not None:
        force_viz.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
