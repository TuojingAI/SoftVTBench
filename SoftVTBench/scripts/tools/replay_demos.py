# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# 确保项目根目录在 sys.path 中，便于在不同运行方式下导入 `benchmarks.*` 等顶层包。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--demo_id",
    type=int,
    default=None,
    help=(
        "Replay a single demo/episode id (for debugging or manual inspection). "
        "For Libero tasks, (task_suite, task_id, demo_id) is the recommended way to pick a single demo."
    ),
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="datasets/dataset.hdf5",
    help=(
        "Input HDF5 dataset file to replay. Useful for manually collected datasets (e.g., from record_demos.py), "
        "or for ad-hoc debugging. For Libero tasks, you can also rely on TASK_SUITE/TASK_ID + env vars to auto-resolve."
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
    default="./replay_demos.hdf5",
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
    default=0,
    help=(
        "Task ID to generate dataset for, select from: [416, 425, 443, 461, 470, 480] for xhumanoid, [0-90] for libero."
    ),
)
parser.add_argument(
    "--output_failure_record_file",
    type=str,
    default="failure.jsonl",
    help="File path to the failure record file.",
)
parser.add_argument(
    "--viz_force",
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

# 兼容旧版行为：支持一次传多个 --task_suite。
# - 若用户未显式传 --task_id：自动遍历每个 suite 的所有 task_id
# - 若用户显式传了 --task_id：对每个 suite 运行该 task_id
# 注意：这里通过子进程逐个 (suite, task_id) 调用自身脚本，避免在单进程里反复重建 Isaac/Kit 环境导致不稳定。
if args_cli.task_suite is not None and (len(args_cli.task_suite) > 1 or "--task_id" not in sys.argv):
    import json
    import subprocess

    def _infer_task_ids_for_suite(task_suite: str) -> list[int]:
        if task_suite.startswith("libero"):
            cfg_path = _PROJECT_ROOT / "benchmarks" / "datasets" / "libero" / "config" / f"{task_suite}.json"
            if not cfg_path.exists():
                raise FileNotFoundError(f"Cannot find suite config: {cfg_path}")
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            tasks = cfg.get("tasks", [])
            ids = []
            for t in tasks:
                try:
                    ids.append(int(t.get("task_id")))
                except Exception:
                    pass
            if ids:
                return sorted(set(ids))
            return list(range(len(tasks)))
        raise ValueError(
            f"--task_id is required for task_suite='{task_suite}'. "
            "Auto-enumeration currently supports libero_* suites via benchmarks/datasets/libero/config/<suite>.json."
        )

    suites = list(args_cli.task_suite)
    explicit_task_id = "--task_id" in sys.argv

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
        if explicit_task_id and isinstance(args_cli.task_id, list) and len(args_cli.task_id) > 0:
            task_ids = [int(x) for x in args_cli.task_id]
            print(f"[replay_demos] Will run suite='{suite}' with task_ids={task_ids}")
        elif explicit_task_id:
            task_ids = [int(args_cli.task_id)]
            print(f"[replay_demos] Will run suite='{suite}' with task_id={task_ids[0]}")
        else:
            task_ids = _infer_task_ids_for_suite(suite)
            print(f"[replay_demos] --task_id not set; will run all tasks in suite '{suite}': {task_ids}")

        for tid in task_ids:
            cmd = (
                [sys.executable, str(Path(__file__).resolve())]
                + base_args
                + ["--task_suite", suite, "--task_id", str(tid)]
            )
            print(f"\n[replay_demos] Running suite='{suite}' task_id={tid}: {' '.join(cmd)}")
            r = subprocess.run(cmd, env=os.environ.copy())
            if r.returncode != 0:
                failed.append((suite, tid))
                print(f"[replay_demos] ⚠️ failed suite='{suite}' task_id={tid} (exit={r.returncode}), continue...")
    if failed:
        raise SystemExit(f"[replay_demos] Some runs failed: {failed}")
    raise SystemExit(0)

# Normalize to a single suite for the rest of the script (single-run mode).
if args_cli.task_suite is not None and isinstance(args_cli.task_suite, list):
    args_cli.task_suite = args_cli.task_suite[0]


# 默认开启 cameras，避免忘记加 --enable_cameras 导致相机初始化报错。
args_cli.enable_cameras = True

# 对于带 Camera 的环境（Libero *_Camera* 系列），必须启用 cameras，否则相机传感器初始化会报错。
# 这里保留基于 task 名的逻辑（即使上面已经默认打开），主要用于文档提示含义。
if args_cli.task is not None and "Camera" in args_cli.task:
    args_cli.enable_cameras = True

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
    from common.replay_utils import (
        resolve_input_hdf5,
    )

    setup_task_objects(args_cli.task_suite, args_cli.task_id)
    # If user didn't pass --dataset_file, resolve it via env vars in one place.
    if args_cli.dataset_file == "datasets/dataset.hdf5":
        # Manual recording support (record_demos.py) has explicit precedence when configured.
        if os.environ.get("RECORDED_DEMOS_PATH", "").strip() or os.environ.get("RECORDED_DEMOS_DIR", "").strip():
            dataset_file, desc = resolve_input_hdf5(
                task=args_cli.task,
                task_suite=args_cli.task_suite,
                task_id=args_cli.task_id,
                prefer="recorded",
            )
            args_cli.dataset_file = dataset_file
            print(f"📊 {desc}")
        else:
            # Libero: auto picks replayed vs assembled by env type. Others: treat as replayed_demos resolver.
            prefer = "auto" if args_cli.task_suite.startswith("libero") else "replayed"
            dataset_file, desc = resolve_input_hdf5(
                task=args_cli.task,
                task_suite=args_cli.task_suite,
                task_id=args_cli.task_id,
                prefer=prefer,
            )
            args_cli.dataset_file = dataset_file
            print(f"📊 {desc}")

    # Strict: if user didn't explicitly pass --dataset_file, then one of the env-based resolvers must succeed.
    if args_cli.dataset_file == "datasets/dataset.hdf5":
        raise ValueError(
            "Input HDF5 is not set.\n"
            "Strict mode requires either:\n"
            "  - pass --dataset_file /path/to/xxx.hdf5\n"
            "  - OR export REPLAYED_DEMOS_DIR=/path/to/replayed_demos (and provide --task_suite/--task_id)\n"
            "  - OR export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5 (for joint-space source; auto-resolve)\n"
            "  - OR export RECORDED_DEMOS_PATH=/path/to/recorded_demo.hdf5 (manual recording)\n"
            "  - OR export RECORDED_DEMOS_DIR=/path/to/recorded_demos_dir (manual recording; requires task_suite/task_id)\n"
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
from benchmarks.common.metrics import compute_contact_force_metrics_from_13d, compute_topk_mean

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from common.replay_utils import (
    compare_states,
    get_episode_map,
    write_failure_jsonl,
    display_action_info,
    convert_action_axisangle_to_quat,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def main():  # noqa: C901
    """Replay episodes loaded from a file."""
    global is_paused

    # Load dataset
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
        episode_indices_to_replay = sorted(episode_map.keys())
        print(f"Found {len(episode_indices_to_replay)} episodes in dataset:")
        print(f"Episode indices: {episode_indices_to_replay}")
        if len(episode_indices_to_replay) != episode_count:
            print(f"Warning: Duplicate episodes found in the dataset: {episode_indices_to_replay}")
    else:
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

    num_envs = args_cli.num_envs

    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.env_name = args_cli.task

    # Extract success checking function for main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        print(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # Disable timeout to allow episodes to run until goal is reached
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # Enable continuous collision detection for better physics accuracy
    env_cfg.sim.physx.enable_ccd = True

    # create environment from loaded config
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # 可选：为 Hybrid 环境开启力–位混合调试可视化（仅支持单 env）
    force_viz = None
    if args_cli.viz_force and "Isaac-Libero-Franka-Hybrid-" in args_cli.task and num_envs == 1:
        # 延迟导入 matplotlib 相关依赖，只有真的需要可视化时才导入
        from common.force_position_debug_viz import ForcePositionDebugVisualizer

        force_viz = ForcePositionDebugVisualizer()

    # 夹爪挤压力的 per-episode 统计（无论是否打开可视化，都统计）
    fsq_pred_sum = 0.0
    fsq_meas_sum = 0.0
    fsq_count = 0
    # 每个 episode 内逐帧挤压力 / 加持力模长（用于 Top5% 统计，env0）
    fsq_meas_values: list[float] = []
    ap_pred_values: list[float] = []
    ap_meas_values: list[float] = []

    # 所有成功 episode 的平均挤压力（用于整个任务的统计）
    succ_squeeze_pred_sum = 0.0
    succ_squeeze_meas_sum = 0.0
    succ_squeeze_count = 0

    # Hybrid 力学 metrics 的任务级统计（仅在 action_dim == 13 时启用）
    succ_metrics_count = 0
    succ_squeeze_max_sum = 0.0
    succ_squeeze_mean_sum = 0.0
    succ_app_max_sum = 0.0
    succ_app_mean_sum = 0.0

    # 统计：跨所有成功 episode 的「实测」Top5% 最大挤压力 / 加持力及平均加持力
    succ_squeeze_max_meas_sum = 0.0
    succ_squeeze_max_meas_count = 0
    succ_ap_mean_meas_sum = 0.0
    succ_ap_mean_meas_count = 0
    succ_ap_max_meas_sum = 0.0
    succ_ap_max_meas_count = 0

    def _update_squeeze_stats(debug: dict):
        """在每个 step 更新本 episode 的挤压力 / 加持力统计."""
        nonlocal fsq_pred_sum, fsq_meas_sum, fsq_count, fsq_meas_values, ap_pred_values, ap_meas_values
        if not debug:
            return
        f_pred = debug.get("f_sq_pred", None)
        f_meas = debug.get("f_sq_meas", None)
        if f_pred is None or f_meas is None:
            return
        try:
            f_pred_val = float(f_pred)
            f_meas_val = float(f_meas)
            fsq_pred_sum += f_pred_val
            fsq_meas_sum += f_meas_val
            fsq_count += 1
            fsq_meas_values.append(f_meas_val)

            # 同时记录加持力模长（若可用），用于 Ap 相关统计
            ap_pred = debug.get("F_app_norm_pred", None)
            ap_meas = debug.get("F_app_norm_meas", None)
            if ap_pred is not None:
                ap_pred_values.append(float(ap_pred))
            if ap_meas is not None:
                ap_meas_values.append(float(ap_meas))
        except Exception:
            pass

    def _reset_squeeze_stats():
        """重置当前 episode 的挤压力 / 加持力统计（不打印）。"""
        nonlocal fsq_pred_sum, fsq_meas_sum, fsq_count, fsq_meas_values, ap_pred_values, ap_meas_values
        fsq_pred_sum = 0.0
        fsq_meas_sum = 0.0
        fsq_count = 0
        fsq_meas_values = []
        ap_pred_values = []
        ap_meas_values = []

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

    # Main replay loop - run in inference mode
    replayed_episode_count = 0
    recorded_episode_count = 0
    current_episode_indices = [None] * num_envs
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
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index is not None:
                                episode_ended[env_id] = False
                                break
                            next_episode_index = None

                        # Check if episode was successful after completion
                        if success_term is not None and current_episode_indices[env_id] is not None:
                            if (
                                bool(success_term.func(env, **success_term.params)[env_id])
                                and not episode_ended[env_id]
                            ):
                                recorded_episode_count += 1
                                plural_trailing_s = "s" if recorded_episode_count > 1 else ""
                                episode_ended[env_id] = True
                                if args_cli.dump_data:
                                    env.recorder_manager.set_success_to_episodes(
                                        [env_id], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                                    )
                                    # export the episode with the original episode name
                                    env.recorder_manager.export_episodes(
                                        [env_id], demo_ids=[current_episode_indices[env_id]]
                                    )
                                    env.recorder_manager.reset([env_id])

                                    print(
                                        f"Finished dumping {recorded_episode_count} episode{plural_trailing_s} for env"
                                        f" {env_id} to abs_eef_pose actions."
                                    )
                                else:
                                    print(
                                        f"Successfully replayed {recorded_episode_count} episode{plural_trailing_s} out"
                                        f" of {replayed_episode_count} demos."
                                    )
                                # 仅在成功 episode 时，输出整集挤压力均值（env0），并累计到全局统计
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

                                        # 统计当前成功 episode 的「实测」挤压力 / 加持力指标（仅 env0）
                                        # 1) 实测挤压力 Top5% 最大值（均值）
                                        if fsq_meas_values:
                                            sq_max_meas_top5 = compute_topk_mean(fsq_meas_values, frac=0.05)
                                            succ_squeeze_max_meas_sum += sq_max_meas_top5
                                            succ_squeeze_max_meas_count += 1

                                        # 2) 实测加持力平均值
                                        if ap_meas_values:
                                            ap_mean_meas = sum(ap_meas_values) / float(len(ap_meas_values))
                                            succ_ap_mean_meas_sum += ap_mean_meas
                                            succ_ap_mean_meas_count += 1

                                        # 3) 实测加持力 Top5% 最大值（均值）
                                        if ap_meas_values:
                                            ap_max_meas_top5 = compute_topk_mean(ap_meas_values, frac=0.05)
                                            succ_ap_max_meas_sum += ap_max_meas_top5
                                            succ_ap_max_meas_count += 1

                                    # 若当前环境的动作维度为 13（Hybrid），则基于 13D 动作计算一次 episode 级的力学 metrics
                                    try:
                                        # 注意：这里不要覆盖用于 env.step(...) 的 actions 张量
                                        # 从当前 env 对应的 EpisodeData 里取出完整的 13D 动作序列，仅用于离线统计
                                        episode = env_episode_data_map.get(env_id, None)
                                        episode_actions = None if episode is None else episode.data.get('actions', None)
                                        if (
                                            episode_actions is not None
                                            and isinstance(episode_actions, torch.Tensor)
                                            and episode_actions.ndim == 2
                                            and episode_actions.shape[1] == 13
                                        ):
                                            metrics = compute_contact_force_metrics_from_13d(episode_actions)
                                            succ_metrics_count += 1
                                            succ_squeeze_max_sum += metrics.squeeze_max
                                            succ_squeeze_mean_sum += metrics.squeeze_mean
                                            succ_app_max_sum += metrics.external_norm_max
                                            succ_app_mean_sum += metrics.external_norm_mean

                                            print(
                                                f"[Hybrid-Metrics] Episode #{current_episode_indices[env_id]} "
                                                f"squeeze_max={metrics.squeeze_max:.4f}, "
                                                f"squeeze_mean={metrics.squeeze_mean:.4f}, "
                                                f"app_max={metrics.external_norm_max:.4f}, "
                                                f"app_mean={metrics.external_norm_mean:.4f}"
                                            )
                                    except Exception:
                                        # metrics 计算失败不影响主流程
                                        pass

                                    _reset_squeeze_stats()
                                # 成功 episode 也重置可视化
                                if force_viz is not None and env_id == 0:
                                    try:
                                        force_viz.reset()
                                    except Exception:
                                        pass
                            else:
                                # if not successful, add to failed demo IDs list
                                if current_episode_indices[env_id] is not None:
                                    failed_demo_ids.append(current_episode_indices[env_id])
                                # 失败 episode：只重置统计与可视化，不打印均值
                                if env_id == 0:
                                    _reset_squeeze_stats()
                                    if force_viz is not None:
                                        try:
                                            force_viz.reset()
                                        except Exception:
                                            pass

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            current_episode_indices[env_id] = next_episode_index
                            print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_map[next_episode_index], env.device
                            )

                            # 如果是 IK 环境，且 HDF5 里的 actions 是 7D 轴角格式（7d2），
                            # 而 env 期望 8D (pos+quat+gripper)，则在加载 episode 时一次性转换为 8D。
                            if (
                                "IK" in args_cli.task
                                and isinstance(episode_data.data.get("actions", None), torch.Tensor)
                                and episode_data.data["actions"].ndim == 2
                                and episode_data.data["actions"].shape[1] == 7
                                and env.action_space.shape[-1] == 8
                            ):
                                actions_7d = episode_data.data["actions"]  # (T, 7)
                                converted_actions = []
                                for a in actions_7d:
                                    converted_actions.append(convert_action_axisangle_to_quat(a))
                                episode_data.data["actions"] = torch.stack(converted_actions, dim=0)  # (T, 8)

                            # 若是 Hybrid(-Tactile) 环境，且 HDF5 中仅有 7D 动作，
                            # 则尝试从 obs/gripper_net_force 中提取当前帧 6D 指力，拼接成 13D 动作。
                            actions_tensor = episode_data.data.get("actions", None)
                            if (
                                "Isaac-Libero-Franka-Hybrid-" in args_cli.task
                                and isinstance(actions_tensor, torch.Tensor)
                                and actions_tensor.ndim == 2
                                and actions_tensor.shape[1] == 7
                                and env.action_space.shape[-1] == 13
                            ):
                                base_actions = actions_tensor  # (T, 7)
                                inst_force = None
                                try:
                                    # EpisodeData.data 通常将 HDF5 的 "obs" 组映射为一个 dict
                                    obs_group = episode_data.data.get("obs", None)
                                    if isinstance(obs_group, dict):
                                        gnf = obs_group.get("gripper_net_force", None)
                                        # 期望形状: (T, H, 2, 3)
                                        if isinstance(gnf, torch.Tensor) and gnf.ndim == 4:
                                            # 取 index=0 作为“当前帧”，flatten 为 6D = [fL(3), fR(3)]
                                            T, H_hist, _, _ = gnf.shape
                                            inst_force = (
                                                gnf[:, 0, :, :]
                                                .reshape(T, 2 * 3)
                                                .to(device=base_actions.device, dtype=base_actions.dtype)
                                            )
                                except Exception:
                                    inst_force = None

                                if inst_force is None:
                                    # 若无法从 obs 中解析指力，则用 0 force 进行回放以避免维度错误
                                    inst_force = torch.zeros(
                                        (base_actions.shape[0], 6),
                                        device=base_actions.device,
                                        dtype=base_actions.dtype,
                                    )
                                    print(
                                        "⚠️  [Hybrid] actions 为 7D 且未能从 obs/gripper_net_force 解析出指力，"
                                        "将使用零力向量扩展为 13D 动作。"
                                    )

                                episode_data.data["actions"] = torch.cat([base_actions, inst_force], dim=1)  # (T,13)

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
                    # apply actions
                    env.step(actions)

                    # 更新力–位混合调试可视化 / 统计挤压力（仅 Hybrid + 单 env 的 term 有 debug_info）
                    try:
                        term = env.action_manager.get_term("arm_action")
                        debug = getattr(term, "debug_info", None)
                    except Exception:
                        debug = None

                    if debug:
                        _update_squeeze_stats(debug)
                        if force_viz is not None:
                            try:
                                force_viz.update(debug)
                            except Exception:
                                # 可视化不应影响正常重放流程
                                pass

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

    # Write failed demo IDs to failure.jsonl
    if failed_demo_ids:
        if args_cli.task_suite is not None:
            key = f"{args_cli.task_suite}_task{args_cli.task_id}"
        else:
            key = args_cli.task
        write_failure_jsonl(args_cli.output_failure_record_file, key, failed_demo_ids)

    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")

    # 输出整个任务（所有成功 episode）的挤压力 / 加持力 metrics 统计（统一使用 Hybrid-Metrics 一行）
    task_avg_pred = None
    task_avg_meas = None
    if succ_squeeze_count > 0:
        task_avg_pred = succ_squeeze_pred_sum / succ_squeeze_count
        task_avg_meas = succ_squeeze_meas_sum / succ_squeeze_count

    if succ_metrics_count > 0:
        task_squeeze_max_mean = succ_squeeze_max_sum / succ_metrics_count
        task_squeeze_mean_mean = succ_squeeze_mean_sum / succ_metrics_count
        task_app_max_mean = succ_app_max_sum / succ_metrics_count
        task_app_mean_mean = succ_app_mean_sum / succ_metrics_count

        # 实测挤压力 / 加持力（若有）
        task_squeeze_max_meas_mean = (
            succ_squeeze_max_meas_sum / succ_squeeze_max_meas_count
            if succ_squeeze_max_meas_count > 0
            else None
        )
        task_ap_mean_meas_mean = (
            succ_ap_mean_meas_sum / succ_ap_mean_meas_count if succ_ap_mean_meas_count > 0 else None
        )
        task_ap_max_meas_mean = (
            succ_ap_max_meas_sum / succ_ap_max_meas_count if succ_ap_max_meas_count > 0 else None
        )

        fragments: list[str] = []
        if task_avg_pred is not None:
            fragments.append(f"squeeze_avg_pred={task_avg_pred:.4f}")
        if task_avg_meas is not None:
            fragments.append(f"squeeze_avg_meas={task_avg_meas:.4f}")

        fragments.extend(
            [
                f"squeeze_max_mean={task_squeeze_max_mean:.4f}",
                f"squeeze_mean_mean={task_squeeze_mean_mean:.4f}",
                f"app_max_mean={task_app_max_mean:.4f}",
                f"app_mean_mean={task_app_mean_mean:.4f}",
            ]
        )
        if task_squeeze_max_meas_mean is not None:
            fragments.append(f"squeeze_max_meas_mean={task_squeeze_max_meas_mean:.4f}")
        if task_ap_max_meas_mean is not None:
            fragments.append(f"ap_max_meas_mean={task_ap_max_meas_mean:.4f}")
        if task_ap_mean_meas_mean is not None:
            fragments.append(f"ap_mean_meas_mean={task_ap_mean_meas_mean:.4f}")

        print(
            "[Hybrid-Metrics] Task contact_metrics "
            + ", ".join(fragments)
            + f" over {succ_metrics_count} successes"
        )
    env.close()
    if force_viz is not None:
        force_viz.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
