#!/usr/bin/env python3
# ruff: noqa
# flake8: noqa
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""回放 Libero demo，同时记录/可视化左右指局部接触力（gripper_net_force）。"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

# 确保项目根目录在 sys.path 中，便于导入 `benchmarks.*` 等顶层包。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class LocalForcePlotter:
    """左右指局部力曲线（6 条：L/R × x/y/z）。"""

    def __init__(self, title: str = "Local finger forces (gripper frame)", interactive: bool = True):
        # 为 headless 保存图做准备：需要在 import pyplot 前设置 backend
        if not interactive:
            import matplotlib

            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        if interactive:
            plt.ion()
        self.plt = plt
        self._interactive = interactive
        self.fig, self.axs = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
        self.fig.suptitle(title)
        self.t = []
        self.series = {
            "Lx": [],
            "Ly": [],
            "Lz": [],
            "Rx": [],
            "Ry": [],
            "Rz": [],
        }
        self._step = 0

    def update(self, f_local_lr: np.ndarray):
        """f_local_lr: shape (2, 3) = [[Lx,Ly,Lz],[Rx,Ry,Rz]]"""
        self._step += 1
        self.t.append(self._step)
        t = np.asarray(self.t, dtype=np.int32)

        L = f_local_lr[0]
        R = f_local_lr[1]
        self.series["Lx"].append(float(L[0]))
        self.series["Ly"].append(float(L[1]))
        self.series["Lz"].append(float(L[2]))
        self.series["Rx"].append(float(R[0]))
        self.series["Ry"].append(float(R[1]))
        self.series["Rz"].append(float(R[2]))

        # Left row
        left_titles = ["Left Fx", "Left Fy", "Left Fz"]
        right_titles = ["Right Fx", "Right Fy", "Right Fz"]
        keys_left = ["Lx", "Ly", "Lz"]
        keys_right = ["Rx", "Ry", "Rz"]

        for j in range(3):
            ax = self.axs[0, j]
            ax.cla()
            ax.plot(t, self.series[keys_left[j]], "b-")
            ax.set_title(left_titles[j])
            ax.grid(True, alpha=0.2)
        for j in range(3):
            ax = self.axs[1, j]
            ax.cla()
            ax.plot(t, self.series[keys_right[j]], "g-")
            ax.set_title(right_titles[j])
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("step")

        self.fig.tight_layout()
        if self._interactive:
            try:
                self.plt.pause(0.001)
            except Exception:
                pass

    def save(self, path: str):
        self.fig.tight_layout()
        self.fig.savefig(path, dpi=150)

    def close(self):
        try:
            self.plt.ioff()
            self.plt.close(self.fig)
        except Exception:
            pass


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Libero demo 回放 + gripper_net_force 记录/可视化。")
    parser.add_argument(
        "--env_variant",
        type=str,
        choices=["contactforce", "tactile"],
        default="contactforce",
        help="选择环境：contactforce（finger）或 tactile（gelsight/gelpad）。",
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_goal",
        help="Libero suite（默认 libero_goal）。",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=5,
        help="Libero task id（从 0 开始，默认 5）。",
    )
    parser.add_argument(
        "--demo_id",
        type=int,
        default=0,
        help="要回放的 demo/episode id（默认 0）。",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="输入 HDF5 路径；不提供则按 (task_suite, task_id) 自动解析（需要相关 env var）。",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="限制最多回放多少步（默认回放完整 demo）。",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="在线绘制局部力曲线（需要图形环境；headless 下建议用 --save_plot）。",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="每隔多少步打印一次局部力。",
    )
    parser.add_argument(
        "--save_npz",
        type=str,
        default=None,
        help="保存局部力序列 npz（shape=(T,2,3)）。",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="保存局部力曲线图 png（适用于 headless）。",
    )

    # 追加 AppLauncher 的通用参数（headless/device 等）
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _set_task_env_vars(task_suite: str, task_id: int) -> None:
    os.environ["TASK_SUITE"] = str(task_suite)
    os.environ["TASK_ID"] = str(int(task_id))


def extract_local_gripper_force(obs) -> np.ndarray | None:
    """从 obs 中提取左右指局部力 (2,3)。返回 None 表示该 obs 没有该键。"""
    try:
        f = obs["policy"]["gripper_net_force"]  # (N,H,2,3)
        f_curr = f[0, -1].detach().cpu().numpy()  # (2,3)
        return f_curr
    except Exception:
        return None


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # 关键：Camera 传感器必须打开
    args.enable_cameras = True

    # 启动模拟（此时会正确加载 Isaac Sim / Omniverse 相关模块）
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 在 SimulationApp 初始化完成后，再导入 tac_manip，确保其中的 Gym 环境完成注册
    # （注册代码链路：tac_manip.__init__ -> tac_manip.tasks.__init__ -> import_packages(...) -> franka.__init__ -> gym.register）
    try:
        import tac_manip  # noqa: F401
    except Exception as exc:
        print(f"[ForceDebug] Warning: failed to import tac_manip package (Gym envs may be missing): {exc}")

    if args.env_variant == "contactforce":
        task = "Isaac-Libero-Franka-Replay-Camera-ContactForce-v0"
    else:
        task = "Isaac-Libero-Franka-Replay-Camera-Tactile-v0"

    _set_task_env_vars(args.task_suite, args.task_id)

    # 延后导入：确保在 AppLauncher 启动后加载依赖
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    from scripts.tools.common.replay_utils import resolve_input_hdf5

    if args.dataset_file is None:
        dataset_file, desc = resolve_input_hdf5(task=task, task_suite=args.task_suite, task_id=args.task_id, prefer="auto")
        print(f"[ForceDebug] {desc}")
    else:
        dataset_file = args.dataset_file

    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env_cfg.env_name = task
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.sim.physx.enable_ccd = True

    env = gym.make(task, cfg=env_cfg).unwrapped
    env.reset()

    ds = HDF5DatasetFileHandler()
    ds.open(dataset_file)

    episode_names = list(ds.get_episode_names())
    if len(episode_names) == 0:
        raise RuntimeError(f"No episodes found in dataset: {dataset_file}")

    demo_id = int(args.demo_id)
    if demo_id < 0 or demo_id >= len(episode_names):
        raise ValueError(f"demo_id out of range: {demo_id}, available: [0..{len(episode_names)-1}]")

    episode_name = episode_names[demo_id]
    episode = ds.load_episode(episode_name, env.device)
    if "initial_state" in episode.data:
        init_state = episode.get_initial_state()
        env.reset_to(init_state, torch.tensor([0], device=env.device), is_relative=True)
    else:
        env.reset()

    actions_seq = episode.data.get("actions", None)
    if actions_seq is None or not isinstance(actions_seq, torch.Tensor) or actions_seq.ndim != 2:
        raise RuntimeError("Episode actions not found or invalid in HDF5 (expect Tensor[T, D]).")
    if actions_seq.shape[1] != env.action_space.shape[-1]:
        raise RuntimeError(
            f"Action dim mismatch: dataset D={actions_seq.shape[1]} vs env D={env.action_space.shape[-1]}. "
            f"dataset={dataset_file}, episode='{episode_name}'"
        )

    local_force_plotter = None
    if args.viz or args.save_plot is not None:
        local_force_plotter = LocalForcePlotter(
            title=f"Local finger forces | {task} | {args.task_suite} task{args.task_id} demo{demo_id}",
            interactive=bool(args.viz),
        )

    recorded_forces = []
    max_steps = int(args.num_steps) if args.num_steps is not None else int(actions_seq.shape[0])
    max_steps = min(max_steps, int(actions_seq.shape[0]))

    print(f"[ForceDebug] Env: {task}")
    print(f"[ForceDebug] Dataset: {dataset_file}")
    print(
        f"[ForceDebug] Replay: suite={args.task_suite} task_id={args.task_id} "
        f"demo_id={demo_id} episode='{episode_name}' steps={max_steps}"
    )

    try:
        for step_idx in range(max_steps):
            obs, _, _, _, _ = env.step(actions_seq[step_idx : step_idx + 1])

            f_local = extract_local_gripper_force(obs)
            if f_local is not None:
                recorded_forces.append(f_local.copy())
                if args.print_every > 0 and step_idx % int(args.print_every) == 0:
                    L = f_local[0]
                    R = f_local[1]
                    print(
                        f"[ForceDebug] step {step_idx:04d} | "
                        f"L=[{L[0]: .3f},{L[1]: .3f},{L[2]: .3f}] N, "
                        f"R=[{R[0]: .3f},{R[1]: .3f},{R[2]: .3f}] N"
                    )
                if local_force_plotter is not None:
                    local_force_plotter.update(f_local)

            # 渲染一帧（即便 headless=True，也保持与 AppLauncher 使用方式一致）
            env.sim.render()

    except KeyboardInterrupt:
        print("\n[ForceDebug] Interrupted by user.")
    finally:
        # 保存记录
        if args.save_npz is not None and len(recorded_forces) > 0:
            try:
                arr = np.asarray(recorded_forces, dtype=np.float32)  # (T,2,3)
                np.savez_compressed(
                    args.save_npz,
                    forces_local_lr=arr,
                    env=str(task),
                    dataset=str(dataset_file),
                    task_suite=str(args.task_suite),
                    task_id=int(args.task_id),
                    demo_id=int(demo_id),
                )
                from pathlib import Path as _Path

                print(f"[ForceDebug] Saved local force series to: {_Path(args.save_npz).resolve()} (shape={arr.shape})")
            except Exception as exc:
                print(f"[ForceDebug] Failed to save npz to {args.save_npz}: {exc}")

        if args.save_plot is not None and local_force_plotter is not None:
            try:
                local_force_plotter.save(args.save_plot)
                from pathlib import Path as _Path

                print(f"[ForceDebug] Saved local force plot to: {_Path(args.save_plot).resolve()}")
            except Exception as exc:
                print(f"[ForceDebug] Failed to save plot to {args.save_plot}: {exc}")

        if local_force_plotter is not None:
            local_force_plotter.close()
        # NOTE:
        # IsaacSim/IsaacLab 的 camera（tiled_camera）在部分环境下 teardown 会触发 C++ 侧异常并 abort（pybind11::error_already_set）。
        # 我们这里是纯调试/离线导出脚本：数据/图片已落盘后，直接快速退出以避免 teardown 崩溃。
        try:
            simulation_app.close()
        except Exception:
            pass
        import os as _os

        _os._exit(0)


if __name__ == "__main__":
    main()


