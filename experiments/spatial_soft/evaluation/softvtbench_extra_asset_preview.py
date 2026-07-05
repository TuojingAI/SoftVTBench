#!/usr/bin/env python3
"""Preview a SoftVTBench scene with an injected extra asset, without LIBERO replay."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


SOFTVTBENCH_REPO = Path(os.environ.get("SOFTVTBENCH_REPO", Path(__file__).resolve().parents[3] / "SoftVTBench"))
if str(SOFTVTBENCH_REPO) not in sys.path:
    sys.path.insert(0, str(SOFTVTBENCH_REPO))
TOOLS_DIR = SOFTVTBENCH_REPO / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Libero-Franka-IK-v0")
parser.add_argument("--task-suite", default="libero_object")
parser.add_argument("--task-id", type=int, default=0)
parser.add_argument("--demo-id", type=int, default=0)
parser.add_argument("--num-steps", type=int, default=80)
parser.add_argument("--out-root", required=True)
parser.add_argument("--camera-view-list", nargs="+", default=["agentview", "eye_in_hand"])
parser.add_argument(
    "--render-only",
    action="store_true",
    help="Do not step physics or send robot actions. Only render the reset scene.",
)
parser.add_argument(
    "--physics-only",
    action="store_true",
    help="Step simulator physics directly without env actions or robot commands.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

kit_args = args_cli.kit_args.split() if args_cli.kit_args else []
for stable_arg in ("--/ngx/enabled=false", "--/rtx/post/dlss/enabled=false"):
    key = stable_arg.split("=", maxsplit=1)[0]
    if not any(arg == key or arg.startswith(f"{key}=") for arg in kit_args):
        kit_args.append(stable_arg)
args_cli.kit_args = " ".join(kit_args)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from common.replay_utils import (
    process_successful_demo_videos,
    save_camera_images,
    setup_replay_output_directories,
)
from tac_manip.utils.task_configs import setup_task_objects


def _to_obs_dict(value):
    return value[0] if isinstance(value, tuple) else value


def _eef_pose(obs) -> torch.Tensor:
    pose = obs["policy"]["eef_pose"]
    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose)
    return pose[0, :7].detach()


def _make_hold_action(env, obs) -> torch.Tensor:
    """Hold the current absolute IK target instead of sending zero pose."""
    action = torch.zeros(env.action_space.shape, device=env.device)
    action_dim = action.shape[-1]
    if action_dim >= 8 and "policy" in obs and "eef_pose" in obs["policy"]:
        pose = _eef_pose(obs).to(env.device)
        quat = pose[3:7]
        if torch.linalg.norm(quat).item() < 1e-5:
            raise RuntimeError(f"Invalid eef quaternion in observation: {pose.detach().cpu().tolist()}")
        action[0, :3] = pose[:3]
        action[0, 3:7] = quat / torch.linalg.norm(quat)
        action[0, -1] = 1.0
    elif action_dim >= 1:
        action[0, -1] = 1.0
    return action


def _make_overview(out_root: Path, video_save_dir: Path) -> Path | None:
    agent = video_save_dir / f"demo_{args_cli.demo_id}_agentview_rgb.mp4"
    wrist = video_save_dir / f"demo_{args_cli.demo_id}_eye_in_hand_rgb.mp4"
    if not agent.exists() or not wrist.exists():
        return None
    out_path = out_root / "softvtbench_extra_asset_preview_2x1.mp4"
    subprocess.run(
        [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(agent),
            "-i",
            str(wrist),
            "-filter_complex",
            "[0:v]scale=640:-2,setpts=PTS-STARTPTS[v0];"
            "[1:v]scale=640:-2,setpts=PTS-STARTPTS[v1];"
            "[v0][v1]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "veryfast",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def main() -> None:
    print("[softvtbench-preview] main_enter", flush=True)
    setup_task_objects(args_cli.task_suite, args_cli.task_id, customized_file_paths=True)

    out_root = Path(args_cli.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    video_root_prefix = str(out_root / "video_datasets") + "/"
    video_save_dir, _ = setup_replay_output_directories(
        True,
        [],
        args_cli.task_suite,
        args_cli.task_id,
        args_cli.task,
        root_dir_prefix=video_root_prefix,
    )

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.sim.physx.enable_ccd = True

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    obs = _to_obs_dict(env.reset())
    print("[softvtbench-preview] env_reset_done", flush=True)

    if args_cli.render_only and args_cli.physics_only:
        raise ValueError("--render-only and --physics-only are mutually exclusive")

    sim_dt = float(getattr(env.sim.cfg, "dt", 1.0 / 60.0))

    with torch.inference_mode():
        for frame_idx in range(args_cli.num_steps):
            if args_cli.physics_only:
                env.sim.step(render=False)
                with contextlib.suppress(Exception):
                    env.scene.update(sim_dt)
            elif not args_cli.render_only:
                action = _make_hold_action(env, obs)
                obs = _to_obs_dict(env.step(action))
            with contextlib.suppress(Exception):
                env.sim.render()
            save_camera_images(
                env,
                0,
                args_cli.demo_id,
                frame_idx,
                True,
                args_cli.camera_view_list,
                task_suite=args_cli.task_suite,
                task_id=args_cli.task_id,
                task=args_cli.task,
                root_dir_prefix=video_root_prefix,
            )
            if frame_idx % 20 == 0:
                print(f"[softvtbench-preview] frame={frame_idx}", flush=True)

    process_successful_demo_videos(
        args_cli.demo_id,
        video_save_dir,
        True,
        args_cli.camera_view_list,
        task_suite=args_cli.task_suite,
        task_id=args_cli.task_id,
        task=args_cli.task,
        root_dir_prefix=video_root_prefix,
    )
    overview = _make_overview(out_root, Path(video_save_dir))
    print(f"SOFTVTBENCH_PREVIEW_DONE out_root={out_root} overview={overview}", flush=True)

    if os.environ.get("ISAAC_FAST_EXIT", "1").lower() in ("1", "true", "yes", "y", "on"):
        os._exit(0)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        with contextlib.suppress(Exception):
            simulation_app.close()
