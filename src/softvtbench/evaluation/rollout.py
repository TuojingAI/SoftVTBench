"""Closed-loop rollout -- single implementation, policy-agnostic.

Per-step sequence (transcribed from the formal client's validated semantics):
  reset -> (optional) scene visuals -> reset_to(recorded initial state) -> render warmup
  -> (soft body) second reset_to + warmup
  loop: observe(EnvObs) feeds the policy buffer every step -> every k steps predict a
        (k,7) chunk -> axis-angle -> quaternion + GripperExecutor -> env.step ->
        success requires 8 consecutive steps -> FEM D_peak
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from softvtbench.evaluation import metrics
from softvtbench.evaluation.preprocessing.rotation import axis_angle_to_quat_wxyz


def _to_uint8_rgb(img) -> np.ndarray:
    """tensor/float image -> uint8 HWC RGB. Transcribed from the client."""
    import torch
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.max() <= 1.0 + 1e-6 \
            else np.clip(img, 0, 255).astype(np.uint8)
    return img


def _apply_scene_visuals(env) -> None:
    if getattr(env, "_softvt_visuals_applied", False):   # env instance attribute; naturally resets on rebuild
        return
    path = os.environ.get("SOFTVTBENCH_SCENE_VISUALS_MODULE", "").strip()
    if not path or os.environ.get("SOFTVTBENCH_APPLY_COLLECTION_SCENE_VISUALS", "0") != "1":
        return
    spec = importlib.util.spec_from_file_location("softvtbench_collection_scene_visuals", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_collection_scene_visuals()
    env._softvt_visuals_applied = True


def _render(env, frames: int) -> None:
    if frames > 0 and not hasattr(env, "_softvt_first_render_had_visuals"):
        env._softvt_first_render_had_visuals = bool(
            getattr(env, "_softvt_visuals_applied", False)
        )
    for _ in range(frames):
        env.unwrapped.sim.render()


def capture_env_obs(env, obs, camera_names=("agentview_cam", "eye_in_hand_cam"),
                    tactile_sensors=("gsmini_left", "gsmini_right"),
                    tactile_output_type="markers_rgb", with_tactile=True, idx: int = 0) -> dict:
    """Assemble the canonical policy-level EnvObs from env + obs (keys map 1:1 to training-data fields). idx = env index."""
    p = obs["policy"]
    out = {
        "eef_pose": p["eef_pose"][idx].detach().cpu().numpy().astype(np.float32),
        "gripper_pos": p["gripper_pos"][idx].detach().cpu().numpy().astype(np.float32),
    }
    scene = env.unwrapped.scene
    cams = []
    for name in camera_names:
        cams.append(_to_uint8_rgb(scene[name].data.output["rgb"][idx]))
    out["agentview_rgb"], out["eye_in_hand_rgb"] = cams[0], cams[1]
    if with_tactile:
        if "gripper_marker_motion" in p:
            out["gripper_marker_motion"] = p["gripper_marker_motion"][idx].detach().cpu().numpy().astype(np.float32)
        if "gripper_net_force" in p:
            out["gripper_net_force"] = p["gripper_net_force"][idx].detach().cpu().numpy().astype(np.float32)
        for key, sensor in zip(("tactile_left_rgb", "tactile_right_rgb"), tactile_sensors):
            out[key] = _to_uint8_rgb(scene.sensors[sensor].data.output[tactile_output_type][idx])
    return out



class _VideoWriter:
    """2x2 video: top-left third-person / top-right wrist / bottom-left left tactile / bottom-right right tactile.
    Enabled via SOFTVT_RECORD_VIDEO=1 (off by default to avoid slowing batch evaluation)."""

    CELL = 384

    def __init__(self, path: str, fps: int = 20):
        self.path, self.fps, self._w = path, fps, None

    def add(self, env_obs: dict) -> None:
        import cv2
        import numpy as _np
        c = self.CELL
        def cell(key):
            img = env_obs.get(key)
            if img is None:
                return _np.zeros((c, c, 3), dtype=_np.uint8)
            return cv2.resize(_np.asarray(img, dtype=_np.uint8), (c, c))
        grid = _np.concatenate([
            _np.concatenate([cell("agentview_rgb"), cell("eye_in_hand_rgb")], axis=1),
            _np.concatenate([cell("tactile_left_rgb"), cell("tactile_right_rgb")], axis=1),
        ], axis=0)
        frame = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        if self._w is None:
            self._w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      self.fps, (frame.shape[1], frame.shape[0]))
        self._w.write(frame)

    def close(self) -> None:
        if self._w is not None:
            self._w.release()


def _finger_pos(env) -> float:
    """Current single-finger joint position (mean of both fingers); used for stall detection."""
    robot = env.unwrapped.scene["robot"]
    ids, _ = robot.find_joints("panda_finger_joint.*")
    return float(robot.data.joint_pos[0, ids].mean())


def prepare_episode(env, initial_state, *, suite: dict, idx: int = 0):
    """Restore one episode exactly as rollout does, without invoking a policy.

    This is also the scene-receipt preflight entry point.  Keeping reset logic
    here prevents the audit path and the actual rollout from silently drifting.
    """
    import torch

    env_ids = torch.tensor([idx], device=env.unwrapped.device)
    warmup = int(suite.get("scene_render_warmup_frames", 4))
    after_state = bool(suite.get("scene_visuals_after_recorded_state", False))
    post_restore = int(suite.get("scene_post_restore_warmup_frames", 0))

    obs, _ = env.reset()
    if not after_state:
        _apply_scene_visuals(env)
    if initial_state is not None:
        obs, _ = env.reset_to(initial_state, env_ids, is_relative=True)
        if after_state:
            _apply_scene_visuals(env)
        _render(env, warmup)
        if post_restore > 0:
            obs, _ = env.reset_to(initial_state, env_ids, is_relative=True)
            _render(env, post_restore)
    else:
        _apply_scene_visuals(env)
        _render(env, warmup)
    return obs


def apply_episode_ood(env, ood, debug_dir: str | None = None) -> None:
    """Apply one episode's physical OOD before policy inference.

    Kept separate from ``run_episode`` so the runner can read back and validate
    the post-reset/post-OOD live scene into a per-episode receipt before any
    model request is sent.
    """
    if ood is None or not ood.enabled:
        return
    ood.apply_physical()
    if ood.requires_render:
        # Lighting changes are authored after reset. Render them into camera
        # products before the receipt and before the first policy observation.
        _render(env, 4)
    if debug_dir:
        import json as _json
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "ood_receipt.json"), "w") as f:
            _json.dump(ood.receipt(), f, indent=2)


def run_episode(env, success_term, policy, gripper_exec, initial_state, *, suite: dict, control: dict,
                fem_asset: str | None = None, grip_width: float | None = None,
                finger_lower_limit: float | None = None,
                debug_dir: str | None = None, ood=None, episode_seed: int | None = None,
                prepared_obs=None, ood_already_applied: bool = False,
                evaluation_protocol: str = "native_env_steps") -> dict:
    """Run one episode (initial_state = pre-filtered nested tensor dict or None); returns a result dict."""
    import torch

    obs = (prepared_obs if prepared_obs is not None
           else prepare_episode(env, initial_state, suite=suite))

    video = None    # see record_video below (must be created after with_tactile)
    if not ood_already_applied:
        apply_episode_ood(env, ood, debug_dir)

    policy.reset(episode_seed=episode_seed)
    gripper_exec.reset(
        finger_lower_limit=(
            finger_lower_limit
            if finger_lower_limit is not None
            else (float(grip_width) / 2.0 if grip_width is not None else None)
        )
    )

    fem_ref = metrics.get_nodal_pos(env, fem_asset) if fem_asset else None
    d_peak: float | None = None
    d_series: list[float] = []          # full deformation series: threshold convention TBD, keep the distribution

    # Total step budget = replan x max_inference (formal max_executed_action_steps=300).
    # Each predict executes k steps (pi05 k=10 -> 30 inferences; ACT k=1 -> 300; DP k=8 -> ~38).
    max_env_steps = int(control["max_inference_steps"]) * int(control["replan_steps"])
    max_inferences = int(control["max_inference_steps"])
    required_chunk = int(control["replan_steps"])
    if evaluation_protocol not in {
        "diagnostic_replay", "native_env_steps", "chunked_30x10"
    }:
        raise ValueError(f"unknown evaluation protocol {evaluation_protocol!r}")
    num_success_steps = int(control["num_success_steps"])
    record_video = bool(debug_dir) and os.environ.get("SOFTVT_RECORD_VIDEO") == "1"
    with_tactile = policy.modality == "vt" or record_video   # 2x2 video needs the tactile frames
    if record_video:
        os.makedirs(debug_dir, exist_ok=True)
        video = _VideoWriter(os.path.join(debug_dir, "episode.mp4"))
    tactile_output_type = suite.get("tactile_output_type", "markers_rgb")

    # Collection protocol: after the close action completes, hard-pin the fingers at the
    # safe width (transcribed from _pin_fingers_to_width). Clamping only the lower limit
    # lets soft-body elastic resistance stop the fingers above the limit; deformation only
    # reaches 63% of collection.
    # Triggering is by **finger stall**, not a fixed step count: collection pins only after
    # the close action completes, and each demo contacts the object at a different time; a
    # fixed step count squeezes before the object has settled (measured: ejects 2/3).
    # Switch (off by default): stall-triggered pinning brings deformation close to
    # collection (6.2 vs recorded 6.5), but replay success drops from 5/5 to 3/5 (some
    # demos get squeezed out). Trigger condition needs tuning; default off, with a known
    # ~37% deformation underestimate.
    PIN_ENABLED = os.environ.get("SOFTVT_FINGER_PIN") == "1"
    STALL_EPS, STALL_N = 2.0e-4, 3      # single-finger displacement < 0.2mm for 3 consecutive steps = stalled
    close_run, pinned, stall_run, prev_fw = 0, False, 0, None
    if PIN_ENABLED and grip_width is not None:   # clear any pin left over from the previous episode (a pinned upper limit keeps the gripper from opening)
        from softvtbench.evaluation.envs.build import unpin_fingers
        unpin_fingers(env, grip_width, float(gripper_exec.open_finger))
    success_streak, success, steps = 0, False, 0
    inference_count = 0
    while (
        steps < max_env_steps
        and (
            evaluation_protocol != "chunked_30x10"
            or inference_count < max_inferences
        )
    ):
        infer_i = inference_count
        inference_count += 1
        env_obs = capture_env_obs(env, obs, with_tactile=with_tactile,
                                  tactile_output_type=tactile_output_type)
        if ood is not None and ood.enabled:
            env_obs["agentview_rgb"], env_obs["eye_in_hand_rgb"] = ood.corrupt_external_rgb(
                env_obs["agentview_rgb"], env_obs["eye_in_hand_rgb"])
        policy.observe(env_obs)
        if video:
            video.add(env_obs)
        chunk = np.asarray(policy.predict(), dtype=np.float32)       # (k,7)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) == 0:
            raise RuntimeError(f"policy returned invalid action chunk {chunk.shape}")
        if evaluation_protocol == "chunked_30x10" and len(chunk) != required_chunk:
            raise RuntimeError(
                f"chunked_30x10 requires exactly {required_chunk} actions per inference; "
                f"got {len(chunk)}"
            )
        k = chunk.shape[0]
        quats = np.stack([axis_angle_to_quat_wxyz(a[3:6]) for a in chunk])

        if debug_dir:                          # log the gripper prediction trace per inference (behavior-level diagnostics)
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, "gripper_trace.jsonl"), "a") as gf:
                gf.write('{"step": %d, "g_min": %.4f, "g_max": %.4f}\n'
                         % (steps, float(chunk[:, 6].min()), float(chunk[:, 6].max())))
        if debug_dir and infer_i == 0:        # dump the first inference for offline comparison against dataset ground truth
            import cv2
            os.makedirs(debug_dir, exist_ok=True)
            np.savez(os.path.join(debug_dir, "first_infer.npz"),
                     eef_pose=env_obs["eef_pose"], gripper_pos=env_obs["gripper_pos"],
                     chunk=chunk)
            cv2.imwrite(os.path.join(debug_dir, "agentview.png"),
                        cv2.cvtColor(env_obs["agentview_rgb"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(debug_dir, "wrist.png"),
                        cv2.cvtColor(env_obs["eye_in_hand_rgb"], cv2.COLOR_RGB2BGR))

        done = False
        for i in range(k):
            if steps >= max_env_steps:      # enforce the 300-step cap inside the chunk too (DP's 8-step block must not overrun)
                done = True
                break
            grip = gripper_exec.step(chunk[i, 6])       # advance the latch per executed step (early termination never runs ahead)
            if debug_dir and gripper_exec.mode == "continuous_fixed_position":
                import json as _json
                with open(os.path.join(debug_dir, "gripper_execution_trace.jsonl"), "a") as ef:
                    ef.write(_json.dumps(
                        {"step": steps, **gripper_exec.last_diag},
                        sort_keys=True,
                    ) + "\n")
            if PIN_ENABLED and grip_width is not None:  # replicate collection's hard finger pin
                closing = float(grip.reshape(-1)[0]) < 0.0
                if not closing and pinned:
                    from softvtbench.evaluation.envs.build import unpin_fingers
                    unpin_fingers(env, grip_width, float(gripper_exec.open_finger))
                    pinned, stall_run, prev_fw = False, 0, None
            action = np.concatenate([chunk[i, :3], quats[i], grip])[None]
            action_t = torch.from_numpy(action).float().to(env.unwrapped.device)
            obs, _, terminated, truncated, _ = env.step(action_t)
            steps += 1
            if PIN_ENABLED and grip_width is not None and closing and not pinned:
                fw = _finger_pos(env)
                if prev_fw is not None and abs(fw - prev_fw) < STALL_EPS:
                    stall_run += 1
                    if stall_run >= STALL_N:            # stalled -> replicate collection's hard pin
                        from softvtbench.evaluation.envs.build import pin_fingers_to_width
                        pin_fingers_to_width(env, grip_width)
                        pinned = True
                else:
                    stall_run = 0
                prev_fw = fw
            if fem_ref is not None:
                d = metrics.fem_deformation_pct(fem_ref, metrics.get_nodal_pos(env, fem_asset))
                if d is not None:
                    d_series.append(d)
                    d_peak = d if d_peak is None else max(d_peak, d)
            if bool(terminated[0]) or bool(truncated[0]):
                done = True
                break
            if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                success_streak += 1
                if success_streak >= num_success_steps:
                    success, done = True, True
                    break
            else:
                success_streak = 0
            # Subsequent steps inside the chunk must feed the policy buffer step by step
            # (tactile/history continuity).
            #
            # Lesson from 7/26: an `and with_tactile` gate was once added to skip the VO
            # image copy, on the theory that "VO's observe() only does
            # self._latest = env_obs, the next round overwrites it, skipping is an
            # identity transform". That holds for ACT (policies/act.py) and openpi
            # (policies/openpi.py), but **not for DP** -- DP's observe() pushes into a
            # maxlen=2 deque to maintain the n_obs_steps=2 adjacent frames required at
            # training. With the skip, DP-VO's inter-frame gap went from 1 step to k steps
            # (k=8), pushing inter-frame motion straight out of the training distribution.
            # Conclusion: this gate must be declared by the policy itself, not guessed
            # from modality. Until a `needs_every_step` flag exists, always observe every
            # step.
            if i < k - 1:
                env_obs = capture_env_obs(env, obs, with_tactile=with_tactile,
                                          tactile_output_type=tactile_output_type)
                if ood is not None and ood.enabled:
                    env_obs["agentview_rgb"], env_obs["eye_in_hand_rgb"] = ood.corrupt_external_rgb(
                        env_obs["agentview_rgb"], env_obs["eye_in_hand_rgb"])
                policy.observe(env_obs)
                if video:
                    video.add(env_obs)
        if done:
            break

    if video:
        video.close()


    return {"success": bool(success), "steps": steps,
            "inference_count": inference_count,
            "evaluation_protocol": evaluation_protocol,
            **metrics.deformation_stats(d_series)}
