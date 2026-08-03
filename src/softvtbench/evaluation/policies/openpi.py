"""pi05 (openpi) adapter (VO + VT), websocket to serve_policy.

Train/inference consistency source = the formal client (openpi_inference_client.py). Key conventions:
  - observation/state = **7D** [xyz, axis_angle, gripper_pos[0]]; quaternion branch is
    canonicalized per frame against the reference [0,1,0,0] (+x half-space)
    (_canonicalize_quat_like_training).
  - images = raw camera frame -> resize_frames_with_padding -> 224^2 uint8 (the original
    function is imported from the benchmark repo, zero drift; the repo is already on
    PYTHONPATH, same source as tac_manip).
  - VT extras: tactile_image = 8-frame 4x4 mosaic; tactile_gripper_force (8,6);
    tactile_marker_motion (1+2, 198, 2) = init + 2 frames of current history.
  - observe() updates the tactile buffers every env step (the 8 mosaic frames must be
    consecutive steps); predict() is called every replan_steps.
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np

from softvtbench.evaluation.determinism import inference_seed
from softvtbench.evaluation.policies.base import Policy, register

TARGET_HW = (224, 224, 3)


def _canonicalize_quat_like_training(quat_wxyz: np.ndarray) -> np.ndarray:
    """Transcribed from the client: same half-space as the reference [0,1,0,0]."""
    q = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1).copy()
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        raise ValueError("zero-norm quaternion")
    q /= n
    if float(np.dot(q, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32))) < 0.0:
        q *= -1.0
    return q


def _pad_front(frames: list, n: int) -> list:
    return [frames[0]] * (n - len(frames)) + frames if len(frames) < n else frames[-n:]


@register("openpi")
class OpenPIPolicy(Policy):
    def __init__(self, *, server_host: str, server_port: int, modality: str,
                 language_instruction: str, replan_steps: int = 10,
                 mosaic_layout: str = "rows"):
        # the tactile mosaic layout must match the training converter (soft=rows / rigid=cols)
        os.environ["SOFTVTBENCH_TACTILE_MOSAIC_LAYOUT"] = mosaic_layout
        from openpi_client import websocket_client_policy as _wc
        # image resize and quat->axisangle shared with the formal client (zero drift)
        from softvtbench.compat.openpi_env import resize_frames_with_padding, quat2axisangle
        self._resize = resize_frames_with_padding
        self._quat2aa = quat2axisangle

        self.modality = modality
        self._is_vt = modality == "vt"
        self._prompt = language_instruction
        self._replan = int(replan_steps)
        self._client = _wc.WebsocketClientPolicy(server_host, server_port)
        self.reset()

    def reset(self, *, episode_seed: int | None = None) -> None:
        self._element_dumped = False
        self._episode_seed = episode_seed
        self._inference_index = 0
        self._latest: dict | None = None
        self._left: deque = deque(maxlen=8)
        self._right: deque = deque(maxlen=8)
        self._force: deque = deque(maxlen=8)
        self._marker: deque = deque(maxlen=2)
        self._marker_init: np.ndarray | None = None

    def observe(self, env_obs: dict) -> None:
        self._latest = env_obs
        if not self._is_vt:
            return
        for key in ("tactile_left_rgb", "tactile_right_rgb"):
            if key not in env_obs:
                raise RuntimeError(f"VT observation missing {key}")
            image = np.asarray(env_obs[key])
            if image.ndim != 3 or image.shape[-1] != 3 or not np.all(np.isfinite(image)):
                raise RuntimeError(f"VT {key} invalid: shape={image.shape}")
            (self._left if key == "tactile_left_rgb" else self._right).append(
                image.astype(np.uint8)
            )
        if "gripper_net_force" not in env_obs:
            raise RuntimeError("VT observation missing gripper_net_force")
        gnf = np.asarray(env_obs["gripper_net_force"], dtype=np.float32)
        if gnf.shape == (2, 3):
            inst = gnf
        elif gnf.ndim == 3 and gnf.shape[1:] == (2, 3) and gnf.shape[0] > 0:
            inst = gnf[0]
        else:
            raise RuntimeError(f"VT gripper_net_force invalid shape {gnf.shape}")
        if not np.all(np.isfinite(inst)):
            raise RuntimeError("VT gripper_net_force contains non-finite values")
        self._force.append(inst.reshape(6))
        if "gripper_marker_motion" not in env_obs:
            raise RuntimeError("VT observation missing gripper_marker_motion")
        gmm = np.asarray(env_obs["gripper_marker_motion"], dtype=np.float32)
        if gmm.shape != (2, 2, 99, 2) or not np.all(np.isfinite(gmm)):
            raise RuntimeError(
                "VT gripper_marker_motion contract violation: "
                f"shape={gmm.shape} finite={bool(np.all(np.isfinite(gmm)))}"
            )
        init = gmm[:, 0].reshape(198, 2)
        curr = gmm[:, 1].reshape(198, 2)
        if self._marker_init is None:
            self._marker_init = init
        self._marker.append(curr)

    def _image(self, raw_hwc: np.ndarray) -> np.ndarray:
        import torch
        frames = torch.from_numpy(np.asarray(raw_hwc, dtype=np.uint8)[None])
        out = self._resize(frames, TARGET_HW, bgr_conversion=False, pad_img=True)
        return np.asarray(out[0], dtype=np.uint8)

    def _mosaic(self) -> np.ndarray:
        import cv2
        H, W = TARGET_HW[0], TARGET_HW[1]
        ch, cw = H // 4, W // 4
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        left = _pad_front(list(self._left), 8)
        right = _pad_front(list(self._right), 8)
        if os.environ.get("SOFTVTBENCH_TACTILE_MOSAIC_LAYOUT", "cols").lower() == "rows":
            for off, hist in enumerate((left, right)):
                for k in range(8):
                    r, c = divmod(off * 8 + k, 4)
                    canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cv2.resize(hist[k], (cw, ch))
        else:
            for k in range(8):
                r, c = k // 2, k % 2
                canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cv2.resize(left[k], (cw, ch))
                canvas[r * ch:(r + 1) * ch, (c + 2) * cw:(c + 3) * cw] = cv2.resize(right[k], (cw, ch))
        return canvas

    def predict(self) -> np.ndarray:
        if self._episode_seed is None:
            raise RuntimeError("OpenPIPolicy requires episode_seed")
        o = self._latest
        pose = np.asarray(o["eef_pose"], dtype=np.float32).reshape(-1)
        quat = _canonicalize_quat_like_training(pose[3:7])
        aa = np.asarray(self._quat2aa(quat.copy()), dtype=np.float32).reshape(3)
        g = np.asarray(o["gripper_pos"], dtype=np.float32).reshape(-1)[:1]
        state7 = np.concatenate([pose[:3], aa, g]).astype(np.float32)

        element = {
            "observation/image": self._image(o["agentview_rgb"]),
            "observation/wrist_image": self._image(o["eye_in_hand_rgb"]),
            "observation/state": state7,
            "prompt": self._prompt,
            "__softvtbench_rng_seed": inference_seed(
                self._episode_seed, self._inference_index
            ),
        }
        self._inference_index += 1
        if self._is_vt:
            element["observation/tactile_image"] = element["tactile_image"] = self._mosaic()
            fh = np.stack(_pad_front(list(self._force), 8)).astype(np.float32)
            if fh.shape != (8, 6) or not np.all(np.isfinite(fh)):
                raise RuntimeError(f"model force history invalid: {fh.shape}")
            element["observation/tactile_gripper_force"] = element["tactile_gripper_force"] = fh
            hist = _pad_front(list(self._marker), 2)
            mm = np.stack([self._marker_init, *hist]).astype(np.float32)
            if mm.shape != (3, 198, 2) or not np.all(np.isfinite(mm)):
                raise RuntimeError(f"model marker history invalid: {mm.shape}")
            element["observation/tactile_marker_motion"] = element["tactile_marker_motion"] = mm
        dbg = os.environ.get("SOFTVT_DEBUG_ELEMENT_DIR", "")
        if dbg and not getattr(self, "_element_dumped", False):   # dump the first element per episode (content-level diagnostics)
            os.makedirs(dbg, exist_ok=True)
            np.savez(os.path.join(dbg, "element0.npz"),
                     **{k.replace("/", "__"): v for k, v in element.items() if k != "prompt"})
            self._element_dumped = True
        chunk = np.asarray(self._client.infer(element)["actions"], dtype=np.float32)
        return chunk[: self._replan, :7]
