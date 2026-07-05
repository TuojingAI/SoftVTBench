# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import random
from dataclasses import dataclass

import os
import gymnasium as gym
import numpy as np
import omni.log
import torch


@dataclass
class ClosedLoopArguments:
    task: str
    num_envs: int = 1
    seed: int = 10
    device: str = "cuda"


class ClosedLoopPolicyInference:
    """
    This base class is used to run inference on a closed loop action policy.
    """

    def __init__(self, args: ClosedLoopArguments):
        self.args = args

    def create_sim_environment(self):
        """
        Creates a simulation environment based on the given arguments.

        Args:
            args (ClosedLoopArguments): The arguments for the simulation environment.
            device (str, optional): The device to use for the simulation. Defaults to 'cuda'.

        Returns:
            gym.Env: The created simulation environment.
        """
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_name = self.args.task
        env_cfg = parse_env_cfg(env_name, device=self.args.device, num_envs=self.args.num_envs)
        env_cfg.env_name = env_name

        # Disable all recorders
        env_cfg.recorders = {}
        # extract success checking function to invoke in the main loop
        success_term = None
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            print(
                "No success termination term was found in the environment.",
                " Will not be able to mark policy evaluation result as successful.",
            )
        # modify configuration such that the environment runs indefinitely until
        # the goal is reached or other termination conditions are met
        env_cfg.terminations.time_out = None
        
        # Enable CCD for stable simulation in inference mode
        env_cfg.sim.physx.enable_ccd = True

        # Optionally match the collection-time camera resolution (the spatial-soft
        # collection expert renders agentview at BOWEN_CAMERA_WIDTH^2, default env
        # cfg differs). Gated on the same flag as the scene-visuals reapply.
        if os.environ.get("SOFTVTBENCH_APPLY_COLLECTION_SCENE_VISUALS", "0") == "1":
            import re as _re

            _cam_w = os.environ.get("BOWEN_CAMERA_WIDTH", "").strip()
            if _cam_w:
                _w = int(_cam_w)
                _h = int(os.environ.get("BOWEN_CAMERA_HEIGHT", _cam_w))
                _views = [v for v in _re.split(r"[\s,]+", os.environ.get("BOWEN_CAMERA_VIEWS", "agentview").strip()) if v]
                for _view in _views:
                    _sensor = getattr(env_cfg.scene, f"{_view}_cam", None)
                    if _sensor is None:
                        continue
                    if hasattr(_sensor, "width"):
                        _sensor.width = _w
                    if hasattr(_sensor, "height"):
                        _sensor.height = _h
                    print(f"[scene-visuals] set {_view}_cam resolution: {_w}x{_h}", flush=True)

        # Disable object_dropped terminations that can cause unexpected episode endings
        termination_attrs = dir(env_cfg.terminations)
        for attr in termination_attrs:
            if attr.endswith('_dropped') or attr == 'object_dropped':
                print(f"Disabling termination condition: {attr}")
                setattr(env_cfg.terminations, attr, None)

        # create environment from loaded config
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        # Set seed
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        env.seed(self.args.seed)

        return env, env_cfg, success_term

    def run_inference(self):
        raise NotImplementedError("Subclasses must implement this method: run_inference")
