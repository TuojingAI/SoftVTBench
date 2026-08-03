"""Policy protocol + registry -- rollout depends only on this layer.

Contract:
  reset()          at the start of every episode. Clears all internal state (history
                   buffers, quaternion branch).
  observe(env_obs) called by rollout once **per env step**; the policy maintains its own
                   temporal buffers (pi05's 8-frame tactile mosaic, DP's 2-frame obs
                   history, ACT has no history).
  predict()        returns a (k,7) absolute target-next action chunk
                   [x,y,z, ax,ay,az, gripper] from the current buffer. k is the number of
                   executed steps per inference: pi05=replan(10) / DP=n_action_steps(8) /
                   ACT=1 (temporal ensemble).
Gripper execution semantics (+-1 / absolute finger width / Schmitt decoding) live outside
the policy, in gripper_execution.
Adding a policy = subclass + @register, without touching rollout/runner.
"""
from __future__ import annotations

import abc

import numpy as np

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise KeyError(f"policy backend already registered: {name}")
        _REGISTRY[name] = cls
        return cls
    return deco


def make(backend: str, **kwargs) -> "Policy":
    if backend not in _REGISTRY:
        raise KeyError(f"unknown policy backend {backend!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[backend](**kwargs)


class Policy(abc.ABC):
    modality: str = "vo"          # "vo" | "vt"

    @abc.abstractmethod
    def reset(self, *, episode_seed: int | None = None) -> None: ...

    @abc.abstractmethod
    def observe(self, env_obs: dict) -> None: ...

    @abc.abstractmethod
    def predict(self) -> np.ndarray:
        """Return a (k, 7) target-next action chunk."""
