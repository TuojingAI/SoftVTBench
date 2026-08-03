"""Replay "policy": plays back the dataset-recorded target-next actions chunk by chunk (open loop).

Purpose = data/env pipeline self-check (the role of the internal stack's
run_data_evaluations): bypass the real policy to verify the whole chain of env build +
reset_to + action execution + success_term.
Low replay success -> the environment pipeline is broken; high replay but low policy ->
the problem is policy obs/action consistency.
The runner calls set_episode(actions) before each episode.
"""
from __future__ import annotations

import numpy as np

from softvtbench.evaluation.policies.base import Policy, register


@register("replay")
class ReplayPolicy(Policy):
    def __init__(self, *, modality: str = "vo", replan_steps: int = 10, **_):
        self.modality = modality
        self._replan = int(replan_steps)
        self._actions: np.ndarray | None = None
        self._t = 0

    def set_episode(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)[:, :7]
        self._t = 0

    def reset(self, *, episode_seed: int | None = None) -> None:
        self._t = 0

    def observe(self, env_obs: dict) -> None:
        pass

    def predict(self) -> np.ndarray:
        a = self._actions
        chunk = a[self._t:self._t + self._replan]
        if len(chunk) == 0:                       # actions exhausted: hold the last frame
            chunk = a[-1:][:]
        self._t += len(chunk)
        return chunk
