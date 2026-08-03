"""ACT/DP cross-conda-env bridge: Isaac process <-> policy worker, stdlib HTTP + pickle, zero dependencies.

Isaac's python environment does not (and should not) have the diffusion_policy/detr
dependencies; ACT/DP run the real adapter inside their own conda envs via
serve_policy_worker.py, and this class only forwards.
localhost trusted environment; pickle is local-machine only.
"""
from __future__ import annotations

import http.client
import pickle

import numpy as np

from softvtbench.evaluation.policies.base import Policy, register


@register("remote")
class RemotePolicy(Policy):
    def __init__(self, *, host: str = "127.0.0.1", port: int, modality: str, **_):
        self.modality = modality
        self._addr = (host, int(port))
        self._rpc("ping", {})

    def _rpc(self, op: str, payload: dict):
        conn = http.client.HTTPConnection(*self._addr, timeout=600)
        try:
            conn.request("POST", f"/{op}", body=pickle.dumps(payload, protocol=4))
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"policy worker {op} failed: {data[:500]!r}")
            return pickle.loads(data)
        finally:
            conn.close()

    def reset(self, *, episode_seed: int | None = None) -> None:
        self._rpc("reset", {"episode_seed": episode_seed})   # DP uses this to seed DDPM sampling

    def set_language(self, language: str) -> None:
        self._rpc("set_language", {"language": language})   # fastwam swaps its prompt cache per task

    def observe(self, env_obs: dict) -> None:
        self._rpc("observe", {"env_obs": env_obs})

    def predict(self) -> np.ndarray:
        return np.asarray(self._rpc("predict", {}), dtype=np.float32)
