"""Versioned stable seeds for paired stochastic evaluation."""
from __future__ import annotations

import hashlib

SCHEME = "softvtbench-sha256-u32-v1"


def stable_u32_seed(*parts: object) -> int:
    payload = "\x1f".join([SCHEME, *(str(x) for x in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def episode_seed(run_seed: int, suite: str, task_id: int, episode: str) -> int:
    """The same task/demo gets identical stochastic policy noise in every condition."""
    return stable_u32_seed("episode", int(run_seed), suite, int(task_id), episode)


def inference_seed(base_episode_seed: int, inference_index: int) -> int:
    return stable_u32_seed("inference", int(base_episode_seed), int(inference_index))
