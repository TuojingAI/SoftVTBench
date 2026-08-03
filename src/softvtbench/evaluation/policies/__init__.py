"""Benchmark-side policy clients.

Model-specific implementations live in SoftVTBench-Models and are reached through
the local worker protocol. Importing this module registers the three benchmark-side
backends.
"""

from . import openpi, remote, replay  # noqa: F401
from .base import Policy, make, register

__all__ = ["Policy", "make", "register"]
