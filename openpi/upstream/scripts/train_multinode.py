#!/usr/bin/env python3
"""Multi-node JAX training entrypoint for OpenPI.

This wraps ``scripts/train.py`` without modifying it.  It initializes JAX
distributed from Volc/MLP environment variables, applies the opt-in data-loader
patch, and disables W&B on non-zero JAX processes.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import jax


def _env_int(*names: str, default: int | None = None) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def _default_local_device_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([part for part in visible.split(",") if part.strip()])
    return 8


def init_jax_distributed_from_env() -> None:
    num_processes = _env_int("MLP_WORKER_NUM", "JAX_NUM_PROCESSES", "NNODES", default=1)
    if num_processes is None or num_processes <= 1:
        return

    master_addr = (
        os.environ.get("MLP_WORKER_0_PRIMARY_HOST")
        or os.environ.get("MLP_WORKER_0_HOST")
        or os.environ.get("MASTER_ADDR")
    )
    master_port = os.environ.get("MLP_WORKER_0_PORT") or os.environ.get("MASTER_PORT", "12345")
    process_id = _env_int("MLP_ROLE_INDEX", "JAX_PROCESS_ID", "NODE_RANK", default=0)
    local_device_count = _env_int(
        "MLP_WORKER_GPU",
        "JAX_LOCAL_DEVICE_COUNT",
        "LOCAL_DEVICE_COUNT",
        default=_default_local_device_count(),
    )
    timeout = _env_int("JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT", default=3600)

    if master_addr is None:
        raise RuntimeError("Missing MLP_WORKER_0_HOST / MLP_WORKER_0_PRIMARY_HOST / MASTER_ADDR for multi-node JAX.")
    if process_id is None or local_device_count is None:
        raise RuntimeError("Missing JAX process id or local device count.")

    kwargs = {}
    if process_id == 0:
        kwargs["coordinator_bind_address"] = f"0.0.0.0:{master_port}"

    jax.distributed.initialize(
        coordinator_address=f"{master_addr}:{master_port}",
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=list(range(local_device_count)),
        initialization_timeout=timeout,
        **kwargs,
    )


def _load_single_node_train_module():
    train_path = Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("_openpi_single_node_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load train.py from {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    init_jax_distributed_from_env()

    from openpi.training import data_loader_multinode_patch

    data_loader_multinode_patch.apply()

    train_module = _load_single_node_train_module()
    original_init_wandb = train_module.init_wandb

    def init_wandb_rank0(config, *, resuming: bool, log_code: bool = False, enabled: bool = True):
        if jax.process_index() != 0:
            import wandb

            wandb.init(mode="disabled")
            return
        return original_init_wandb(config, resuming=resuming, log_code=log_code, enabled=enabled)

    train_module.init_wandb = init_wandb_rank0

    config = train_module._config.cli()
    train_module.main(config)


if __name__ == "__main__":
    main()

