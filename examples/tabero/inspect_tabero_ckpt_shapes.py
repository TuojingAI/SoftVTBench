#!/usr/bin/env python3
"""Compare Tabero tactile checkpoint parameter shapes against local model code."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from flax import traverse_util
import flax.nnx as nnx
import jax
import orbax.checkpoint as ocp

from openpi.models import pi0_tabero


CRITICAL_KEYS = (
    "state_proj/kernel",
    "action_in_proj/kernel",
    "action_out_proj/kernel",
)


def _resolve_params_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "params").is_dir():
        return path / "params"
    return path


def _flatten_metadata(tree, path: tuple[str, ...] = ()) -> dict[str, tuple[tuple[int, ...], str]]:
    if isinstance(tree, dict):
        out: dict[str, tuple[tuple[int, ...], str]] = {}
        for key, value in tree.items():
            out.update(_flatten_metadata(value, path + (str(key),)))
        return out

    parts = list(path)
    if parts and parts[0] == "params":
        parts = parts[1:]
    if parts and parts[-1] == "value":
        parts = parts[:-1]
    return {"/".join(parts): (tuple(tree.shape), str(tree.dtype))}


def _local_model_shapes() -> dict[str, tuple[tuple[int, ...], str]]:
    config = pi0_tabero.TaberoPi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=32,
        action_horizon=50,
        image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    )
    model = nnx.eval_shape(config.create, jax.random.key(0))
    _, state = nnx.split(model)
    flat = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    return {key: (tuple(value.shape), str(value.dtype)) for key, value in flat.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to an OpenPI Orbax params directory, or to a checkpoint step directory containing params/.",
    )
    args = parser.parse_args()

    params_dir = _resolve_params_dir(args.checkpoint)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpoint_shapes = _flatten_metadata(checkpointer.metadata(params_dir).tree)

    local_shapes = _local_model_shapes()
    ckpt_tactile = {k: v for k, v in checkpoint_shapes.items() if "tactile" in k.lower()}
    local_tactile = {k: v for k, v in local_shapes.items() if "tactile" in k.lower()}

    missing_in_local = sorted(set(ckpt_tactile) - set(local_tactile))
    missing_in_ckpt = sorted(set(local_tactile) - set(ckpt_tactile))
    shape_mismatch = sorted(
        (key, ckpt_tactile[key], local_tactile[key])
        for key in set(ckpt_tactile) & set(local_tactile)
        if ckpt_tactile[key][0] != local_tactile[key][0]
    )

    print(f"params_dir: {params_dir}")
    print(f"ckpt_tactile_count: {len(ckpt_tactile)}")
    print(f"local_tactile_count: {len(local_tactile)}")
    print(f"missing_in_local: {len(missing_in_local)}")
    print(f"missing_in_ckpt: {len(missing_in_ckpt)}")
    print(f"shape_mismatch: {len(shape_mismatch)}")
    for key in CRITICAL_KEYS:
        print(f"{key}: ckpt={checkpoint_shapes.get(key)} local={local_shapes.get(key)}")

    if missing_in_local or missing_in_ckpt or shape_mismatch:
        for key in missing_in_local[:20]:
            print(f"missing_in_local_key: {key}")
        for key in missing_in_ckpt[:20]:
            print(f"missing_in_ckpt_key: {key}")
        for key, ckpt_shape, local_shape in shape_mismatch[:20]:
            print(f"shape_mismatch_key: {key} ckpt={ckpt_shape} local={local_shape}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
