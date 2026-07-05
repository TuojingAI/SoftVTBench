"""Opt-in multi-process JAX data-loader patch.

This module intentionally leaves ``openpi.training.data_loader`` untouched on
disk.  Import ``apply`` from a dedicated multi-node entrypoint before calling the
normal training code.
"""

from __future__ import annotations

import logging
import multiprocessing
import typing

import jax
import torch

import openpi.training.data_loader as _dl


_APPLIED = False


def _torch_loader_init(
    self,
    dataset,
    local_batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    sampler: torch.utils.data.Sampler | None = None,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
):
    if len(dataset) < local_batch_size:
        raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

    self._sharding = sharding
    if sharding is None and framework == "jax":
        self._sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(jax.devices(), ("B",)),
            jax.sharding.PartitionSpec("B"),
        )
    self._num_batches = num_batches
    self._sampler = sampler

    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    generator = torch.Generator()
    generator.manual_seed(seed)
    self._data_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_dl._collate_fn,
        worker_init_fn=_dl._worker_init_fn,
        drop_last=True,
        generator=generator,
    )


def _torch_loader_iter(self):
    num_items = 0
    epoch = 0
    while True:
        if self._sampler is not None and hasattr(self._sampler, "set_epoch"):
            self._sampler.set_epoch(epoch)
        data_iter = iter(self._data_loader)
        while True:
            if self._num_batches is not None and num_items >= self._num_batches:
                return
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            num_items += 1
            if self._sharding is not None:
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
            else:
                yield jax.tree.map(torch.as_tensor, batch)
        epoch += 1


def _create_torch_data_loader(
    data_config,
    model_config,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
):
    dataset = _dl.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _dl.transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        process_count = jax.process_count()
        if batch_size % process_count != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by JAX process_count {process_count}.")
        local_batch_size = batch_size // process_count
        if process_count > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=process_count,
                rank=jax.process_index(),
                shuffle=shuffle,
                drop_last=True,
                seed=seed,
            )

    logging.info(
        "local_batch_size: %s process_index=%s process_count=%s sampler=%s",
        local_batch_size,
        jax.process_index() if framework == "jax" else "-",
        jax.process_count() if framework == "jax" else "-",
        type(sampler).__name__ if sampler is not None else "None",
    )
    data_loader = _dl.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )
    return _dl.DataLoaderImpl(data_config, data_loader)


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _dl.TorchDataLoader.__init__ = _torch_loader_init
    _dl.TorchDataLoader.__iter__ = _torch_loader_iter
    _dl.create_torch_data_loader = _create_torch_data_loader
    _APPLIED = True

