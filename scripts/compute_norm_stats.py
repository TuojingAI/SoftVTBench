"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import json
import pathlib

from datasets import load_dataset
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _lerobot_root(data_config: _config.DataConfig) -> pathlib.Path:
    if data_config.root is not None:
        return pathlib.Path(data_config.root)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    return pathlib.Path.cwd() / ".cache" / "lerobot" / data_config.repo_id


def _episode_ends(root: pathlib.Path) -> np.ndarray:
    ends = []
    total = 0
    with (root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            total += int(json.loads(line)["length"])
            ends.append(total)
    return np.asarray(ends, dtype=np.int64)


def _low_dim_sample(
    idx: int,
    action_horizon: int,
    data_config: _config.DataConfig,
    *,
    state_array: np.ndarray | None,
    actions_array: np.ndarray | None,
    episode_indices: np.ndarray,
    episode_ends: np.ndarray,
    force_array: np.ndarray | None,
    marker_column,
) -> dict:
    ep_idx = int(episode_indices[idx])
    ep_end = int(episode_ends[ep_idx])
    sample = {}
    if state_array is not None:
        sample["state"] = state_array[idx]

    if "actions" in data_config.action_sequence_keys and actions_array is not None:
        query_indices = [min(ep_end - 1, idx + delta) for delta in range(action_horizon)]
        sample["actions"] = actions_array[query_indices]

    if force_array is not None:
        sample["tactile_gripper_force"] = force_array[idx]
    if marker_column is not None:
        sample["tactile_marker_motion"] = np.asarray(marker_column[idx], dtype=np.float32)

    if data_config.prompt_from_task:
        # Low-dim stats do not use prompt, but some transforms expect the key to exist.
        sample["prompt"] = ""

    return sample


def compute_low_dim_stats(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> dict[str, normalize.NormStats]:
    """Compute stats without decoding image/video columns.

    Normalization only uses low-dimensional keys. Reading videos here is both expensive
    and unnecessary, so this path materializes only the parquet columns that feed state,
    actions, and Tabero tactile low-dimensional encoders.
    """

    root = _lerobot_root(data_config)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text())

    hf_dataset = load_dataset("parquet", data_dir=str(root / "data"), split="train").with_format(None)
    low_dim_columns = [
        "state",
        "actions",
        "episode_index",
        "index",
        "tactile_gripper_force",
        "tactile_marker_motion",
        "task_index",
    ]
    hf_dataset = hf_dataset.select_columns([col for col in low_dim_columns if col in hf_dataset.column_names])
    limit = len(hf_dataset) if max_frames is None else min(max_frames, len(hf_dataset))
    episode_ends = _episode_ends(root)
    episode_indices = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    state_array = np.asarray(hf_dataset["state"], dtype=np.float32) if "state" in hf_dataset.column_names else None
    actions_array = (
        np.asarray(hf_dataset["actions"], dtype=np.float32) if "actions" in hf_dataset.column_names else None
    )
    force_array = (
        np.asarray(hf_dataset["tactile_gripper_force"], dtype=np.float32)
        if "tactile_gripper_force" in hf_dataset.column_names
        else None
    )
    marker_column = (
        np.asarray(hf_dataset["tactile_marker_motion"], dtype=np.float32)
        if "tactile_marker_motion" in hf_dataset.column_names
        else None
    )

    transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ]
    )

    keys = ["state", "actions", "tactile_prefix", "tactile_suffix"]
    stats = {key: normalize.RunningStats() for key in keys}
    seen = {key: False for key in keys}

    batch = {key: [] for key in keys}
    for idx in tqdm.tqdm(range(limit), desc="Computing low-dim stats"):
        sample = _low_dim_sample(
            idx,
            action_horizon,
            data_config,
            state_array=state_array,
            actions_array=actions_array,
            episode_indices=episode_indices,
            episode_ends=episode_ends,
            force_array=force_array,
            marker_column=marker_column,
        )
        # Provide dummy images only for transforms that require image keys.
        dummy = np.zeros((1, 1, 3), dtype=np.uint8)
        sample.setdefault("image", dummy)
        sample.setdefault("wrist_image", dummy)
        sample.setdefault("tactile_image", dummy)
        transformed = transform(sample)
        for key in keys:
            if key in transformed:
                batch[key].append(np.asarray(transformed[key]))
                seen[key] = True

        if (idx + 1) % batch_size == 0:
            for key in keys:
                if batch[key]:
                    stats[key].update(np.asarray(batch[key]))
                    batch[key].clear()

    for key in keys:
        if batch[key]:
            stats[key].update(np.asarray(batch[key]))

    if int(info.get("total_frames", len(hf_dataset))) != len(hf_dataset):
        print(f"Warning: metadata total_frames={info.get('total_frames')} but parquet rows={len(hf_dataset)}")

    return {key: stats[key].get_statistics() for key in keys if seen[key]}


def main(
    config_name: str,
    max_frames: int | None = None,
    repo_id: str | None = None,
    root: str | None = None,
    assets_dir: str | None = None,
    asset_id: str | None = None,
    low_dim_only: bool = False,
):
    config = _config.get_config(config_name)
    if repo_id is not None or root is not None or assets_dir is not None or asset_id is not None:
        updates = {}
        if repo_id is not None:
            updates["repo_id"] = repo_id
        if root is not None:
            updates["root"] = root
        if assets_dir is not None or asset_id is not None:
            updates["assets"] = dataclasses.replace(
                config.data.assets,
                **({"assets_dir": assets_dir} if assets_dir is not None else {}),
                **({"asset_id": asset_id} if asset_id is not None else {}),
            )
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, **updates))
    data_config = config.data.create(config.assets_dirs, config.model)

    if low_dim_only:
        norm_stats = compute_low_dim_stats(data_config, config.model.action_horizon, config.batch_size, max_frames)
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

        keys = ["state", "actions", "tactile_prefix", "tactile_suffix"]
        stats = {key: normalize.RunningStats() for key in keys}
        seen = {key: False for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                if key not in batch:
                    continue
                stats[key].update(np.asarray(batch[key]))
                seen[key] = True

        norm_stats = {key: stats[key].get_statistics() for key in keys if seen[key]}

    if data_config.asset_id is None:
        raise ValueError("Data config must have an asset_id")
    output_path = pathlib.Path(config.data.assets.assets_dir or config.assets_dirs) / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
