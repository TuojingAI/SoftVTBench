# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

# Task description dictionary for xhumanoid task_suite
xhumanoid_task_dict = {
    416: "Put the cup on the second shelf",
    425: "Put the blue bowl on the pink bowl",
    443: "Put the blue bowl on the pink plate",
    461: "Put the strawberry from the pink plate into the blue bowl",
    470: "Put the apple from the pink plate into the blue bowl",
    480: "Put the cup on the blue bowl",
}


# Object name dictionary for xhumanoid task_suite
xhumanoid_object_name_dict = {
    416: ["Custom_cup_holder", "Plate_rack"],
    425: ["Custom_blue_bowl", "Custom_pink_bowl"],
    443: ["Custom_blue_bowl", "Plate_dataset"],
    461: ["Strawberry", "Custom_blue_bowl", "Plate_dataset"],
    470: ["Apple", "Custom_blue_bowl", "Plate_dataset"],
    480: ["Custom_cup_no_handle", "Custom_blue_bowl"],
}

libero_path_dict = {
    # Strict: user must provide the assembled_hdf5 directory via env var.
    "assembled_hdf5": os.getenv("HDF5_TRAJ_SOURCE_DIR", ""),
    # Directory containing replayed demos (input source for replay/evaluation).
    "replayed_demos": os.getenv("REPLAYED_DEMOS_DIR", ""),
    # Directory containing recorded demos (teleop output).
    "recorded_demos": os.getenv("RECORDED_DEMOS_DIR", ""),
}


def find_hdf5_file(hdf5_folder: Path, task_suite: str, task_id: int) -> Path | None:
    pattern = f"{task_suite}_task{task_id}_*_demo.hdf5"
    # Convert to absolute path if it's relative
    hdf5_folder_abs = hdf5_folder if hdf5_folder.is_absolute() else Path.cwd() / hdf5_folder
    matching_files = list(hdf5_folder_abs.glob(pattern))
    return matching_files[0] if matching_files else None


def setup_task_objects(task_suite, task_id, customized_file_paths: bool = False):
    """
    Set up task-related object environment variables
    Args:
        task_suite: Task suite name (e.g., "libero_10", "xhumanoid")
        task_id: Task ID number
        customized_file_paths: Whether to use customized file paths
    """

    if task_suite == "xhumanoid":
        if task_id not in xhumanoid_object_name_dict:
            print(f"[ERROR] Task ID {task_id} not found in xhumanoid_object_name_dict")
            return

        objects = xhumanoid_object_name_dict[task_id]
        if len(objects) == 2:
            os.environ["OBJECT_A_NAME"] = objects[0]
            os.environ["OBJECT_B_NAME"] = objects[1]
        else:
            os.environ["OBJECT_A_NAME"] = objects[0]
            os.environ["OBJECT_B_NAME"] = objects[1]
            os.environ["OBJECT_C_NAME"] = objects[2]

    elif task_suite.startswith("libero"):
        # Preferred task identifiers
        os.environ["TASK_SUITE"] = task_suite
        os.environ["TASK_ID"] = str(task_id)

        if customized_file_paths:
            return
        assembled_dir = libero_path_dict["assembled_hdf5"].strip()
        if not assembled_dir:
            raise ValueError(
                "Missing required env var: HDF5_TRAJ_SOURCE_DIR\n"
                "Please set:\n"
                "  export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5"
            )

        # Find the file name from assembled_hdf5 folder (auto-resolve by task_suite/task_id)
        assembled_file = find_hdf5_file(Path(assembled_dir), task_suite, task_id)

        # Reuse the same filename for assembled source.
        if assembled_file:
            # 始终根据 assembled_hdf5 自动推断并设置默认的 assembled 路径
            os.environ["HDF5_TRAJ_SOURCE_PATH"] = str(assembled_file)

            # Note (important):
            # We intentionally do NOT auto-fill REPLAYED_DEMOS_PATH / RECORDED_DEMOS_PATH here.
            # Those paths are user-controlled (manual recording/replay) and should not be "guessed"
            # by reusing the assembled filename. This avoids hidden defaults that can mislead debugging.

            print(f"[setup_task_objects] Task: {task_suite}, ID: {task_id}")
            print(f"  TRAJ_SRC:  {os.environ['HDF5_TRAJ_SOURCE_PATH']}")
        else:
            print(f"[ERROR] Could not find HDF5 file for {task_suite} task {task_id} in {libero_path_dict['assembled_hdf5']}")

        if (task_suite == "libero_90" and task_id > 89) or (task_suite != "libero_90" and task_id > 9):
            print(f"[ERROR] Task ID {task_id} not found in {task_suite}.")
            return
    else:
        print(f"[NOT IMPLEMENTED] Task suite {task_suite} not implemented.")
        return
