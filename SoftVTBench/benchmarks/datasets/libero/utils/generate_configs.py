# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script is used to generate the task suite information for the Libero benchmark.
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
import copy
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def get_robot_table_info(env):
    # Get the robot's base position and orientation
    robot_base_pos = env.robots[0].base_pos.tolist()
    robot_base_ori = env.robots[0].base_ori.tolist()  # xyzw quaternion
    robot_base_ori = [robot_base_ori[3], robot_base_ori[0], robot_base_ori[1], robot_base_ori[2]]

    # get the table/floor length and offset
    workspace_name = env.env.workspace_name.lstrip("main_")
    if hasattr(env.env, f"{workspace_name}_full_size"):
        workspace_full_size = getattr(env.env, f"{workspace_name}_full_size")
    else:
        workspace_full_size = [0, 0, 0]
    if hasattr(env.env, f"{workspace_name}_offset"):
        workspace_offset = getattr(env.env, f"{workspace_name}_offset")
    else:
        workspace_offset = [0, 0, 0]

    return robot_base_pos, robot_base_ori, workspace_name, workspace_full_size, workspace_offset


def find_xml_path(libero_assets_path, obj_type):
    """
    Find the XML file with the name obj_type.xml in all subfolders of libero_assets_path.

    Args:
        libero_assets_path (str): Path to the libero assets directory
        obj_type (str): Name of the object type to find (without .xml extension)

    Returns:
        str: Full path to the found XML file, or None if not found
    """
    # Add .xml extension if not present
    if not obj_type.endswith(".xml"):
        obj_type = f"{obj_type}.xml"

    # Walk through all subdirectories
    for root, dirs, files in os.walk(libero_assets_path):
        if obj_type in files:
            return os.path.join(root, obj_type)

    return None


def extract_scale_from_xml(xml_path):
    """
    Extract scale information from an XML file.

    Args:
        xml_path (str): Path to the XML file

    Returns:
        tuple: Scale values (x, y, z) as floats, or None if not found
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Look for mesh elements in the asset section
        asset = root.find("asset")
        if asset is not None:
            # Find the first mesh element with a scale attribute
            for mesh in asset.findall("mesh"):
                scale = mesh.get("scale")
                if scale:
                    # Convert space-separated string to tuple of floats
                    return tuple(float(x) for x in scale.split())
    except Exception as e:
        print(f"Error reading XML file {xml_path}: {e}")

    return None


def validate_objects_regions(objects, regions):
    """
    Validate that all objects' initial regions are within the regions dictionary.
    If not, replace the object's initial region with the initial region of the specific object.

    Args:
        objects (dict): Dictionary of objects with their information
        regions (dict): Dictionary of regions with their pose ranges

    Returns:
        tuple: (is_valid, missing_regions, objects_without_regions)
    """

    # Check each object's initial region
    for obj_name, obj_info in objects.items():
        if "initial_region" in obj_info:
            initial_region = obj_info["initial_region"]
            if initial_region not in regions:
                print(f"\n[Warning] Object {obj_name} has initial region {initial_region}, which is not in regions")
                for name, info in objects.items():
                    if name in initial_region:  # find the object that the initial region is related to
                        new_init_region = info["initial_region"]
                        obj_info["initial_region"] = new_init_region + "_" + obj_name
                        print(f"[Info] Replace Object {obj_name} initial region to: {new_init_region + '_' + obj_name}\n")
                        # create a new region for the object
                        regions[new_init_region + "_" + obj_name] = copy.deepcopy(regions[new_init_region])
        else:
            # Object doesn't have an initial region (might be a fixture)
            print(f"[Info] Object {obj_name} has no initial_region (likely a fixture)")

    return objects, regions


def _process_object_info(parsed_data):
    """Process basic information for rigid body objects and fixtures"""
    objects = {}
    libero_assets_path = get_libero_path("assets")

    # Process rigid objects from parsed data
    for obj_type, obj_list in parsed_data["objects"].items():
        for obj_name in obj_list:
            objects[obj_name] = {"type": obj_type}
            _add_xml_path_and_scale(objects[obj_name], obj_type, obj_name, libero_assets_path)

    # Process fixtures (static objects) from parsed data
    for obj_type, obj_list in parsed_data["fixtures"].items():
        for obj_name in obj_list:
            objects[obj_name] = {"type": obj_type}
            _add_xml_path_and_scale(objects[obj_name], obj_type, obj_name, libero_assets_path)

    return objects


def _add_xml_path_and_scale(obj_info, obj_type, obj_name, libero_assets_path):
    """Add XML path and scale information to objects"""
    obj_xml_path = find_xml_path(libero_assets_path, obj_type)
    if obj_xml_path:
        obj_info["xml_path"] = os.path.join(get_libero_path("assets"), obj_xml_path)
        scale = extract_scale_from_xml(obj_xml_path)
        obj_info["scale"] = scale if scale else [1.0, 1.0, 1.0]
    else:
        obj_info["scale"] = [1.0, 1.0, 1.0]
        print(f"[Warning] XML file for {obj_name} not found.")


def _process_initial_states(parsed_data, objects):
    """Process initial states to get object placement information"""
    for state in parsed_data["initial_state"]:
        if state[0] == "on" or state[0] == "in":
            obj_name = state[1]
            region = state[2]
            if obj_name in objects:
                objects[obj_name]["initial_region"] = region
            else:
                print(f"[Warning] Object {obj_name} not found in objects dictionary.")


def _create_fixtures_dict(objects, env):
    """Create fixtures dictionary and process articulation properties"""
    fixtures = {}
    objects_to_remove = []

    for obj_name, obj_info in objects.items():
        # Process objects without initial regions
        if "initial_region" not in obj_info:
            fixtures[obj_name] = obj_info
            fixtures[obj_name]["initial_pose"] = getattr(env.env, f"{obj_name.lstrip('main_')}_offset")
            objects_to_remove.append(obj_name)

        # Process articulation properties
        if obj_name in env.env.fixtures_dict:
            _add_articulation_properties(objects[obj_name], env.env.fixtures_dict[obj_name])

    # Remove moved objects from objects dictionary
    for obj_name in objects_to_remove:
        del objects[obj_name]

    return fixtures


def _add_articulation_properties(obj_info, fixture_obj):
    """Add articulation properties to objects"""
    properties = getattr(fixture_obj, "object_properties")
    if "articulation" in properties:
        articulation = properties["articulation"]
        if "default_turnon_ranges" in articulation:
            obj_info["default_turnon_ranges"] = articulation["default_turnon_ranges"]
            obj_info["default_turnoff_ranges"] = articulation["default_turnoff_ranges"]
        if "default_open_ranges" in articulation:
            obj_info["default_open_ranges"] = articulation["default_open_ranges"]
            obj_info["default_close_ranges"] = articulation["default_close_ranges"]


def _process_goals(parsed_data, objects):
    """Process goals to prepare contact sensors for bending in Lab"""
    goals = []
    targets = set()

    for goal in parsed_data["goal_state"]:
        if goal[0] == "on" or goal[0] == "in":
            obj_name = goal[1]
            region = goal[2]
            target = next((target_name for target_name in objects if target_name in region), region)
            goals.append({
                "relationship": goal[0],
                "ref_obj": obj_name,
                "target": target,
                "xy_threshold": 0.1,
                "height_threshold": 0.1,
                "height_diff": 0.0,
                "enable_force_threshold": "True",
                "force_threshold": 0.05,
            })
            targets.add(target)
        elif len(goal) == 2:
            region = goal[1]
            target = next((target_name for target_name in objects if target_name in region), region)
            goals.append({"operation": goal[0], "target": target})
        else:
            print(f"[Warning] Invalid goal: {goal}")

    return goals, targets


def _process_obj_of_interest(env, targets):
    """Process objects of interest"""
    obj_to_grasp = env.obj_of_interest.copy()
    for obj_name in env.obj_of_interest:
        if any(target in obj_name for target in targets):
            obj_to_grasp.remove(obj_name)
            print(f"[Warning] Object {obj_name} contains a target, removed from obj_to_grasp.")
    return obj_to_grasp


def _get_object_bottom_offset(env, object_name):
    """Get object bottom offset"""
    try:
        return getattr(env.env.objects_dict[object_name], "bottom_offset")
    except (AttributeError, KeyError):
        try:
            return getattr(env.env.fixtures_dict[object_name], "bottom_offset")
        except (AttributeError, KeyError):
            print(f"[Warning] Could not find bottom_offset for {object_name} in either objects_dict or fixtures_dict")
            return [0, 0, 0]


def _process_regions(parsed_data, objects, env):
    """Process region information"""
    regions = {}

    for region_name, region_data in parsed_data["regions"].items():
        if not region_data["ranges"]:
            continue

        # Get target object for this region
        target = region_data["target"]
        z_height = getattr(env.env, f"{target.lstrip('main_')}_offset")[2]

        # Find object name with error handling
        matching_objects = [name for name, obj_info in objects.items() if obj_info.get("initial_region") == region_name]
        if not matching_objects:
            print(f"[Warning] No object found for region {region_name}, skipping...")
            continue

        # If multiple objects, create separate regions for each
        for object_name in matching_objects:
            new_region_name = region_name
            if len(matching_objects) > 1:
                new_region_name = f"{region_name}_{object_name}"
                print(f"[Info] Create new region {new_region_name} for object {object_name}")
                # Assign new region_name to object's initial_region
                if "initial_region" in objects[object_name]:
                    objects[object_name]["initial_region"] = new_region_name
            object_bottom_offset = _get_object_bottom_offset(env, object_name)
            adj_z_height = z_height - object_bottom_offset[2]

            # Extract ranges and yaw rotation
            x_range = (region_data["ranges"][0][0], region_data["ranges"][0][1])
            y_range = (region_data["ranges"][0][2], region_data["ranges"][0][3])
            yaw = tuple(region_data["yaw_rotation"])

            regions[new_region_name] = {
                "pose_range": {"x": x_range, "y": y_range, "z": (adj_z_height, adj_z_height), "yaw": yaw},
                "target": target,
            }

    return regions


def extract_objects_and_poses(bddl_file, env):
    """Extract objects and poses information from BDDL file and environment"""
    # Use robosuite_parse_problem to parse BDDL file
    parsed_data = BDDLUtils.robosuite_parse_problem(bddl_file)

    # Process object information
    objects = _process_object_info(parsed_data)

    # Process initial states
    _process_initial_states(parsed_data, objects)

    # Create fixtures dictionary
    fixtures = _create_fixtures_dict(objects, env)

    # Process goals
    goals, targets = _process_goals(parsed_data, objects)

    # Process objects of interest
    obj_to_grasp = _process_obj_of_interest(env, targets)

    # Process regions
    regions = _process_regions(parsed_data, objects, env)

    # Validate objects and regions, if object's initial region is not in regions, replace it with the initial region of the specific object.
    objects, regions = validate_objects_regions(objects, regions)


    return fixtures, objects, regions, goals, obj_to_grasp, list(targets)


def save_task_suite_info(task_suite, config_dir, task_suite_name):
    """
    Collect and save all task information from a task suite to a JSON file.
    """
    all_tasks_info = []

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        # Create environment to get additional information
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 128, "camera_widths": 128}
        env = OffScreenRenderEnv(**env_args)

        # get robot and table info
        robot_base_pos, robot_base_ori, workspace_name, workspace_full_size, workspace_offset = get_robot_table_info(
            env
        )

        # Get objects and regions information
        fixtures, objects, regions, goals, obj_to_grasp, targets = extract_objects_and_poses(task_bddl_file, env)

        # Collect all task information
        task_info = {
            "task_id": task_id,
            "task_name": task.name,
            "language_instruction": task.language,
            "problem_name": env.problem_name,
            "bddl_file": task_bddl_file,
            "obj_of_interest": obj_to_grasp,
            "targets": targets,
            "goals": goals,
            "fixtures": fixtures,
            "objects": objects,
            "regions": regions,
            "workspace_name": workspace_name,
            "workspace_full_size": workspace_full_size,
            "workspace_offset": workspace_offset,
            "robot_base_pos": robot_base_pos,
            "robot_base_ori": robot_base_ori,
            "workspace_name": workspace_name,
            "workspace_full_size": workspace_full_size,
            "workspace_offset": workspace_offset,
        }

        all_tasks_info.append(task_info)
        env.close()

    # Create filename
    filename = f"{config_dir}/{task_suite_name}.json"

    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(
            {"task_suite_name": task_suite_name, "total_tasks": task_suite.n_tasks, "tasks": all_tasks_info},
            f,
            indent=2,
        )

    print(f"\nTask suite information saved to {filename}")
    return filename


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate HDF5 files for LIBERO task replay")
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_10",
        help="Name of the task suite (e.g., libero_10, libero_spatial, libero_object, libero_goal)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize the task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    config_dir = "benchmarks/datasets/libero/config_"
    os.makedirs(config_dir, exist_ok=True)

    # Save all task information
    output_file = save_task_suite_info(task_suite, config_dir, args.task_suite)
    print(f"Task information has been saved to: {output_file}")
