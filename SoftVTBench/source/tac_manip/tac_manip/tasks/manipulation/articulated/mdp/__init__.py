# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the cabinet environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.manipulation.cabinet.mdp import *  # noqa: F401, F403
try:
    from isaaclab_tasks.manager_based.manipulation.place.mdp import *  # noqa: F401, F403
    from isaaclab_tasks.manager_based.manipulation.place.mdp import (  # noqa: F401, F403
        object_grasped as object_grasped_w_force,
    )
except ModuleNotFoundError:
    # Isaac Lab 2.1.x uses pick_place instead of place.
    from isaaclab_tasks.manager_based.manipulation.pick_place.mdp import *  # noqa: F401, F403
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import (  # noqa: F401, F403
        object_grasped as object_grasped_w_force,
    )
from isaaclab_tasks.manager_based.manipulation.stack.mdp import *  # noqa: F401, F403
from tac_manip.tasks.manipulation.libero.mdp import *  # noqa: F401, F403

from .terminations import *  # noqa: F401, F403
