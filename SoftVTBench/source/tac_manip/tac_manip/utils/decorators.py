# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from functools import wraps

# Store all decorated subtask functions
_REGISTERED_SUBTASKS: dict[str, Callable] = {}


def subtask_termination(func: Callable) -> Callable:
    """
    Decorator to mark subtask termination conditions.

    Args:
        func: The function to decorate

    Returns:
        The decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Register the function to the global dictionary
    _REGISTERED_SUBTASKS[func.__name__] = func
    return wrapper
