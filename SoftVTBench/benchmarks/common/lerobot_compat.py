"""LeRobot API compatibility helpers.

LeRobotDataset.add_frame has had API differences across versions:
- Some versions accept: add_frame(frame, task=...)
- Some versions accept: add_frame(frame) and expect frame['task'] to exist (then pops it internally).
"""

from __future__ import annotations


def lerobot_add_frame(dataset, frame: dict, task: str) -> None:
    """Add one frame with task description, compatible with multiple LeRobot versions."""
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        frame["task"] = task
        dataset.add_frame(frame)


