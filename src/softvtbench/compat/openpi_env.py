# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Image and rotation helpers shared with the OpenPI benchmark client.

`quat2axisangle` is derived from robosuite (MIT); see the docstring for the
upstream reference. Both helpers are imported by
`softvtbench.evaluation.policies.openpi` and are kept here so the OpenPI client
reproduces the preprocessing used during data collection bit for bit.
"""

import math

import cv2
import numpy as np
import torch


def resize_frames_with_padding(
    frames: torch.Tensor, target_image_size: tuple, bgr_conversion: bool = False, pad_img: bool = True
):
    """Process batch of frames with padding and resizing vectorized
    Args:
        frames: np.ndarray of shape [N, 256, 160, 3]
        target_image_size: target size
        bgr_conversion: whether to convert BGR to RGB
        pad_img: whether to resize images
    """
    # convert to cpu
    frames = frames.cpu().numpy()
    if bgr_conversion:
        frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames])

    if pad_img:
        top_padding = (frames.shape[2] - frames.shape[1]) // 2
        bottom_padding = top_padding

        # Add padding to all frames at once
        frames = np.pad(
            frames,
            pad_width=((0, 0), (top_padding, bottom_padding), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    # Resize all frames at once
    if frames.shape[1:] != target_image_size:
        frames = np.stack([cv2.resize(f, target_image_size[:2]) for f in frames])

    return frames


def quat2axisangle(quat):
    """
    Copied and modified from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    Args:
        quat (np.array): (w,x,y,z) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[0] > 1.0:
        quat[0] = 1.0
    elif quat[0] < -1.0:
        quat[0] = -1.0

    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[1:] * 2.0 * math.acos(quat[0])) / den
