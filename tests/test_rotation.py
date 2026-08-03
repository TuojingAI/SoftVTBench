"""Characterization tests for the sealed quaternion branch convention."""

import unittest

import numpy as np

from softvtbench.evaluation.preprocessing.rotation import (
    QuatBranchTracker,
    _canonical_wpos,
    axis_angle_to_quat_wxyz,
    quat_wxyz_to_axis_angle,
)


class RotationContractTest(unittest.TestCase):
    def test_sealed_rotx_trajectory(self):
        quaternions = np.array(
            [
                [0.02, 0.999, 0.01, -0.01],
                [-0.01, -0.999, -0.02, 0.01],
                [0.03, 0.998, -0.015, 0.02],
                [0.04, -0.997, 0.01, -0.02],
            ],
            dtype=np.float64,
        )
        expected = np.array(
            [
                [3.1012511, 0.03104356, -0.03104356],
                [3.1207967, 0.06247841, -0.03123921],
                [3.0805430, -0.04630075, 0.06173433],
                [3.2209601, -0.03230652, 0.06461304],
            ],
            dtype=np.float32,
        )
        tracker = QuatBranchTracker()
        actual = np.stack([tracker.to_axis_angle(q) for q in quaternions])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_sign_equivalent_quaternions_match_after_reset(self):
        q = np.array([0.02, 0.999, 0.01, -0.01], dtype=np.float64)
        left = QuatBranchTracker().to_axis_angle(q)
        right = QuatBranchTracker().to_axis_angle(-q)
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-7)

    def test_axis_angle_roundtrip_in_principal_domain(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            angle = axis * rng.uniform(0, np.pi - 1e-3)
            recovered = quat_wxyz_to_axis_angle(
                _canonical_wpos(axis_angle_to_quat_wxyz(angle))
            )
            np.testing.assert_allclose(recovered, angle, rtol=0.0, atol=1e-5)

    def test_zero_norm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            _canonical_wpos(np.zeros(4))


if __name__ == "__main__":
    unittest.main()
