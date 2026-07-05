from __future__ import annotations

import unittest

import numpy as np

from experiments.common.deformation import deformation_series


class DeformationMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            dtype=np.float64,
        )

    def test_translation_and_rotation_are_removed(self) -> None:
        angle = np.deg2rad(37.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        transformed = self.reference @ rotation.T + np.array([4.0, -2.0, 7.0])
        rms, max_deformation = deformation_series(self.reference, transformed)
        self.assertLess(float(rms[0]), 1.0e-5)
        self.assertLess(float(max_deformation[0]), 1.0e-5)

    def test_nonrigid_motion_is_positive(self) -> None:
        deformed = self.reference.copy()
        deformed[-1, 2] += 0.5
        rms, max_deformation = deformation_series(self.reference, deformed)
        self.assertGreater(float(rms[0]), 0.0)
        self.assertGreaterEqual(float(max_deformation[0]), float(rms[0]))

    def test_rejects_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            deformation_series(self.reference, np.zeros((5, 2)))


if __name__ == "__main__":
    unittest.main()
