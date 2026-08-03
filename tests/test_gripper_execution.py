import unittest

import numpy as np

from softvtbench.evaluation.envs.build import calibrated_finger_lower_limit
from softvtbench.evaluation.gripper_execution import (
    ABSOLUTE_GRIPPER_MODES,
    GripperExecutor,
)


class ContinuousDirectTest(unittest.TestCase):
    """Lock the numerical contract of the policy_abs mode migrated from the internal collection stack."""

    def test_chunked_policy_abs_parity(self):
        executor = GripperExecutor(
            "continuous_direct", open_finger=0.04, closure_gain=1.0
        )
        self.assertEqual(executor.action_dim, 9)
        for raw in (-0.01, 0.0, 0.0037, 0.02, 0.04, 0.06):
            expected = np.float32(np.clip(raw, 0.0, 0.04))
            np.testing.assert_array_equal(
                executor.step(raw), np.asarray([expected, expected], dtype=np.float32)
            )

    def test_old_name_is_exact_alias(self):
        canonical = GripperExecutor("continuous_direct")
        legacy = GripperExecutor("measured_aperture_direct")
        self.assertIn(canonical.mode, ABSOLUTE_GRIPPER_MODES)
        self.assertIn(legacy.mode, ABSOLUTE_GRIPPER_MODES)
        for raw in (-0.1, 0.0034, 0.019, 0.0406):
            np.testing.assert_array_equal(canonical.step(raw), legacy.step(raw))

    def test_continuous_direct_has_no_binary_hold_or_latch(self):
        executor = GripperExecutor(
            "continuous_direct", close_threshold=0.0398, open_threshold=0.03995,
            min_hold_steps=10
        )
        commands = [0.04, 0.01, 0.03, 0.04]
        executed = [float(executor.step(value)[0]) for value in commands]
        np.testing.assert_allclose(executed, commands, rtol=0.0, atol=1e-7)

    def test_continuous_fixed_position_maps_intent_to_episode_target(self):
        executor = GripperExecutor(
            "continuous_fixed_position",
            close_threshold=0.0398,
            open_threshold=0.03995,
            min_hold_steps=2,
        )
        executor.reset(finger_lower_limit=0.006)
        np.testing.assert_allclose(executor.step(0.040), [0.040, 0.040])
        np.testing.assert_allclose(executor.step(0.020), [0.000, 0.000])
        # First open prediction is still covered by the declared hold.
        np.testing.assert_allclose(executor.step(0.040), [0.000, 0.000])
        np.testing.assert_allclose(executor.step(0.040), [0.040, 0.040])
        self.assertEqual(executor.last_diag["physical_close_limit"], 0.006)
        self.assertEqual(executor.action_dim, 9)

    def test_continuous_fixed_position_fails_without_episode_calibration(self):
        executor = GripperExecutor(
            "continuous_fixed_position",
            close_threshold=0.0398,
            open_threshold=0.03995,
        )
        with self.assertRaisesRegex(RuntimeError, "gripper_width / 2"):
            executor.step(0.02)

    def test_vo_total_width_tightening_is_split_across_fingers(self):
        self.assertAlmostEqual(
            calibrated_finger_lower_limit(0.012, 0.0006),
            0.0057,
        )
        self.assertAlmostEqual(
            calibrated_finger_lower_limit(0.012),
            0.006,
        )

    def test_invalid_total_width_tightening_fails_closed(self):
        with self.assertRaises(ValueError):
            calibrated_finger_lower_limit(0.0005, 0.0006)

    def test_non_finite_prediction_fails_closed(self):
        executor = GripperExecutor("binary")
        with self.assertRaisesRegex(ValueError, "finite"):
            executor.step(float("nan"))


class RelativeApertureTest(unittest.TestCase):
    def test_binary_closes_on_drop_and_opens_on_rebound(self):
        executor = GripperExecutor(
            "relative_decoded_binary",
            close_delta=0.003,
            open_delta=0.003,
            min_hold_steps=2,
        )
        executor.reset(finger_lower_limit=0.02)
        for raw in (0.0395, 0.0394, 0.0397, 0.0393):
            np.testing.assert_array_equal(executor.step(raw), [1.0])

        np.testing.assert_array_equal(executor.step(0.0366), [-1.0])
        np.testing.assert_array_equal(executor.step(0.0350), [-1.0])
        # The first rebound after the declared hold releases the latch.
        np.testing.assert_array_equal(executor.step(0.0381), [1.0])

    def test_threshold_adapts_to_narrow_calibrated_travel(self):
        executor = GripperExecutor(
            "relative_decoded_binary",
            close_delta=0.003,
            open_delta=0.003,
            aperture_noise_tolerance=0.0005,
        )
        executor.reset(finger_lower_limit=0.038)
        np.testing.assert_array_equal(executor.step(0.0396), [1.0])
        np.testing.assert_array_equal(executor.step(0.0392), [1.0])
        # 2 mm physical travel -> adaptive 1 mm trigger, not fixed 3 mm.
        np.testing.assert_array_equal(executor.step(0.0385), [-1.0])
        self.assertAlmostEqual(executor.last_diag["effective_close_delta"], 0.001)

    def test_unobservable_travel_precloses_without_changing_gap(self):
        binary = GripperExecutor(
            "relative_decoded_binary",
            close_delta=0.003,
            open_delta=0.003,
            aperture_noise_tolerance=0.0005,
        )
        binary.reset(finger_lower_limit=0.0398)
        np.testing.assert_array_equal(binary.step(0.0399), [-1.0])
        self.assertTrue(binary.last_diag["preclosed_unobservable"])

        fixed = GripperExecutor(
            "relative_fixed_position",
            close_delta=0.003,
            open_delta=0.003,
            aperture_noise_tolerance=0.0005,
        )
        fixed.reset(finger_lower_limit=0.0398)
        np.testing.assert_array_equal(fixed.step(0.0399), [0.0, 0.0])
        self.assertEqual(fixed.action_dim, 9)
        self.assertEqual(fixed.last_diag["executed_q_target"], 0.0)

    def test_relative_mode_requires_episode_calibration(self):
        executor = GripperExecutor(
            "relative_decoded_binary",
            close_delta=0.003,
            open_delta=0.003,
        )
        with self.assertRaisesRegex(RuntimeError, "gripper_width / 2"):
            executor.step(0.04)

    def test_invalid_relative_configuration_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "close_delta/open_delta"):
            GripperExecutor(
                "relative_decoded_binary",
                close_delta=0.0,
                open_delta=0.003,
            )
        executor = GripperExecutor(
            "relative_decoded_binary",
            close_delta=0.003,
            open_delta=0.003,
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            executor.reset(finger_lower_limit=0.041)

    def test_one_shot_release_is_terminal(self):
        executor = GripperExecutor(
            "relative_decoded_binary",
            close_delta=0.003,
            open_delta=0.003,
            min_hold_steps=2,
            one_shot=True,
        )
        executor.reset(finger_lower_limit=0.02)
        commands = [
            float(executor.step(raw)[0])
            for raw in (
                0.040, 0.036,       # first close
                0.034, 0.038,       # first release after hold
                0.040, 0.030, 0.025 # later drop must not re-close
            )
        ]
        self.assertEqual(commands, [1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertTrue(executor.last_diag["release_committed"])


if __name__ == "__main__":
    unittest.main()
