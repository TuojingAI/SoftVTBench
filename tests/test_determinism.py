import unittest

import numpy as np

from softvtbench.evaluation.determinism import episode_seed, inference_seed, stable_u32_seed
from softvtbench.evaluation.policies.openpi import OpenPIPolicy


class DeterminismTest(unittest.TestCase):
    def test_stable_and_condition_independent_identity(self):
        value = episode_seed(11, "object_soft", 3, "demo_80")
        self.assertEqual(value, episode_seed(11, "object_soft", 3, "demo_80"))
        self.assertLessEqual(value, 0xFFFFFFFF)
        self.assertNotEqual(value, episode_seed(11, "object_soft", 4, "demo_80"))

    def test_per_inference_key(self):
        base = episode_seed(11, "object_soft", 3, "demo_80")
        self.assertEqual(inference_seed(base, 0), inference_seed(base, 0))
        self.assertNotEqual(inference_seed(base, 0), inference_seed(base, 1))

    def test_delimiter_safe(self):
        self.assertNotEqual(stable_u32_seed("ab", "c"), stable_u32_seed("a", "bc"))

    def test_openpi_requests_are_keyed(self):
        requests = []

        class Client:
            def infer(self, element):
                requests.append(element)
                return {"actions": np.zeros((50, 7), dtype=np.float32)}

        policy = OpenPIPolicy.__new__(OpenPIPolicy)
        policy.modality = "vo"
        policy._is_vt = False
        policy._prompt = "test"
        policy._replan = 10
        policy._client = Client()
        policy._image = lambda x: np.asarray(x, dtype=np.uint8)
        policy._quat2aa = lambda _: np.zeros(3, dtype=np.float32)
        base = episode_seed(11, "object_soft", 3, "demo_80")
        obs = {
            "eef_pose": np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            "gripper_pos": np.asarray([0.04, -0.04], dtype=np.float32),
            "agentview_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "eye_in_hand_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        policy.reset(episode_seed=base)
        policy.observe(obs)
        policy.predict()
        policy.predict()
        self.assertEqual(
            requests[0]["__softvtbench_rng_seed"], inference_seed(base, 0)
        )
        self.assertEqual(
            requests[1]["__softvtbench_rng_seed"], inference_seed(base, 1)
        )

    def test_vt_marker_is_fail_closed(self):
        policy = OpenPIPolicy.__new__(OpenPIPolicy)
        policy._is_vt = True
        policy.reset(episode_seed=1)
        with self.assertRaisesRegex(RuntimeError, "tactile_left_rgb"):
            policy.observe({})


if __name__ == "__main__":
    unittest.main()
