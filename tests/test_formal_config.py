import os
from pathlib import Path
import unittest
from unittest.mock import patch

from softvtbench.config import load_policy_manifest


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "SoftVTBench-Models"


class FormalConfigContractTest(unittest.TestCase):
    def _policies(self):
        with patch.dict(
            os.environ,
            {
                "SOFTVTBENCH_ROOT": str(ROOT),
                "SOFTVT_MODELS_ROOT": str(MODELS),
                "SOFTVT_CHECKPOINT_ROOT": "/checkpoints",
            },
            clear=False,
        ):
            return load_policy_manifest()["policies"]

    def test_formal_matrix_is_exact(self):
        policies = self._policies()
        suffixes = {
            "pi05_full_vo_c",
            "pi05_full_vt_c",
            "pi05_vo_c",
            "pi05_vt_c",
            "dp_vo_c",
            "dp_vt_c",
            "fastwam_vo_c",
            "fastwam_vt_c",
        }
        expected = {
            f"{suite}/{suffix}"
            for suite in ("object_soft", "spatial_soft")
            for suffix in suffixes
        }
        self.assertEqual({policy["id"] for policy in policies}, expected)
        self.assertEqual(len(policies), 16)

    def test_execution_profiles_lock_formal_semantics(self):
        policies = self._policies()
        for policy in policies:
            grip = policy["gripper_execution"]
            self.assertEqual(grip.get("total_width_tighten_m", 0), 0)
            if policy["backend"] == "openpi":
                self.assertEqual(policy["evaluation_protocol"], "chunked_30x10")
                self.assertEqual(grip["mode"], "continuous_fixed_position")
                self.assertEqual(grip["min_hold_steps"], 10)
            elif policy["backend"] == "diffusion":
                self.assertEqual(policy["evaluation_protocol"], "native_env_steps")
                self.assertEqual(policy["execution"], "native8")
            else:
                self.assertEqual(policy["evaluation_protocol"], "chunked_30x10")
                self.assertEqual(grip["mode"], "relative_fixed_position")
                expected_hold = 75 if policy["suite"] == "object_soft" else 68
                self.assertEqual(grip["min_hold_steps"], expected_hold)

    def test_four_suites_have_ordered_ten_tasks(self):
        import yaml

        for name in ("object_rigid", "spatial_rigid", "object_soft", "spatial_soft"):
            suite = yaml.safe_load((ROOT / "config/suites" / f"{name}.yaml").read_text())
            self.assertEqual([task["id"] for task in suite["tasks"]], list(range(10)))

    def test_ood_matrix_has_nine_unique_conditions(self):
        lines = [
            line.split()
            for line in (ROOT / "config/ood/formal_n50/conditions_9.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(lines), 9)
        self.assertEqual(len({parts[0] for parts in lines}), 9)


if __name__ == "__main__":
    unittest.main()

