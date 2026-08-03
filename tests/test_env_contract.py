from pathlib import Path
import tempfile
import unittest

from softvtbench.evaluation.envs.build import task_env_vars


class TaskEnvVarsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = {
            "libero_assets_data_dir": "/assets",
            "scene_visuals_module": "/scene_visuals.py",
            "eval_usd_dir": str(self.root),
        }
        self.suite = {
            "libero_suite": "libero_object",
            "libero_config_dir": "/config",
            "camera": {"width": 512, "height": 512, "views": ["agentview"]},
        }
        self.physics = {
            "deformable_body": {
                "simulation_hexahedral_resolution": 6,
                "solver_position_iteration_count": 64,
                "vertex_velocity_damping": 0.1,
                "contact_offset": 0.001,
                "rest_offset": 0.0,
                "max_depenetration_velocity": 1.0,
            }
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_rigid_card_does_not_inject_deformable_duplicate(self):
        extra = self.root / "rigid.json"
        extra.write_text('[{"name":"rigid_pastry001","rigid":true,"deformable":false}]')
        task = {
            "id": 0,
            "asset_name": "rigid_pastry001",
            "extra_assets_file": str(extra),
            "removed_source_assets": "alphabet_soup_1",
        }
        env = task_env_vars(
            self.paths, self.suite, task,
            {"name": "rigid_object_pastry001", "rigid": True, "deformable": False},
            self.physics,
        )
        self.assertIn("rigid_pastry001", env["SOFTVTBENCH_EXTRA_ASSETS_JSON"])
        self.assertEqual(env["SOFTVTBENCH_EXTRA_ASSET_USD"], "")
        self.assertEqual(env["SOFTVTBENCH_EXTRA_ASSET_DEFORMABLE"], "")
        self.assertEqual(env["SOFTVTBENCH_REMOVED_SOURCE_ASSETS"], "alphabet_soup_1")

    def test_soft_card_injects_exactly_one_deformable_target(self):
        usd_dir = self.root / "pastry001"
        usd_dir.mkdir()
        (usd_dir / "pastry001.usd").write_text("#usda 1.0\n")
        task = {
            "id": 0,
            "asset_name": "soft_pastry001",
            "asset_scale": "1 1 1",
            "asset_spawn_pos": "0 0 0",
        }
        env = task_env_vars(
            self.paths, self.suite, task,
            {"name": "soft_pastry001", "deformable": True, "usd_asset": "pastry001"},
            self.physics,
        )
        self.assertEqual(env["SOFTVTBENCH_EXTRA_ASSET_NAME"], "soft_pastry001")
        self.assertEqual(env["SOFTVTBENCH_EXTRA_ASSET_DEFORMABLE"], "1")
        self.assertEqual(env["SOFTVTBENCH_EXTRA_HEX_RES"], "6")


if __name__ == "__main__":
    unittest.main()
