import copy
import unittest

from softvtbench.evaluation import receipt


SUITE = {
    "name": "object_rigid",
    "libero_suite": "libero_object",
    "camera": {"width": 512, "height": 512, "views": ["agentview"]},
    "apply_scene_visuals": False,
}


def good_receipt():
    return {
        "schema": "softvtbench_scene_receipt_v3_fail_closed",
        "missing_scene_prims": [],
        "unexpected_scene_prims": [],
        "removed_sources_present": [],
        "expected_hidden_sources": {},
        "tiny_prims": {},
        "target_asset": {
            "name": "rigid_pastry001",
            "present": True,
            "scale": [1, 1, 1],
            "configured_scale": [1, 1, 1],
            "physics": {},
        },
        "extra_assets": {},
        "cameras_actual": {"agentview_cam": {"width": 512, "height": 512}},
        "language": "pick up pastry",
        "scene_config_language": "pick up pastry",
        "hdf5_language": "pick up pastry",
        "robot_friction_configured": None,
        "simulator": {
            "python": "3.10",
            "isaac_sim": "4.5.0.0",
            "isaac_lab": "0.41.3",
            "physics_hz": 60.0,
            "control_hz": 20.0,
            "enable_ccd": True,
        },
        "physics_config": {
            "simulator": {
                "python": "3.10",
                "isaac_sim": "4.5.0.0",
                "isaac_lab": "0.41.3",
                "physics_hz": 60,
                "control_hz": 20,
            },
            "deformable_body": {"enable_ccd": True},
        },
        "release": {"commit": "deadbeef", "dirty": "0"},
    }


class FailClosedReceiptTest(unittest.TestCase):
    def test_complete_receipt_passes(self):
        rec = good_receipt()
        receipt.assert_contract(rec, suite=SUITE, obj=None)
        self.assertTrue(rec["contract"]["passed"])

    def test_missing_camera_is_failure(self):
        rec = good_receipt()
        rec["cameras_actual"] = {}
        with self.assertRaisesRegex(RuntimeError, "required camera"):
            receipt.assert_contract(rec, suite=SUITE, obj=None)

    def test_missing_target_is_failure(self):
        rec = good_receipt()
        rec["target_asset"]["present"] = False
        with self.assertRaisesRegex(RuntimeError, "target absent"):
            receipt.assert_contract(rec, suite=SUITE, obj=None)

    def test_unexpected_prim_is_failure(self):
        rec = good_receipt()
        rec["unexpected_scene_prims"] = ["akita_black_bowl_2"]
        with self.assertRaisesRegex(RuntimeError, "unexpected scene prims"):
            receipt.assert_contract(rec, suite=SUITE, obj=None)

    def test_missing_runtime_material_is_failure(self):
        rec = good_receipt()
        obj = {
            "deformable": True,
            "fem_nodes": 182,
            "expected": {
                "youngsModulus": 60000,
                "poissonsRatio": 0.35,
                "density": 240,
                "dynamicFriction": 1.5,
            },
        }
        rec["fem_nodal_count"] = 182
        with self.assertRaisesRegex(RuntimeError, "runtime target physics"):
            receipt.assert_contract(rec, suite=SUITE, obj=obj)

    def test_runtime_package_version_mismatch_is_failure(self):
        rec = good_receipt()
        rec["simulator"]["isaac_sim"] = "5.1.0.0"
        with self.assertRaisesRegex(RuntimeError, "isaac_sim"):
            receipt.assert_contract(rec, suite=SUITE, obj=None)

    def test_hdf5_language_is_authoritative(self):
        rec = good_receipt()
        rec["scene_config_language"] = "harmless BDDL paraphrase"
        receipt.assert_contract(rec, suite=SUITE, obj=None)
        rec = good_receipt()
        rec["hdf5_language"] = "different collected prompt"
        with self.assertRaisesRegex(RuntimeError, "hdf5"):
            receipt.assert_contract(rec, suite=SUITE, obj=None)


def good_episode_receipt():
    return {
        "schema": "softvtbench_episode_receipt_v1_fail_closed",
        "suite": "object_rigid",
        "task_id": 0,
        "episode": "demo_7",
        "condition": "id",
        "condition_kind": "id",
        "episode_seed": 123,
        "language": "pick up pastry",
        "release": {"commit": "deadbeef", "dirty": "1"},
        "gripper_constraint_mode": "lower_limit_only",
        "static_contract_passed": True,
        "hdf5": {
            "exists": True,
            "episode_group": {
                "present": True,
                "episode": "demo_7",
                "attrs": {
                    "task_id": 0,
                    "task_suite": "libero_object",
                    "asset_name": "rigid_pastry001",
                    "language": "pick up pastry",
                    "original_demo_id": 7007,
                },
                "initial_state": {
                    "present": True,
                    "dataset_count": 3,
                    "sha256": "abc",
                },
            },
        },
        "scene_params": {},
        "target_asset": {
            "name": "rigid_pastry001",
            "present": True,
            "scale": [1, 1, 1],
            "configured_scale": [1, 1, 1],
        },
        "extra_assets": {},
        "initial_state_readback": {
            "missing": [],
            "checks": {
                "articulation/robot/joint_position": {
                    "shape": [9],
                    "max_abs_error": 0.0,
                },
            },
        },
        "simulator": {"first_render_had_collection_visuals": True},
        "dome_light": {"present": True, "active": True, "intensity": 135.0},
        "ood": None,
    }


class EpisodeReceiptTest(unittest.TestCase):
    def test_complete_episode_receipt_passes(self):
        rec = good_episode_receipt()
        receipt.assert_episode_contract(rec, suite=SUITE)
        self.assertTrue(rec["contract"]["passed"])

    def test_missing_initial_state_fails(self):
        rec = good_episode_receipt()
        rec["hdf5"]["episode_group"]["initial_state"]["dataset_count"] = 0
        with self.assertRaisesRegex(RuntimeError, "initial_state"):
            receipt.assert_episode_contract(rec, suite=SUITE)

    def test_nonformal_gripper_constraint_fails(self):
        rec = good_episode_receipt()
        rec["gripper_constraint_mode"] = "hard_pin"
        with self.assertRaisesRegex(RuntimeError, "constraint mode"):
            receipt.assert_episode_contract(rec, suite=SUITE)

    def test_scene_seed_mismatch_fails(self):
        rec = good_episode_receipt()
        rec["scene_params"] = {
            "asset_name": "rigid_pastry001",
            "collection_demo_id": 999,
        }
        with self.assertRaisesRegex(RuntimeError, "collection_demo_id"):
            receipt.assert_episode_contract(rec, suite=SUITE)

    def test_ood_condition_requires_ood_receipt(self):
        rec = good_episode_receipt()
        rec["condition"] = "mass_l1"
        rec["condition_kind"] = "ood"
        with self.assertRaisesRegex(RuntimeError, "no enabled OOD receipt"):
            receipt.assert_episode_contract(rec, suite=SUITE)

    def test_id_label_does_not_need_to_be_literal_id(self):
        rec = good_episode_receipt()
        rec["condition"] = "ID_L0"
        receipt.assert_episode_contract(rec, suite=SUITE)
        self.assertTrue(rec["contract"]["passed"])

    def test_soft_vo_c_receipt_accepts_total_width_tightening(self):
        suite = dict(SUITE, name="object_soft")
        for mode in ("continuous_fixed_position", "relative_fixed_position"):
            with self.subTest(mode=mode):
                rec = good_episode_receipt()
                rec["suite"] = "object_soft"
                rec["grip_width_m"] = 0.012
                rec["gripper_joint_limits"] = {"lower": [0.0057, 0.0057]}
                rec["policy"] = {
                    "modality": "vo",
                    "gripper_execution": {
                        "mode": mode,
                        "total_width_tighten_m": 0.0006,
                    },
                }
                receipt.assert_episode_contract(rec, suite=suite)
                self.assertTrue(rec["contract"]["passed"])

    def test_soft_vt_c_receipt_requires_exact_hdf5_width(self):
        rec = good_episode_receipt()
        rec["suite"] = "object_soft"
        rec["grip_width_m"] = 0.012
        rec["gripper_joint_limits"] = {"lower": [0.006, 0.006]}
        rec["policy"] = {
            "modality": "vt",
            "gripper_execution": {"mode": "continuous_fixed_position"},
        }
        suite = dict(SUITE, name="object_soft")
        receipt.assert_episode_contract(rec, suite=suite)
        self.assertTrue(rec["contract"]["passed"])

    def test_soft_vt_c_rejects_vo_only_tightening(self):
        rec = good_episode_receipt()
        rec["suite"] = "object_soft"
        rec["grip_width_m"] = 0.012
        rec["gripper_joint_limits"] = {"lower": [0.0057, 0.0057]}
        rec["policy"] = {
            "modality": "vt",
            "gripper_execution": {
                "mode": "continuous_fixed_position",
                "total_width_tighten_m": 0.0006,
            },
        }
        suite = dict(SUITE, name="object_soft")
        with self.assertRaisesRegex(RuntimeError, "restricted to soft VO"):
            receipt.assert_episode_contract(rec, suite=suite)

    def test_rigid_relative_decoder_requires_episode_calibration(self):
        rec = good_episode_receipt()
        rec["policy"] = {
            "modality": "vo",
            "gripper_execution": {"mode": "relative_decoded_binary"},
        }
        with self.assertRaisesRegex(RuntimeError, "no recorded gripper width"):
            receipt.assert_episode_contract(rec, suite=SUITE)

        rec["grip_width_m"] = 0.076
        rec["gripper_joint_limits"] = {"lower": [0.038, 0.038]}
        receipt.assert_episode_contract(rec, suite=SUITE)
        self.assertTrue(rec["contract"]["passed"])


if __name__ == "__main__":
    unittest.main()
