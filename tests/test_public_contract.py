from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicContractTest(unittest.TestCase):
    def test_protocol_is_pi05_goal_and_safe(self) -> None:
        protocol = json.loads((ROOT / "configs/benchmark_protocol_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["release_scope"]["models"], ["pi05"])
        self.assertEqual(set(protocol["metrics"]), {"goal_success", "safe_success", "d_peak"})
        self.assertNotIn("nodrop", json.dumps(protocol).lower())

    def test_simulator_versions_match_the_released_environment(self) -> None:
        physics = json.loads((ROOT / "configs/simulation_physics_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(physics["simulator"]["isaac_sim"], "4.5.0")
        self.assertEqual(physics["simulator"]["isaac_lab"], "2.1.1")
        lock = (ROOT / "requirements.lock.txt").read_text(encoding="utf-8")
        self.assertIn("90b79bb2d44feb8d833f260f2bf37da3487180ba", lock)
        self.assertNotIn("requirements-" + "openpi.txt", lock)

    def test_public_launchers_are_pi05_only(self) -> None:
        for relative in ("openpi/scripts/train_softvtbench.sh", "openpi/scripts/evaluate_softvtbench.sh"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('MODEL}" != "pi05', text)

    def test_unified_task_subset_covers_public_object_and_spatial_tasks(self) -> None:
        relative = "openpi/configs/task_subset_softvtbench.json"
        subset = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        self.assertEqual(subset["libero_object"], list(range(10)))
        self.assertEqual(subset["libero_spatial"], list(range(10)))
        self.assertEqual(subset["libero_10"], [])
        self.assertEqual(subset["libero_goal"], [])

        for launcher in (
            "openpi/scripts/softvtbench_train_object_soft_pi05_vision_8gpu_20260625.sh",
            "openpi/scripts/train_object_soft_pi05_tactile_targetnext_7d_a800_20260626.sh",
            "openpi/scripts/softvtbench_train_spatial_soft_pi05_vision_8gpu_20260627.sh",
            "openpi/scripts/train_spatial_soft_pastry005_pi05_tactile_7d_parquet_norm_a800_20260626.sh",
            "openpi/scripts/train_rigid_pi05.sh",
        ):
            text = (ROOT / launcher).read_text(encoding="utf-8")
            self.assertIn("task_subset_softvtbench.json", text)

    def test_all_evaluations_default_to_50_rollouts(self) -> None:
        for relative in (
            "openpi/scripts/evaluate_softvtbench.sh",
            "experiments/object_soft/evaluation/run_object_soft_10tasks_pi05_eval.sh",
            "experiments/spatial_soft/evaluation/run_spatial_pastry005_pi05_eval.sh",
            "experiments/rigid/evaluation/run_rigid_pi05_eval.sh",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('N="${N:-50}"', text)
            self.assertIn("N must be a positive integer", text)

    def test_soft_eval_configs_cover_ten_tasks_and_50_initializations(self) -> None:
        for suite, task_config in (
            ("object_soft", "libero_object.json"),
            ("spatial_soft", "libero_spatial.json"),
        ):
            config_root = ROOT / "configs" / suite
            tasks = json.loads((config_root / task_config).read_text(encoding="utf-8"))
            scene_params = json.loads((config_root / "scene_params.json").read_text(encoding="utf-8"))
            self.assertEqual(tasks["total_tasks"], 10)
            self.assertEqual([task["task_id"] for task in tasks["tasks"]], list(range(10)))
            self.assertEqual(set(scene_params), {str(index) for index in range(10)})
            for task_id in range(10):
                self.assertEqual(set(scene_params[str(task_id)]), {f"demo_{index}" for index in range(50)})

        thresholds = json.loads((ROOT / "configs/safety_thresholds.json").read_text(encoding="utf-8"))
        self.assertEqual(len(thresholds["thresholds"]), 10)

    def test_rigid_eval_uses_training_config_norm_stats_ids(self) -> None:
        text = (ROOT / "experiments/rigid/evaluation/run_rigid_pi05_eval.sh").read_text(encoding="utf-8")
        self.assertIn('dst_asset_id="local/softvtbench_vision"', text)
        self.assertIn('dst_asset_id="local/softvtbench_tactile"', text)

    def test_soft_eval_accepts_the_default_training_asset_ids(self) -> None:
        expected = {
            "experiments/object_soft/evaluation/run_object_soft_10tasks_pi05_eval.sh": (
                "object_soft_vision_pi05_h50_targetnext_20260625",
                "object_soft_10assets_pi05_tactile_h50_targetnext_7d",
            ),
            "experiments/spatial_soft/evaluation/run_spatial_pastry005_pi05_eval.sh": (
                "spatial_soft_pastry005_pi05_vision_h50_targetnext_20260627",
                "spatial_soft_pastry005_pi05_tactile_h50_targetnext_7d",
            ),
        }
        for relative, asset_ids in expected.items():
            text = (ROOT / relative).read_text(encoding="utf-8")
            for asset_id in asset_ids:
                self.assertIn(asset_id, text)

    def test_soft_external_server_mode_does_not_require_a_checkpoint_view(self) -> None:
        for relative in (
            "experiments/object_soft/evaluation/run_object_soft_10tasks_pi05_eval.sh",
            "experiments/spatial_soft/evaluation/run_spatial_pastry005_pi05_eval.sh",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('CKPT=${CKPT:-}', text)
            self.assertIn('EXP=${EXP:-external_server}', text)
            self.assertIn('if [[ "${EXTERNAL_SERVER}" != "1" ]]; then\n  prepare_checkpoint_view\nfi', text)
            self.assertIn('--server_host "${SERVER_HOST}"', text)

    def test_softvtbench_train_configs_are_public_and_scoped(self) -> None:
        config = (ROOT / "openpi/upstream/src/openpi/training/config.py").read_text(encoding="utf-8")
        self.assertIn('name="pi05_lora_vision_softvtbench"', config)
        self.assertIn('name="pi05_lora_tacall_softvtbench"', config)
        self.assertNotIn("dim48", config)
        self.assertNotIn("univtac", config.lower())

        vision_converter = (
            ROOT / "openpi/upstream/examples/softvtbench/convert_softvtbench_vision_data_to_lerobot.py"
        ).read_text(encoding="utf-8")
        tactile_converter = (
            ROOT / "openpi/upstream/examples/softvtbench/convert_softvtbench_tactile_data_to_lerobot.py"
        ).read_text(encoding="utf-8")
        self.assertIn('repo_name: str = "local/softvtbench_vision"', vision_converter)
        self.assertIn('repo_name: str = "local/softvtbench_tactile"', tactile_converter)

    def test_public_training_phases_are_explicit(self) -> None:
        launcher = (ROOT / "openpi/scripts/train_softvtbench.sh").read_text(encoding="utf-8")
        docs = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn('PHASE="${PHASE:-all}"', launcher)
        for phase in ("convert", "stats", "train", "all"):
            self.assertIn(f"PHASE={phase}", docs)
        self.assertIn("SOFTVTBENCH_STOP_AFTER_CONVERT", launcher)

    def test_readme_uses_the_current_hosted_data_count(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("1,628 currently hosted", readme)
        self.assertIn("500 Object-Soft, 500 Spatial-Soft, 421 Object-Rigid, and 207 Spatial-Rigid", readme)
        self.assertNotIn("2,000 episodes", readme)
        self.assertIn("hf download Arthur12137/SoftVTBench", readme)
        self.assertNotIn("huggingface-cli", readme)

    def test_eval_preflight_checks_both_external_asset_bundles(self) -> None:
        doctor = (ROOT / "tools/doctor.py").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn('"--runtime-assets"', doctor)
        self.assertIn("check_runtime_assets(report", doctor)
        self.assertIn('--runtime-assets "$SOFTVTBENCH_DATA/tactile-runtime-assets/assets/data"', readme)

    def test_training_and_evaluation_share_one_openpi_client(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        lock = (ROOT / "requirements.lock.txt").read_text(encoding="utf-8")
        shared = "openpi/upstream/packages/openpi-client"
        self.assertIn(f"python -m pip install -e {shared}", readme)
        self.assertIn(f"-e ./{shared}", lock)
        for relative in (
            "experiments/object_soft/evaluation/run_object_soft_10tasks_pi05_eval.sh",
            "experiments/spatial_soft/evaluation/run_spatial_pastry005_pi05_eval.sh",
            "experiments/rigid/evaluation/run_rigid_pi05_eval.sh",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("${OPENPI_DIR}/packages/openpi-client/src", text)
            self.assertNotIn("${SOFTVTBENCH_DIR}/benchmarks/openpi/openpi-client/src", text)

    def test_soft_eval_uses_training_tactile_layout(self) -> None:
        for relative in (
            "experiments/object_soft/evaluation/run_object_soft_10tasks_pi05_eval.sh",
            "experiments/spatial_soft/evaluation/run_spatial_pastry005_pi05_eval.sh",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("SOFTVTBENCH_TACTILE_MOSAIC_LAYOUT=rows", text)
            self.assertNotIn("OPENPI_EVAL_PYTHON", text)

    def test_soft_eval_maps_public_suite_names_and_scores_only_rollouts(self) -> None:
        expected = {
            "experiments/object_soft/evaluation/run_object_soft_10tasks_pi05_eval.sh": (
                "object-soft",
                "libero_object",
            ),
            "experiments/spatial_soft/evaluation/run_spatial_pastry005_pi05_eval.sh": (
                "spatial-soft",
                "libero_spatial",
            ),
        }
        for relative, (public_name, internal_name) in expected.items():
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(f'[[ "${{SUITE:-}}" == "{public_name}" ]]', text)
            self.assertIn(f"SUITE={internal_name}", text)
            self.assertIn('"${METRICS}" "${DEBUG_ROOT}"', text)
            self.assertNotIn('"${METRICS}" "${OUT_ROOT}"', text)

        client = (ROOT / "SoftVTBench/benchmarks/openpi/openpi_inference_client.py").read_text(encoding="utf-8")
        self.assertIn("np.isfinite(gripper_threshold)", client)
        self.assertIn("np.isfinite(controller_gripper_norm[i])", client)

    def test_checkpoint_view_never_writes_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            view = base / "view"
            norm = source / "assets" / "training_asset" / "norm_stats.json"
            norm.parent.mkdir(parents=True)
            norm.write_text("{}\n", encoding="utf-8")
            (source / "params").mkdir()
            command = f"""
set -euo pipefail
source '{ROOT / 'openpi/scripts/softvtbench_paths.sh'}'
softvtbench_make_checkpoint_view '{source}' '{view}'
softvtbench_alias_norm_stats '{source}' '{view}' 'local/softvtbench_tactile' training_asset >/dev/null
test -L '{view / 'assets/local/softvtbench_tactile/norm_stats.json'}'
test ! -e '{source / 'assets/local'}'
"""
            subprocess.run(["bash", "-c", command], check=True)


if __name__ == "__main__":
    unittest.main()
