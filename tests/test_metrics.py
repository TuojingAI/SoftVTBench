from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.common.deformation import METRIC_ID
from experiments.common.softvtbench_metrics import Episode, _episode_from_jsonl, summarize


class PublicMetricsTest(unittest.TestCase):
    def test_goal_and_safe_success_only(self) -> None:
        episodes = [
            Episode("jsonl", "a", "t", "0", "asset", True, 4.0, 5.0, 2, ""),
            Episode("jsonl", "b", "t", "1", "asset", True, 6.0, 5.0, 2, ""),
            Episode("jsonl", "c", "t", "2", "asset", False, 1.0, 5.0, 2, ""),
        ]
        result = summarize(episodes, rigid=False)
        self.assertAlmostEqual(result["goal_success_rate"], 200.0 / 3.0)
        self.assertAlmostEqual(result["safe_success_rate"], 100.0 / 3.0)
        self.assertNotIn("nodrop", json.dumps(result).lower())

    def test_online_metric_requires_canonical_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy_action_debug.jsonl"
            rows = [
                {
                    "frame": 0,
                    "exp_idx": 0,
                    "task_id": "libero_object_task0",
                    "success_now": False,
                    "fem_deformation_known": True,
                    "fem_deformation_metric_id": METRIC_ID,
                    "fem_deformation_rms": 2.0,
                    "fem_deformation_asset": "soft_a",
                },
                {
                    "frame": 1,
                    "exp_idx": 0,
                    "task_id": "libero_object_task0",
                    "success_now": True,
                    "fem_deformation_known": True,
                    "fem_deformation_metric_id": METRIC_ID,
                    "fem_deformation_rms": 7.0,
                    "fem_deformation_asset": "soft_a",
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            episode = _episode_from_jsonl(path, {"soft_a": 6.0})
        self.assertIsNotNone(episode)
        assert episode is not None
        self.assertTrue(episode.goal_success)
        self.assertEqual(episode.d_peak, 7.0)
        self.assertFalse(episode.safe_success(False))


if __name__ == "__main__":
    unittest.main()
