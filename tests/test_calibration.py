from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.common.calibrate_safety_thresholds import build_payload, collect
from experiments.common.deformation import METRIC_ID


class CompressionCalibrationTest(unittest.TestCase):
    def test_threshold_uses_largest_stable_peak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.jsonl"
            rows = [
                {"asset": "a", "stable": True, "d_peak": 10.0},
                {"asset": "a", "stable": "true", "d_peak": 14.0},
                {"asset": "a", "stable": False, "d_peak": 30.0},
                {"asset": "b", "stable": True, "d_peak": 8.0},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            payload = build_payload([path], 0.5)
        self.assertEqual(payload["metric_id"], METRIC_ID)
        self.assertEqual(payload["thresholds"]["a"]["d_ref"], 14.0)
        self.assertEqual(payload["thresholds"]["a"]["tau"], 7.0)
        self.assertEqual(payload["thresholds"]["a"]["n_stable_trials"], 2)
        self.assertEqual(payload["thresholds"]["b"]["tau"], 4.0)

    def test_rejects_non_boolean_stability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"asset":"a","stable":1,"d_peak":2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                collect([path])


if __name__ == "__main__":
    unittest.main()
