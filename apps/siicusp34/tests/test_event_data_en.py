"""A budget-limited F run must never inherit C's completion time or solved status."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_event_data_en as event


class EventDataTests(unittest.TestCase):
    def test_unfinished_extension_uses_latest_feasible_F_tour_without_C_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix, limit, solved in [("F", 300, False), ("F", 900, False), ("C", 21600, True)]:
                folder = root / f"{prefix}-ours-test-{limit}s"
                folder.mkdir()
                (folder / "plan.json").write_text(json.dumps({"time_limit": limit}))
                (folder / "runs.jsonl").write_text(json.dumps({"instance": "hard", "time_limit": limit,
                    "solved": solved, "result": {"path": [[0, 0], [1, 0], [0, 0]]}}) + "\n")
            pending = root / "F-ours-test-21600s"
            pending.mkdir()
            (pending / "plan.json").write_text(json.dumps({"time_limit": 21600}))
            tour = event.our_tours(root)["hard"]
            self.assertFalse(tour["solved"])
            self.assertEqual(tour["time_limit"], 900)

    def test_completion_time_requires_three_short_runs_or_a_closed_extension(self):
        rows = [{"case": "1", "configuration": "F", "root": "ours", "time_limit": "300",
                 "solved": "2", "repetitions": "3", "seconds_median": "1"}]
        self.assertEqual(event.solved_times(rows, "F", "ours"), {})
        rows.append({**rows[0], "time_limit": "900", "solved": "1", "repetitions": "1", "seconds_median": "350"})
        self.assertEqual(event.solved_times(rows, "F", "ours"), {0: 350})
        self.assertEqual(event.solved_within(rows, "F", "ours", 300), 0)
        self.assertEqual(event.solved_within(rows, "F", "ours", 900), 1)


if __name__ == "__main__":
    unittest.main()
