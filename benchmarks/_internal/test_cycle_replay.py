"""Focused failure-path tests for captured cycle replay persistence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cycle_replay


class CycleReplayFailurePersistenceTest(unittest.TestCase):
	def setUp(self):
		self.temp = tempfile.TemporaryDirectory()
		self.root = Path(self.temp.name)
		self.capture = self.root / "capture.jsonl"
		self.capture.write_text(json.dumps({"event": "begin", "id": 1, "proposal_bound": True}) + "\n")
		self.binary = self.root / "replay-runner"
		self.binary.write_text("synthetic executable placeholder\n")

	def tearDown(self):
		self.temp.cleanup()

	def run_main(self, output, fake_run):
		arguments = ["--capture", str(self.capture), "--call-id", "1", "--seconds", "1",
			"--repetitions", "2", "--binary", str(self.binary), "--skip-build", "--output", str(output)]
		with patch.object(cycle_replay.subprocess, "check_output", return_value="test-commit\n"), \
				patch.object(cycle_replay.subprocess, "run", side_effect=fake_run), \
				patch.object(cycle_replay.platform, "platform", return_value="test-platform"):
			return cycle_replay.main(arguments)

	def test_nonzero_exit_keeps_complete_rows_and_marks_remainder_censored(self):
		output = self.root / "failed"
		line = json.dumps({"call_id": 1, "repeat": 0, "status": "completed", "gap_satisfied": "false",
			"proposal_bound": "true", "proposal_calls": "1", "proposal_accepts": "0"})
		process = CompletedProcess([], 7, stdout=line + "\n", stderr="runner failed after first row\n")
		self.assertEqual(self.run_main(output, lambda *args, **kwargs: process), 7)
		rows = (output / "raw.jsonl").read_text().splitlines()
		self.assertEqual(len(rows), 1)
		self.assertIs(json.loads(rows[0])["gap_satisfied"], False)
		self.assertIs(json.loads(rows[0])["proposal_bound"], True)
		self.assertEqual(json.loads(rows[0])["proposal_calls"], 1)
		self.assertEqual(json.loads(rows[0])["proposal_accepts"], 0)
		self.assertTrue(json.loads((output / "config.json").read_text())["options_by_call"]["1"]["proposal_bound"])
		progress = json.loads((output / "progress.json").read_text())
		self.assertTrue(progress["censored"])
		self.assertEqual(progress["planned_records"], 2)
		self.assertIn(line, (output / "stdout.log").read_text())

	def test_hard_timeout_keeps_complete_rows_and_marks_partial_tail_censored(self):
		output = self.root / "timeout"
		line = json.dumps({"call_id": 1, "repeat": 0, "status": "interrupted"})
		partial = line + "\n{\"call_id\":2"
		exception = TimeoutExpired([], 31, output=partial.encode(), stderr=b"time limit\n")
		self.assertEqual(self.run_main(output, lambda *args, **kwargs: (_ for _ in ()).throw(exception)), 1)
		self.assertEqual(len((output / "raw.jsonl").read_text().splitlines()), 1)
		progress = json.loads((output / "progress.json").read_text())
		self.assertEqual(progress["status"], "process_timeout")
		self.assertTrue(progress["censored"])
		self.assertEqual(progress["malformed_complete_lines"], 1)
		self.assertIn("{\"call_id\":2", (output / "stdout.log").read_text())


if __name__ == "__main__":
	unittest.main()
