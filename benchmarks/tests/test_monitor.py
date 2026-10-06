import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import jobs
import live_progress
import monitor


class MonitorDataTests(unittest.TestCase):
	def test_tail_returns_last_lines_of_a_large_file(self):
		with tempfile.TemporaryDirectory() as folder:
			path = Path(folder) / "solver.log"
			path.write_text("".join(f"line {number}\n" for number in range(50000)))
			lines = monitor.tail(path, max_bytes=1000)
			self.assertEqual(lines[-1], "line 49999")
			self.assertLess(len(lines), 200)
			self.assertEqual(monitor.tail(Path(folder) / "missing.log")[0][:12], "(cannot read")

	def test_logs_found_newest_first(self):
		with tempfile.TemporaryDirectory() as folder:
			root = Path(folder)
			(root / "a" / "run").mkdir(parents=True)
			old, new = root / "a" / "run" / "solver.log", root / "a" / "progress.jsonl"
			old.write_text("x")
			new.write_text("y")
			os.utime(old, (time.time() - 100, time.time() - 100))
			(root / "a" / "report.json").write_text("{}")
			self.assertEqual(monitor.find_logs(root), [new, old])

	def test_instances_skip_snapshots_of_dead_processes(self):
		with tempfile.TemporaryDirectory() as folder:
			root = Path(folder)
			record = {"key": "1", "label": "free case 3/9", "elapsed_seconds": 5, "calls": 1, "nodes": 1,
				"open_nodes": 1, "reported_at": time.time(), "child_pid": 4242, "stop_signal": "SIGINT"}
			for name, pid in (("alive", os.getpid()), ("dead", 2**22 + 11)):
				(root / name).mkdir()
				(root / name / "live.json").write_text(json.dumps(
					{"pid": pid, "host": None, "updated_at": time.time(), "running": [record]}))
			rows = monitor.collect_instances(root)
			self.assertEqual([row.child["run"] for row in rows], ["alive"])
			with mock.patch.object(live_progress, "stop_child", return_value=True) as stop:
				self.assertIn("4242", monitor.stop_row(rows[0]))
				stop.assert_called_once()

	def test_jobs_listed_with_status_and_stoppable(self):
		with tempfile.TemporaryDirectory() as folder:
			job = Path(folder) / "20260101-000000-demo"
			job.mkdir()
			jobs._write(job, {"id": job.name, "command": ["tpp.py", "bench"], "exit_code": 0})
			with mock.patch.object(jobs, "jobs_dir", return_value=Path(folder)):
				rows = monitor.collect_jobs()
				self.assertEqual(len(rows), 1)
				self.assertTrue(rows[0].text.startswith("completed"))
				self.assertIn("not running", monitor.stop_row(rows[0]))


if __name__ == "__main__":
	unittest.main()
