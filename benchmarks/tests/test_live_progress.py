import json
import stat
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import live_progress as live
import unordered_runner


def report(
	worker=0,
	seconds=10.0,
	lower=100.0,
	upper=150.0,
	calls=1000,
	nodes=100,
	open_nodes=50,
	depth=7,
):
	return {
		"progress": 1,
		"worker": worker,
		"elapsed_seconds": seconds,
		"lower_bound": lower,
		"upper_bound": upper,
		"calls": calls,
		"nodes": nodes,
		"open_nodes": open_nodes,
		"peak_open_nodes": open_nodes,
		"pruned_nodes": 3,
		"max_sequence_depth": depth,
		"regions": 20,
	}


class CombineTests(unittest.TestCase):
	def test_portfolio_workers_share_the_best_valid_bounds(self):
		record = live.combine(
			{
				0: report(0, lower=100, upper=150, nodes=10, open_nodes=5),
				1: report(1, lower=120, upper=140, nodes=30, open_nodes=7, depth=9),
			}
		)
		self.assertEqual((record["lower_bound"], record["upper_bound"]), (120, 140))
		self.assertEqual(
			(
				record["nodes"],
				record["open_nodes"],
				record["max_sequence_depth"],
				record["searches"],
			),
			(40, 12, 9, 2),
		)

	def test_gap_is_relative_to_the_incumbent_and_absent_without_one(self):
		self.assertEqual(live.gap_fraction(90, 100), 0.1)
		self.assertIsNone(live.gap_fraction(0, None))
		self.assertIsNone(live.gap_fraction(0, float("inf")))
		self.assertEqual(live.gap_fraction(120, 100), 0)


class EstimateTests(unittest.TestCase):
	def history(self, gaps, opens):
		return [
			{"elapsed_seconds": 100.0 * i, "gap": gap, "open_nodes": open_nodes}
			for i, (gap, open_nodes) in enumerate(zip(gaps, opens))
		]

	def test_a_shrinking_gap_gives_a_rough_remaining_time_and_a_growing_queue_is_reported(
		self,
	):
		result = live.estimate(self.history([0.4, 0.2, 0.1], [100, 150, 220]), 0.0125)
		self.assertEqual(result["queue"], "growing")
		self.assertAlmostEqual(result["remaining_seconds"], 300.0, delta=1)
		self.assertGreater(result["gap_per_hour"], 0)

	def test_no_estimate_when_the_gap_is_not_clearly_falling(self):
		result = live.estimate(self.history([0.2, 0.2, 0.2], [100, 100, 100]), 0.01)
		self.assertEqual(result["queue"], "steady")
		self.assertIsNone(result["remaining_seconds"])
		self.assertEqual(
			live.estimate(self.history([0.2], [100]), 0.01),
			{"queue": None, "gap_per_hour": None, "remaining_seconds": None},
		)
		self.assertIsNone(
			live.estimate(self.history([0.4, 0.2], [10, 10]), None)["remaining_seconds"]
		)


class LiveStatusTests(unittest.TestCase):
	def test_reports_become_lines_and_an_atomic_snapshot_removed_when_finished(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "live.json"
			lines = []
			status = live.LiveStatus(path, echo=lines.append)
			callback = status.reporter(
				"a", "case 1/2", max_seconds=100.0, max_calls=10_000, target_gap=0.01
			)
			callback(report(seconds=10, lower=90, upper=100, calls=2500))
			self.assertIn("case 1/2", lines[0])
			self.assertIn("gap 10%", lines[0])
			self.assertIn("time limit 10% used", lines[0])
			self.assertIn("call limit 25% used", lines[0])
			saved = json.loads(path.read_text())
			self.assertEqual([r["key"] for r in saved["running"]], ["a"])
			self.assertEqual(saved["running"][0]["lower_bound"], 90)
			status.finish("a")
			self.assertEqual(json.loads(path.read_text())["running"], [])

	def test_close_removes_the_snapshot_of_a_finished_run(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "live.json"
			status = live.LiveStatus(path, echo=lambda line: None)
			status.reporter("a", "x")(report())
			status.finish("a")
			self.assertTrue(path.exists())
			status.close()
			self.assertFalse(path.exists())
			status.close()  # idempotent

	def test_a_broken_report_or_unwritable_snapshot_never_raises(self):
		status = live.LiveStatus(
			Path("/nonexistent-directory/live.json"), echo=lambda line: None
		)
		callback = status.reporter("a", "x")
		callback({"worker": 0})  # missing fields
		callback(report())
		status.finish("a")

	def test_no_call_limit_is_never_reported_as_a_fraction_used(self):
		status = live.LiveStatus(None, echo=lambda line: None)
		callback = status.reporter("a", "x", max_calls=-1)
		callback(report())
		self.assertNotIn("calls_used", status.last("a"))

	def test_snapshots_flag_silent_and_vanished_runs(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			(root / "run").mkdir()
			status = live.LiveStatus(root / "run" / "live.json", echo=lambda line: None)
			status.reporter("a", "case 1")(report())
			now = time.time()
			fresh = live.render_snapshots(root, now=now + 10)
			self.assertEqual(len(fresh), 1)
			self.assertNotIn("!!", fresh[0])
			self.assertIn(
				"no report for",
				live.render_snapshots(root, now=now + 1000, stale_after=300)[0],
			)
			data = json.loads((root / "run" / "live.json").read_text())
			data["pid"] = 2**22 + 12345
			(root / "run" / "live.json").write_text(json.dumps(data))
			self.assertEqual(live.render_snapshots(root, now=now + 10), [])
			entries, hidden = live.read_status(root, now=now + 10)
			self.assertEqual((entries, hidden), ([], 1))
			self.assertTrue((root / "run" / "live.json").exists())
			self.assertTrue((root / "run" / "live.json").exists())
			self.assertIn(
				"no longer running",
				live.render_snapshots(root, now=now + 10, show_gone=True)[0],
			)

	def test_dead_snapshots_are_deleted_only_when_they_belong_to_this_machine(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			for name, host in (("mine", live.socket.gethostname()), ("theirs", "another-machine")):
				(root / name).mkdir()
				(root / name / "live.json").write_text(
					json.dumps({"updated_at": time.time(), "pid": 2**22 + 7, "host": host, "running": []})
				)
			_, removed = live.read_status(root, remove_gone=True, show_idle=False)
			self.assertEqual(removed, 1)
			self.assertFalse((root / "mine" / "live.json").exists())
			self.assertTrue((root / "theirs" / "live.json").exists())

	def test_dead_snapshots_are_deleted_only_when_they_belong_to_this_machine(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			for name, host in (("mine", live.socket.gethostname()), ("theirs", "another-machine")):
				(root / name).mkdir()
				(root / name / "live.json").write_text(
					json.dumps({"updated_at": time.time(), "pid": 2**22 + 7, "host": host, "running": []})
				)
			_, removed = live.read_status(root, remove_gone=True, show_idle=False)
			self.assertEqual(removed, 1)
			self.assertFalse((root / "mine" / "live.json").exists())
			self.assertTrue((root / "theirs" / "live.json").exists())

	def test_idle_snapshots_are_hidden_unless_asked_for(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "live.json"
			path.write_text(json.dumps({"updated_at": time.time(), "running": []}))
			self.assertEqual(
				live.render_snapshots(Path(directory), show_idle=False), []
			)
			self.assertIn("nothing running", live.render_snapshots(Path(directory))[0])


class StreamingRunnerTests(unittest.TestCase):
	def fake_solver(self, directory, body):
		path = Path(directory) / "solver"
		path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
		path.chmod(path.stat().st_mode | stat.S_IEXEC)
		return path

	def run_solver(self, solver, **options):
		return unordered_runner.run_unordered_solver(
			solver, (0, 0), (0, 0), [[(0, 0), (1, 0), (0, 1)]], 100, 5, **options
		)

	def test_progress_lines_reach_the_callback_and_the_result_is_unchanged(self):
		with tempfile.TemporaryDirectory() as directory:
			solver = self.fake_solver(
				directory,
				"""
				import json, sys
				assert '--progress-interval' in sys.argv and sys.argv[sys.argv.index('--progress-interval') + 1] == '2.5'
				sys.stdin.read()
				print(json.dumps({'progress': 1, 'worker': 0, 'calls': 5}), file=sys.stderr, flush=True)
				print('some warning', file=sys.stderr, flush=True)
				print(json.dumps({'progress': 1, 'worker': 0, 'calls': 9}), file=sys.stderr, flush=True)
				print(json.dumps({'termination': 'optimal', 'calls': 9}))
			""",
			)
			seen = []
			result = self.run_solver(
				solver, progress=seen.append, progress_interval=2.5
			)
			self.assertEqual(result, {"termination": "optimal", "calls": 9})
			self.assertEqual([r["calls"] for r in seen], [5, 9])

	def test_without_a_callback_nothing_is_passed_and_stderr_is_not_streamed(self):
		with tempfile.TemporaryDirectory() as directory:
			solver = self.fake_solver(
				directory,
				"""
				import json, sys
				assert '--progress-interval' not in sys.argv
				sys.stdin.read()
				print(json.dumps({'ok': True}))
			""",
			)
			self.assertEqual(self.run_solver(solver), {"ok": True})
			self.assertEqual(
				self.run_solver(
					solver, progress=lambda record: None, progress_interval=0
				),
				{"ok": True},
			)

	def test_a_failing_callback_does_not_break_the_run_and_errors_keep_their_message(
		self,
	):
		with tempfile.TemporaryDirectory() as directory:
			good = self.fake_solver(
				directory,
				"""
				import json, sys
				sys.stdin.read()
				print(json.dumps({'progress': 1}), file=sys.stderr, flush=True)
				print(json.dumps({'ok': True}))
			""",
			)

			def explode(record):
				raise RuntimeError("boom")

			self.assertEqual(
				self.run_solver(good, progress=explode, progress_interval=1),
				{"ok": True},
			)
		with tempfile.TemporaryDirectory() as directory:
			bad = self.fake_solver(
				directory,
				"""
				import sys
				sys.stdin.read()
				print('real failure', file=sys.stderr)
				sys.exit(3)
			""",
			)
			with self.assertRaisesRegex(RuntimeError, "real failure"):
				self.run_solver(bad, progress=lambda record: None, progress_interval=1)

	def test_the_process_timeout_still_applies_while_streaming(self):
		with tempfile.TemporaryDirectory() as directory:
			slow = self.fake_solver(
				directory,
				"""
				import sys, time
				sys.stdin.read()
				time.sleep(30)
			""",
			)
			began = time.time()
			with self.assertRaises(unordered_runner.subprocess.TimeoutExpired):
				self.run_solver(
					slow,
					progress=lambda record: None,
					progress_interval=1,
					process_timeout=1,
				)
			self.assertLess(time.time() - began, 10)


if __name__ == "__main__":
	unittest.main()
