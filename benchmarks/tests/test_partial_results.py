"""A solver that is stopped, killed or runs out of memory still leaves its bounds behind."""

import io
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import struct
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import free_order_campaign
import live_progress as live
import process_guard
import tspn_run_comparison
import unordered_runner


def _two_case_suite() -> bytes:
	encoded = bytearray()
	for index in range(2):
		encoded.extend(struct.pack("<ddddQ", 0.0, 0.0, 10.0, 0.0, 1))
		encoded.extend(struct.pack("<Q", 4))
		for x, y in ((index * 10.0, 1.0), (index * 10.0 + 1.0, 1.0),
			(index * 10.0 + 1.0, 2.0), (index * 10.0, 2.0)):
			encoded.extend(struct.pack("<dd", x, y))
		encoded.extend(struct.pack("<Q", 0))
	return bytes(encoded)


def fake_solver(directory, body):
	path = Path(directory) / "solver"
	path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
	path.chmod(path.stat().st_mode | stat.S_IEXEC)
	return path


def run_solver(solver, **options):
	return unordered_runner.run_unordered_solver(
		solver, (0, 0), (0, 0), [[(0, 0), (1, 0), (0, 1)]], 100, 5, **options
	)


class JournalTests(unittest.TestCase):
	def test_every_report_is_appended_to_a_journal_that_outlives_the_run(self):
		with tempfile.TemporaryDirectory() as directory:
			status = live.LiveStatus(Path(directory) / "live.json", echo=lambda line: None)
			callback = status.reporter("a", "free case 1/2 tpp-ours")
			callback({"worker": 0, "elapsed_seconds": 1.0, "lower_bound": 1.0, "upper_bound": 3.0,
				"calls": 5, "nodes": 2, "open_nodes": 1, "max_sequence_depth": 1})
			callback({"worker": 0, "elapsed_seconds": 2.0, "lower_bound": 2.0, "upper_bound": 3.0,
				"calls": 9, "nodes": 4, "open_nodes": 2, "max_sequence_depth": 1})
			last = status.finish("a")
			status.close()
			lines = [json.loads(line) for line in (Path(directory) / "progress.jsonl").read_text().splitlines()]
			self.assertEqual([line["lower_bound"] for line in lines], [1.0, 2.0])
			self.assertEqual(lines[0]["label"], "free case 1/2 tpp-ours")
			self.assertEqual(last["upper_bound"], 3.0)
			self.assertFalse((Path(directory) / "live.json").exists())

	def test_the_snapshot_names_the_process_and_how_to_stop_it(self):
		with tempfile.TemporaryDirectory() as directory:
			status = live.LiveStatus(Path(directory) / "live.json", echo=lambda line: None)
			callback = status.reporter("a", "free case 3/9 tpp-fekete", stop_signal=signal.SIGTERM)
			status.set_pid("a", 4242)
			callback({"worker": 0, "elapsed_seconds": 1.0, "calls": 1, "nodes": 1, "open_nodes": 0, "max_sequence_depth": 0})
			(record,) = status.snapshot()["running"]
			self.assertEqual((record["child_pid"], record["stop_signal"]), (4242, "SIGTERM"))


class SolverDeathTests(unittest.TestCase):
	def test_a_solver_killed_by_a_signal_is_reported_as_killed_with_its_pid(self):
		with tempfile.TemporaryDirectory() as directory:
			solver = fake_solver(directory, """
				import json, os, signal, sys
				sys.stdin.read()
				print(json.dumps({'progress': 1, 'worker': 0, 'calls': 5}), file=sys.stderr, flush=True)
				os.kill(os.getpid(), signal.SIGKILL)
			""")
			seen, pids = [], []
			with self.assertRaises(unordered_runner.SolverKilled) as caught:
				run_solver(solver, progress=seen.append, progress_interval=1, on_start=pids.append)
			self.assertEqual(caught.exception.signal_number, signal.SIGKILL)
			self.assertEqual(len(pids), 1)
			self.assertEqual(len(seen), 1)

	def test_the_memory_guard_asks_the_solver_to_stop_and_the_solver_still_answers(self):
		with tempfile.TemporaryDirectory() as directory:
			solver = fake_solver(directory, """
				import json, signal, sys, time
				stopped = []
				signal.signal(signal.SIGINT, lambda *_: stopped.append(1))
				sys.stdin.read()
				print(json.dumps({'progress': 1, 'worker': 0, 'calls': 5}), file=sys.stderr, flush=True)
				deadline = time.time() + 20
				while not stopped and time.time() < deadline:
					time.sleep(0.05)
				print(json.dumps({'termination': 'interrupted', 'lower_bound': 1.0, 'upper_bound': 2.0}))
			""")
			used = []
			with patch.object(process_guard, "POLL_SECONDS", 0.1), patch.object(
				process_guard, "rss_bytes", return_value=10 * 2**30
			):
				result = run_solver(solver, max_memory_bytes=2**30, on_memory_limit=used.append)
			self.assertEqual(result["termination"], "interrupted")
			self.assertEqual(used, [10 * 2**30])

	def test_the_guard_reads_a_real_process_and_ignores_one_below_the_limit(self):
		process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
		try:
			self.assertGreater(process_guard.rss_bytes(process.pid), 1_000_000)
			guard = process_guard.MemoryGuard(process.pid, 2**40, poll_seconds=0.05)
			time.sleep(0.3)
			self.assertFalse(guard.triggered)
			guard.close()
		finally:
			process.kill()
			process.wait()
		self.assertIsNone(process_guard.rss_bytes(process.pid))


class CampaignRowTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		root = Path(self.directory.name)
		self.campaign = root / "campaign"
		self.campaign.mkdir()
		(self.campaign / "tiny.bin").write_bytes(_two_case_suite())
		(self.campaign / "campaign.json").write_text(json.dumps({"inputs": [{"file": "tiny.bin"}]}))
		self.binary = root / "tpp"
		self.binary.write_bytes(b"stub")

	def run_main(self, solver, *extra):
		with patch.object(free_order_campaign, "BINARY", self.binary), patch.object(
			free_order_campaign, "ensure_binary"
		), patch.object(free_order_campaign, "run_unordered_solver", side_effect=solver), patch.object(
			free_order_campaign, "validate_path", return_value={"valid": True}
		), patch("sys.stdout", io.StringIO()):
			return free_order_campaign.main(
				[str(self.campaign), "--solver", "tpp-ours", "--max-seconds", "60", "--no-build",
				 "--progress-interval", "1", *extra]
			)

	def report(self):
		return json.loads(next((self.campaign / "results").glob("*/report.json")).read_text())

	def test_a_killed_solver_keeps_the_last_bounds_it_reported(self):
		def solver(*_args, progress=None, on_start=None, **_options):
			if on_start:
				on_start(1234)
			progress({"worker": 0, "elapsed_seconds": 90.0, "lower_bound": 5.0, "upper_bound": 6.0,
				"calls": 77, "nodes": 7, "open_nodes": 3, "max_sequence_depth": 2})
			raise unordered_runner.SolverKilled("solver killed by SIGKILL", signal.SIGKILL)

		self.assertEqual(self.run_main(solver), 2)
		report = self.report()
		self.assertEqual(report["status"], "completed_with_errors")
		row = report["rows"][0]
		self.assertEqual(
			(row["status"], row["lower_bound"], row["upper_bound"], row["seconds"], row["partial"]),
			("killed", 5.0, 6.0, 90.0, True),
		)
		self.assertEqual(report["solver_errors"]["tpp-ours"]["first_error"], "solver killed by SIGKILL")
		path = next((self.campaign / "results").glob("*/progress.jsonl"))
		self.assertEqual(len(path.read_text().splitlines()), 2)  # one report for each of the two cases

	def test_the_memory_limit_is_recorded_and_the_instance_is_retried_on_resume(self):
		def solver(*_args, on_memory_limit=None, **_options):
			on_memory_limit(3 * 2**30)
			return {"termination": "interrupted", "status": "interrupted", "lower_bound": 1.0, "upper_bound": 2.0}

		self.assertEqual(self.run_main(solver, "--max-memory-gb", "1"), 2)
		row = self.report()["rows"][0]
		self.assertEqual((row["status"], row["upper_bound"]), ("memory_limit", 2.0))
		self.assertIn("memory limit", row["error"])


class FeketeWorkerTests(unittest.TestCase):
	def args(self, root):
		return SimpleNamespace(worker_python=Path(sys.executable), tspn_repo=root, suite=root / "tiny.bin",
			mode="path", time_limit=3600, threads=1, eps=.001, feasibility_tolerance=.001,
			validation_tolerance=1e-7, oracle_backend="socp", oracle_tolerance=1e-7)

	def spawn(self, root, *, returncode, stdout="", stderr="", checkpoint=None):
		def fake(command, **_kwargs):
			result = Path(command[command.index("--worker-result") + 1])
			if checkpoint is not None:
				result.write_text(json.dumps(checkpoint))
			process = Mock(returncode=returncode, stdout=io.StringIO(stdout), stderr=io.StringIO(stderr), pid=999999)
			process.wait.return_value = returncode
			return process
		return patch.object(tspn_run_comparison.subprocess, "Popen", side_effect=fake)

	def test_gurobi_feasibility_tol_reaches_the_worker_as_a_gurobi_env_file_in_its_directory(self):
		seen = {}

		def fake(command, **kwargs):
			environment_file = Path(kwargs["cwd"]) / "gurobi.env"
			seen["cwd"] = kwargs["cwd"]
			seen["content"] = environment_file.read_text() if environment_file.exists() else None
			process = Mock(returncode=-9, stdout=io.StringIO(""), stderr=io.StringIO(""), pid=999999)
			process.wait.return_value = -9
			return process

		with tempfile.TemporaryDirectory() as directory:
			for tolerance, expected in ((1e-7, "FeasibilityTol 1e-07\n"), (None, None)):
				arguments = self.args(Path(directory))
				arguments.gurobi_feasibility_tol = tolerance
				with patch.object(tspn_run_comparison.subprocess, "Popen", side_effect=fake):
					tspn_run_comparison.run_case(arguments, 0, io.StringIO())
				self.assertEqual(seen["content"], expected)
				if tolerance is None:
					self.assertEqual(Path(seen["cwd"]), tspn_run_comparison.PROJECT_ROOT)

	def test_a_killed_worker_keeps_the_incumbent_checkpoint_it_had_written(self):
		with tempfile.TemporaryDirectory() as directory:
			checkpoint = {"status": "interrupted", "lower_bound": 2713.99, "upper_bound": 2719.56,
				"trajectory": [[0, 0], [1, 1]], "solve_seconds": 48590.0}
			with self.spawn(Path(directory), returncode=-9, checkpoint=checkpoint):
				row = tspn_run_comparison.run_case(self.args(Path(directory)), 0, io.StringIO())
			self.assertEqual(row["status"], "killed")
			self.assertEqual((row["lower_bound"], row["upper_bound"]), (2713.99, 2719.56))
			self.assertEqual(row["trajectory"], [[0, 0], [1, 1]])
			self.assertIn("SIGKILL", row["error"])

	def test_without_a_checkpoint_the_last_trace_line_gives_the_bounds(self):
		with tempfile.TemporaryDirectory() as directory:
			trace = "10\t518.169\t|\tinf\t|\t0.06s\n300\t604\t|\t731.614\t|\t7.276s\n"
			with self.spawn(Path(directory), returncode=-9, stdout=trace):
				row = tspn_run_comparison.run_case(self.args(Path(directory)), 0, io.StringIO())
			self.assertEqual((row["status"], row["lower_bound"], row["upper_bound"]), ("killed", 604.0, 731.614))

	def test_a_stop_request_is_an_interruption_not_a_failure_to_explain(self):
		with tempfile.TemporaryDirectory() as directory:
			with self.spawn(Path(directory), returncode=-signal.SIGTERM, stdout="5\t1\t|\t2\t|\t1s\n"):
				row = tspn_run_comparison.run_case(self.args(Path(directory)), 0, io.StringIO())
			self.assertEqual(row["status"], "interrupted")

	def test_bound_lines_reach_the_progress_callback_as_they_arrive(self):
		with tempfile.TemporaryDirectory() as directory:
			trace = "1\t500\t|\tinf\t|\t0.0s\n2\t510\t|\t900\t|\t0.1s\n"
			seen, pids = [], []
			with self.spawn(Path(directory), returncode=-9, stdout=trace):
				tspn_run_comparison.run_case(
					self.args(Path(directory)), 0, io.StringIO(),
					progress=seen.append, progress_interval=60, on_start=pids.append,
				)
			self.assertEqual(pids, [999999])
			self.assertEqual(len(seen), 1)  # the first line at once, the next within the interval is skipped
			self.assertEqual((seen[0]["lower_bound"], seen[0]["upper_bound"]), (500.0, None))


class StopCommandTests(unittest.TestCase):
	def test_stop_signals_the_chosen_instance_only(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			(root / "run").mkdir()
			status = live.LiveStatus(root / "run" / "live.json", echo=lambda line: None)
			process = subprocess.Popen(
				[sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
			)
			try:
				callback = status.reporter("a", "free case 7/9 tpp-ours")
				status.set_pid("a", process.pid)
				callback({"worker": 0, "elapsed_seconds": 1.0, "calls": 1, "nodes": 1, "open_nodes": 0, "max_sequence_depth": 0})
				status._write(force=True)
				(child,) = live.running_children(root)
				self.assertEqual((child["child_pid"], child["run"]), (process.pid, "run"))
				self.assertTrue(live.stop_child(child))
				self.assertIsNotNone(process.wait(timeout=10))
				self.assertEqual(process.returncode, -signal.SIGINT)
				self.assertFalse(live.stop_child(child))  # already gone
			finally:
				process.kill()


if __name__ == "__main__":
	unittest.main()
