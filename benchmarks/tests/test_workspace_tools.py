from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

INTERNAL = Path(__file__).resolve().parents[1] / "_internal"
sys.path.insert(0, str(INTERNAL))

import jobs
import native_build
import remote
import tpp
import workspace


class NativeBuildTests(unittest.TestCase):
	def test_tool_names_follow_the_cmake_rule(self):
		tools = native_build.available_tools()
		self.assertIn("tpp-unordered", tools)
		self.assertIn("tpp-unordered-tests", tools)
		self.assertIn("tpp-bnb", tools)  # main-tpp_bnb drops the tpp_ prefix
		self.assertIn("tpp-convex-intersection-tests", tools)
		self.assertTrue(native_build.GUROBI_ONLY_TOOLS <= tools.keys())

	def test_every_cmake_package_declares_named_tools(self):
		for package, prefix in native_build.TOOL_PACKAGES.items():
			text = (native_build.ROOT / "packages" / package / "cpp/CMakeLists.txt").read_text()
			self.assertIn("src/main-*.cpp", text, package)
			self.assertIn(prefix.rstrip("-"), text, package)

	def test_relocated_cache_is_detected(self):
		with tempfile.TemporaryDirectory() as directory:
			build = Path(directory)
			(build / "CMakeCache.txt").write_text(
				"CMAKE_HOME_DIRECTORY:INTERNAL=/elsewhere/packages/nonconvex-tpp/cpp\n"
				f"CMAKE_CACHEFILE_DIR:INTERNAL={build.resolve()}\n"
			)
			self.assertFalse(native_build.cache_matches_checkout(build))
			self.assertTrue(native_build.cache_matches_checkout(build / "missing"))

	def test_no_build_reports_missing_tools(self):
		with patch.object(native_build, "BUILD_ROOT", Path("/nonexistent/build")), self.assertRaises(SystemExit):
			native_build.ensure_tools(["tpp-unordered"], no_build=True)

	def test_unknown_tool_is_rejected(self):
		with self.assertRaises(SystemExit):
			native_build.ensure_tools(["tpp-does-not-exist"])


class WorkspaceTests(unittest.TestCase):
	def setUp(self):
		self.temporary = tempfile.TemporaryDirectory()
		self.addCleanup(self.temporary.cleanup)
		self.root = Path(self.temporary.name)
		patcher = patch.dict(os.environ, {"TPP_WORKSPACE": str(self.root / "ws"), "TPP_ORIGIN": "cli"})
		patcher.start()
		self.addCleanup(patcher.stop)

	def test_names_and_paths_resolve_inside_the_workspace(self):
		self.assertEqual(workspace.campaign_path("abc"), self.root.resolve() / "ws/campaigns/abc")
		self.assertEqual(workspace.campaign_path("/tmp/x"), Path("/tmp/x").resolve())
		self.assertEqual(workspace.run_path("gap"), self.root.resolve() / "ws/runs/gap")

	def test_local_data_prefers_campaigns_then_experiments(self):
		(workspace.experiments_dir() / "usp").mkdir(parents=True)
		self.assertEqual(workspace.local_data("usp"), workspace.experiments_dir() / "usp")
		(workspace.campaigns_dir() / "usp").mkdir(parents=True)
		self.assertEqual(workspace.local_data("usp"), workspace.campaigns_dir() / "usp")

	def test_recorded_run_appends_attempts_with_machine_and_status(self):
		directory = self.root / "run"
		with workspace.recorded_run(directory, kind="demo", argv=["x"], parameters={"a": 1}):
			pass
		with self.assertRaises(KeyboardInterrupt), workspace.recorded_run(directory, kind="demo", argv=["x"]):
			raise KeyboardInterrupt
		manifest = json.loads((directory / "run.json").read_text())
		self.assertEqual([attempt["status"] for attempt in manifest["attempts"]], ["completed", "interrupted"])
		self.assertEqual(manifest["status"], "interrupted")
		machine = manifest["attempts"][0]["machine"]
		for key in ("host", "cpus", "cpu_model", "memory_bytes", "system"):
			self.assertIn(key, machine)

	def test_rewrite_paths_only_touches_whole_components(self):
		directory = self.root / "data"
		directory.mkdir()
		path = directory / "index.csv"
		path.write_text('/a/results/run-1/x.csv,"/a/results/run-10",/a/results/run-1\n')
		workspace.rewrite_paths(directory, [("/a/results/run-1", "/b/runs/run-1")])
		self.assertEqual(path.read_text(), '/b/runs/run-1/x.csv,"/a/results/run-10",/b/runs/run-1\n')

	def test_migration_moves_rewrites_and_links(self):
		fake_root = self.root / "repo"
		campaigns = fake_root / "benchmarks/campaigns"
		results = fake_root / "benchmarks/results"
		suite = fake_root / "benchmarks/suites/s.bin"
		suite.parent.mkdir(parents=True)
		suite.write_bytes(b"x")
		campaign = campaigns / "c1"
		(campaign / "results").mkdir(parents=True)
		(campaign / "campaign.json").write_text(json.dumps({"inputs": [{"file": "../../suites/s.bin"}]}))
		(campaign / "results/run-index.csv").write_text(f"{campaign}/results/a.csv\n")
		(campaigns / "notes").mkdir()
		(results / "r1").mkdir(parents=True)
		workspace_root = fake_root / "benchmarks/workspace"
		with patch.multiple(workspace, ROOT=fake_root, LEGACY_CAMPAIGNS=campaigns, LEGACY_RESULTS=results), \
			patch.dict(os.environ, {"TPP_WORKSPACE": str(workspace_root)}):
			workspace.main(["migrate"])
			moved = (workspace_root / "campaigns/c1").resolve()
			self.assertTrue((workspace_root / "experiments/notes").is_dir())
			self.assertTrue((workspace_root / "runs/r1").is_dir())
			data = json.loads((moved / "campaign.json").read_text())
			self.assertEqual((moved / data["inputs"][0]["file"]).resolve(), suite.resolve())
			self.assertEqual((moved / "results/run-index.csv").read_text(), f"{moved}/results/a.csv\n")
			self.assertTrue(campaigns.is_symlink() and results.is_symlink())
			self.assertEqual(campaigns.resolve(), (workspace_root / "campaigns").resolve())


class CliTests(unittest.TestCase):
	def test_every_command_is_listed_once_and_old_names_remain(self):
		names = [name for group in tpp.GROUPS.values() for name in group]
		self.assertEqual(len(names), len(set(names)))
		for legacy in ("create", "run", "status", "free-order", "free-order-run", "compare-gaps",
				"run-fekete", "tspn-benchmark", "split", "list-groups", "run-groups", "generate-matrix"):
			self.assertIn(legacy, tpp.COMMANDS)

	def test_history_records_exit_codes(self):
		with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TPP_WORKSPACE": directory}), \
			patch.dict(tpp.COMMANDS, {"boom": tpp.Command("", "", lambda argv: 3)}):
			self.assertEqual(tpp.main(["boom", "x"]), 3)
			entry = json.loads((Path(directory) / "history.jsonl").read_text().splitlines()[-1])
			self.assertEqual((entry["command"], entry["exit_code"]), (["tpp.py", "boom", "x"], 3))


class JobsAndRemoteTests(unittest.TestCase):
	def test_detached_job_records_exit_code_and_label(self):
		with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TPP_WORKSPACE": directory}):
			script = [sys.executable, str(INTERNAL / "jobs.py"), "start", "--name", "probe", "--", "build", "--list"]
			job_id = subprocess.run(script, capture_output=True, text=True, check=True,
				env=os.environ | {"TPP_WORKSPACE": directory}).stdout.strip()
			self.assertTrue(job_id.endswith("-probe"))
			job = Path(directory) / "jobs" / job_id
			deadline = time.monotonic() + 30
			while jobs.status(jobs._read(job)) == "running" and time.monotonic() < deadline:
				time.sleep(0.2)
			self.assertEqual(jobs.status(jobs._read(job)), "completed")
			self.assertIn("tpp-unordered", (job / "output.log").read_text())

	def test_remote_run_keeps_options_and_quotes_the_command(self):
		scripts = []
		with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TPP_WORKSPACE": directory}), \
			patch.object(remote, "ssh", lambda host, script, **_: scripts.append((host, script)) or
				subprocess.CompletedProcess([], 0)):
			remote.main(["run", "lab", "--dir", "~/tpp dir", "--name", "x", "--", "free-order", "a b", "--threads", "8"])
		host, script = scripts[0]
		self.assertEqual(host, "lab")
		self.assertIn("cd ~/'tpp dir' &&", script)
		self.assertIn("jobs start --name x -- free-order 'a b' --threads 8", script)
		self.assertTrue(re.search(r"TPP_WORKSPACE=~/'tpp dir/benchmarks/workspace'", script))


if __name__ == "__main__":
	unittest.main()
