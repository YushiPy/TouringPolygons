from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

INTERNAL = Path(__file__).resolve().parents[1] / "_internal"
sys.path.insert(0, str(INTERNAL))

import benchmark_environment as environment
import tspn_oracle_backends
import tspn_run_comparison


class BenchmarkEnvironmentTests(unittest.TestCase):
	def setUp(self):
		self.temporary = tempfile.TemporaryDirectory()
		self.addCleanup(self.temporary.cleanup)
		self.root = Path(self.temporary.name)
		self.project = self.root / "benchmarks"
		self.project.mkdir()
		self.venv = self.project / ".venv"
		self.ready = self.venv / ".tpp-environment-ready"
		for name in ("pyproject.toml", "uv.lock", ".python-version"):
			(self.project / name).write_text(name)
		for name, value in (
			("ROOT", self.root),
			("PROJECT", self.project),
			("VENV", self.venv),
			("READY", self.ready),
		):
			patcher = patch.object(environment, name, value)
			patcher.start()
			self.addCleanup(patcher.stop)

	def prepare(self):
		python = environment.environment_python()
		python.parent.mkdir(parents=True)
		python.symlink_to(sys.executable)
		self.ready.write_text(environment.environment_fingerprint())
		return python

	def test_help_and_setup_work_without_an_environment(self):
		with patch.object(environment.os, "execve") as execute:
			for argv in (
				[],
				["--help"],
				["setup"],
				["setup", "--offline"],
				["free-order", "x", "--help"],
			):
				environment.enter_environment(argv)
			execute.assert_not_called()

	def test_missing_and_stale_environments_require_setup(self):
		with self.assertRaisesRegex(SystemExit, "python3 benchmarks/tpp.py setup"):
			environment.enter_environment(["status", "campaign"])
		self.prepare()
		(self.project / "uv.lock").write_text("updated dependencies")
		with self.assertRaisesRegex(SystemExit, "python3 benchmarks/tpp.py setup"):
			environment.enter_environment(["status", "campaign"])

	def test_public_cli_preserves_venv_entry_point_arguments_and_environment(self):
		python = self.prepare()
		argv = ["status", "campaign with spaces"]
		with (
			patch.object(environment.sys, "prefix", "/unrelated/.venv"),
			patch.dict(
				os.environ, {"PATH": "/usr/bin", "VIRTUAL_ENV": "/unrelated/.venv"}
			),
			patch.object(environment.os, "execve") as execute,
		):
			environment.enter_environment(argv)
			entry, command, child_environment = execute.call_args.args
		self.assertEqual(entry, str(python))
		self.assertNotEqual(entry, str(python.resolve()))
		self.assertEqual(command, [str(python), str(self.project / "tpp.py"), *argv])
		self.assertEqual(
			child_environment["PATH"], str(python.parent) + os.pathsep + "/usr/bin"
		)
		self.assertEqual(child_environment["VIRTUAL_ENV"], str(self.venv))

	def test_running_inside_own_environment_does_not_reexecute(self):
		python = self.prepare()
		with (
			patch.object(environment.sys, "prefix", str(self.venv)),
			patch.dict(os.environ, {"PATH": "/usr/bin"}),
			patch.object(environment.os, "execve") as execute,
		):
			environment.enter_environment(["status", "campaign"])
			execute.assert_not_called()
			self.assertTrue(
				os.environ["PATH"].startswith(str(python.parent) + os.pathsep)
			)

	def test_setup_uses_own_locked_environment_and_checks_imports(self):
		self.prepare()
		with (
			patch.object(environment.shutil, "which", return_value="/tools/uv"),
			patch.dict(os.environ, {"UV_PROJECT_ENVIRONMENT": "/unrelated/.venv"}),
			patch.object(
				environment.subprocess,
				"run",
				return_value=SimpleNamespace(returncode=0),
			) as run,
			contextlib.redirect_stdout(io.StringIO()),
		):
			self.assertEqual(environment.main(["--offline", "--python", "3.12"]), 0)
		self.assertEqual(run.call_count, 2)
		sync, check = run.call_args_list
		self.assertEqual(
			sync.args[0],
			[
				"/tools/uv",
				"sync",
				"--project",
				str(self.project),
				"--locked",
				"--python",
				"3.12",
				"--offline",
			],
		)
		self.assertEqual(sync.kwargs["cwd"], self.root)
		self.assertEqual(sync.kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(self.venv))
		self.assertEqual(
			check.args[0][:2], [str(environment.environment_python()), "-c"]
		)
		self.assertIn("import shapely", check.args[0][2])
		self.assertTrue(environment.environment_ready())

	def test_failed_sync_or_import_leaves_environment_unready(self):
		self.prepare()
		for failures in (
			[SimpleNamespace(returncode=9)],
			[SimpleNamespace(returncode=0), SimpleNamespace(returncode=7)],
		):
			self.ready.write_text(environment.environment_fingerprint())
			with (
				patch.object(environment.shutil, "which", return_value="/tools/uv"),
				patch.object(
					environment.subprocess, "run", side_effect=failures
				) as run,
			):
				self.assertEqual(environment.main([]), failures[-1].returncode)
				self.assertEqual(run.call_count, len(failures))
			self.assertFalse(environment.environment_ready())

	def test_missing_uv_gives_bootstrap_instruction(self):
		with (
			patch.object(environment.shutil, "which", return_value=None),
			patch.object(environment.subprocess, "run") as run,
			contextlib.redirect_stderr(io.StringIO()) as message,
		):
			self.assertEqual(environment.main([]), 2)
			run.assert_not_called()
		self.assertIn("install_dependencies.sh", message.getvalue())

	def test_oracle_binding_runs_in_external_python_without_loading_in_parent(self):
		python = self.prepare()
		argv = [
			"--suite",
			"unused.bin",
			"--tspn-repo",
			str(self.root),
			"--output",
			"unused.jsonl",
			"--external-python",
			str(python),
		]
		with (
			patch.object(
				tspn_oracle_backends.subprocess, "call", return_value=17
			) as child,
			patch.object(tspn_oracle_backends, "load_core") as load,
		):
			self.assertEqual(tspn_oracle_backends.main(argv), 17)
			load.assert_not_called()
		command = child.call_args.args[0]
		self.assertEqual(command[0], str(python))
		self.assertEqual(command[2:], ["--worker", *argv])

	def test_external_workers_default_to_repository_venv_not_parent_python(self):
		repository = self.root / "external"
		python = environment.environment_python(repository / ".venv")
		python.parent.mkdir(parents=True)
		python.symlink_to(sys.executable)
		args = SimpleNamespace(
			tspn_repo=repository,
			suite=self.root / "unused.bin",
			mode="path",
			time_limit=1,
			threads=1,
			eps=0.001,
			feasibility_tolerance=0.001,
			validation_tolerance=1e-7,
			oracle_backend="socp",
			oracle_tolerance=1e-7,
		)

		def fake_spawn(command, **_kwargs):
			self.assertEqual(command[0], str(python))
			output = Path(command[command.index("--worker-result") + 1])
			output.write_text('{"status": "optimal"}')
			process = Mock(returncode=0, stdout=io.StringIO(""), stderr=io.StringIO(""))
			process.wait.return_value = 0
			return process

		with (
			patch.object(
				tspn_run_comparison.subprocess, "Popen", side_effect=fake_spawn
			),
			patch.object(sys, "prefix", str(self.venv)),
		):
			result = tspn_run_comparison.run_case(args, 0, io.StringIO())
		self.assertEqual(result["status"], "optimal")


if __name__ == "__main__":
	unittest.main()
