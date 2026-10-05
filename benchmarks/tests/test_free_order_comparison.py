import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import free_order_comparison as comparison


class FakeProcess:
	def __init__(self, stdout="", returncode=0):
		self.stdout, self.returncode = stdout, returncode


class CampaignTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		patcher = patch.dict(os.environ, {"TPP_WORKSPACE": self.directory.name})
		patcher.start()
		self.addCleanup(patcher.stop)

	def test_the_campaign_points_at_the_pinned_suite_and_is_reused(self):
		campaign = comparison.ensure_campaign("c")
		metadata = json.loads((campaign / "campaign.json").read_text())
		self.assertEqual(metadata["source"]["sha256"], comparison.EXPECTED_SUITE_SHA256)
		self.assertEqual((campaign / metadata["inputs"][0]["file"]).resolve(), comparison.SUITE.resolve())
		self.assertEqual(comparison.ensure_campaign("c"), campaign)

	def test_a_foreign_campaign_or_directory_is_never_overwritten(self):
		campaign = comparison.ensure_campaign("c")
		(campaign / "campaign.json").write_text(json.dumps({"inputs": [{"file": "other.bin"}]}))
		with self.assertRaises(SystemExit):
			comparison.ensure_campaign("c")
		stray = Path(self.directory.name) / "campaigns" / "stray"
		stray.mkdir(parents=True)
		(stray / "notes.txt").write_text("keep me")
		with self.assertRaises(SystemExit):
			comparison.ensure_campaign("stray")
		self.assertEqual((stray / "notes.txt").read_text(), "keep me")


class SetupFlowTests(unittest.TestCase):
	def run_main(self, *argv):
		calls = []
		with patch.object(comparison, "verify_suite"), patch.object(
			comparison, "prepare_submodule", side_effect=lambda: calls.append("submodule") or "abc"
		), patch.object(comparison, "build_fekete", side_effect=lambda *_: calls.append("fekete")), patch.object(
			comparison, "use_conan_packages", side_effect=lambda *_: calls.append("conan")
		), patch.object(comparison, "build_ours", side_effect=lambda: calls.append("ours")):
			code = comparison.main(["--setup-only", *argv])
		return code, calls

	def test_ours_only_does_not_touch_fekete(self):
		self.assertEqual(self.run_main("--solver", "tpp-ours"), (0, ["conan", "ours"]))

	def test_fekete_and_both_prepare_the_submodule_and_the_binding_first(self):
		self.assertEqual(self.run_main("--solver", "tpp-fekete"), (0, ["submodule", "fekete", "conan"]))
		self.assertEqual(self.run_main(), (0, ["submodule", "fekete", "conan", "ours"]))

	def test_the_campaign_name_follows_the_thread_count_unless_given(self):
		captured = []
		with patch.object(comparison, "verify_suite"), patch.object(comparison, "use_conan_packages"), patch.object(
			comparison, "build_ours"
		), patch.object(comparison, "ensure_campaign", side_effect=lambda name: captured.append(name) or Path("/x")), patch.object(
			sys.modules.setdefault("free_order_campaign", __import__("free_order_campaign")), "main", return_value=0
		) as run:
			comparison.main(["--solver", "tpp-ours", "--threads-per-instance", "8"])
			comparison.main(["--solver", "tpp-ours", "--campaign", "mine"])
			comparison.main(["--solver", "tpp-ours"])
		self.assertEqual(captured, ["fekete-free-order-comparison-8threads", "mine", "fekete-free-order-comparison-v1"])
		first = run.call_args_list[0].args[0]
		self.assertEqual(first[first.index("--threads-per-instance") + 1], "8")
		self.assertIn("tpp-ours", first)
		self.assertNotIn("tpp-fekete", first)

	def test_bad_options_are_rejected(self):
		for bad in (["--workers", "0"], ["--max-seconds", "0"], ["--campaign", "a/b"], ["--solver", "x"]):
			with self.assertRaises(SystemExit), patch("sys.stderr"):
				comparison.main(bad)


class SubmoduleTests(unittest.TestCase):
	def test_only_the_managed_patches_may_modify_the_submodule(self):
		def fake_git(*arguments, cwd=comparison.ROOT, check=False):
			if arguments[:2] == ("apply", "--reverse"):
				return FakeProcess(returncode=0)
			return FakeProcess()

		outputs = {"rev-parse": "abc", "status": " M README.md", "ls-tree": "160000 commit abc\tthird_party/tspn-socg"}

		def fake_output(*arguments, cwd=comparison.ROOT):
			return outputs[arguments[0]]

		with tempfile.TemporaryDirectory() as directory:
			source = Path(directory)
			(source / ".git").write_text("gitdir: x")
			with patch.object(comparison, "EXTERNAL_SOURCE", source), patch.object(
				comparison, "git", side_effect=fake_git
			), patch.object(comparison, "git_output", side_effect=fake_output):
				with self.assertRaises(SystemExit), patch("sys.stderr"):
					comparison.prepare_submodule()  # a stray modification: stop and preserve it
				outputs["status"] = comparison.PATCHED_STATUS
				self.assertEqual(comparison.prepare_submodule(), "abc")


class BuildTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		self.source = Path(self.directory.name)
		bin_dir = self.source / ".venv/bin"
		bin_dir.mkdir(parents=True)
		for name in ("python", "conan", "cmake", "ninja"):
			(bin_dir / name).write_text("#!/bin/sh\n")
			(bin_dir / name).chmod(0o755)
		self.binding_dir = self.source / "python/tspn_bnb2/core"
		self.binding_dir.mkdir(parents=True)
		(self.binding_dir / "_tspn_bindings.cpython-312.so").write_bytes(b"x")
		self.marker = self.source / ".venv/.touring-polygons-fekete-build-fingerprint"
		(self.source / ".venv/.touring-polygons-editable-installed").write_text("")
		self.commands = []

	def fake_run(self, command, **_options):
		command = [str(part) for part in command]
		self.commands.append(command)
		text = " ".join(command)
		if "fekete_fingerprint.py" in text:
			return FakeProcess("FINGERPRINT\n")
		if "--version" in command and "cmake" in command[0]:
			return FakeProcess("cmake version 3.31\nextra\n")
		return FakeProcess("ok\n")

	def run_build(self):
		with patch.object(comparison, "EXTERNAL_SOURCE", self.source), patch.object(
			comparison.subprocess, "run", side_effect=self.fake_run
		), patch.object(comparison, "_build_with_retries") as build, patch.dict(os.environ):
			comparison.build_fekete("rev", 4)
		return build

	def test_an_up_to_date_build_is_not_repeated(self):
		self.marker.write_text("FINGERPRINT\n")
		build = self.run_build()
		build.assert_not_called()
		self.assertEqual(self.marker.read_text(), "FINGERPRINT\n")
		self.assertTrue(any("import importlib.util" in " ".join(command) for command in self.commands))  # binding verified
		self.assertTrue(any("gurobipy" in " ".join(command) for command in self.commands))  # license checked

	def test_a_changed_fingerprint_rebuilds_and_records_the_new_one(self):
		self.marker.write_text("OLD\n")
		build = self.run_build()
		build.assert_called_once()
		self.assertEqual(self.marker.read_text(), "FINGERPRINT\n")

	def test_a_missing_binding_after_the_build_is_an_error(self):
		(self.binding_dir / "_tspn_bindings.cpython-312.so").unlink()
		with self.assertRaises(SystemExit), patch("sys.stderr"):
			self.run_build()

	def test_transient_download_failures_are_retried_and_others_are_not(self):
		class Process:
			def __init__(self, lines, code):
				self.stdout, self._code = iter(lines), code

			def wait(self):
				return self._code

		sequences = [Process(["too many 503 error responses\n"], 1), Process(["fine\n"], 0)]
		with patch.object(comparison.subprocess, "Popen", side_effect=sequences), patch.object(comparison.time, "sleep"):
			comparison._build_with_retries(Path("python"))
		with patch.object(comparison.subprocess, "Popen", side_effect=[Process(["compile error\n"], 2)]), patch.object(
			comparison.time, "sleep"
		) as sleep:
			with self.assertRaises(SystemExit) as caught:
				comparison._build_with_retries(Path("python"))
		self.assertEqual(caught.exception.code, 2)
		sleep.assert_not_called()


if __name__ == "__main__":
	unittest.main()
