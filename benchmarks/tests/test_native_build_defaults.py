import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import native_build


class DefaultBuildTests(unittest.TestCase):
	def run_main(self, argv, fail=()):
		calls = []

		def fake(tools, **_options):
			calls.append(list(tools))
			if list(tools) == list(fail):
				raise subprocess.CalledProcessError(2, "cmake")
			return [Path("/bin") / tools[0]]

		with patch.object(native_build, "ensure_tools", side_effect=fake), patch.object(
			native_build, "fetch_header_dependencies"
		) as fetch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
			code = native_build.main(argv)
		return code, calls, fetch.called, errors.getvalue()

	def test_the_fixed_order_benchmark_is_optional_in_the_default_set(self):
		code, calls, _, errors = self.run_main([], fail=["tpp-bnb-workload-benchmark"])
		self.assertEqual((code, calls), (0, [["tpp-unordered"], ["tpp-bnb-workload-benchmark"]]))
		self.assertIn("Skipped tpp-bnb-workload-benchmark", errors)

	def test_the_free_order_solver_failing_is_still_an_error(self):
		with self.assertRaises(subprocess.CalledProcessError):
			self.run_main([], fail=["tpp-unordered"])

	def test_fetch_deps_alone_only_downloads_headers(self):
		self.assertEqual(self.run_main(["--fetch-deps"])[:3], (0, [], True))

	def test_named_tools_are_built_as_asked_even_after_fetching(self):
		code, calls, fetched, _ = self.run_main(["--fetch-deps", "tpp-unordered"])
		self.assertEqual((code, calls, fetched), (0, [["tpp-unordered"]], True))


class CompilerProbeTests(unittest.TestCase):
	"""GCC 13 has <format> but not <print>: only the tools that include <print> need GCC 14+."""

	def select(self, compilers, *, require_print):
		# compilers: CXX -> whether its library has <print> (all have <format>).
		def probe(_cc, cxx, _standard, *, with_print=True):
			return None if compilers[cxx] or not with_print else "no <print>"

		with patch.object(native_build, "compiler_candidates", return_value=[("cc", cxx) for cxx in compilers]), \
				patch.object(native_build, "probe_compiler", side_effect=probe), patch.dict("os.environ", {}, clear=False), \
				contextlib.redirect_stdout(io.StringIO()):
			native_build.os.environ.pop("TPP_CXX_STANDARD", None)
			return native_build.select_toolchain(require_print=require_print)

	def test_the_paper_tools_do_not_need_print(self):
		tools = native_build.available_tools()
		for tool in ("tpp-convex-path-benchmark", "tpp-convex-cycle-sequences", "tpp-unordered"):
			self.assertFalse(native_build.needs_print(tools[tool]), tool)
		self.assertTrue(native_build.needs_print(tools["tpp-convex-intersection-tests"]))

	def test_a_compiler_with_print_is_preferred_for_every_tool(self):
		chosen = self.select({"g++-13": False, "g++-14": True}, require_print=False)
		self.assertEqual((chosen["CXX"], chosen.get("print", True)), ("g++-14", True))

	def test_format_only_compiler_builds_tools_without_print(self):
		chosen = self.select({"g++-13": False}, require_print=False)
		self.assertEqual((chosen["CXX"], chosen["print"]), ("g++-13", False))

	def test_format_only_compiler_is_refused_for_print_tools(self):
		with self.assertRaises(SystemExit):
			self.select({"g++-13": False}, require_print=True)


class NewToolReconfigureTests(unittest.TestCase):
	"""A build directory configured before src/main-foo.cpp existed has no
	target for it, and a Makefile build fails before CMake re-globs."""

	def build(self, configured_tools):
		import json
		import tempfile
		calls = []
		with tempfile.TemporaryDirectory() as temporary:
			directory = Path(temporary) / "tools"
			(directory / "bin").mkdir(parents=True)
			(directory / "CMakeCache.txt").write_text("")
			(directory / "bin" / "tpp-unordered").write_text("")
			toolchain = {"CC": "cc", "CXX": "c++", "TPP_CXX_STANDARD": "23"}
			previous = native_build.configuration(gurobi=False, toolchain=toolchain) | {"tools": configured_tools}
			(directory / native_build.CONFIG_FILE).write_text(json.dumps(previous))
			with patch.object(native_build, "build_directory", return_value=directory), \
					patch.object(native_build, "cache_matches_checkout", return_value=True), \
					patch.object(native_build.subprocess, "run", side_effect=lambda command, **_: calls.append(command)), \
					contextlib.redirect_stdout(io.StringIO()):
				native_build.ensure_tools(["tpp-unordered"])
			written = json.loads((directory / native_build.CONFIG_FILE).read_text())
		return [command[1] for command in calls], written["tools"]

	def test_a_new_tool_source_reconfigures_before_building(self):
		tools = sorted(native_build.available_tools())
		steps, written = self.build([tool for tool in tools if tool != "tpp-unordered"])
		self.assertEqual((steps, written), (["-S", "--build"], tools))

	def test_a_directory_that_knows_every_tool_is_not_reconfigured(self):
		steps, _ = self.build(sorted(native_build.available_tools()))
		self.assertEqual(steps, ["--build"])


if __name__ == "__main__":
	unittest.main()
