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


if __name__ == "__main__":
	unittest.main()
