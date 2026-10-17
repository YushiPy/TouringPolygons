import ast
import json
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import fekete_reference as reference
import fekete_reference_child as child

ROOT = Path(__file__).resolve().parents[2]
PAPER_SCRIPT = ROOT / "third_party/tspn-socg/evaluation/rq4_scalability/01_run_solver.py"


def paper_call_keywords() -> dict:
	"""Keyword arguments of solve_annotated_instance in the paper's experiment script."""
	tree = ast.parse(PAPER_SCRIPT.read_text())
	for node in ast.walk(tree):
		if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "solve_annotated_instance":
			return {keyword.arg: keyword.value for keyword in node.keywords}
	raise AssertionError("solve_annotated_instance call not found")


class ChildCallTests(unittest.TestCase):
	def call(self, **options):
		captured = {}
		operations = types.ModuleType("tspn_bnb2.operations")
		operations.solve_annotated_instance = lambda instance, **keywords: captured.update(keywords) or "solution"
		modules = {"tspn_bnb2": types.ModuleType("tspn_bnb2"), "tspn_bnb2.operations": operations}
		with patch.dict(sys.modules, modules):
			solution, seconds, perf = child.solve("instance", **options)
		self.assertEqual(solution, "solution")
		self.assertGreaterEqual(seconds, 0)
		self.assertGreaterEqual(perf, 0)
		return captured

	def test_the_call_passes_the_configuration(self):
		keywords = self.call(root="OrderRoot", threads=24, time_limit=300, eps=0.001)
		self.assertEqual((keywords["root"], keywords["num_threads"], keywords["timelimit"], keywords["eps"]),
			("OrderRoot", 24, 300, 0.001))

	@unittest.skipUnless(PAPER_SCRIPT.is_file(), "third_party/tspn-socg is not checked out")
	def test_every_other_argument_is_the_paper_scripts(self):
		paper = paper_call_keywords()
		ours = self.call(root="OrderRoot", threads=1, time_limit=300, eps=0.001)
		self.assertEqual(set(ours), set(paper))
		# num_threads is the one argument configuration B changes (A keeps the paper's 24).
		self.assertEqual(ast.literal_eval(paper["num_threads"]), reference.CONFIGURATIONS["A"].threads)
		for name, value in ours.items():
			if name != "num_threads" and isinstance(paper[name], (ast.Constant, ast.List)):
				with self.subTest(name=name):
					self.assertEqual(value, ast.literal_eval(paper[name]))
		# The values that vary come from the configuration, as they do from alg_params there.
		self.assertEqual(ast.unparse(paper["root"]), "alg_params['root']")
		self.assertEqual(ast.unparse(paper["timelimit"]), "time_limit")

	@unittest.skipUnless(PAPER_SCRIPT.is_file(), "third_party/tspn-socg is not checked out")
	def test_the_configurations_are_ones_the_paper_ran(self):
		text = PAPER_SCRIPT.read_text()
		self.assertIn("time_limit = 300", text)
		self.assertIn('"search": "DfsBfs"', text.replace("'", '"'))
		for root in ("OrderRoot", "LongestEdgePlusFurthestSite"):
			self.assertIn(f'"{root}"', text)
		self.assertIn("num_threads=24", text)
		self.assertIn("0.001", text)
		self.assertEqual(reference.CONFIGURATIONS["A"].threads, 24)
		self.assertEqual(reference.EPS, 0.001)
		self.assertEqual(reference.DEFAULT_SECONDS, 300)


SETUP_SCRIPT = ROOT / "third_party/tspn-socg/setup.py"


class DependencyPinTests(unittest.TestCase):
	@unittest.skipUnless(SETUP_SCRIPT.is_file(), "third_party/tspn-socg is not checked out")
	def test_the_pins_cover_every_conan_requirement_within_its_range(self):
		tree = ast.parse(SETUP_SCRIPT.read_text())
		requirements = next(ast.literal_eval(keyword.value) for node in ast.walk(tree) if isinstance(node, ast.Call)
			for keyword in node.keywords if keyword.arg == "conan_requirements")
		version = lambda text: tuple(int(part) for part in text.split("."))
		pins = dict(pin.partition("/")[::2] for pin in reference.PINNED_REQUIRES)
		ranges = dict(requirement.partition("/")[::2] for requirement in requirements)
		self.assertEqual(set(pins), set(ranges))
		for name, wanted in ranges.items():
			with self.subTest(name=name):
				if wanted.startswith("[>="):
					self.assertGreaterEqual(version(pins[name]), version(wanted[3:-1]))
				else:
					self.assertEqual(pins[name], wanted)

	def test_the_lockfile_lists_every_package_once(self):
		lockfile = json.dumps({"version": "0.5", "requires": ["fmt/11.2.0#abc%1", "boost/1.83.0#def%2"],
			"build_requires": ["cmake/4.4.4#x%3"], "python_requires": []})
		self.assertEqual(reference.resolved_packages(lockfile), {"boost": "1.83.0", "cmake": "4.4.4", "fmt": "11.2.0"})


class PlanTests(unittest.TestCase):
	def test_every_repetition_runs_every_instance_with_every_root_once(self):
		instances = [f"case{index}" for index in range(10)]
		roots = ("OrderRoot", "LongestEdgePlusFurthestSite")
		tasks = reference.plan_tasks(instances, roots, 3)
		self.assertEqual(len(tasks), 60)
		self.assertEqual(len({task.key for task in tasks}), 60)
		self.assertEqual(tasks, reference.plan_tasks(instances, roots, 3))
		orders = [[task.instance for task in tasks if task.repetition == repetition][::2] for repetition in range(3)]
		self.assertEqual(len({tuple(order) for order in orders}), 3)
		first = [task.root for task in tasks if task.repetition == 0]
		self.assertEqual(first[:4], [roots[0], roots[1], roots[1], roots[0]])

	def test_the_smoke_test_takes_the_smallest_instances(self):
		sizes = {f"case{index}": 60 - index for index in range(20)}
		self.assertEqual(reference.smoke_instances(sizes), sorted(f"case{index}" for index in range(14, 20)))

	def test_finished_runs_are_kept_and_errors_are_repeated(self):
		rows = [
			{"repetition": 0, "instance": "a", "root": "R", "status": "ok"},
			{"repetition": 0, "instance": "b", "root": "R", "status": "error"},
			{"repetition": 0, "instance": "c", "root": "R", "status": "error"},
			{"repetition": 0, "instance": "c", "root": "R", "status": "killed"},
		]
		self.assertEqual(reference.completed_keys(rows), {(0, "a", "R"), (0, "c", "R")})

	def test_the_summary_counts_by_root_and_repetition(self):
		configuration = reference.CONFIGURATIONS["B"]
		tasks = reference.plan_tasks(["a", "b"], configuration.roots, 2)

		def row(repetition, instance, root, optimal, seconds, status="ok"):
			return {"repetition": repetition, "instance": instance, "root": root, "status": status,
				"solve_time": seconds, "solution": {"is_optimal": optimal}}

		rows = [row(r, "a", root, True, 1.0 + r) for r in range(2) for root in configuration.roots]
		rows += [row(0, "b", "OrderRoot", False, 300), row(1, "b", "OrderRoot", True, 250)]
		rows += [row(0, "b", "LongestEdgePlusFurthestSite", False, 0, status="error")]
		summary = reference.summarize(configuration, rows, tasks)
		chr_ = summary["roots"]["CHR"]
		self.assertEqual(chr_["repetitions"][1], {"runs": 2, "solved": 1, "not_solved_in_time": 1, "killed": 0,
			"errors": 0, "valid_tours": 0, "solved_seconds_sum": 1.0})
		self.assertEqual((chr_["instances_solved_in_every_repetition"], chr_["instances_solved_in_some_repetition"]), (1, 2))
		self.assertEqual(chr_["median_of_instance_medians_seconds"], 1.5)
		self.assertEqual(summary["roots"]["LEFP"]["repetitions"][1]["errors"], 1)
		self.assertIn("| CHR | 2 | 2 | 2 |", reference.render(configuration, 300, summary, {}))


class JudgeTests(unittest.TestCase):
	def test_the_two_stopping_rules_differ_near_the_tolerance(self):
		self.assertEqual(reference.criteria(1000, 1000.9), {"fekete": True, "strict": True})
		self.assertEqual(reference.criteria(1000, 1001.002), {"fekete": True, "strict": False})
		self.assertEqual(reference.criteria(1000, 1002), {"fekete": False, "strict": False})
		self.assertEqual(reference.criteria(None, float("inf")), {"fekete": False, "strict": False})

	def test_fekete_tours_are_read_from_wkt(self):
		self.assertEqual(reference.trajectory_points("LINESTRING (0 0, 1.5 2, 0 0)"), [(0.0, 0.0), (1.5, 2.0), (0.0, 0.0)])
		self.assertEqual(reference.trajectory_points("LINESTRING EMPTY"), [])

	def test_our_runs_have_their_own_directory_per_commit(self):
		with patch.object(reference, "our_revision", return_value="abc1234"):
			directory = reference.run_directory(reference.CONFIGURATIONS["C"], 300, False)
		self.assertEqual(directory.name, "C-ours-abc1234-300s")
		self.assertEqual(reference.run_directory(reference.CONFIGURATIONS["B"], 300, False).name, "B-single-thread-300s")


class LockTests(unittest.TestCase):
	def test_a_lock_of_a_dead_process_is_taken_over_and_a_live_one_is_not(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "x.lock"
			mine = {"pid": os.getpid(), "host": reference.platform.node()}
			path.write_text(json.dumps({"pid": 2**22 + 12345, "host": reference.platform.node()}))
			with patch.object(reference, "alive", side_effect=lambda pid: pid == os.getpid()):
				self.assertTrue(reference.try_lock(path, mine))
				self.assertFalse(reference.try_lock(path, mine | {"pid": 7}))
				reference.release_lock(path)
			self.assertFalse(path.exists())


FAKE_CHILD = textwrap.dedent("""
	import argparse, json, sys, time
	parser = argparse.ArgumentParser()
	for name in ("--archive", "--instance", "--root", "--threads", "--time-limit", "--eps", "--output"):
		parser.add_argument(name)
	args = parser.parse_args()
	if args.instance == "slow":
		time.sleep(60)
	print("solver output")
	ok = args.instance != "broken"
	result = {"ok": ok, "solve_time": 0.5, "solve_time_perf": 0.5,
		"solution": {"is_optimal": True, "gap": 0.0, "upper_bound": 1.0, "threads": args.threads}} if ok else {
		"ok": False, "error": "ValueError: no solution"}
	open(args.output, "w").write(json.dumps(result))
""")


class RunnerTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		base = Path(self.directory.name)
		(base / "venv/bin").mkdir(parents=True)
		(base / "venv/bin/python").symlink_to(sys.executable)
		(base / "child.py").write_text(FAKE_CHILD)
		self.output = base / "run"
		self.output.mkdir()
		for name, value in (("VENV", base / "venv"), ("CHILD", base / "child.py"), ("SOURCE", base)):
			patcher = patch.object(reference, name, value)
			patcher.start()
			self.addCleanup(patcher.stop)

	def runner(self):
		configuration = reference.CONFIGURATIONS["A"]
		sizes = {"fine": 5, "slow": 6, "broken": 7}
		return reference.Runner(configuration, 1, 1, self.output, sizes, None, total=3, done=0)

	def test_rows_record_results_errors_and_runs_killed_after_the_hard_limit(self):
		runner = self.runner()
		with patch.object(reference.Runner, "hard_limit", return_value=2.0), patch("builtins.print"):
			reference.work(runner, [reference.Task(0, name, "OrderRoot") for name in ("fine", "slow", "broken")])
		rows = {row["instance"]: row for row in reference.read_rows(self.output / "runs.jsonl")}
		self.assertEqual(rows["fine"]["status"], "ok")
		self.assertEqual(rows["fine"]["solution"]["threads"], "24")
		self.assertEqual((rows["fine"]["polygons"], rows["fine"]["root_label"]), (5, "CHR"))
		self.assertGreater(rows["fine"]["process"]["wall_seconds"], 0)
		self.assertEqual(rows["slow"]["status"], "killed")
		self.assertLess(rows["slow"]["process"]["wall_seconds"], 30)
		self.assertEqual(rows["broken"]["status"], "error")
		self.assertIn("ValueError", rows["broken"]["error"])
		self.assertIn("solver output", rows["broken"]["log_tail"])
		self.assertEqual(reference.completed_keys(rows.values()), {(0, "fine", "OrderRoot"), (0, "slow", "OrderRoot")})


FAKE_OURS = textwrap.dedent("""
	import json, sys
	header, *rings = sys.stdin.read().split("\\n")
	size = int(header.split()[4])
	closed = size == 2 and "--cycle" in sys.argv
	lower = 6.0 if closed else 5.0
	print(json.dumps({"lower_bound": lower, "upper_bound": 6.0, "seconds": 0.25, "path": [[0.5, 0.5], [3.5, 0.5], [0.5, 0.5]],
		"arguments": sys.argv[1:]}))
""")


class OurRunnerTests(unittest.TestCase):
	def test_our_rows_carry_the_result_the_verdict_and_the_validation(self):
		with tempfile.TemporaryDirectory() as directory:
			base = Path(directory)
			binary = base / "tpp-unordered"
			binary.write_text(f"#!{sys.executable}\n" + FAKE_OURS)
			binary.chmod(0o755)
			square = lambda x: [(x, 0.0), (x + 1, 0.0), (x + 1, 1.0), (x, 1.0)]
			polygons = {"two": [square(0), square(3)], "three": [square(0), square(3), square(6)]}
			runner = reference.Runner(reference.CONFIGURATIONS["C"], 300, 2, base, {"two": 2, "three": 3}, None,
				total=2, done=0, polygons=polygons, binary=binary)
			with patch("builtins.print"):
				reference.work(runner, [reference.Task(0, name, "ours") for name in ("two", "three")])
			rows = {row["instance"]: row for row in reference.read_rows(base / "runs.jsonl")}
		two, three = rows["two"], rows["three"]
		self.assertEqual((two["status"], two["solve_time"], two["solved"], two["root_label"]), ("ok", 0.25, True, "nosso"))
		self.assertEqual(two["result"]["arguments"], list(reference.OUR_ARGUMENTS))
		self.assertTrue(two["validation"]["valid"])
		self.assertEqual((three["solved"], three["criteria"]), (False, {"fekete": False, "strict": False}))
		self.assertFalse(three["validation"]["valid"])  # the tour misses the third square


class CommandTests(unittest.TestCase):
	def test_plan_only_prints_the_plan(self):
		sizes = {f"case{index}": 10 for index in range(reference.EXPECTED_INSTANCES)}
		with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TPP_WORKSPACE": directory}), \
				patch.object(reference, "read_setup", return_value={"fingerprint": "x"}), \
				patch.object(reference, "load_instances", return_value=sizes), \
				patch.object(reference, "prepare") as prepare, patch("builtins.print") as printed:
			self.assertEqual(reference.single_thread_main(["600", "--plan", "--workers", "8"]), 0)
		prepare.assert_not_called()
		text = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
		self.assertIn(f"{reference.EXPECTED_INSTANCES * 2 * 3} execuções", text)
		self.assertIn("8 por vez, limite de 600 s", text)
		self.assertIn("B-single-thread-600s", text)


if __name__ == "__main__":
	unittest.main()
