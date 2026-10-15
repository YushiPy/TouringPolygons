import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import native_build
import paula_campaign as campaign

# Synthetic rows in the layout of the draft's appendix table (no third-party values).
TABLE = r"""
\multicolumn{1}{|c|}{} & 11eil51 & 11 & 51 & 16 & 61 & 126 & 475 & 1,199 & 110 \\
7demo1             &   7 &       1,000.00 &         990.00 & 10 &     3.20 &    -1.000 &    -0.900 &    -0.500 \\
100pcb1173-10x10   & 100 &      10,000.00 &       9,950.00 &  4 &   900.00 &    -0.500 &    -0.499 &    +0.100 \\
"""


def ils_row(name, seed, length, history=(), valid=True, complete=True, lower=None):
	return {"phase": "ils", "name": name, "seed": seed, "m": 7, "length": length, "valid": valid,
		"complete": complete, "lower_bound": lower, "history": [list(item) for item in history]}


class ReferenceTests(unittest.TestCase):
	def test_reads_only_result_rows_and_maps_the_known_typo(self):
		rows = campaign.parse_reference(TABLE)
		self.assertEqual(set(rows), {"7demo1", "100prb1173-10x10"})
		demo = rows["7demo1"]
		self.assertEqual((demo["m"], demo["gtsp"], demo["best"], demo["runs_le_gtsp"]), (7, 1000.0, 990.0, 10))
		self.assertEqual(demo["gap_percent"], [-1.0, -0.9, -0.5])
		self.assertAlmostEqual(demo["mean"], 991.0)
		self.assertAlmostEqual(demo["worst"], 995.0)

	def test_rounded_mean_is_never_below_the_printed_best(self):
		row = campaign.parse_reference("9x & 9 & 100.00 & 99.47 & 10 & 1.00 & -0.531 & -0.531 & -0.531 \\\\")["9x"]
		self.assertEqual(row["mean"], 99.47)
		self.assertEqual(row["worst"], 99.47)

	def test_lengths_equal_within_her_rounding(self):
		self.assertEqual(campaign.compare(990.004, 990.0), "equal")
		self.assertEqual(campaign.compare(989.99, 990.0), "better")
		self.assertEqual(campaign.compare(990.02, 990.0), "worse")


class SelectionTests(unittest.TestCase):
	def test_open_all_and_lists(self):
		names = [*campaign.OPEN_INSTANCES, "8i600z"]
		self.assertEqual(campaign.select("open", names), list(campaign.OPEN_INSTANCES))
		self.assertEqual(campaign.select("all", names), sorted(names))
		self.assertEqual(campaign.select("8i600z, 50eil101", names), ["8i600z", "50eil101"])
		for value in ("nope", "8i600z,8i600z"):
			with self.subTest(value=value), self.assertRaises(SystemExit):
				campaign.select(value, names)


class RowTests(unittest.TestCase):
	def test_best_tour_needs_a_valid_tour_and_lower_bound_takes_any_run(self):
		rows = [ils_row("a", 0, 5.0, valid=False, lower=1.0), ils_row("a", 1, 6.0, lower=4.0),
			{"phase": "bnb", "name": "a", "seed": 0, "length": 6.0, "valid": True, "lower_bound": 5.5}]
		self.assertEqual(campaign.best_tour(rows, "a")["seed"], 1)
		self.assertEqual(campaign.lower_bound(rows, "a"), 5.5)
		self.assertIsNone(campaign.best_tour(rows, "b"))

	def test_only_complete_runs_count_as_done(self):
		rows = [ils_row("a", 0, 5.0), ils_row("a", 1, 5.0, complete=False)]
		self.assertEqual(campaign.completed(rows), {("ils", "a", 0)})

	def test_time_to_target_uses_the_band(self):
		row = ils_row("a", 0, 990.0, history=[(0.5, 1000.0), (2.0, 990.004), (9.0, 990.0)])
		self.assertEqual(campaign.time_to(row, 990.0), 2.0)
		self.assertIsNone(campaign.time_to(row, 900.0))

	def test_truncated_last_line_is_skipped(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "runs.jsonl"
			path.write_text(json.dumps(ils_row("a", 0, 5.0)) + "\n{\"phase\": \"il")
			self.assertEqual(len(campaign.read_rows(path)), 1)


class ReportTests(unittest.TestCase):
	def test_summary_compares_best_and_mean_with_hers(self):
		reference = campaign.parse_reference(TABLE)
		rows = [ils_row("7demo1", 0, 990.0, history=[(1.0, 990.0)], lower=900.0),
			ils_row("7demo1", 1, 993.0, history=[(1.0, 993.0)], lower=950.0)]
		protocol = campaign.protocol_of(campaign.build_parser().parse_args([]))
		report = campaign.summarize(rows, ["7demo1"], reference, protocol)
		item = report["instances"][0]
		self.assertEqual((item["vs_paula_best"], item["seeds_reaching_paula_best"], item["ils_valid_runs"]), ("equal", 1, 2))
		self.assertEqual(item["mean_vs_paula_mean"], "worse")
		self.assertAlmostEqual(item["gap"], 1 - 950.0 / 990.0)
		self.assertFalse(item["certified"])
		self.assertEqual(report["summary"]["vs_paula_best"], {"better": 0, "equal": 1, "worse": 0})
		text = campaign.markdown(report, ["host"], True)
		self.assertIn("| 7demo1 | 7 | 2 | 990.00 | 990.00 | +0.0000% | 1/2 | 1.0 |", text)

	def test_closed_gap_is_a_certified_optimum(self):
		rows = [ils_row("x", 0, 10.0, lower=10.0 * (1 - 1e-7))]
		report = campaign.summarize(rows, ["x"], {}, {})
		self.assertTrue(report["instances"][0]["certified"])
		self.assertEqual(report["summary"]["certified"], 1)


class ProtocolTests(unittest.TestCase):
	def test_only_the_seeds_and_the_bnb_limit_may_change(self):
		parser = campaign.build_parser()
		with tempfile.TemporaryDirectory() as directory:
			output = Path(directory)
			first = campaign.protocol_of(parser.parse_args([]))
			(output / campaign.PROTOCOL_FILE).write_text(json.dumps(first))
			campaign.check_protocol(output, campaign.protocol_of(parser.parse_args(["--seeds", "20"])))
			campaign.check_protocol(output, campaign.protocol_of(parser.parse_args(["--bnb-seconds", "86400"])))
			with self.assertRaises(SystemExit):
				campaign.check_protocol(output, campaign.protocol_of(parser.parse_args(["--ils-seconds", "60"])))
			with self.assertRaises(SystemExit):
				campaign.check_protocol(output, campaign.protocol_of(parser.parse_args(["--bnb-args", "--dive-interval 1"])))


class SolverArgumentTests(unittest.TestCase):
	def test_bnb_runs_get_threads_and_a_checkpoint(self):
		protocol = campaign.protocol_of(campaign.build_parser().parse_args(["--bnb-threads", "8", "--phases", "bnb"]))
		bnb = campaign.solver_arguments("bnb", protocol, 128, 2, 8, Path("c/checkpoints/x.ckpt"))
		self.assertEqual(bnb[bnb.index("--subtree-threads") + 1], "8")
		self.assertEqual(bnb[bnb.index("--checkpoint") + 1], str(Path("c/checkpoints/x.ckpt")))
		self.assertEqual(bnb[bnb.index("--max-frontier-mib") + 1], "128")
		for flag in protocol["bnb_arguments"]:
			self.assertIn(flag, bnb)
		ils = campaign.solver_arguments("ils", protocol, 128, 3, 8, Path("unused"))
		self.assertNotIn("--subtree-threads", ils)
		self.assertNotIn("--checkpoint", ils)
		self.assertEqual(ils[ils.index("--primal-ils-seed") + 1], "3")


class SchedulerTests(unittest.TestCase):
	def test_ils_first_and_bnb_after_its_instance(self):
		scheduler = campaign.Scheduler([("a", 0), ("b", 0), ("a", 1)], [("a", 0), ("b", 0)])
		first, second, third = scheduler.next(), scheduler.next(), scheduler.next()
		self.assertEqual([first, second, third], [("ils", "a", 0), ("ils", "b", 0), ("ils", "a", 1)])
		scheduler.done(second)
		self.assertEqual(scheduler.next(), ("bnb", "b", 0))
		received = []
		waiter = threading.Thread(target=lambda: received.append(scheduler.next()))
		waiter.start()
		scheduler.done(first)
		self.assertTrue(waiter.is_alive() or received == [])
		scheduler.done(third)
		waiter.join(timeout=5)
		self.assertEqual(received, [("bnb", "a", 0)])
		self.assertIsNone(scheduler.next())

	def test_stop_releases_waiting_workers(self):
		scheduler = campaign.Scheduler([("a", 0)], [("a", 2)])
		scheduler.next()
		received = []
		waiter = threading.Thread(target=lambda: received.append(scheduler.next()))
		waiter.start()
		scheduler.stop()
		waiter.join(timeout=5)
		self.assertEqual(received, [None])


class RepeatTests(unittest.TestCase):
	def test_repeated_bnb_gets_the_next_attempt(self):
		with tempfile.TemporaryDirectory() as directory:
			output = Path(directory)
			rows = [{"phase": "bnb", "name": "x", "seed": 0, "complete": True},
				{"phase": "bnb", "name": "x", "seed": 1, "complete": False}]
			(output / campaign.RUNS_FILE).write_text("".join(json.dumps(row) + "\n" for row in rows))
			args = campaign.build_parser().parse_args(["--campaign", str(output), "--phases", "bnb", "--repeat-bnb",
				"--instances", "x", "--dry-run"])
			with patch.object(campaign, "load_instances", return_value={"x": {"polygons": [], "sha256": ""}}), \
					patch.object(campaign, "select", return_value=["x"]), patch("builtins.print") as printed:
				campaign.run(args, [])
			plan = " ".join(str(call.args[0]) for call in printed.call_args_list)
			self.assertIn("1 B&B runs", plan)


class LocalGmpTests(unittest.TestCase):
	def test_a_built_gmp_is_passed_to_cmake(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			toolchain = {"CC": "cc", "CXX": "c++", "TPP_CXX_STANDARD": "23"}
			with patch.object(native_build, "DEPS_ROOT", root):
				self.assertNotIn("TPP_GMP", " ".join(native_build.configuration(gurobi=False, toolchain=toolchain)["arguments"]))
				prefix = root / "gmp" / native_build.SOURCE_DEPENDENCIES["gmp"]["version"]
				(prefix / "lib").mkdir(parents=True)
				(prefix / "lib/libgmp.a").write_bytes(b"")
				arguments = native_build.configuration(gurobi=False, toolchain=toolchain)["arguments"]
			self.assertIn(f"-DTPP_GMP_INCLUDE_DIR={prefix / 'include'}", arguments)
			self.assertIn(f"-DTPP_GMP_LIBRARY={prefix / 'lib/libgmp.a'}", arguments)

	def test_reads_whether_the_build_found_gmp(self):
		with tempfile.TemporaryDirectory() as directory:
			build = Path(directory)
			self.assertIsNone(native_build.build_uses_gmp(build))
			(build / "CMakeCache.txt").write_text("TPP_GMP_LIBRARY:FILEPATH=TPP_GMP_LIBRARY-NOTFOUND\n")
			self.assertFalse(native_build.build_uses_gmp(build))
			(build / "CMakeCache.txt").write_text("TPP_GMP_LIBRARY:FILEPATH=/usr/lib/libgmp.so\n")
			self.assertTrue(native_build.build_uses_gmp(build))


if __name__ == "__main__":
	unittest.main()
