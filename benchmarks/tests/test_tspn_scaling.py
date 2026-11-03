import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import tspn_scaling as scaling


def run_row(phase, n, seed, length, lower, exact=False, valid=True, history=(), trajectory=(), nodes=10, seconds=1.0):
	return {"phase": phase, "name": scaling.instance_name(n, seed), "n": n, "seed": seed, "length": length,
		"lower_bound": lower, "exact": exact, "valid": valid, "complete": True, "nodes": nodes, "seconds": seconds,
		"final_gap": 1 - lower / length, "history": [list(item) for item in history],
		"trajectory": [list(item) for item in trajectory], "termination": "optimal" if exact else "time_limit"}


class GeneratorTests(unittest.TestCase):
	def test_deterministic_simple_nonconvex_on_the_grid(self):
		first = scaling.generate_instance(12, 3)
		self.assertEqual(first, scaling.generate_instance(12, 3))
		self.assertNotEqual(first["polygons"], scaling.generate_instance(12, 4)["polygons"])
		self.assertEqual(len(first["polygons"]), 12)
		for polygon in first["polygons"]:
			self.assertEqual(len(polygon), scaling.VERTICES)
			ring = [(x * scaling.GRID, y * scaling.GRID) for x, y in polygon]
			self.assertTrue(all(value == int(value) for point in ring for value in point))
			self.assertTrue(scaling.simple_ccw_nonconvex([(int(x), int(y)) for x, y in ring]))

	def test_density_is_the_same_at_every_size(self):
		for n in (10, 80):
			instance = scaling.generate_instance(n, 0)
			area = sum(scaling._area(polygon) for polygon in instance["polygons"])
			self.assertAlmostEqual(area / instance["side"] ** 2, scaling.COVERAGE, delta=1e-3)

	def test_rejects_convex_self_crossing_and_clockwise_rings(self):
		self.assertFalse(scaling.simple_ccw_nonconvex([(0, 0), (4, 0), (4, 4), (0, 4)]))
		self.assertFalse(scaling.simple_ccw_nonconvex([(0, 0), (4, 4), (4, 0), (0, 4)]))
		self.assertTrue(scaling.simple_ccw_nonconvex([(0, 0), (4, 0), (2, 1), (4, 4), (0, 4)]))
		self.assertFalse(scaling.simple_ccw_nonconvex([(0, 4), (4, 4), (2, 1), (4, 0), (0, 0)]))

	def test_stored_instances_must_match_the_generator(self):
		with tempfile.TemporaryDirectory() as directory:
			output = Path(directory)
			scaling.load_instances(output, [5], 1, scaling.VERTICES, scaling.COVERAGE)
			path = output / "instances" / "n005-s00.json"
			stored = json.loads(path.read_text())
			stored["sha256"] = "0" * 64
			path.write_text(json.dumps(stored))
			with self.assertRaises(SystemExit):
				scaling.load_instances(output, [5], 1, scaling.VERTICES, scaling.COVERAGE)


class SummaryTests(unittest.TestCase):
	def test_optimum_comes_from_a_closed_valid_run_and_rates_the_ils(self):
		rows = [run_row("bnb", 20, 0, 100.0, 100.0 * (1 - 1e-7), exact=True, history=[(0.1, 101.0), (0.4, 100.0)]),
			run_row("ils", 20, 0, 100.00005, 90.0, history=[(0.01, 102.0), (3.0, 100.00005)])]
		item = scaling.instance_summary(rows)
		self.assertEqual(item["optimum"], 100.0)
		self.assertTrue(item["ils_found_optimum"])
		self.assertEqual(item["ils_time_to_optimum"], 3.0)
		self.assertEqual(item["ils_vs_bnb"], "equal")
		self.assertAlmostEqual(item["ils_excess_lower"], 100.00005 / (100.0 * (1 - 1e-7)) - 1)

	def test_open_instance_has_no_optimum(self):
		rows = [run_row("bnb", 90, 1, 110.0, 100.0), run_row("ils", 90, 1, 108.0, 80.0, exact=True, valid=False)]
		item = scaling.instance_summary(rows)
		self.assertIsNone(item["optimum"])
		self.assertNotIn("ils_found_optimum", item)
		self.assertEqual(item["lower"], 100.0)
		self.assertEqual(item["best"], 110.0)

	def test_loose_gap_time_from_the_trajectory_or_the_final_gap(self):
		row = run_row("bnb", 50, 0, 100.0, 99.0, trajectory=[(2.0, 0.05, 5), (4.0, 0.0009, 9)], seconds=600.0)
		self.assertEqual(scaling.time_to_gap(row, 1e-3), 4.0)
		closed = run_row("bnb", 10, 0, 100.0, 100.0, exact=True, seconds=0.2)
		self.assertEqual(scaling.time_to_gap(closed, 1e-3), 0.2)
		self.assertIsNone(scaling.time_to_gap(run_row("bnb", 90, 0, 100.0, 90.0), 1e-3))

	def test_unclosed_runs_count_at_the_limit(self):
		protocol = scaling.protocol_of(scaling.build_parser().parse_args(["--seconds", "60"]))
		rows = [run_row("bnb", 30, 0, 10.0, 10.0, exact=True, seconds=2.0, nodes=100),
			run_row("bnb", 30, 1, 10.0, 9.0, seconds=60.3, nodes=900),
			run_row("ils", 30, 0, 10.0, 8.0), run_row("ils", 30, 1, 9.9, 8.0)]
		entry = scaling.summarize(rows, protocol)["sizes"][0]
		self.assertEqual((entry["bnb_closed"], entry["bnb_runs"]), (1, 2))
		self.assertEqual(entry["bnb_seconds"], (16.5, 31.0, 45.5))  # 60.3 s counts as the 60 s limit
		self.assertEqual(entry["bnb_seconds_closed"], (2.0, 2.0, 2.0))
		self.assertEqual(entry["ils_vs_bnb"], {"better": 1, "equal": 1, "worse": 0})
		self.assertIn("| 30 | 1/2 |", scaling.markdown(scaling.summarize(rows, protocol), []))

	def test_protocol_may_not_change_under_a_campaign(self):
		parser = scaling.build_parser()
		with tempfile.TemporaryDirectory() as directory:
			output = Path(directory)
			protocol = scaling.protocol_of(parser.parse_args([]))
			(output / scaling.PROTOCOL_FILE).write_text(json.dumps(protocol))
			scaling.check_protocol(output, json.loads(json.dumps(protocol)))
			with self.assertRaises(SystemExit):
				scaling.check_protocol(output, scaling.protocol_of(parser.parse_args(["--seconds", "60"])))


if __name__ == "__main__":
	unittest.main()
