"""End-to-end checks of tpp-convex-cetsp on small instances of benchmarks/suites/cetsp:
known optima (Coutinho et al. 2016, three decimals), the floor kept when children
are dropped for memory, strict targets, interruption, a zero-length tour, and a
written tour that meets every decimal disk within its written bound (rational
arithmetic). Skipped when the tool is not built (sanity_check.sh builds it)."""

import json
import math
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_internal"))

import native_build

SUITE = Path(__file__).resolve().parents[1] / "suites" / "cetsp"
TOOL = native_build.tool_path("tpp-convex-cetsp")
KROD100_010 = SUITE / "mennell-tsplib-2d" / "kroD100-0.10.txt"
KROD100_030 = SUITE / "mennell-tsplib-2d" / "kroD100-0.30.txt"


def strict_json(text):
	def reject(name):
		raise ValueError(f"non-JSON constant {name}")
	return json.loads(text, parse_constant=reject)


def run(*arguments, timeout=60):
	completed = subprocess.run([str(TOOL), *map(str, arguments)], capture_output=True, text=True, check=True, timeout=timeout)
	return strict_json(completed.stdout)


def read_disks(path):
	"""The depot and the disks in Mennell's format: disks as exact decimals, the depot
	(a point, which no tour can meet in decimal) as the solver reads it, in binary64."""
	depot, disks = None, []
	for line in path.read_text().splitlines():
		if line.startswith("//"):
			if "Depot" in line:
				text = line.split("Depot", 1)[1].lstrip(" is:").split(",")
				depot = (Fraction(float(text[0])), Fraction(float(text[1])), Fraction(0))
			continue
		fields = line.split()
		if len(fields) >= 4:
			disks.append((Fraction(fields[0]), Fraction(fields[1]), Fraction(fields[3])))
	return [depot, *disks]


def squared_distance_to_polyline(x, y, points):
	best = None
	for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1]):
		ex, ey, fx, fy = bx - ax, by - ay, x - ax, y - ay
		ee = ex * ex + ey * ey
		t = min(Fraction(1), max(Fraction(0), (fx * ex + fy * ey) / ee)) if ee else Fraction(0)
		gx, gy = fx - ex * t, fy - ey * t
		d2 = gx * gx + gy * gy
		best = d2 if best is None else min(best, d2)
	return best


def sqrt_up(square):
	root = math.sqrt(float(square))
	while Fraction(root) ** 2 < square:
		root = math.nextafter(root, math.inf)
	return Fraction(root)


@unittest.skipUnless(TOOL.is_file(), "tpp-convex-cetsp is not built")
class CetspCliTests(unittest.TestCase):
	def test_known_optima(self):
		for path, optimum in ((KROD100_030, 58.541), (KROD100_010, 89.668)):
			result = run(path, 30, 1e-6, "-")
			self.assertTrue(result["closed"], path.name)
			self.assertLessEqual(result["lower_bound"], result["upper_bound"])
			self.assertLess(abs(result["upper_bound"] - optimum), 6e-4, path.name)

	def test_written_tour_meets_every_disk_within_its_bound(self):
		with tempfile.TemporaryDirectory() as directory:
			tour_file = Path(directory) / "tour.json"
			result = run(KROD100_010, 30, 1e-6, tour_file)
			tour = strict_json(tour_file.read_text())
		disks = read_disks(KROD100_010)
		points = [(Fraction(x), Fraction(y)) for x, y in tour["contacts"]]
		self.assertEqual(len(points), len(tour["sequence"]))
		for x, y, radius in disks:
			self.assertLessEqual(squared_distance_to_polyline(x, y, points), radius * radius)
		length = sum(sqrt_up((bx - ax) ** 2 + (by - ay) ** 2) for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1]))
		self.assertLessEqual(length, Fraction(tour["upper_bound"]))
		self.assertLessEqual(result["lower_bound"], tour["upper_bound"])

	def test_dropped_children_keep_a_valid_floor(self):
		result = run(KROD100_010, 30, 1e-6, "-", "--max-memory", 0.002)
		self.assertGreater(result["dropped"], 0)
		self.assertIn(result["stopped"], ("dropped", "closed"))
		self.assertLessEqual(result["lower_bound"], 89.668 + 5e-4)
		self.assertGreaterEqual(result["upper_bound"], 89.668 - 5e-4)

	def test_targets_are_strict(self):
		closed = run(KROD100_030, 30, 1e-6, "-")
		self.assertTrue(closed["closed"])
		same = run(KROD100_030, 30, 1e-6, "-", "--target-lower", repr(closed["lower_bound"]))
		self.assertEqual(same["stopped"], "closed")
		early = run(KROD100_030, 30, 1e-6, "-", "--target-lower", 1)
		self.assertEqual(early["stopped"], "target")
		self.assertGreater(early["lower_bound"], 1)
		self.assertTrue(early["trace"])

	def test_zero_length_tour(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "zero.txt"
			path.write_text("//Depot is 0, 0\n0 0 0 1\n1 0 0 2\n")
			result = run(path, 5, 1e-6, "-")
		self.assertTrue(result["closed"])
		self.assertEqual(result["gap"], 0)
		self.assertEqual(result["upper_bound"], 0)

	def test_interrupt_reports_valid_bounds(self):
		process = subprocess.Popen([str(TOOL), str(SUITE / "mennell-tsplib-2d" / "dsj1000-0.02.txt"), "60", "1e-6", "-"],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		time.sleep(1.5)
		process.send_signal(signal.SIGINT)
		output, _ = process.communicate(timeout=30)
		result = strict_json(output)
		self.assertEqual(result["stopped"], "interrupted")
		self.assertLessEqual(result["lower_bound"], result["upper_bound"])
		self.assertTrue(math.isfinite(result["upper_bound"]))


if __name__ == "__main__":
	unittest.main()
