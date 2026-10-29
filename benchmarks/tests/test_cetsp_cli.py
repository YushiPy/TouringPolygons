"""End-to-end checks of tpp-convex-cetsp on small instances of benchmarks/suites/cetsp:
known optima (Coutinho et al. 2016, three decimals), the floor kept when children
are dropped for memory, strict targets, open tours to a fixed end, interruption, a
zero-length tour, and a written tour that meets every decimal disk within its written bound (rational
arithmetic), also when pulling its contacts would lose a disk met outside the sequence, and
below the normal range. Skipped when the tool is not built, except under sanity_check.sh,
which builds it and sets TPP_REQUIRE_NATIVE=1."""

import json
import math
import os
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
REQUIRED = os.environ.get("TPP_REQUIRE_NATIVE") == "1"
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


def squared_distance_to_polyline(x, y, points, closed=True):
	best = None
	for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1] if closed else points[1:]):
		ex, ey, fx, fy = bx - ax, by - ay, x - ax, y - ay
		ee = ex * ex + ey * ey
		t = min(Fraction(1), max(Fraction(0), (fx * ex + fy * ey) / ee)) if ee else Fraction(0)
		gx, gy = fx - ex * t, fy - ey * t
		d2 = gx * gx + gy * gy
		best = d2 if best is None else min(best, d2)
	return best


def sqrt_up(square):
	"""A rational at least the square root of `square`, within 2^-80 of it (relative): an
	integer square root, which, unlike float(square), neither underflows nor overflows."""
	return Fraction(math.isqrt(square.numerator * square.denominator << 160) + 1, square.denominator << 80)


def check_written_tour(test, path, tour, closed=True):
	"""Every disk met by the written polyline, and its length within the written bound."""
	points = [(Fraction(x), Fraction(y)) for x, y in tour["contacts"]]
	test.assertEqual(len(points), len(tour["sequence"]))
	test.assertTrue(tour["verified"])
	for x, y, radius in read_disks(path):
		test.assertLessEqual(squared_distance_to_polyline(x, y, points, closed), radius * radius)
	pairs = zip(points, points[1:] + points[:1] if closed else points[1:])
	length = sum(sqrt_up((bx - ax) ** 2 + (by - ay) ** 2) for (ax, ay), (bx, by) in pairs)
	test.assertLessEqual(length, Fraction(tour["upper_bound"]))


def run_with_tour(path, *arguments):
	with tempfile.TemporaryDirectory() as directory:
		tour_file = Path(directory) / "tour.json"
		result = run(path, *arguments[:2], tour_file, *arguments[2:])
		return result, strict_json(tour_file.read_text())


@unittest.skipUnless(TOOL.is_file() or REQUIRED, "tpp-convex-cetsp is not built")
class CetspCliTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		if not TOOL.is_file():
			raise AssertionError(f"{TOOL} is not built")

	def test_known_optima(self):
		for path, optimum in ((KROD100_030, 58.541), (KROD100_010, 89.668)):
			result = run(path, 30, 1e-6, "-")
			self.assertTrue(result["closed"], path.name)
			self.assertLessEqual(result["lower_bound"], result["upper_bound"])
			self.assertLess(abs(result["upper_bound"] - optimum), 6e-4, path.name)

	def test_written_tour_meets_every_disk_within_its_bound(self):
		result, tour = run_with_tour(KROD100_010, 30, 1e-6)
		check_written_tour(self, KROD100_010, tour)
		self.assertLessEqual(result["lower_bound"], tour["upper_bound"])

	def test_written_tour_keeps_the_disks_met_outside_its_sequence(self):
		# The tour turns at the bottom of a disk of radius 10^6, (0, 5), which
		# meets the disk of radius 10^-7 there without visiting it; pulling that
		# contact 2^-40 of 10^6 toward its center used to miss the small disk
		# by 8 radii (review of 2026-10-28).
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "pulled.txt"
			path.write_text("//Depot is -10, 0\n10 0 0 0\n0 1000005 0 1000000\n0 5 0 0.0000001\n")
			_, tour = run_with_tour(path, 10, 1e-9, "--heuristic", 0)
			check_written_tour(self, path, tour)

	def test_written_bound_below_the_normal_range(self):
		# Squared coordinates of 1e-200 underflow; the bound used to be 5e-324.
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "tiny.txt"
			path.write_text("//Depot is 0, 0\n1e-200 0 0 1e-210\n")
			_, tour = run_with_tour(path, 5, 1e-9, "--heuristic", 0)
			check_written_tour(self, path, tour)

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

	def test_open_tours(self):
		# From (0, 0) to (10, 0): two unit disks on the way cost nothing; with a
		# third centered at (5, 4) the optimum turns at its bottom, (5, 3), and
		# at the first two, symmetrically. That angle alone, searched on a fine
		# grid outside the solver, gives 11.865339530283185.
		with tempfile.TemporaryDirectory() as directory:
			two, three = Path(directory) / "two.txt", Path(directory) / "three.txt"
			two.write_text("//Depot is 0, 0\n3 0 0 1\n7 0 0 1\n")
			three.write_text("//Depot is 0, 0\n3 0 0 1\n7 0 0 1\n5 4 0 1\n")
			for path, optimum in ((two, 10.0), (three, 11.865339530283185)):
				result, tour = run_with_tour(path, 10, 1e-9, "--end", "10,0")
				check_written_tour(self, path, tour, closed=False)
				self.assertTrue(result["closed"] and result["open"] and tour["open"])
				self.assertEqual(result["disks"], 2 if path == two else 3)
				self.assertLessEqual(result["lower_bound"], optimum + 1e-9)
				self.assertLess(abs(result["upper_bound"] - optimum), 1e-7)

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
