
from collections.abc import Sequence
import math
import random

import sys
sys.setrecursionlimit(10 ** 7)

import solutions

reference_solution = solutions.tpp_solve
test_solution = solutions.tpp_solve_naive_cpp

#import LegacySolutions
#test_solution = LegacySolutions.u_tpp_filtered.tpp_solve


type Point = tuple[float, float]
type Points = Sequence[Point]

type TestCase = tuple[Point, Point, Sequence[Points], Points | None]

# If True, the test suite will stop at the first failed test and raise an exception, 
# otherwise it will continue running all tests and print a summary at the end.
STRICT = False

# If True, the test suite will print detailed information about each failed test, 
# including the input and the expected and actual outputs. Otherwise, it will only print 
# a summary of how many tests passed and failed.
VERBOSE = True

EPSILON = 1e-10

# If True, tests will compare solution by total length rather than by matching individual points, 
# which can allows for solutions with collinear points to be considered correct even if they don't match exactly.
TEST_TOTAL_LENGTH = False 

def cross_product(o: Point, a: Point, b: Point) -> float:
	"""
	Returns the cross product of vectors OA and OB, where O is the origin point, A and B are the other two points.
	"""
	return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def clean_polygon(polygon: Points, eps: float = EPSILON) -> Points:
	"""
	Cleans a polygon by removing collinear points and making the vertices counter-clockwise.

	:param Points polygon: The polygon to clean.
	:param float eps: A small epsilon value for numerical stability.

	:return: The cleaned polygon.
	"""

	cleaned = []
	n = len(polygon)

	# Remove collinear points
	for i in range(n):

		prev = polygon[i - 1]
		curr = polygon[i]
		next = polygon[(i + 1) % n]

		cross = cross_product(prev, curr, next)

		if abs(cross) > eps ** 2:
			cleaned.append(curr)

	# Ensure counter-clockwise order
	area = 0.0

	for i in range(len(cleaned)):
		v1 = cleaned[i]
		v2 = cleaned[(i + 1) % len(cleaned)]
		area += (v1[0] * v2[1] - v2[0] * v1[1])

	if area < 0:
		cleaned.reverse()

	return cleaned

def is_convex(polygon: Points) -> bool:	
	return len(polygon) >= 3 and all(cross_product(polygon[i - 1], polygon[i], polygon[(i + 1) % len(polygon)]) > 0 for i in range(len(polygon)))


def solutions_equal(sol1: Sequence[Point], sol2: Sequence[Point], eps: float = EPSILON) -> bool:

	if TEST_TOTAL_LENGTH:

		def path_length(path: Sequence[Point]) -> float:
			return sum(((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5 for i in range(1, len(path)))

		return math.isclose(path_length(sol1), path_length(sol2), rel_tol=eps)
	
	else:

		if len(sol1) != len(sol2):
			return False

		return all(math.isclose(p1[0], p2[0], rel_tol=eps) and math.isclose(p1[1], p2[1], rel_tol=eps) for p1, p2 in zip(sol1, sol2))

def test_suite(name: str, test_cases: Sequence[TestCase]) -> None:
	"""
	Runs a suite of tests and prints the total time taken by both the reference solution and the tested solution,
	as well as the ratio
	"""

	fail_count = 0

	for i, (start, target, polygons, expected) in enumerate(test_cases):

		try:
			tested_path = test_solution(start, target, polygons)
		except Exception as e:
			
			print(f"❌ Tested solution raised an exception: {e}")

			if VERBOSE:
				print(f"Input: {(start, target, polygons)}")

			if STRICT:
				exit(1)
			else:
				continue

		if expected is None:
			try:
				expected = reference_solution(start, target, polygons)
			except Exception as e:
	
				print(f"Reference solution raised an exception: {e}")
	
				if VERBOSE:
					print(f"Input: {(start, target, polygons)}")
	
				continue
		
		if not solutions_equal(tested_path, expected):

			if fail_count == 0:
				print(f"{name}:", flush=True)

			fail_count += 1

			if STRICT:
				raise AssertionError(f"Test {i + 1} failed; expected {expected}, got {tested_path}.\nInput: {(start, target, polygons)}")
			elif VERBOSE:
				print(f"❌ Test {i + 1} failed; expected {expected}, got {tested_path}.\nInput: {(start, target, polygons)}")
	
	if fail_count == 0:
		print(f"✅ All {name} passed!", flush=True)
	else:
		print(f"⚠️ {fail_count}/{len(test_cases)} failed tests, but continuing due to non-strict mode.", flush=True)

def regular_polygon(n: int, radius: float, center: Point = (0.0, 0.0), angle: float = 0.0) -> list[Point]:
	"""
	Generates a regular polygon with `n` sides, given a `radius`, `center`, and starting `angle`.
	"""

	return [(center[0] + radius * math.cos(angle + 2 * math.pi * i / n), center[1] + radius * math.sin(angle + 2 * math.pi * i / n)) for i in range(n)]

def make_test(sides: Sequence[int], compactness: float) -> TestCase:
	"""
	Creates a test case with the given list of polygon sides. 
	Each polygon will be a regular polygon with a random radius and angle.
	The `compactness` parameter controls how tightly packed the polygons are, 
	with `0` being very tightly packed and `1` being more spread out, 
	and values greater than `1` creating even more spread out polygons.
	"""

	if not (0 <= compactness):
		raise ValueError("Compactness must be greater than or equal to 0.")

	count = len(sides)
	squares = math.ceil(count ** (1 + compactness))

	width = math.ceil(math.sqrt(squares))
	height = math.ceil(squares / width)
	squares = width * height

	indeces = random.sample(range(squares), count)

	polygons = []

	min_r = 0.33
	max_r = 1.0

	def index_to_point(index: int) -> Point:

		i = index // width
		j = index % width

		return 2 * max_r * j, 2 * max_r * i

	for index in indeces:

		radius = random.uniform(min_r, max_r)
		angle = random.uniform(0, math.tau)

		x, y = index_to_point(index)
		center = x + max_r, y + max_r

		polygon = regular_polygon(sides[len(polygons)], radius, center, angle)

		polygons.append(polygon)

	start_index, target_index = random.sample(indeces, 2) if len(indeces) >= 2 else (-1, 1)

	start = index_to_point(start_index)
	target = index_to_point(target_index)

	return start, target, polygons, None

def make_tests(sides_list: Sequence[Sequence[int]], compactness: float = 0.5) -> list[TestCase]:
	"""
	Creates a list of test cases from a list of sides lists, using the `make_test` function.

	:param list sides_list: A list of lists of integers, where each inner list represents the number of sides for each polygon in a test case.
	:param float compactness: The compactness parameter to pass to the `make_test` function.

	:return: A list of test cases.
	"""

	return [make_test(sides, compactness) for sides in sides_list]


if __name__ == "__main__":

	# (start, target, polygons, solution)
	basic_tests = [
		((0.0, 1.0), (2.0, 1.0), [[(2.0, 3.0), (-1.0, 2.0), (2.0, 2.0)]], [(0.0, 1.0), (1.0, 2.0), (2.0, 1.0)]),
		((0.0, 1.0), (3.0, 2.0), [[(2.0, 3.0), (-1.0, 2.0), (2.0, 2.0)]], [(0.0, 1.0), (2.0, 2.0), (3.0, 2.0)]),
		((0.0, 1.0), (0.0, 3.0), [[(2.0, 3.0), (-1.0, 2.0), (2.0, 2.0)]], [(0.0, 1.0), (0.0, 3.0)]),
		((0.0, 0.0), (-2.0, 1.0), [[(-1.0, 2.0), (0.7745461995542708, 1.0969540776602265), (2.0, 2.0), (2.0, 3.0), (0.01042765904450249, 3.997485680411619)]], [(0.0, 0.0), (-0.3877793180272373, 1.688448015291771), (-2.0, 1.0)]),
		((-1.0, 0.5), (3.0, 0.5), [[(-1.0, 2.0), (1.0, 1.0), (3.0, 2.0), (1.0, 3.0)]], [(-1.0, 0.5), (1.0, 1.0), (3.0, 0.5)]),
		((-1.0, 0.5), (3.0, 0.5), [[(-1.0, 2.0), (1.0, 1.0), (3.0, 2.0), (1.0, 3.0)], [(2.0, -2.5), (0.0, -2.0), (0.0, -4.0)]], [(-1.0, 0.5), (0.44179104477611975, 1.2791044776119402), (0.9670014347202298, -2.241750358680058), (3.0, 0.5)]),
		((4.0, 1.5), (-1.5, -1.0), [[(-1.0, 2.0), (1.0, 1.0), (3.0, 2.0), (1.0, 3.0)], [(2.0, -2.0), (0.0, -2.0), (1.0, -4.0)]], [(4.0, 1.5), (2.2857142857142856, 1.6428571428571428), (0.0, -2.0), (-1.5, -1.0)]),
		((-2.0, 0.5), (-1.0, 5.5), [[(-1.0, 2.0), (1.0, 1.0), (3.0, 2.0), (1.0, 3.0)], [(2.0, -2.0), (0.0, -2.0), (1.0, -4.0)], [(-5.5892251770067904, -2.0635401226162013), (-4.0, -2.5), (-3.0, -1.5), (-4.0, 1.0), (-5.5, 1.5)], [(-0.5, 3.5), (-3.5, 5.5), (-3.0, 2.5)]], [(-2.0, 0.5), (-0.25454545454545463, 1.6272727272727272), (0.0, -2.0), (-3.417422867513612, -0.45644283121597096), (-1.0, 5.5)])
	]

	edge_cases = [
		((-2.0, 0.0), (2.0, 0.0), [[(-1.0, 1.0), (1.0, 1.0), (0.0, 2.0)]], [(-2.0, 0.0), (0.0, 1.0), (2.0, 0.0)]),
		((-1.0, 5.0), (-0.5, 4.5), [[(0.0, 4.0), (0.0, 2.0), (2.0, 2.0)]], [(-1.0, 5.0), (0.0, 4.0), (-0.5, 4.5)]),
		((-0.6, 5.2), (-0.6, 4.8), [[(0.0, 5.0), (0.0, 2.0), (2.8, 2.0)]], [(-0.6, 5.2), (0.0, 5.0), (-0.6, 4.8)]),
		((4.5, 2.0), (3.5, 2.0), [[(0.0, 5.0), (0.0, 2.0), (2.8, 2.0)]], [(4.5, 2.0), (2.8, 2.0), (3.5, 2.0)]),
		((0.0, 4.0), (4.0, 0.0), [[(3.0, 3.0), (1.0, 3.0), (1.0, 1.0), (3.0, 1.0)]], [(0.0, 4.0), (4.0, 0.0)]),
		((-1.7926052592116688, 2.0041878360341565), (-1.2727245469438635, -2.1926283895019223), [[(-2.8043080390114095, -3.654186702016027), (-1.7541858929206575, -5.033836318424196), (-1.0844353498351025, -3.4345790546288875)], [(2.0975681598044416, 2.0395046385241002), (3.094175815858267, 3.1866802982459888), (1.9470001561363786, 4.183287954299814), (0.950392500082553, 3.0361122945779258)]], [(-1.7926052592116688, 2.0041878360341565), (-1.0844353498351025, -3.4345790546288875), (2.0975681598044416, 2.0395046385241002), (-1.2727245469438635, -2.1926283895019223)]),
		((-1.0, 0.0), (-1.0, 2.8000000000000003), [[(-1.0, 3.0), (-1.0, 2.0), (2.0, 2.0), (0.2, 3.0)]], [(-1.0, 0.0), (-1.0, 2.8000000000000003)]),
		((-1.0, 1.0), (-1.0, 2.6), [[(-1.0, 3.0), (-1.0, 2.0), (2.0, 2.0), (2.0, 3.0)]], [(-1.0, 1.0), (-1.0, 2.6)]),
		((-2.0, 2.0), (16.0, 2.0), [[(2.0, 2.0), (0.0, 4.0), (0.0, 2.0)], [(5.0, 2.0), (3.0, 4.0), (3.0, 2.0)], [(8.0, 2.0), (6.0, 4.0), (6.0, 2.0)], [(11.0, 2.0), (9.0, 4.0), (9.0, 2.0)], [(14.0, 2.0), (12.0, 4.0), (12.0, 2.0)]], [(-2.0, 2.0), (16.0, 2.0)]),
		((0.0, 0.0), (0.0, 1.2000000000000002), [[(0.0, 3.0), (-1.0, 2.0), (1.0, 2.0)]], [(0.0, 0.0), (0.0, 2.0), (0.0, 1.2000000000000002)]),
	]

	random_tests = [
		("Single Polygon", [
			[n] for n in range(3, 50)
		], 0.5, 100),
		("Small Tests", [
			[3, 3, 3, 3],
			[4, 4, 4, 4],
			[5, 5, 5, 5],
			[6, 6, 6, 6],
			[7, 7, 7, 7],
		], 0.5, 100),
		("Medium Tests", [
			[3, 4, 5, 8, 6, 7, 4, 4, 10, 7, 3, 5],
			[5] * 10,
			[30] * 10,
			[4, 5, 6] * 3
		], 1.0, 10),
		("Many Small Polygons Tests", [
			[3] * 150,
			[4] * 150,
			[5] * 150,
			[6] * 150,
			[7] * 150,
		], 0.0, 10),
		("Many Medium Polygons Tests", [
			[10] * 30,
			[20] * 30,
			[30] * 30,
			[40] * 30,
			[50] * 30,
		], 0.5, 20),
		("Few Large Polygons Tests", [
			[100] * 1,
			[200] * 2,
			[300] * 3,
			[400] * 4,
			[10 ** 4] * 2,
			[10 ** 5] * 2,
			[10 ** 6] * 1,
		] * 10, 0.5, 20),
	]

	test_suite("Basic", basic_tests)
	test_suite("Edge Cases", edge_cases)

	for name, sides_list, compactness, number in random_tests:
		test_suite(name, make_tests(sides_list, compactness))
