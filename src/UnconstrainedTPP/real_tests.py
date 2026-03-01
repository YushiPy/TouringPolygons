
from collections.abc import Sequence
import math
import random

from vector2 import Vector2
from polygon2 import Polygon2

from u_tpp import tpp_solve as reference_solution
from u_tpp_final import tpp_solve as test_solution
#from u_tpp_naive import tpp_solve as test_solution

from time import perf_counter

type Point = tuple[float, float]
type Points = Sequence[Point]

type TestCase = tuple[Point, Point, list[Points]]


STRICT = False
EPSILON = 1e-10

# If True, tests will compare solution by total length rather than by matching individual points, 
# which can allows for solutions with collinear points to be considered correct even if they don't match exactly.
TEST_TOTAL_LENGTH = False 

# If true, reference solution will be a dummy solution that instantly returns an empty list. 
# This is useful for testing the performance of the tested solution without the overhead of the reference solution, 
# but should be set to False for accurate testing.
DUMMY_REFERENCE = False

if DUMMY_REFERENCE:
	reference_solution = lambda start, target, polygons: []

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


def solutions_equal(sol1: list[Point], sol2: list[Point], eps: float = EPSILON) -> bool:

	if DUMMY_REFERENCE:
		return True

	if TEST_TOTAL_LENGTH:

		def path_length(path: list[Point]) -> float:
			return sum(((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5 for i in range(1, len(path)))

		return abs(path_length(sol1) - path_length(sol2)) < eps
	
	else:

		if len(sol1) != len(sol2):
			return False

		return all(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < eps ** 2 for p1, p2 in zip(sol1, sol2))

def do_test(test: TestCase, number: int = 10) -> tuple[float, float]:
	"""
	Runs both the reference solution and the tested solution on the given test case, and returns the time taken by each respective solution.
	"""

	test = (test[0], test[1], [clean_polygon(polygon) for polygon in test[2]])

	if any(not is_convex(polygon) for polygon in test[2]):
		if STRICT:
			raise ValueError("All polygons must be convex.")
		else:
			print("⚠️​ Warning: All polygons should be convex for accurate testing.")
			return (0.0, 0.0)

	reference_time = 0.0
	tested_time = 0.0

	def run_reference() -> tuple[float, list[Point]]:
		start_time = perf_counter()

		try:
			result = reference_solution(*test)
		except Exception as e:
			if STRICT:
				raise ValueError(f"Reference solution raised an exception for input={test}.\nException: {e}.")
			else:
				print(f"⚠️​ Warning: Reference solution raised an exception for input={test}.\nException: {e}.")
				return (0.0, [])

		return perf_counter() - start_time, result

	def run_tested() -> tuple[float, list[Point]]:

		start_time = perf_counter()

		try:
			result = test_solution(*test)
		except Exception as e:
			if STRICT:
				raise ValueError(f"Tested solution raised an exception for input={test}.\nException: {e}.")
			else:
				print(f"⚠️​ Warning: Tested solution raised an exception for input={test}.\nException: {e}.")
				return (0.0, [])

		return perf_counter() - start_time, result

	for i in range(number):

		if random.random() < 0.5:
			ellapsed_reference, reference_result = run_reference()
			ellapsed_tested, tested_result = run_tested()
		else:
			ellapsed_tested, tested_result = run_tested()
			ellapsed_reference, reference_result = run_reference()

		reference_time += ellapsed_reference
		tested_time += ellapsed_tested

		# Only check the first result to avoid slowing down the tests too much, since the results should be deterministic.
		if i == 0 and not solutions_equal(reference_result, tested_result):
			if STRICT:
				raise ValueError(f"Test failed for input={test}.\nReference solution: {reference_result}.\nTested solution: {tested_result}.")
			else:
				print(f"⚠️​ Warning: Test failed for input={test}.\nReference solution: {reference_result}.\nTested solution: {tested_result}.")

	return reference_time, tested_time

def test_suite(name: str, test_cases: list[TestCase], number: int = 10) -> None:
	"""
	Runs a suite of tests and prints the total time taken by both the reference solution and the tested solution,
	as well as the ratio
	"""

	print(f"{name} Tests:", flush=True)

	reference_times = []
	tested_times = []

	for test in test_cases:
		a, b = do_test(test, number=number)
		reference_times.append(a)
		tested_times.append(b)

	ratios = [round(ref / test if test > 0 else float('inf'), 2) for ref, test in zip(reference_times, tested_times)]
	min_ratio = min(ratios)
	max_ratio = max(ratios)

	speedup = round(sum(ratios) / len(ratios), 2) if ratios else 1.0
	reference_time = round(sum(reference_times), 6)
	tested_time = round(sum(tested_times), 6)

	print(f"- Ref/Test: {speedup}x ({min_ratio}x - {max_ratio}x) | {reference_solution.__module__}: {reference_time}s | {test_solution.__module__}: {tested_time}s", flush=True)


def regular_polygon(n: int, radius: float, center: Point = (0.0, 0.0), angle: float = 0.0) -> list[Point]:
	"""
	Generates a regular polygon with `n` sides, given a `radius`, `center`, and starting `angle`.
	"""

	return [(center[0] + radius * math.cos(angle + 2 * math.pi * i / n), center[1] + radius * math.sin(angle + 2 * math.pi * i / n)) for i in range(n)]

def make_test(sides: list[int], compactness: float) -> TestCase:
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

	return start, target, polygons

def make_tests(sides_list: list[list[int]], compactness: float = 0.5) -> list[TestCase]:
	"""
	Creates a list of test cases from a list of sides lists, using the `make_test` function.

	:param list sides_list: A list of lists of integers, where each inner list represents the number of sides for each polygon in a test case.
	:param float compactness: The compactness parameter to pass to the `make_test` function.

	:return: A list of test cases.
	"""

	return [make_test(sides, compactness) for sides in sides_list]


if __name__ == "__main__":

	fixed = [
		((-2.0, 0.0), (2.0, 0.0), [[(-1.0, 1.0), (1.0, 1.0), (0.0, 2.0)]]),
		((-1.0, 5.0), (-0.5, 4.5), [[(2.0, 2.0), (0.0, 2.0), (0.0, 4.0)]]),
		((-0.6, 5.2), (-0.6, 4.8), [[(2.8, 2.0), (0.0, 2.0), (0.0, 5.0)]]),
		((4.5, 2.0), (3.5, 2.0), [[(2.8, 2.0), (0.0, 2.0), (0.0, 5.0)]]),
		((0.0, 4.0), (4.0, 0.0), [[(3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (3.0, 3.0)]]),
		((-1.7926052592116688, 2.0041878360341565), (-1.2727245469438635, -2.1926283895019223), [[(-2.8043080390114095, -3.654186702016027), (-1.7541858929206575, -5.033836318424196), (-1.0844353498351025, -3.4345790546288875)], [(2.0975681598044416, 2.0395046385241002), (3.094175815858267, 3.1866802982459888), (1.9470001561363786, 4.183287954299814), (0.950392500082553, 3.0361122945779258)]]),
		((-1.0, 0.0), (-1.0, 2.8000000000000003), [[(0.2, 3.0), (2.0, 2.0), (-1.0, 2.0), (-1.0, 3.0)]]),
		((-1.0, 1.0), (-1.0, 2.6), [[(2.0, 3.0), (2.0, 2.0), (-1.0, 2.0), (-1.0, 3.0)]]),
	]

	random_tests = [
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
			[30] * 7,
			[4, 5, 6] * 3
		], 1.0, 10),
		("Many Small Polygons", [
			[3] * 150,
			[4] * 150,
			[5] * 150,
			[6] * 150,
			[7] * 150,
		], 0.0, 10),
		("Few Large Polygons", [
			[100] * 1,
			[200] * 2,
			[300] * 3,
			[400] * 4,
			#[10 ** 5] * 2,
			#[10 ** 6] * 1,
			#[10 ** 5] * 10,
		], 0.5, 2),
	]

	test_suite("Fixed", fixed, number=100) # type: ignore

	for name, sides_list, compactness, number in random_tests:
		test_suite(name, make_tests(sides_list, compactness), number=number)
