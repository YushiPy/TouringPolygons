
from collections.abc import Iterable
import random
import math


from vector2 import Vector2
from polygon2 import Polygon2

from u_tpp_naive import tpp_solve as reference_tpp_solve
from u_tpp import tpp_solve as test_tpp_solve

type _Vector2 = Iterable[float]
type _Polygon2 = Iterable[_Vector2]
type TestCase = tuple[_Vector2, _Vector2, list[_Polygon2]]


TOLERANCE = 1e-5
TEST_LENGTH = True

def reference_solution(start: Vector2, target: Vector2, polygons: list[Polygon2]) -> list[Vector2]:
	return list(map(Vector2, reference_tpp_solve(start, target, polygons))) # type: ignore

def test_solution(start: Vector2, target: Vector2, polygons: list[Polygon2]) -> list[Vector2]:
	return list(map(Vector2, test_tpp_solve(start, target, polygons))) # type: ignore


def regular_polygon(n: int, r: float, center: Vector2 = Vector2(), angle: float = 0) -> Polygon2:
	"""
	Create a regular polygon with `n` vertices and radius `r`.

	:param int n: The number of vertices.
	:param float r: The radius of the polygon.
	:param Vector2 center: The center of the polygon.
	:param float angle: The rotation angle of the polygon.

	:return: A Polygon2 object representing the regular polygon.
	"""
	return Polygon2(center + Vector2.from_polar(r, i * math.tau / n + angle) for i in range(n))

def make_test(sides: list[int]) -> TestCase:
	"""
	Create a test case with polygons of given number of sides.

	:param list sides: A list of integers representing the number of sides for each polygon.

	:return: A tuple containing start point, target point, and list of polygons.
	"""

	def random_point(range_size: float) -> Vector2:
		return Vector2(
			random.uniform(-range_size, range_size),
			random.uniform(-range_size, range_size)
		)
	
	def get_random_point(radius: float, initial_range: float = 1.0, tries: int = 10) -> Vector2:

		range_size = initial_range

		while True:

			for _ in range(tries):

				point = random_point(range_size)

				if rounding:
					point = round(point, 0)

				if all((point - c).magnitude() >= 1.5 + radius for c in centers):
					return point

				range_size *= 1.5

	rounding = random.randint(0, 1)

	polygons: list[Polygon2] = []
	centers: list[Vector2] = []
	
	for n in sides:

		radius = random.uniform(0.5, 1.5)
		angle = random.uniform(0, math.tau)

		center = get_random_point(radius, initial_range=1.0, tries=20)
		polygon = regular_polygon(n, radius, center, angle)

		polygons.append(polygon)
		centers.append(center)

	start = get_random_point(0, initial_range=1.0, tries=50)
	target = get_random_point(0, initial_range=1.0, tries=50)

	return start, target, polygons # type: ignore

def print_test(test: TestCase) -> None:
	print((tuple(test[0]), tuple(test[1]), [list(map(tuple, polygon)) for polygon in test[2]]))

def do_test(test: TestCase) -> bool:

	test = (Vector2(*test[0]), Vector2(*test[1]), list(map(Polygon2, test[2])))

	try:
		expected = reference_solution(*test) # type: ignore
	except Exception as e:
		print("❌ Reference solution raised an exception:", e)
		print_test(test)
		exit(1)
		return False
	
	try:
		actual = test_solution(*test) # type: ignore
	except Exception as e:
		print("❌ Test solution raised an exception:", e)
		print_test(test)
		exit(1)
		return False

	if TEST_LENGTH:
		length_expected = sum((expected[i] - expected[i + 1]).magnitude() for i in range(len(expected) - 1))
		length_actual = sum((actual[i] - actual[i + 1]).magnitude() for i in range(len(actual) - 1))
		passed = abs(length_expected - length_actual) <= TOLERANCE
	else:
		passed = all((e - a).magnitude() <= TOLERANCE for e, a in zip(expected, actual))
	
	if not passed:
		print("❌ Test failed!")
		print("Expected:", [tuple(p) for p in expected])
		print("Actual:", [tuple(p) for p in actual])
		print_test(test)
		return False

	return True

def test_suite(name: str, test_cases: Iterable[TestCase]) -> None:

	print("	Runnning", name, "tests...", flush=True)

	for test in test_cases:		
		if not do_test(test):
			break
	else:
		print("	✅ All", name, "tests passed!", flush=True)

def test_block(name: str, tests: Iterable[tuple[str, Iterable[TestCase]]]) -> None:

	print(f"{name}:")
	for subname, sides_list in tests:
		test_suite(f"{subname}", sides_list)

def random_test_suit(name: str, sides_list: list[list[int]]) -> None:
	return test_suite(name, map(make_test, sides_list))

def random_test_block(name: str, tests: Iterable[tuple[str, list[list[int]]]]) -> None:
	return test_block(name, ((subname, map(make_test, sides_list)) for subname, sides_list in tests))


if __name__ == "__main__":

	fixed = [
		((-2.0, 0.0), (2.0, 0.0), [[(-1.0, 1.0), (1.0, 1.0), (0.0, 2.0)]]),
		((-1.0, 5.0), (-0.5, 4.5), [[(2.0, 2.0), (0.0, 2.0), (0.0, 4.0)]]),
		((-0.6, 5.2), (-0.6, 4.8), [[(2.8, 2.0), (0.0, 2.0), (0.0, 5.0)]]),
		((4.5, 2.0), (3.5, 2.0), [[(2.8, 2.0), (0.0, 2.0), (0.0, 5.0)]]),
		((0.0, 4.0), (4.0, 0.0), [[(3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (3.0, 3.0)]]),
		((-1.7926052592116688, 2.0041878360341565), (-1.2727245469438635, -2.1926283895019223), [[(-2.8043080390114095, -3.654186702016027), (-1.7541858929206575, -5.033836318424196), (-1.0844353498351025, -3.4345790546288875)], [(2.0975681598044416, 2.0395046385241002), (3.094175815858267, 3.1866802982459888), (1.9470001561363786, 4.183287954299814), (0.950392500082553, 3.0361122945779258)]])
	]

	test_block("Fixed", [
		("Basic", fixed), # type: ignore
	])

	tests1 = ("Small", [
		[3], 
		[4], 
		[5], 
		[3, 4], 
		[4, 5], 
		[3, 4, 5], 
		[6, 7, 8], 
		[3, 5, 7, 9], 
		[3, 4, 5, 6, 7], 
		[3, 3, 3, 3, 3, 3],	
	])

	tests2 = ("Medium", [
		[10, 12],
		[8, 10, 12],
		[5, 6, 7, 8, 9, 10],
		[4, 5, 6, 7, 8, 9, 10],
		[3, 4, 5, 6, 7, 8, 9, 10],
	])

	tests3 = ("Large", [
		[15, 18, 20],
		[10, 12, 14, 16, 18, 20],
		[8, 9, 10, 11, 12, 13, 14, 15, 16],
	])

	random_test_block("Random 1", [tests1, tests2, tests3])

	small_random = ("Small", [
		[random.randint(3, 10) for _ in range(random.randint(1, 5))]
		for _ in range(20)
	])

	medium_random = ("Medium", [
		[random.randint(3, 20) for _ in range(random.randint(1, 10))]
		for _ in range(50)
	])

	large_random = ("Large", [
		[random.randint(3, 50) for _ in range(random.randint(1, 20))]
		for _ in range(50)
	])

	random_test_block("Random 2", [small_random, medium_random, large_random])
