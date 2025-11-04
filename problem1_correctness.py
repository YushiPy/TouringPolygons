
import random
import math


from vector2 import Vector2
from polygon2 import Polygon2

from problem1 import Solution
from problem1_fast import Solution as FastSolution

type TestCase = tuple[Vector2, Vector2, list[Polygon2]]

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

				if all((point - c).magnitude() >= 1.5 + radius for c in centers):
					return point

				range_size *= 1.5

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

	return start, target, polygons


def do_test(test: TestCase) -> bool:

	expected = Solution(*test).solve()
	actual = FastSolution(*test).solve()

	if expected != actual:
		print("❌ Test failed!")
		print(test)
		return False

	return True

def test_suite(name: str, sides_list: list[list[int]]) -> None:

	print("	Runnning", name, "tests...", flush=True)

	for i, (sides) in enumerate(sides_list, 1):
		
		test = make_test(sides)
		
		if not do_test(test):
			break

		#if i % 10 == 0:
		#	print(f"		Completed {i} tests...", flush=True)
	else:
		print("	✅ All", name, "tests passed!", flush=True)

def test_block(name: str, tests: list[tuple[str, list[list[int]]]]) -> None:

	print(f"{name}:")
	for subname, sides_list in tests:
		test_suite(f"{subname}", sides_list)

if __name__ == "__main__":

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

	test_block("Fixed", [tests1, tests2, tests3])

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

	test_block("Random", [small_random, medium_random, large_random])
