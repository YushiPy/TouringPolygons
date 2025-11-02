
import math
import random
from timeit import timeit

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

def do_test(test: TestCase, number: int = 10) -> tuple[float, float]:

	slow = timeit(lambda: Solution(*test).solve(), number=number)
	fast = timeit(lambda: FastSolution(*test).solve(), number=number)

	return slow, fast

def test_suite(name: str, sides_list: list[list[int]], number: int = 10) -> None:

	def justified(value: float) -> str:
		return str(round(value, 6)).rjust(10, " ")

	print(f"{name} Tests:")

	for sides in sides_list:

		test = make_test(sides)
		slow, fast = do_test(test, number=number)

		sides_string = str(sides) if len(sides) < 10 else "[" + ", ".join(map(str, sides[:5])) + ", ..., " + ", ".join(map(str, sides[-5:])) + "]"

		print(f"\tSpeedup: {justified(round(slow / fast, 2))}x | Slow: {justified(slow)}s | Fast: {justified(fast)}s | Sides: {sides_string}", flush=True)

		#print(f"\tSides: {sides} | Slow: {slow:.6f}s | Fast: {fast:.6f}s | Speedup: {slow / fast:.2f}x", flush=True)

if __name__ == "__main__":

	small_tests = [
		[3, 3, 3, 3],
		[3, 5, 4, 6],
		[5, 5, 5],
		[7],
		[10, 8], 
	]

	numerous_small_tests = [
		[3] * 10,
		[4] * 20,
		[5] * 30,
		[6] * 40,
		[7] * 50,
		[3] * 100,
	]

	large_tests = [
		[100, 150, 90],
		[200, 100, 150, 120],
		[300, 250],
		[400],
		[500, 400, 300, 200, 100],
		[1000],
	]

	test_suite("Small", small_tests, number=100)
	test_suite("Numerous Small", numerous_small_tests, number=10)
	test_suite("Large", large_tests, number=1)