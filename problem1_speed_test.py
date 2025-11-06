
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

def time_test(test: TestCase, number: int = 10) -> float:
	return timeit(lambda: FastSolution(*test).solve(), number=number)

def time_test_suite(name: str, sides_list: list[list[int]], number: int = 10) -> None:

	def justified(value: float) -> str:
		return str(round(value, 6)).rjust(10, " ")

	print(f"{name} Timing Tests:")

	for sides in sides_list:

		test = make_test(sides)
		fast = time_test(test, number=number)

		sides_string = str(sides) if len(sides) < 10 else "[" + ", ".join(map(str, sides[:5])) + ", ..., " + ", ".join(map(str, sides[-5:])) + "]"

		print(f"\tFast: {justified(fast)}s | Sides: {sides_string}", flush=True)

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

	#test_suite("Small", small_tests, number=100)
	#test_suite("Numerous Small", numerous_small_tests, number=10)
	#test_suite("Large", large_tests, number=10)

	time_test_suite("Timing Small", small_tests, number=1000)
	time_test_suite("Timing Numerous Small", numerous_small_tests, number=100)
	time_test_suite("Timing Large", large_tests, number=100)

"""
Small Tests:
	Speedup:       0.94x | Slow:   0.017235s | Fast:   0.018328s | Sides: [3, 3, 3, 3]
	Speedup:       1.08x | Slow:   0.028904s | Fast:   0.026814s | Sides: [3, 5, 4, 6]
	Speedup:       1.01x | Slow:   0.019049s | Fast:   0.018871s | Sides: [5, 5, 5]
	Speedup:       1.86x | Slow:   0.009437s | Fast:   0.005073s | Sides: [7]
	Speedup:       2.54x | Slow:   0.056886s | Fast:   0.022369s | Sides: [10, 8]
Numerous Small Tests:
	Speedup:       1.15x | Slow:   0.004057s | Fast:   0.003521s | Sides: [3, 3, 3, 3, 3, ..., 3, 3, 3, 3, 3]
	Speedup:       1.07x | Slow:   0.011139s | Fast:   0.010456s | Sides: [4, 4, 4, 4, 4, ..., 4, 4, 4, 4, 4]
	Speedup:       1.09x | Slow:   0.023372s | Fast:   0.021489s | Sides: [5, 5, 5, 5, 5, ..., 5, 5, 5, 5, 5]
	Speedup:       1.16x | Slow:   0.037994s | Fast:   0.032619s | Sides: [6, 6, 6, 6, 6, ..., 6, 6, 6, 6, 6]
	Speedup:        1.1x | Slow:   0.061807s | Fast:   0.056424s | Sides: [7, 7, 7, 7, 7, ..., 7, 7, 7, 7, 7]
	Speedup:       0.93x | Slow:   0.038155s | Fast:    0.04112s | Sides: [3, 3, 3, 3, 3, ..., 3, 3, 3, 3, 3]
Large Tests:
	Speedup:       8.14x | Slow:   0.475136s | Fast:   0.058399s | Sides: [100, 150, 90]
	Speedup:       10.7x | Slow:   1.199237s | Fast:   0.112121s | Sides: [200, 100, 150, 120]
	Speedup:      27.44x | Slow:   2.030551s | Fast:   0.073988s | Sides: [300, 250]
	Speedup:      65.05x | Slow:   1.519299s | Fast:   0.023355s | Sides: [400]
	Speedup:      19.14x | Slow:   5.469802s | Fast:   0.285727s | Sides: [500, 400, 300, 200, 100]
	Speedup:     133.06x | Slow:   7.935566s | Fast:   0.059638s | Sides: [1000]

Timing Small Timing Tests:
	Fast:   0.145585s | Sides: [3, 3, 3, 3]
	Fast:   0.270796s | Sides: [3, 5, 4, 6]
	Fast:    0.22928s | Sides: [5, 5, 5]
	Fast:   0.051113s | Sides: [7]
	Fast:   0.206471s | Sides: [10, 8]
Timing Numerous Small Timing Tests:
	Fast:   0.038306s | Sides: [3, 3, 3, 3, 3, ..., 3, 3, 3, 3, 3]
	Fast:   0.111474s | Sides: [4, 4, 4, 4, 4, ..., 4, 4, 4, 4, 4]
	Fast:   0.233947s | Sides: [5, 5, 5, 5, 5, ..., 5, 5, 5, 5, 5]
	Fast:    0.35686s | Sides: [6, 6, 6, 6, 6, ..., 6, 6, 6, 6, 6]
	Fast:   0.536957s | Sides: [7, 7, 7, 7, 7, ..., 7, 7, 7, 7, 7]
	Fast:   0.426815s | Sides: [3, 3, 3, 3, 3, ..., 3, 3, 3, 3, 3]
Timing Large Timing Tests:
	Fast:   0.600902s | Sides: [100, 150, 90]
	Fast:   0.959572s | Sides: [200, 100, 150, 120]
	Fast:   0.759498s | Sides: [300, 250]
	Fast:    0.24136s | Sides: [400]
	Fast:   3.194541s | Sides: [500, 400, 300, 200, 100]
	Fast:   0.611497s | Sides: [1000]	
"""