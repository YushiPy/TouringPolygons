
from collections.abc import Sequence
import math

type Point = tuple[float, float]

def regular_polygon(n: int, radius: float, center: Point = (0.0, 0.0), angle: float = 0.0) -> list[Point]:
	"""
	Generates a regular polygon with `n` sides, given a `radius`, `center`, and starting `angle`.
	"""

	return [(center[0] + radius * math.cos(angle + 2 * math.pi * i / n), center[1] + radius * math.sin(angle + 2 * math.pi * i / n)) for i in range(n)]

def make_test(sides: Sequence[int], compactness: float) -> list[list[Point]]:
	"""
	Creates a test case with the given list of polygon sides. 
	Each polygon will be a regular polygon with a random radius and angle.
	The `compactness` parameter controls how tightly packed the polygons are, 
	with `0` being very tightly packed and `1` being more spread out, 
	and values greater than `1` creating even more spread out polygons.
	"""

	import random
	

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

	def index_to_point(index: int) -> tuple[float, float]:

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

	return polygons

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
	("Many Small Polygons", [
		[3] * 150,
		[4] * 150,
		[5] * 150,
		[6] * 150,
		[7] * 150,
	], 0.0, 10),
	("Many Medium Polygons", [
		[10] * 30,
		[20] * 30,
		[30] * 30,
		[40] * 30,
		[50] * 30,
	], 0.5, 20),
]

a = []

for name, polygon_sizes, hole_probability, num_tests in random_tests[1:2]:
	for _ in range(1):
		for size in polygon_sizes:
			a.append(make_test(size, hole_probability))

print(",\n".join(str(test) for test in a))