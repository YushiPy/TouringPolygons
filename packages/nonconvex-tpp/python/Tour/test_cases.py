
from test_solution import make_test

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

for name, polygon_sizes, hole_probability, num_tests in random_tests:
	for _ in range(1):
		for size in polygon_sizes:
			a.append(make_test(size, hole_probability)[:3])

print(",\n".join(str(test) for test in a))