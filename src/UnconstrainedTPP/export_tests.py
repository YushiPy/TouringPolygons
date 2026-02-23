"""
Converts python tests to a binary file that is easily read by the C++ code. 
This is used to test the C++ implementation against the python implementation.
"""

from vector2 import Vector2


def test_case_to_binary(start: Vector2, target: Vector2, polygons: list[list[Vector2]]) -> bytes:
	"""
	Converts a test case to a binary format that can be read by the C++ code. 
	Each point is represented as two 64-bit floats (16 bytes), and the number of polygons and vertices are represented as 64-bit integers (8 bytes).
	The format is as follows:
	
	- `16` bytes: `start`
	- `16` bytes: `target`
	- `8` bytes: number of polygons (`k`)
	- For each polygon `P_i`:
		- `8` bytes: number of vertices (`|P_i|`)
		- For each `vertex`:
			- `16` bytes: `vertex`
	"""

	result = bytearray()

	result.extend(start.to_bytes())
	result.extend(target.to_bytes())

	result.extend(len(polygons).to_bytes(8, byteorder='little'))

	for polygon in polygons:
		result.extend(len(polygon).to_bytes(8, byteorder='little'))
		for vertex in polygon:
			result.extend(vertex.to_bytes())
	
	return bytes(result)

def export_test_cases(test_cases: list[tuple[Vector2, Vector2, list[list[Vector2]]]], filename: str) -> None:
	"""
	Exports a list of test cases to a binary file. 
	Each test case is converted to binary format using the `test_case_to_binary` function and written to the file sequentially.
	The format of the file is as follows:

	- `8` bytes: number of test cases (`m`)
	- For each test case:
		- `16` bytes: `start`
		- `16` bytes: `target`
		- `8` bytes: number of polygons (`k`)
		- For each polygon `P_i`:
			- `8` bytes: number of vertices (`|P_i|`)
			- For each `vertex`:
				- `16` bytes: `vertex`
	"""

	with open(filename, 'wb') as f:
		f.write(len(test_cases).to_bytes(8, byteorder='little'))
		for start, target, polygons in test_cases:
			f.write(test_case_to_binary(start, target, polygons))

def read_test_results(filename: str) -> list[tuple[float, list[Vector2]]]:
	"""
	Reads the test results from a binary file. 
	The format of the file is as follows:

	- `8` bytes: number of test cases (`m`)
	- For each test case:
		- `8` bytes: time taken in seconds (double)
		- `8` bytes: number of vertices in the solution path (`n`)
		- For each vertex in the solution path:
			- `16` bytes: vertex
		
	The function returns a list of tuples, where each tuple contains the length of the path and the list of points in the path.
	"""

	import struct

	results = []

	with open(filename, 'rb') as f:
		m = int.from_bytes(f.read(8), byteorder='little')
		for _ in range(m):
			time_taken = struct.unpack('d', f.read(8))[0]
			n = int.from_bytes(f.read(8), byteorder='little')
			path = []
			for _ in range(n):
				point = Vector2.from_bytes(f.read(16))
				path.append(point)
			results.append((time_taken, path))
	
	return results

test1 = (
	Vector2(5, 1), 
	Vector2(7, 3),
	[
		([Vector2(3, 0), Vector2(2, 4), Vector2(1, 4), Vector2(-1, 1)]),
		([Vector2(3, 3), Vector2(4, 3), Vector2(4, 4), Vector2(3, 4)]),
		([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)]),
	]
)

from math import pi, tau

test2 = (
	Vector2(-1, -1),
	Vector2(1, -1),
	[
		([Vector2.from_polar(2, i * tau / 6 + pi * 0.35) + Vector2(4, 5) for i in range(6)]),
		([Vector2.from_polar(2, i * tau / 3 + pi /4) + Vector2(-3, 4) for i in range(3)]),
		([Vector2.from_polar(2, i * tau / 10) + Vector2(5, -4) for i in range(10)]),
		([Vector2.from_polar(2, i * tau / 4 + pi / 4) + Vector2(-4, -2) for i in range(4)]),
		([Vector2.from_polar(2, i * tau / 30 + pi / 4) + Vector2(0, -8) for i in range(30)]),
	]
)

test3 = (
	Vector2(4, 1), 
	Vector2(7, 3),
	[
		([Vector2(3, 0), Vector2(1, 4), Vector2(-1, 1)]),
		([Vector2(2.5, 5.), Vector2(4.7, 5), Vector2(4, 6), Vector2(3, 6)]),
		([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)])
	]
)

tests = [
	test1,
	test2,
	test3,
]

export_test_cases(tests, 'test_cases.bin')

"""
{Vector2(5, 1), Vector2(2.518382, 1.926471), Vector2(5.233333, 5), Vector2(7, 3)}
{Vector2(-1, -1), Vector2(2.701298, 3.819644), Vector2(-2.337778, 2.607743), Vector2(3.381966, -2.824429), Vector2(-2.585786, -3.414214), Vector2(-0.517638, -6.068148), Vector2(1, -1)}
{Vector2(4, 1), Vector2(2.415789, 1.168421), Vector2(4.7, 5), Vector2(5, 5), Vector2(7, 3)}
{Vector2(0, 0), Vector2(0, 0)}

[(5, 1), (2.518382352941176, 1.9264705882352944), (5.233333333333333, 5.0), (7, 3)]
[(-1, -1), (2.70129832888558, 3.8196440245126055), (-2.337778030407033, 2.607742731253013), Vector2(3.381966011250105, -2.8244294954150533), Vector2(-2.585786437626905, -3.4142135623730954), Vector2(-0.5176380902050413, -6.068148347421864), (1, -1)]
[(4, 1), (2.415789473684211, 1.1684210526315788), Vector2(4.7, 5), Vector2(5, 5), (7, 3)]
"""

from u_tpp import tpp_solve

for test in tests:
	start, target, polygons = test
	path = tpp_solve(start, target, polygons) # type: ignore
	#print(path)

x = read_test_results('test_results.bin')
for length, path in x:
	print(length, path)