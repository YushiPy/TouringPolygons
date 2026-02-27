"""
Converts python tests to a binary file that is easily read by the C++ code. 
This is used to test the C++ implementation against the python implementation.
"""

from collections.abc import Sequence

from vector2 import Vector2


def point_to_bytes(point: tuple[float, float]) -> bytes:
	"""
	Converts a point to a binary format that can be read by the C++ code. 
	Each point is represented as two 64-bit floats (16 bytes).
	"""
	return Vector2(point).to_bytes()

def test_case_to_binary(start: tuple[float, float], target: tuple[float, float], polygons: list[list[tuple[float, float]]]) -> bytes:
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

	result.extend(point_to_bytes(start))
	result.extend(point_to_bytes(target))

	result.extend(len(polygons).to_bytes(8, byteorder='little'))

	for polygon in polygons:
		result.extend(len(polygon).to_bytes(8, byteorder='little'))
		for vertex in polygon:
			result.extend(point_to_bytes(vertex))
	
	return bytes(result)

def export_test_cases(test_cases: Sequence[tuple[tuple[float, float], tuple[float, float], Sequence[Sequence[tuple[float, float]]]]], filename: str) -> None:
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
		
	The function returns a list of tuples, where each tuple contains the time taken and the solution path as a list of `Vector2` objects.
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
