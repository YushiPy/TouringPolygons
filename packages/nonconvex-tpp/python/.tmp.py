
from collections.abc import Sequence

from vector2 import Vector2

import json
import re
import os
import math
import random

def parse_json(filepath: str) -> list[list[Vector2]]:

	with open(filepath, 'r') as f:
		data = json.load(f)

	polygon_data = data['polygons']
	polygons: list[list[Vector2]] = []

	for string in polygon_data:
		
		values = string.strip("POLYGON() ").replace(",", "").split(" ")

		if len(values) % 2 != 0:
			raise ValueError(f"Invalid polygon data in file {filepath}: {string}")

		points = [Vector2(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]
		polygon = points[:-1]

		polygons.append(polygon)

	return polygons

def fix_polygons(polygons: list[list[Vector2]]) -> list[list[Vector2]]:

	n = len(polygons)
	height = math.isqrt(n)
	width = (n + height - 1) // height

	fixed_polygons: list[list[Vector2]] = []
	left = 0
	bottom = 0

	max_h = 0

	for i, polygon in enumerate(polygons):

		minx = min(point.x for point in polygon)
		maxx = max(point.x for point in polygon)
		miny = min(point.y for point in polygon)
		maxy = max(point.y for point in polygon)

		w = maxx - minx
		h = maxy - miny
		max_h = max(max_h, h)

		fixed_polygon = [Vector2(point.x - minx + left + w / 10, point.y - miny + bottom + h / 10) for point in polygon]

		left += w + w / 10
		fixed_polygons.append(fixed_polygon)

		if (i % width == width - 1):
			left = 0
			bottom += max_h + max_h / 10
			max_h = 0

	random.shuffle(fixed_polygons)

	return fixed_polygons


def add_points(polygons: list[list[Vector2]]) -> tuple[Vector2, Vector2, list[list[Vector2]]]:

	min_x = min(point.x for polygon in polygons for point in polygon)
	max_x = max(point.x for polygon in polygons for point in polygon)
	min_y = min(point.y for polygon in polygons for point in polygon)
	max_y = max(point.y for polygon in polygons for point in polygon)

	bottom_left = Vector2(min_x, min_y)
	top_right = Vector2(max_x, max_y)

	return bottom_left, top_right, polygons


def point_to_bytes(point: tuple[float, float]) -> bytes:
	"""
	Converts a point to a binary format that can be read by the C++ code. 
	Each point is represented as two 64-bit floats (16 bytes).
	"""
	import struct
	return struct.pack('<d', point[0]) + struct.pack('<d', point[1])

def int_to_bytes(value: int) -> bytes:
	"""
	Converts an integer to a binary format that can be read by the C++ code. 
	The integer is represented as a 64-bit integer (8 bytes).
	"""
	return value.to_bytes(8, byteorder='little')

def test_case_to_binary(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], solution: Sequence[tuple[float, float]]) -> bytes:
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
	- `8` bytes: Size of solution
	- For each `vertex` in solution:
		- `16` bytes: `vertex`
	"""

	result = bytearray()

	result.extend(point_to_bytes(start))
	result.extend(point_to_bytes(target))

	result.extend(int_to_bytes(len(polygons)))

	for polygon in polygons:
		result.extend(int_to_bytes(len(polygon)))
		for vertex in polygon:
			result.extend(point_to_bytes(vertex))
	
	result.extend(int_to_bytes(len(solution)))

	for vertex in solution:
		result.extend(point_to_bytes(vertex))
	
	return bytes(result)

def export_test_cases(test_cases: Sequence[tuple[tuple[float, float], tuple[float, float], Sequence[Sequence[tuple[float, float]]], Sequence[tuple[float, float]]]], filename: str) -> None:
	"""
	Exports a list of test cases to a binary file. 
	Each test case is converted to binary format using the `test_case_to_binary` function and written to the file sequentially.
	The format of the file is as follows:

	- For each test case:
		- `16` bytes: `start`
		- `16` bytes: `target`
		- `8` bytes: number of polygons (`k`)
		- For each polygon `P_i`:
			- `8` bytes: number of vertices (`|P_i|`)
			- For each `vertex`:
				- `16` bytes: `vertex`
		- `8` bytes: Size of solution
		- For each `vertex` in solution:
			- `16` bytes: `vertex`
	"""

	with open(filename, 'wb') as f:
		for start, target, polygons, solution in test_cases:
			f.write(test_case_to_binary(start, target, polygons, solution))


if __name__ == '__main__':

	folder_path = 'instances_simplified/'

	test_cases = []

	for filename in os.listdir(folder_path):

		if not filename.endswith('.json'):
			continue

		filepath = os.path.join(folder_path, filename)
		polygons = parse_json(filepath)
		polygons = fix_polygons(polygons)
		bottom_left, top_right, polygons = add_points(polygons)

		test_cases.append((bottom_left, top_right, polygons, []))

	export_test_cases(test_cases, 'test_cases_simplified2.bin')
