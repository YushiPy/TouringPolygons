

from collections.abc import Sequence

from vector2 import Vector2
from tpp_bnb import tpp_solve

import json
import os
import math
import random

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

	filename = 'custom_tests.bin'
	instances = [
		'{"startPoint":[0.5072036691384626,1.164277814518649],"targetPoint":[1.1,0.2],"polygons":[[[1.5595287039847678,1.4066884029028874],[0.9957831496028189,1.732167050479761],[0.38693795087031335,1.6152742580242085],[0.9882665422110595,1.5701746136736525]],[[0.5297534913137407,0.13450260184762008],[0.3831796471744344,0.27731814229104723],[-0.11291644068168127,1.010187362987582],[0.3531132176073971,0.05933652793002675],[1.0521577050410138,-0.3164938416579395]],[[2.461521590995888,-0.023346153379325663],[2.0067668437944493,1.0177039703793411],[2.0969661324955613,0.41261707534271524],[2.0819329177120425,-0.1248203531680766]]],"polygonColors":["#FFFF00","#0000FF","#00FF00"],"currentPolygon":2,"currentPolygonVertex":0,"scrollSensitivity":0.0005,"snapping":false,"showVertexLine":false,"camera":{"position":[0.9582001126440229,0.6963690043816299],"unitsToPixels":266.07748625964655},"drawingName":"untitled"}',
		'{"startPoint":[0.6876022465406872,2.115128649576203],"targetPoint":[1.1235654752627282,-0.3503185749208565],"polygons":[[[1.5595287039847678,1.4066884029028874],[0.9957831496028189,1.732167050479761],[0.38693795087031335,1.6152742580242085],[0.9882665422110595,1.5701746136736525]],[[1.1235654752627278,0.5441577046985033],[0.9769916311234215,0.686973245141931],[0.48089554326730577,1.4198424658384683],[0.9469252015563843,0.46899163078091016],[1.6459696889900008,0.09316126119294393]],[[0.9582001126440262,-0.47810090058076504],[0.5034453654425843,0.5629492231779026],[0.5936446541436964,-0.04213767185872508],[0.5786114393601776,-0.579575100369516]]],"polygonColors":["#FFFF00","#0000FF","#00FF00"],"currentPolygon":1,"currentPolygonVertex":0,"scrollSensitivity":0.0005,"snapping":false,"showVertexLine":false,"camera":{"position":[1.6630849286769074,0.674516706838888],"unitsToPixels":245.04677238481347},"drawingName":"untitled"}',
		'{"startPoint":[0.4959608028758342,2.084451621049625],"targetPoint":[1.1774633518575797,-0.25795833772595544],"polygons":[[[1.5595287039847678,1.4066884029028874],[0.9957831496028189,1.732167050479761],[0.38693795087031335,1.6152742580242085],[0.9882665422110595,1.5701746136736525]],[[1.1235654752627278,0.5441577046985033],[0.9769916311234215,0.686973245141931],[0.48089554326730577,1.4198424658384683],[0.9469252015563843,0.46899163078091016],[1.6459696889900008,0.09316126119294393]],[[0.9582001126440262,-0.47810090058076504],[0.5034453654425843,0.5629492231779026],[0.5936446541436964,-0.04213767185872508],[0.5786114393601776,-0.579575100369516]]],"polygonColors":["#FFFF00","#0000FF","#00FF00"],"currentPolygon":1,"currentPolygonVertex":0,"scrollSensitivity":0.0005,"snapping":false,"showVertexLine":false,"camera":{"position":[1.6630849286769074,0.674516706838888],"unitsToPixels":245.04677238481347},"drawingName":"untitled"}',
	]

	test_cases = []

	for instance in instances:

		# Parse the instance from JSON format
		instance_data = json.loads(instance)

		start = tuple(instance_data['startPoint'])
		target = tuple(instance_data['targetPoint'])
		polygons = [ [tuple(vertex) for vertex in polygon] for polygon in instance_data['polygons'] ]

		solution = tpp_solve(start, target, polygons)

		test_cases.append((start, target, polygons, solution))

		continue

		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(1, 1, figsize=(8, 8))

		ax.scatter(*start, color='green', label='Start')
		ax.scatter(*target, color='red', label='Target')

		for poly in polygons:
			ax.fill(*zip(*poly), alpha=0.5, edgecolor='black')
		
		ax.plot(*zip(*solution), color='blue', label='Solution Path')
		ax.legend()
		plt.show()

	export_test_cases(test_cases, filename)
