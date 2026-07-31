
# print deprecation warning
import warnings
warnings.warn("This module is outdated, please update.", DeprecationWarning, stacklevel=1)
"""
We need to update the correctness check:
- If the path crosses an edge, we must update the currect segment we are checking againt. Otherwise, we might accept incorrect solutions where a segment visits polygon 2 then 1 and we will consider that it visits 1 then 2.
"""

from collections.abc import Sequence
import math

from LegacySolutions.vector2 import Vector2


def segment_segment_intersection(p1: Vector2, p2: Vector2, p3: Vector2, p4: Vector2, eps: float = 1e-8) -> bool:

	denominator = (p2.x - p1.x) * (p4.y - p3.y) - (p2.y - p1.y) * (p4.x - p3.x)

	if denominator == 0:
		return False

	t = ((p3.x - p1.x) * (p4.y - p3.y) - (p3.y - p1.y) * (p4.x - p3.x)) / denominator
	u = ((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / denominator

	return -eps <= t <= 1 + eps and -eps <= u <= 1 + eps

def segment_segment_intersection_point(p1: Vector2, p2: Vector2, p3: Vector2, p4: Vector2, eps: float = 1e-8) -> Vector2 | None:
	
	denominator = (p2.x - p1.x) * (p4.y - p3.y) - (p2.y - p1.y) * (p4.x - p3.x)

	if denominator == 0:
		return None

	t = ((p3.x - p1.x) * (p4.y - p3.y) - (p3.y - p1.y) * (p4.x - p3.x)) / denominator
	u = ((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / denominator

	if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
		return Vector2(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))
	else:
		return None

def segment_polygon_intersection(p1: Vector2, p2: Vector2, polygon: Sequence[Vector2]) -> bool:

	for i in range(len(polygon)):
		if segment_segment_intersection(p1, p2, polygon[i], polygon[(i + 1) % len(polygon)]):
			return True

	return False


def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
	"""
	Check if a `point` is inside the cone defined by `vertex` and rays `ray1` and `ray2`.
	The rays are in counter-clockwise order. 

	This version is used for binary search and considers the case where `ray1` and `ray2` point in the same direction.
	"""

	# Rays are almost parallel and in the same direction, treat as a single ray
	if ray1.is_same_direction(ray2):
		return ray1.is_same_direction(point - vertex)
	
	if ray1.cross(ray2) < 0:
		return ray1.cross(point - vertex) >= 0 or ray2.cross(point - vertex) <= 0
	else:
		return ray1.cross(point - vertex) >= 0 and ray2.cross(point - vertex) <= 0


def is_correct_solution(start: Vector2 | tuple[float, float], target: Vector2 | tuple[float, float], polygons: Sequence[Sequence[Vector2 | tuple[float, float]]], solution: Sequence[Vector2 | tuple[float, float]]) -> str:
	"""
	Checks if the given solution is correct for the given start, target, and polygons. 
	If not, returns a string describing the reason why the solution is not correct. 
	If the solution is correct, returns an empty string.
	
	Some reasons why a solution might not be correct include:
	- The solution does not start at the start point or end at the target point.
	- The solution does not visit all the polygons in order.
	- The visit to a polygon vertex is not optimal, meaning that the path leaves the polygon at a suboptimal angle.
	- The visit to a polygon edge is not optimal, meaning that the path leaves the polygon at an angle which does not correspond to a reflection at that edge.
	"""

	start = Vector2(start)
	target = Vector2(target)
	polygons = [list(map(Vector2, polygon)) for polygon in polygons]
	solution = list(map(Vector2, solution))

	return _is_correct_solution(start, target, polygons, solution) # type: ignore

def _is_correct_solution(start: Vector2, target: Vector2, polygons: Sequence[Sequence[Vector2]], solution: list[Vector2]) -> str:
	"""
	Checks if the given solution is correct for the given start, target, and polygons. 
	If not, returns a string describing the reason why the solution is not correct. 
	If the solution is correct, returns an empty string.
	
	Some reasons why a solution might not be correct include:
	- The solution does not start at the start point or end at the target point.
	- The solution does not visit all the polygons in order.
	- The visit to a polygon vertex is not optimal, meaning that the path leaves the polygon at a suboptimal angle.
	- The visit to a polygon edge is not optimal, meaning that the path leaves the polygon at an angle which does not correspond to a reflection at that edge.
	"""

	if len(solution) < 2:
		return "LengthError: The solution must have at least 2 points (start and target)."

	# Check if the solution starts at the start point and ends at the target point
	if not (solution[0] == start and solution[-1] == target):
		return "StartEndError: The solution must start at the start point and end at the target point."

	if len(solution) > 2:

		for i in range(len(solution) - 2):

			p1, p2, p3 = solution[i], solution[i + 1], solution[i + 2]
			d1 = p2 - p1
			d2 = p3 - p2

			if d1.is_same_direction(d2):
				return f"OptimalityError: The solution has a unnecessary intermediate point at index {i + 1}."

	# Check if the solution visits all the polygons in order
	polygon_index = 0
	path_index = 1

	while polygon_index < len(polygons) and path_index < len(solution):

		polygon = polygons[polygon_index]
		
		new_point = solution[path_index]
		previous_point = solution[path_index - 1]

		if segment_polygon_intersection(previous_point, new_point, polygon):
			polygon_index += 1
		else:
			path_index += 1

	# Check if all polygons were visited
	if polygon_index < len(polygons):
		return f"VisitationError: The solution does not visit polygon {polygon_index}."
	
	polygon_index = 0
	path_index = 1
	visited = False

	while polygon_index < len(polygons) and path_index < len(solution):

		polygon = polygons[polygon_index]
		
		new_point = solution[path_index]
		previous_point = solution[path_index - 1]

		for i in range(len(polygon)):
			
			v1 = polygon[i]
			v2 = polygon[(i + 1) % len(polygon)]

			# Path visits polygon at vertex
			if new_point.is_close(v1):

				visited = True
				
				# The target is at the vertex, which shouldn't happen, 
				# but we let the algorithm continue to check the rest of the path.
				if path_index + 1 == len(solution):
					polygon_index += 1
					continue
				
				v = v1
				v_before = polygon[i - 1]
				v_after = polygon[(i + 1) % len(polygon)]

				last = solution[path_index - 1]
				diff = (v - last)

				ray1 = diff.reflect((v - v_before).perpendicular())
				ray2 = diff.reflect((v - v_after).perpendicular())
				
				visible1 = diff.cross(v - v_before) < 0
				visible2 = diff.cross(v - v_after) > 0

				if not visible1:
					ray1 = diff
				if not visible2:
					ray2 = diff

				next_point = solution[path_index + 1]

				if not point_in_cone(next_point, v, ray1, ray2):
					return f"VertexError: The solution visits polygon {polygon_index} at vertex {i}, but leaves the vertex at a suboptimal angle."

				polygon_index += 1
				break
			
			# This case will be handled when the path visits the next vertex, which will be v1
			elif new_point.is_close(v2):
				continue 
			
			# Path visits polygon at edge
			else:

				if not segment_segment_intersection(previous_point, new_point, v1, v2):
					continue

				visited = True

				if abs((new_point - v1).cross(v2 - v1)) > 1e-8:
					polygon_index += 1
					continue

				# The target is at the edge, which shouldn't happen, 
				# but we let the algorithm continue to check the rest of the path.
				if path_index + 1 == len(solution):
					polygon_index += 1
					continue

				next_point = solution[path_index + 1]
				reflected = next_point.reflect_segment(v1, v2)

				d1 = new_point - previous_point
				d2 = reflected - new_point
				
				if not d1.is_same_direction(d2):
					return f"EdgeError: The solution visits polygon {polygon_index} at edge {i}, but leaves the edge without following the reflection rule."

				polygon_index += 1
				break
		else:
			if not visited:
				return f"OptimalityError: Bends can only occur at the polygon's edge, point {path_index} is not optimal."
			else:
				path_index += 1
				visited = False

	return ""

if __name__ == "__main__":
	
	basic_tests = [
		((0.0, 1.0), (2.0, 1.0), [[(2.0, 3.0), (-1.0, 2.0), (2.0, 2.0)]], [(0.0, 1.0), (1.0, 2.0), (2.0, 1.0)]),
		((0.0, 1.0), (3.0, 2.0), [[(2.0, 3.0), (-1.0, 2.0), (2.0, 2.0)]], [(0.0, 1.0), (2.0, 2.0), (3.0, 2.0)]),
		((0.0, 1.0), (0.0, 3.0), [[(2.0, 3.0), (-1.0, 2.0), (2.0, 2.0)]], [(0.0, 1.0), (0.0, 3.0)]),
		((0.0, 0.0), (-2.0, 1.0), [[(-1.0, 2.0), (0.7745461995542708, 1.0969540776602265), (2.0, 2.0), (2.0, 3.0), (0.01042765904450249, 3.997485680411619)]], [(0.0, 0.0), (-0.3877793180272373, 1.688448015291771), (-2.0, 1.0)]),
		((-1.0, 0.5), (3.0, 0.5), [[(-1.0, 2.0), (1.0, 1.0), (3.0, 2.0), (1.0, 3.0)]], [(-1.0, 0.5), (1.0, 1.0), (3.0, 0.5)]),
		((-1.0, 0.5), (3.0, 0.5), [[(-1.0, 2.0), (1.0, 1.0), (3.0, 2.0), (1.0, 3.0)], [(2.0, -2.5), (0.0, -2.0), (0.0, -4.0)]], [(-1.0, 0.5), (0.44179104477611975, 1.2791044776119402), (0.9670014347202298, -2.241750358680058), (3.0, 0.5)]),
		((4.0, 1.5), (-1.5, -1.0), [[(-1.0, 2.0), (1.0, 1.0), (3.0, 2.0), (1.0, 3.0)], [(2.0, -2.0), (0.0, -2.0), (1.0, -4.0)]], [(4.0, 1.5), (2.2857142857142856, 1.6428571428571428), (0.0, -2.0), (-1.5, -1.0)]),
		((-2.0, 0.5), (-1.0, 5.5), [[(-1.0, 2.0), (1.0, 1.0), (3.0, 2.0), (1.0, 3.0)], [(2.0, -2.0), (0.0, -2.0), (1.0, -4.0)], [(-5.5892251770067904, -2.0635401226162013), (-4.0, -2.5), (-3.0, -1.5), (-4.0, 1.0), (-5.5, 1.5)], [(-0.5, 3.5), (-3.5, 5.5), (-3.0, 2.5)]], [(-2.0, 0.5), (-0.25454545454545463, 1.6272727272727272), (0.0, -2.0), (-3.417422867513612, -0.45644283121597096), (-1.0, 5.5)])
	]

	edge_cases = [
		((-2.0, 0.0), (2.0, 0.0), [[(-1.0, 1.0), (1.0, 1.0), (0.0, 2.0)]], [(-2.0, 0.0), (0.0, 1.0), (2.0, 0.0)]),
		((-1.0, 5.0), (-0.5, 4.5), [[(0.0, 4.0), (0.0, 2.0), (2.0, 2.0)]], [(-1.0, 5.0), (0.0, 4.0), (-0.5, 4.5)]),
		((-0.6, 5.2), (-0.6, 4.8), [[(0.0, 5.0), (0.0, 2.0), (2.8, 2.0)]], [(-0.6, 5.2), (0.0, 5.0), (-0.6, 4.8)]),
		((4.5, 2.0), (3.5, 2.0), [[(0.0, 5.0), (0.0, 2.0), (2.8, 2.0)]], [(4.5, 2.0), (2.8, 2.0), (3.5, 2.0)]),
		((0.0, 4.0), (4.0, 0.0), [[(3.0, 3.0), (1.0, 3.0), (1.0, 1.0), (3.0, 1.0)]], [(0.0, 4.0), (4.0, 0.0)]),
		((-1.7926052592116688, 2.0041878360341565), (-1.2727245469438635, -2.1926283895019223), [[(-2.8043080390114095, -3.654186702016027), (-1.7541858929206575, -5.033836318424196), (-1.0844353498351025, -3.4345790546288875)], [(2.0975681598044416, 2.0395046385241002), (3.094175815858267, 3.1866802982459888), (1.9470001561363786, 4.183287954299814), (0.950392500082553, 3.0361122945779258)]], [(-1.7926052592116688, 2.0041878360341565), (-1.0844353498351025, -3.4345790546288875), (2.0975681598044416, 2.0395046385241002), (-1.2727245469438635, -2.1926283895019223)]),
		((-1.0, 0.0), (-1.0, 2.8000000000000003), [[(-1.0, 3.0), (-1.0, 2.0), (2.0, 2.0), (0.2, 3.0)]], [(-1.0, 0.0), (-1.0, 2.8000000000000003)]),
		((-1.0, 1.0), (-1.0, 2.6), [[(-1.0, 3.0), (-1.0, 2.0), (2.0, 2.0), (2.0, 3.0)]], [(-1.0, 1.0), (-1.0, 2.6)]),
		((-2.0, 2.0), (16.0, 2.0), [[(2.0, 2.0), (0.0, 4.0), (0.0, 2.0)], [(5.0, 2.0), (3.0, 4.0), (3.0, 2.0)], [(8.0, 2.0), (6.0, 4.0), (6.0, 2.0)], [(11.0, 2.0), (9.0, 4.0), (9.0, 2.0)], [(14.0, 2.0), (12.0, 4.0), (12.0, 2.0)]], [(-2.0, 2.0), (16.0, 2.0)]),
		((0.0, 0.0), (0.0, 1.2000000000000002), [[(0.0, 3.0), (-1.0, 2.0), (1.0, 2.0)]], [(0.0, 0.0), (0.0, 2.0), (0.0, 1.2000000000000002)]),
	]

	tests = basic_tests + edge_cases

	def random_tests() -> None:

		for start, target, polygons, expected in tests:

			expected = list(map(Vector2, expected))

			import random

			if len(expected) <= 2:
				continue

			index = random.randint(1, len(expected) - 2)

			if random.random() < 0.5:
				vertex = None

				while True:
					vertex = Vector2(random.choice(polygons[index - 1]))

					if vertex.is_close(expected[index]):
						continue

					break

				expected[index] = vertex
			else:
				edge = random.randrange(len(polygons[index - 1]))
				v1 = Vector2(polygons[index - 1][edge])
				v2 = Vector2(polygons[index - 1][(edge + 1) % len(polygons[index - 1])])

				scale = random.random()
				p = v1.lerp(v2, scale)

				if p.is_close(expected[index]):
					continue
				else:
					expected[index] = p

			result = is_correct_solution(Vector2(start), Vector2(target), [list(map(Vector2, polys)) for polys in polygons], expected)

			if result != "":
				print(f"Test failed: {result}")
			else:
				print("Test passed.")

	start = Vector2(-1, 0.5)
	target = Vector2(3, 0.5)
	polygons = [
		[Vector2(-1, 2), Vector2(1, 1), Vector2(3, 2), Vector2(1, 3)],
		[Vector2(2, -2.5), Vector2(0, -2), Vector2(0, -4)],
	]

	result = [Vector2(-1, 0.5), Vector2(0.15714285714285725, 1.4214285714285713), Vector2(0, -4), Vector2(3, 0.5)]
	expected = [Vector2(-1, 0.5), Vector2(0.44179104477611975, 1.2791044776119402), Vector2(0.9670014347202298, -2.241750358680058), Vector2(3, 0.5)]

	expected = result
	#start, target, polygons, expected = tests[5]

	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(1, 1, figsize=(6, 6))

	ax.scatter(*start, color="green", zorder=5)
	ax.scatter(*target, color="red", zorder=5)

	for polygon in polygons:
		ax.fill(*zip(*polygon), alpha=0.5, edgecolor="black")

	ax.plot(*zip(*expected), color="blue", marker="o")

	i = 1

	tested_path = [
		start,
		expected[1],
		expected[2],
		target,
	]

	ax.plot(*zip(*tested_path), color="orange", marker="o")

	result = is_correct_solution(Vector2(start), Vector2(target), [list(map(Vector2, polys)) for polys in polygons], list(map(Vector2, tested_path)))

	if result == "":
		print("The solution is correct.")
	else:
		print(result)