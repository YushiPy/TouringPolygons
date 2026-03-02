"""
First approach to solve the unconstrained TPP. 
This implementation is not optimized and may not be efficient for large inputs, 
but it serves as a proof of concept and a baseline for further improvements.

See report for details on the algorithm and its complexity analysis.
"""

from collections.abc import Sequence
from common import point_in_cone, point_in_edge, segment_segment_intersection, vector_cross, vector_reflect_ray, vector_reflect_segment, vector_sub

type Vector2 = tuple[float, float]


def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]]) -> list[tuple[float, float]]:

	def get_cone(i: int, j: int) -> tuple[Vector2, Vector2]:
		"""
		Get the cone for vertex `j` of polygon `i`.
		"""

		if cones[i][j] is None:

			before = polygons[i][j - 1]
			vertex = polygons[i][j]
			after = polygons[i][(j + 1) % len(polygons[i])]

			last = query(vertex, i)
			diff = vector_sub(vertex, last)

			ray1 = vector_reflect_ray(diff, vertex, before)
			ray2 = vector_reflect_ray(diff, vertex, after)

			first_contact[i][j - 1] = vector_cross(diff, vector_sub(vertex, before)) < 0
			first_contact[i][j] = vector_cross(diff, vector_sub(after, vertex)) < 0

			if not first_contact[i][j - 1]:
				ray1 = diff

			if not first_contact[i][j]:
				ray2 = diff

			cones[i][j] = (ray1, ray2)

		return cones[i][j] # type: ignore

	def locate_point(point: Vector2, i: int) -> int:
		"""
		Locate `point` in the shortest last step map of `polygon` and return the index of the region as follows:
		- `2n` if the point is in the region of vertex `n`
		- `2n + 1` if the point is between vertices `n` and `n + 1`.
		- `-1` if the point is in the pass through region.
		"""

		polygon = polygons[i]

		for j in range(len(polygon)):

			v = polygon[j]
			ray1, ray2 = get_cone(i, j)
			
			if not first_contact[i][j] and not first_contact[i][j - 1]:
				continue

			if point_in_cone(point, v, ray1, ray2):
				return 2 * j

		for j in range(len(polygon)):

			v1 = polygon[j]
			v2 = polygon[(j + 1) % len(polygon)]

			ray1 = get_cone(i, j)[1]
			ray2 = get_cone(i, (j + 1) % len(polygon))[0]
			
			if point_in_edge(point, v1, ray1, v2, ray2):
				return 2 * j + 1 if first_contact[i][j] else -1

		return -1

	def query_full(point: Vector2, i: int) -> list[Vector2]:
		"""
		Returns the `i`-path to `point`.
		"""

		if i == 0:
			return [start, point]
		
		polygon = polygons[i - 1]
		location = locate_point(point, i - 1)

		if location == -1:
			return query_full(point, i - 1)

		pos = location // 2

		if location % 2 == 0:
			return query_full(polygon[pos], i - 1) + [point]

		v1, v2 = polygon[pos], polygon[(pos + 1) % len(polygon)]

		reflected = vector_reflect_segment(point, v1, v2)
		path = query_full(reflected, i - 1)
		last = path[-2]

		intersection = segment_segment_intersection(last, reflected, v1, v2)

		if intersection is None:
			raise ValueError(f"Intersection not found for point {point} in polygon {i} at edge {pos}")

		return path[:-1] + [intersection, point]

	def query(point: Vector2, i: int) -> Vector2:
		"""
		Returns the last step of the `i`-path to `point`.
		"""

		if i == 0:
			return start
		
		polygon = polygons[i - 1]
		location = locate_point(point, i - 1)

		if location == -1:
			return query(point, i - 1)

		pos = location // 2

		if location % 2 == 0:
			return polygon[pos]

		v1, v2 = polygon[pos], polygon[(pos + 1) % len(polygon)]

		reflected = vector_reflect_segment(point, v1, v2)
		last = query(reflected, i - 1)

		intersection = segment_segment_intersection(last, reflected, v1, v2)

		if intersection is None:
			raise ValueError(f"Intersection not found for point {point} in polygon {i} at edge {pos}")

		return intersection

	cones: list[list[tuple[Vector2, Vector2] | None]] = [[None] * len(polygon) for polygon in polygons]
	first_contact: list[list[bool]] = [[False] * len(polygon) for polygon in polygons]

	return query_full(target, len(polygons))
