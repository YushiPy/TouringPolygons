"""
First approach to solve the unconstrained TPP. 
This implementation is not optimized and may not be efficient for large inputs, 
but it serves as a proof of concept and a baseline for further improvements.

See report for details on the algorithm and its complexity analysis.
"""

from collections.abc import Sequence

import common
from common import vector_cross, vector_reflect_ray, vector_sub


type Vector2 = tuple[float, float]

# Magic number to switch between binary search and linear search in point location.
BINARY_SEARCH_THRESHOLD = 25

def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]]) -> list[tuple[float, float]]:

	def get_cone(i: int, j: int) -> tuple[Vector2, Vector2]:
		"""
		Get the cone of visibility for vertex `j` of polygon `i`.
		Caches the result for future queries.
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

	def locate_point(point: Vector2, index: int) -> int:
		return common.locate_point(point, polygons[index], lambda j: get_cone(index, j), first_contact[index].__getitem__, binary_search=len(polygons[index]) >= BINARY_SEARCH_THRESHOLD)

	def query_full(point: Vector2, i: int) -> list[Vector2]:
		return common.query_full(point, i, start, polygons, locate_point)

	def query(point: Vector2, i: int) -> Vector2:
		return common.query(point, i, start, polygons, locate_point)

	cones: list[list[tuple[Vector2, Vector2] | None]] = [[None] * len(polygon) for polygon in polygons]
	first_contact: list[list[bool]] = [[False] * len(polygon) for polygon in polygons]

	return query_full(target, len(polygons))
