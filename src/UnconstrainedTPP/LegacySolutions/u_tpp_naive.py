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


def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]]) -> list[tuple[float, float]]:

	def locate_point(point: Vector2, index: int) -> int:
		return common.locate_point(point, polygons[index], cones[index].__getitem__, first_contact[index].__getitem__, binary_search=False)

	def query_full(point: Vector2, i: int) -> list[Vector2]:
		return common.query_full(point, i, start, polygons, locate_point)

	def query(point: Vector2, i: int) -> Vector2:
		return common.query(point, i, start, polygons, locate_point)

	def get_first_contact_region(i: int) -> list[bool]:
		"""
		Returns the first contact region of polygon `i`.
		"""

		result = []
		polygon = polygons[i - 1]

		for j in range(len(polygon)):

			v1 = polygon[j]
			v2 = polygon[(j + 1) % len(polygon)]
			last = query(v1, i - 1)

			result.append(vector_cross(vector_sub(v2, v1), vector_sub(last, v1)) < 0)

		return result

	def get_last_step_map(i: int) -> list[tuple[Vector2, Vector2]]:
		"""
		Returns the last step map of polygon `i`.
		"""

		result = []
		polygon = polygons[i - 1]
		_first_contact = first_contact[i - 1]

		for j in range(len(polygon)):

			before = polygon[j - 1]
			vertex = polygon[j]
			after = polygon[(j + 1) % len(polygon)]
			
			last = query(vertex, i - 1)
			diff = vector_sub(vertex, last)

			ray1 = vector_reflect_ray(diff, vertex, before)
			ray2 = vector_reflect_ray(diff, vertex, after)

			if not _first_contact[j - 1]:
				ray1 = diff

			if not _first_contact[j]:
				ray2 = diff

			result.append((ray1, ray2))

		return result

	first_contact: list[list[bool]] = []
	cones: list[list[tuple[Vector2, Vector2]]] = []

	for i in range(1, len(polygons) + 1):
		first_contact.append(get_first_contact_region(i))
		cones.append(get_last_step_map(i))

	return query_full(target, len(polygons))
