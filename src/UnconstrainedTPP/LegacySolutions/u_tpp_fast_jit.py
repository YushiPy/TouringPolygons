
from collections.abc import Sequence

import common
from common import clean_polygon, vector_cross, vector_reflect_ray, vector_sub


type Vector2 = tuple[float, float]

def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[Vector2]:

	def get_cone(i: int, j: int) -> tuple[Vector2, Vector2]:
		"""
		Returns the visibility cone at vertex j of polygon at index i.

		:param int i: The polygon index.
		:param int j: The vertex index.

		:return: A tuple containing the two ray directions defining the cone.
		"""

		j = j % len(polygons[i])

		if cones[i][j] is not None:
			return cones[i][j] # type: ignore
		
		polygon = polygons[i]

		vertex = polygon[j]
		before = polygon[j - 1]
		after = polygon[(j + 1) % len(polygon)]
		
		last = query(vertex, i)
		diff = vector_sub(vertex, last)

		if vector_cross(diff, vector_sub(vertex, before)) > 0:
			ray1 = diff
		else:
			ray1 = vector_reflect_ray(diff, vertex, before)
		
		if vector_cross(diff, vector_sub(after, vertex)) > 0:
			ray2 = diff
		else:
			ray2 = vector_reflect_ray(diff, vertex, after)

		cones[i][j] = (ray1, ray2)

		return ray1, ray2
	
	def is_first_contact(i: int, j: int) -> bool:

		if first_contact[i][j] is None:

			polygon = polygons[i]
			v1 = polygon[j]
			v2 = polygon[(j + 1) % len(polygon)]

			last = query(v1, i)
			first_contact[i][j] = vector_cross(vector_sub(v2, v1), vector_sub(last, v1)) < 0

		return first_contact[i][j] # type: ignore

	def locate_point(point: Vector2, index: int) -> int:
		return common.locate_point(point, polygons[index], lambda j: get_cone(index, j), lambda j: is_first_contact(index, j), binary_search=True)

	def query(point: Vector2, index: int) -> Vector2:
		return common.query(point, index, start, polygons, locate_point)

	def query_full(point: Vector2, index: int) -> list[Vector2]:
		return common.query_full(point, index, start, polygons, locate_point)

	if simplify:
		polygons = [clean_polygon(polygon) for polygon in polygons]

	cones: list[list[tuple[Vector2, Vector2] | None]] = [[None] * len(polygon) for polygon in polygons]
	first_contact: list[list[bool | None]] = [[None] * len(polygon) for polygon in polygons]


	return query_full(target, len(polygons))
