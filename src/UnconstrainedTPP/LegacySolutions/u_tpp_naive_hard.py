"""
First approach to solve the unconstrained TPP. 
This implementation is not optimized and may not be efficient for large inputs, 
but it serves as a proof of concept and a baseline for further improvements.

See report for details on the algorithm and its complexity analysis.
"""


from common import point_in_cone, point_in_edge, segment_segment_intersection, vector_cross, vector_reflect_ray, vector_reflect_segment, vector_sub


type Vector2 = tuple[float, float]
type Polygon2 = Sequence[Vector2]

def locate_point(point: Vector2, polygon: Polygon2, cones: list[tuple[Vector2, Vector2]], first_contact: list[bool]) -> int:
	"""
	Locate `point` in the shortest last step map of `polygon` and return the index of the region as follows:
	- `2n` if the point is in the region of vertex `n`
	- `2n + 1` if the point is between vertices `n` and `n + 1`.
	- `-1` if the point is in the pass through region.
	"""

	for j in range(len(polygon)):

		if not first_contact[j] and not first_contact[j - 1]:
			continue

		v = polygon[j]
		ray1, ray2 = cones[j]

		if point_in_cone(point, v, ray1, ray2):
			return 2 * j

	for j in range(len(polygon)):

		if not first_contact[j]:
			continue

		v1 = polygon[j]
		v2 = polygon[(j + 1) % len(polygon)]

		ray1 = cones[j][1]
		ray2 = cones[(j + 1) % len(cones)][0]

		if point_in_edge(point, v1, ray1, v2, ray2):
			return 2 * j + 1

	return -1


from collections.abc import Sequence

def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]]) -> list[tuple[float, float]]:

	def query_full(point: Vector2, i: int) -> list[Vector2]:
		"""
		Returns the `i`-path to `point`.
		"""

		if i == 0:
			return [start, point]
		
		polygon = polygons[i - 1]
		location = locate_point(point, polygon, cones[i - 1], first_contact[i - 1])

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
		location = locate_point(point, polygon, cones[i - 1], first_contact[i - 1])

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
