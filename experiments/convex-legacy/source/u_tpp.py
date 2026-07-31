
from collections.abc import Sequence
from common import clean_polygon, point_in_edge_plus, segment_segment_intersection, vector_cross, vector_reflect_ray, vector_reflect_segment, vector_sub


type Vector2 = tuple[float, float]

def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[Vector2]:

	if simplify:
		polygons = [clean_polygon(polygon) for polygon in polygons]

	cones: list[list[tuple[Vector2, Vector2] | None]] = [[None] * len(polygon) for polygon in polygons]
	blocked_edges: list[list[bool | None]] = [[None] * len(polygon) for polygon in polygons]

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
	
	def is_blocked_edge(i: int, j: int) -> bool:

		if blocked_edges[i][j] is None:

			polygon = polygons[i]
			v1 = polygon[j]
			v2 = polygon[(j + 1) % len(polygon)]

			last = query(v1, i)
			blocked_edges[i][j] = vector_cross(vector_sub(v2, v1), vector_sub(last, v1)) >= 0

		return blocked_edges[i][j] # type: ignore

	def locate_point(point: Vector2, index: int) -> int:
		"""
		Locates point in cones or edges defined by polygon and cones at the given index.
		Returns index as follows:
		- `2n` -> cone in vertex `n`
		- `2n + 1` -> edge between vertex `n` and `n + 1`

		:param Vector2 point: The point to locate.
		:param int index: The index of the polygon.

		:return: The located index.
		"""

		def get(i: int) -> tuple[Vector2, Vector2]:
			return get_cone(index, i)

		def check(l: int, r: int) -> bool:

			v1 = polygon[l // 2]
			v2 = polygon[r // 2]
			ray1 = get(l // 2)[l % 2]
			ray2 = get(r // 2)[r % 2]
			
			return point_in_edge_plus(point, v1, v2, ray1, ray2)
		
		polygon = polygons[index]
		n = len(polygon)

		left = 0
		right = 2 * n - 1

		if check(right, left):
			return right

		while left + 1 != right:

			mid = (left + right) // 2

			if check(left, mid):
				right = mid
			else:
				left = mid

		if not check(left, right):
			raise ValueError("Point is not located in any cone or edge.")
		
		return left
	
	def query(point: Vector2, index: int) -> Vector2:

		if index == 0:
			return start

		polygon = polygons[index - 1]

		location = locate_point(point, index - 1)
		ind = location // 2

		# Check for vertex region
		if location % 2 == 0:
			return polygon[ind]

		v1 = polygon[ind]
		v2 = polygon[(ind + 1) % len(polygon)]

		# Check if point is in pass through region
		if is_blocked_edge(index - 1, ind):
			return query(point, index - 1)

		# Point is in edge region, need to reflect and find intersection
		reflected = vector_reflect_segment(point, v1, v2)
		last = query(reflected, index - 1)

		intersection = segment_segment_intersection(last, reflected, v1, v2)
	
		if intersection is None:
			raise ValueError("No intersection found between segments.")

		return intersection
	
	def query_full(point: Vector2, index: int) -> list[Vector2]:

		if index == 0:
			return [start, point]

		polygon = polygons[index - 1]
		location = locate_point(point, index - 1)
		ind = location // 2

		# Check for vertex region
		if location % 2 == 0:
			x = query_full(polygon[ind], index - 1)
			x.append(point)
			return x

		# Check if point is in pass through region
		if is_blocked_edge(index - 1, ind):
			return query_full(point, index - 1)

		# Point is in edge region, need to reflect and find intersection
		v1 = polygon[ind]
		v2 = polygon[(ind + 1) % len(polygon)]

		reflected = vector_reflect_segment(point, v1, v2)
		other = query_full(reflected, index - 1)
		last = other[-2]
		intersection = segment_segment_intersection(last, reflected, v1, v2)

		if intersection is None:
			raise ValueError("No intersection found between segments.")

		# return other[:-1] + [intersection, point]
		other.pop()
		other.append(intersection)
		other.append(point)

		return other

	return query_full(target, len(polygons))
