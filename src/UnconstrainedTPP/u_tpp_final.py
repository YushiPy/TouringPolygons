"""
First approach to solve the unconstrained TPP. 
This implementation is not optimized and may not be efficient for large inputs, 
but it serves as a proof of concept and a baseline for further improvements.

See report for details on the algorithm and its complexity analysis.
"""


type Vector2 = tuple[float, float]
type Polygon2 = Sequence[Vector2]


EPSILON = 1e-8

# Vector operations

def vector_add(v1: Vector2, v2: Vector2) -> Vector2:
	return (v1[0] + v2[0], v1[1] + v2[1])

def vector_sub(v1: Vector2, v2: Vector2) -> Vector2:
	return (v1[0] - v2[0], v1[1] - v2[1])

def vector_mul(v: Vector2, scalar: float) -> Vector2:
	return (v[0] * scalar, v[1] * scalar)

def vector_cross(v1: Vector2, v2: Vector2) -> float:
	return v1[0] * v2[1] - v1[1] * v2[0]

def vector_dot(v1: Vector2, v2: Vector2) -> float:
	return v1[0] * v2[0] + v1[1] * v2[1]

def vector_is_same_direction(v1: Vector2, v2: Vector2, eps: float = EPSILON) -> bool:

	cross = vector_cross(v1, v2)
	dot = vector_dot(v1, v2)

	return abs(cross) < eps ** 2 and dot > 0

def vector_is_close(v1: Vector2, v2: Vector2, eps: float = EPSILON) -> bool:
	return abs(v1[0] - v2[0]) < eps and abs(v1[1] - v2[1]) < eps

def vector_length(v: Vector2) -> float:
	return (v[0] ** 2 + v[1] ** 2) ** 0.5

def vector_normalize(v: Vector2) -> Vector2:

	length = vector_length(v)

	if length == 0:
		return (0.0, 0.0)

	return (v[0] / length, v[1] / length)

def vector_reflect(v: Vector2, normal: Vector2) -> Vector2:
	
	normal = vector_normalize(normal)
	dot = v[0] * normal[0] + v[1] * normal[1]

	return vector_sub(v, vector_mul(normal, 2 * dot))

def vector_perpendicular(v: Vector2) -> Vector2:
	return (-v[1], v[0])

def vector_reflect_segment(point: Vector2, start: Vector2, end: Vector2) -> Vector2:
	return vector_add(start, vector_reflect(vector_sub(point, start), vector_perpendicular(vector_sub(end, start))))

def segment_segment_intersection(start1: Vector2, end1: Vector2, start2: Vector2, end2: Vector2) -> Vector2 | None:
	"""
	Returns the intersection point of segments (start1, end1) and (start2, end2) if they intersect, otherwise returns None.
	"""

	direction1 = vector_sub(end1, start1)
	direction2 = vector_sub(end2, start2)

	cross = vector_cross(direction1, direction2)

	if cross == 0:
		return None

	sdiff = vector_sub(start2, start1)
	rate1 = vector_cross(sdiff, direction2) / cross
	rate2 = vector_cross(sdiff, direction1) / cross

	if 0 <= rate1 <= 1 and 0 <= rate2 <= 1:
		return vector_add(start1, vector_mul(direction1, rate1))
	
	return None


# Point location functions

def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
	"""
	Check if `point` is inside the cone defined by `vertex` and rays `ray1` and `ray2`. 
	The rays are in counter-clockwise order.
	"""

	if vector_cross(ray1, ray2) >= 0:
		return vector_cross(ray1, vector_sub(point, vertex)) >= 0 and vector_cross(ray2, vector_sub(point, vertex)) <= 0
	else:
		return vector_cross(ray1, vector_sub(point, vertex)) >= 0 or vector_cross(ray2, vector_sub(point, vertex)) <= 0

def point_in_edge(point: Vector2, vertex1: Vector2, ray1: Vector2, vertex2: Vector2, ray2: Vector2) -> bool:
	"""
	Check if `point` is inside the edge region defined by `ray1` coming from `vertex1` and `ray2` coming from `vertex2`.
	The rays and edge are in counter-clockwise order.
	"""
	return vector_cross(ray1, vector_sub(point, vertex1)) > 0 and vector_cross(ray2, vector_sub(point, vertex2)) < 0 and vector_cross(vector_sub(vertex2, vertex1), vector_sub(point, vertex1)) <= 0

from collections.abc import Sequence

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

			ray1 = vector_reflect(diff, vector_perpendicular(vector_sub(vertex, before)))
			ray2 = vector_reflect(diff, vector_perpendicular(vector_sub(vertex, after)))

			first_contact[i][j - 1] = vector_cross(diff, vector_sub(vertex, before)) < 0
			first_contact[i][j] = vector_cross(diff, vector_sub(after, vertex)) < 0

			if not first_contact[i][j - 1]:
				ray1 = diff

			if not first_contact[i][j]:
				ray2 = diff

			cones[i][j] = (ray1, ray2)

		return cones[i][j] # type: ignore

	def locate_point_naive(point: Vector2, i: int) -> int:
		"""
		Locate `point` in the shortest last step map of `polygon[i]` and return the index of the region as follows:
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
	
	def locate_point_binary_search(point: Vector2, i: int) -> int:
		"""
		Locates point in cones or edges defined by polygon and cones at the given index.
		Returns index as follows:
		- `2n` -> cone in vertex `n`
		- `2n + 1` -> edge between vertex `n` and `n + 1`

		:param Vector2 point: The point to locate.
		:param int index: The index of the polygon.

		:return: The located index.
		"""

		def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2, eps: float = EPSILON) -> bool:
			"""
			Check if a point is inside the cone defined by two rays originating from a vertex.

			:param Vector2 point: The point to check.
			:param Vector2 vertex: The vertex of the cone.
			:param Vector2 ray1: The first ray direction.
			:param Vector2 ray2: The second ray direction.
			:param float eps: A small epsilon value for numerical stability. Positive values expand the cone, negative values contract it.

			:return: True if the point is inside the cone, False otherwise.
			"""

			# Rays are almost parallel and in the same direction, treat as a single ray
			if vector_is_same_direction(ray1, ray2, eps):
				return vector_is_same_direction(vector_sub(point, vertex), ray1, eps)

			eps_squared = eps * eps

			if vector_cross(ray1, ray2) < -eps_squared:
				return vector_cross(ray1, vector_sub(point, vertex)) >= -eps_squared or vector_cross(ray2, vector_sub(point, vertex)) <= eps_squared
			else:
				return vector_cross(ray1, vector_sub(point, vertex)) >= -eps_squared and vector_cross(ray2, vector_sub(point, vertex)) <= eps_squared

		def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2, eps: float = EPSILON) -> bool:

			if vector_is_close(vertex1, vertex2):
				return point_in_cone(point, vertex1, ray1, ray2)

			p1 = vector_sub(point, vertex1)
			p2 = vector_sub(point, vertex2)
			dv = vector_sub(vertex2, vertex1)

			if vector_is_same_direction(ray1, dv, eps) or vector_is_same_direction(vector_mul(ray2, -1), dv, eps):
				return False

			eps_squared = eps * eps

			if vector_cross(dv, ray1) < eps_squared:
				if vector_cross(dv, ray2) < eps_squared:
					return vector_cross(ray1, p1) > -eps_squared and vector_cross(ray2, p2) < eps_squared and vector_cross(dv, p1) < eps_squared
				else:
					return vector_cross(ray1, p1) > -eps_squared if vector_cross(dv, p1) < eps_squared else vector_cross(ray2, p2) < eps_squared
			else:
				if vector_cross(dv, ray2) < eps_squared:
					return vector_cross(ray2, p2) < eps_squared if vector_cross(dv, p2) < eps_squared else vector_cross(ray1, p1) > -eps_squared
				else:
					return vector_cross(ray1, p1) > -eps_squared or vector_cross(ray2, p2) < eps_squared or vector_cross(dv, p1) < eps_squared

		def check(l: int, r: int) -> bool:

			v1 = polygon[l // 2]
			v2 = polygon[r // 2]
			ray1 = get_cone(i, l // 2)[l % 2]
			ray2 = get_cone(i, r // 2)[r % 2]
			
			return point_in_edge(point, v1, v2, ray1, ray2)
		
		polygon = polygons[i]
		n = len(polygon)

		left = 0
		right = 2 * n - 1
		
		if check(right, left):
			return right if first_contact[i][right // 2] else -1
		
		while left + 1 != right:

			mid = (left + right) // 2

			if check(left, mid):
				right = mid
			else:
				left = mid

		if not check(left, right):
			raise ValueError("Point is not located in any cone or edge.")
		
		if first_contact[i][(left - 1) // 2] or first_contact[i][left // 2]:
			return left
		else:
			return -1

	def locate_point(point: Vector2, i: int) -> int:
		"""
		Locate `point` in the shortest last step map of `polygon` and return the index of the region as follows:
		- `2n` if the point is in the region of vertex `n`
		- `2n + 1` if the point is between vertices `n` and `n + 1`.
		- `-1` if the point is in the pass through region.
		"""
	
		if len(polygons[i]) < 20:
			return locate_point_naive(point, i)
		else:
			return locate_point_binary_search(point, i)

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
