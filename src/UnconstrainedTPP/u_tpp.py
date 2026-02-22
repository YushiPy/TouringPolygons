
from collections.abc import Sequence


type Vector2 = tuple[float, float]
type Polygon2 = Sequence[Vector2]


EPSILON = 1e-10

def vector_add(v1: Vector2, v2: Vector2) -> Vector2:
	return (v1[0] + v2[0], v1[1] + v2[1])

def vector_sub(v1: Vector2, v2: Vector2) -> Vector2:
	return (v1[0] - v2[0], v1[1] - v2[1])

def vector_mul(v: Vector2, scalar: float) -> Vector2:
	return (v[0] * scalar, v[1] * scalar)

def vector_cross(v1: Vector2, v2: Vector2) -> float:
	return v1[0] * v2[1] - v1[1] * v2[0]

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
	#(point - start).reflect((end - start).perpendicular()) + start
	return vector_add(start, vector_reflect(vector_sub(point, start), vector_perpendicular(vector_sub(end, start))))


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

	if vector_cross(ray1, ray2) < -eps:
		return vector_cross(ray1, vector_sub(point, vertex)) >= -eps or vector_cross(ray2, vector_sub(point, vertex)) <= eps
	else:
		return vector_cross(ray1, vector_sub(point, vertex)) >= -eps and vector_cross(ray2, vector_sub(point, vertex)) <= eps

def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2, eps: float = EPSILON) -> bool:

	if vector_is_close(vertex1, vertex2):
		return point_in_cone(point, vertex1, ray1, ray2)

	p1 = vector_sub(point, vertex1)
	p2 = vector_sub(point, vertex2)
	dv = vector_sub(vertex2, vertex1)

	if vector_is_close(ray1, ray2):
		return False
		return vector_cross(dv, p1) >= -eps and vector_cross(dv, p2) <= eps

	if vector_cross(dv, ray1) < eps:
		if vector_cross(dv, ray2) < eps:
			return vector_cross(ray1, p1) > -eps and vector_cross(ray2, p2) < eps and vector_cross(dv, p1) < eps
		else:
			return vector_cross(ray1, p1) > -eps if vector_cross(dv, p1) < eps else vector_cross(ray2, p2) < eps
	else:
		if vector_cross(dv, ray2) < eps:
			return vector_cross(ray2, p2) < eps if vector_cross(dv, p2) < eps else vector_cross(ray1, p1) > -eps
		else:
			return vector_cross(ray1, p1) > -eps or vector_cross(ray2, p2) < eps or vector_cross(dv, p1) < eps


def segment_segment_intersection(start1: Vector2, end1: Vector2, start2: Vector2, end2: Vector2, eps: float = EPSILON) -> Vector2 | None:
	"""
	Returns the intersection point of two line segments if they intersect, otherwise returns None.

	:param Vector2 start1: The start point of the first segment as a Vector2.
	:param Vector2 end1: The end point of the first segment as a Vector2.
	:param Vector2 start2: The start point of the second segment as a Vector2.
	:param Vector2 end2: The end point of the second segment as a Vector2.

	:return: The intersection point as a Vector2 if the segments intersect, otherwise None.	
	"""

	diff1 = vector_sub(end1, start1)
	diff2 = vector_sub(end2, start2)

	cross = vector_cross(diff1, diff2)

	if abs(cross) < eps:
		return None
	
	sdiff = vector_sub(start2, start1)

	rate1 = vector_cross(sdiff, diff2) / cross
	rate2 = vector_cross(sdiff, diff1) / cross

	if -eps <= rate1 <= 1 + eps and -eps <= rate2 <= 1 + eps:
		return (start1[0] + rate1 * diff1[0], start1[1] + rate1 * diff1[1])
	
	return None


def clean_polygon(polygon: Polygon2, eps: float = EPSILON) -> Polygon2:
	"""
	Cleans a polygon by removing collinear points and making the vertices counter-clockwise.

	:param Polygon2 polygon: The polygon to clean.
	:param float eps: A small epsilon value for numerical stability.

	:return: The cleaned polygon.
	"""

	cleaned: Polygon2 = []

	n = len(polygon)

	for i in range(n):

		prev = polygon[i - 1]
		curr = polygon[i]
		next = polygon[(i + 1) % n]

		v1 = vector_sub(curr, prev)
		v2 = vector_sub(next, curr)

		if abs(vector_cross(v1, v2)) > eps ** 2:
			cleaned.append(curr)

	# Ensure counter-clockwise order
	area = 0.0

	for i in range(len(cleaned)):
		v1 = cleaned[i]
		v2 = cleaned[(i + 1) % len(cleaned)]
		area += (v1[0] * v2[1] - v2[0] * v1[1])

	if area < 0:
		cleaned.reverse()

	return cleaned

def tpp_solve(start: Sequence[float], target: Sequence[float], polygons: Sequence[Polygon2], *, simplify: bool = True) -> list[Vector2]:

	start = (start[0], start[1])
	target = (target[0], target[1])

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
		diff = vector_normalize(vector_sub(vertex, last))

		if vector_cross(diff, vector_sub(vertex, before)) > 0:
			ray1 = diff
		else:
			ray1 = vector_reflect(diff, vector_perpendicular(vector_sub(before, vertex)))
		
		if vector_cross(diff, vector_sub(after, vertex)) > 0:
			ray2 = diff
		else:
			ray2 = vector_reflect(diff, vector_perpendicular(vector_sub(after, vertex)))

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
			return point_in_edge(point, polygon[l // 2], polygon[r // 2], get(l // 2)[l % 2], get(r // 2)[r % 2])
		
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
