"""
Final solution to the Unconstrained Touring Polygons Problem (TPP).
This implementation creates a last step map for each polygon and uses it to efficiently locate the last step of the path to any point in the plane.
The point location is done using both a linear seach and a binary search, depending on the number of vertices in the polygon, to optimize performance.
Only the cones of visibility of the last step map of that are needed are constructed and then cached for future queries.

This solution is the same as common.tpp_solve_dynamic_jit, but hardcoding all functions, 
slightly improving performance by avoiding function call overhead 
and making this solution a standalone implementation that does not depend on the common module.
"""


type Vector2 = tuple[float, float]
type Polygon2 = Sequence[Vector2]

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

def vector_is_same_direction(v1: Vector2, v2: Vector2) -> bool:
	return vector_cross(v1, v2) == 0 and vector_dot(v1, v2) >= 0

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
	Check if a `point` is inside the cone defined by `vertex` and rays `ray1` and `ray2`.
	The rays are in counter-clockwise order. 

	This version is used for binary search and considers the case where `ray1` and `ray2` point in the same direction.
	"""

	# Rays are almost parallel and in the same direction, treat as a single ray
	#if vector_is_same_direction(ray1, ray2):
	#	return vector_is_same_direction(vector_sub(point, vertex), ray1)

	if vector_cross(ray1, ray2) < 0:
		return vector_cross(ray1, vector_sub(point, vertex)) >= 0 or vector_cross(ray2, vector_sub(point, vertex)) <= 0
	else:
		return vector_cross(ray1, vector_sub(point, vertex)) >= 0 and vector_cross(ray2, vector_sub(point, vertex)) <= 0

def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
	"""
	Check if `point` is inside the edge region defined by `ray1` coming from `vertex1` and `ray2` coming from `vertex2`.
	The rays and edge are in counter-clockwise order. 
	
	This version is used for binary search and considers all possible cases of ray directions.
	"""

	if vertex1 == vertex2:
		return point_in_cone(point, vertex1, ray1, ray2)

	dv = vector_sub(vertex2, vertex1)

	if vector_is_same_direction(ray1, dv) or vector_is_same_direction(vector_mul(ray2, -1), dv):
		return False

	p1 = vector_sub(point, vertex1)
	p2 = vector_sub(point, vertex2)
	
	rp1 = vector_cross(ray1, p1) >= 0
	rp2 = vector_cross(ray2, p2) <= 0
	dp = vector_cross(dv, p1) <= 0

	if vector_cross(dv, ray1) < 0:
		if vector_cross(dv, ray2) < 0:
			return rp1 and rp2 and dp
		else:
			return rp1 if dp else rp2
	else:
		if vector_cross(dv, ray2) < 0:
			return rp2 if dp else rp1
		else:
			return rp1 or rp2 or dp


# Cleanup functions

def remove_collinear_points(points: Sequence[Vector2]) -> list[Vector2]:
	"""
	Removes collinear points from a sequence of points.
	"""

	cleaned: list[Vector2] = [points[0], points[1]]

	for i in range(2, len(points)):

		a = cleaned[-2]
		b = cleaned[-1]
		candidate = points[i]

		v1 = vector_sub(b, a)
		v2 = vector_sub(candidate, b)

		if vector_is_same_direction(v1, v2):
			cleaned[-1] = candidate
		else:
			cleaned.append(candidate)

	return cleaned

def clean_polygon(polygon: Polygon2) -> list[Vector2]:
	"""
	Cleans a polygon by removing collinear points and making the vertices counter-clockwise.
	"""

	cleaned = remove_collinear_points(polygon)

	# Ensure counter-clockwise order
	if vector_cross(vector_sub(cleaned[1], cleaned[0]), vector_sub(cleaned[-1], cleaned[0])) < 0:
		cleaned.reverse()

	return cleaned


from collections.abc import Sequence

def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[tuple[float, float]]:
	"""
	Given a `start` point, a `target` point, and a list of **convex** polygons, returns the shortest path from `start` to `target` that visits each polygon at least once.
	"""

	if simplify:
		polygons = [clean_polygon(polygon) for polygon in polygons]

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

	def locate_point(point: Vector2, i: int) -> int:
		"""
		Locate `point` in the shortest last step map of `polygon[i]` and return the index of the region as follows:
		- `2n` if the point is in the region of vertex `n`
		- `2n + 1` if the point is between vertices `n` and `n + 1`.
		- `-1` if the point is in the pass through region.
		"""

		def check_vertex(j: int) -> bool:
			"""Checks if `point` is in the cone of vertex `j` of polygon `i`."""
			
			ray1, ray2 = get_cone(i, j)

			if visible[j] or visible[j - 1]:
				return point_in_cone(point, polygon[j], ray1, ray2)
			else:
				return False
			
		def check_edge(l: int, r: int) -> bool:
			"""Checks if `point` is in any of the regions of edges `l` to `r`, inclusive."""			

			v1 = polygon[l]
			v2 = polygon[r]
			ray1 = get_cone(i, l)[0]
			ray2 = get_cone(i, r)[0]

			return point_in_edge(point, v1, v2, ray1, ray2)

		polygon = polygons[i]
		visible = first_contact[i]

		left = 0
		right = len(polygon)
		
		while left + 1 != right:

			mid = (left + right) // 2

			if check_edge(left, mid):
				right = mid
			else:
				left = mid

		if check_vertex(left):
			return 2 * left
		else:
			return 2 * left + 1 if visible[left] else -1

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

	return remove_collinear_points(query_full(target, len(polygons)))
