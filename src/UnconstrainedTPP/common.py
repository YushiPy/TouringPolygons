"""
This file compiles all variations of solutions to the Unconstrained Polygons Problem into a single file.
Here, we define all possible vector operations, point location functions, queries and final resolution.
This allows for us to compactly create all six variations of solutions in a single file as such:

`tpp_solve_linear`: Uses linear search and computes all last step maps.
`tpp_solve_binary`: Uses binary search and computes all last step maps.
`tpp_solve_dynamic`: Uses dynamic search and computes all last step maps.
`tpp_solve_linear_jit`: Uses linear search and computes only cones of visibility that are needed.
`tpp_solve_binary_jit`: Uses binary search and computes only cones of visibility that are needed.
`tpp_solve_dynamic_jit`: Uses dynamic search and computes only cones of visibility that are needed.

We also have a `Solution` class that encapsulates the entire solution process, allowing us 
to easily inspect each part of the solution, such as the cones of visibility, first contact regions, and the final path.
"""


from collections.abc import Callable, Sequence


type Vector2 = tuple[float, float]
type Polygon2 = Sequence[Vector2]

# Magic number to switch between binary search and linear search in point location.
BINARY_SEARCH_THRESHOLD = 25


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
	return vector_cross(v1, v2) == 0 and vector_dot(v1, v2) > 0

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

def vector_reflect_ray(direction: Vector2, start: Vector2, end: Vector2) -> Vector2:
	"""Bounces `direction` on the line defined by `start` and `end`."""
	return vector_reflect(direction, vector_perpendicular(vector_sub(end, start)))


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


def point_in_cone_plus(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
	"""
	Check if a `point` is inside the cone defined by `vertex` and rays `ray1` and `ray2`.
	The rays are in counter-clockwise order. 

	This version is used for binary search and considers the case where `ray1` and `ray2` point in the same direction.
	"""

	# Rays are almost parallel and in the same direction, treat as a single ray
	if vector_is_same_direction(ray1, ray2):
		return vector_is_same_direction(vector_sub(point, vertex), ray1)

	if vector_cross(ray1, ray2) < 0:
		return vector_cross(ray1, vector_sub(point, vertex)) >= 0 or vector_cross(ray2, vector_sub(point, vertex)) <= 0
	else:
		return vector_cross(ray1, vector_sub(point, vertex)) >= 0 and vector_cross(ray2, vector_sub(point, vertex)) <= 0

def point_in_edge_plus(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
	"""
	Check if `point` is inside the edge region defined by `ray1` coming from `vertex1` and `ray2` coming from `vertex2`.
	The rays and edge are in counter-clockwise order. 
	
	This version is used for binary search and considers all possible cases of ray directions.
	"""

	if vertex1 == vertex2:
		return point_in_cone_plus(point, vertex1, ray1, ray2)

	dv = vector_sub(vertex2, vertex1)

	if vector_is_same_direction(ray1, dv) or vector_is_same_direction(vector_mul(ray2, -1), dv):
		return False

	p1 = vector_sub(point, vertex1)
	p2 = vector_sub(point, vertex2)
	
	if vector_cross(dv, ray1) < 0:
		if vector_cross(dv, ray2) < 0:
			return vector_cross(ray1, p1) >= 0 and vector_cross(ray2, p2) <= 0 and vector_cross(dv, p1) <= 0
		else:
			if vector_cross(dv, p1) < 0:
				return vector_cross(ray1, p1) >= 0
			else:
				return vector_cross(ray2, p2) <= 0
	else:
		if vector_cross(dv, ray2) < 0:
			if vector_cross(dv, p2) < 0:
				return vector_cross(ray2, p2) <= 0
			else:
				return vector_cross(ray1, p1) >= 0
		else:
			return vector_cross(ray1, p1) >= 0 or vector_cross(ray2, p2) <= 0 or vector_cross(dv, p1) <= 0


def locate_point_linear_search(point: Vector2, polygon: Polygon2, cones: Callable[[int], tuple[Vector2, Vector2]], first_contact: Callable[[int], bool]) -> int:
	"""
	Uses linear search to locate `point` in the visibility map of `polygon[i]` defined by `cones` and `first_contact`.
	Returns index as follows:
	- `2n` -> cone in vertex `n`
	- `2n + 1` -> edge between vertex `n` and `n + 1`
	- `-1` -> pass through region
	"""

	for j in range(len(polygon)):

		v = polygon[j]
		ray1, ray2 = cones(j)
		
		if not first_contact(j) and not first_contact(j - 1):
			continue

		if point_in_cone(point, v, ray1, ray2):
			return 2 * j

	for j in range(len(polygon)):

		v1 = polygon[j]
		v2 = polygon[(j + 1) % len(polygon)]

		ray1 = cones(j)[1]
		ray2 = cones((j + 1) % len(polygon))[0]
		
		if point_in_edge(point, v1, ray1, v2, ray2):
			return 2 * j + 1 if first_contact(j) else -1

	return -1

def locate_point_binary_search(point: Vector2, polygon: Polygon2, cones: Callable[[int], tuple[Vector2, Vector2]]) -> int:
	"""
	Uses binary search to locate `point` in the visibility map of `polygon[i]` defined by `cones`.
	Returns index as follows:
	- `2n` -> cone in vertex `n`
	- `2n + 1` -> edge between vertex `n` and `n + 1`

	The returned vertex or edge may not be in the first contact region, 
	so the caller should check for that and return -1 if it's not in the first contact region.
	"""

	def check(l: int, r: int) -> bool:

		v1 = polygon[l // 2]
		v2 = polygon[r // 2]
		ray1 = cones(l // 2)[l % 2]
		ray2 = cones(r // 2)[r % 2]

		return point_in_edge_plus(point, v1, v2, ray1, ray2)
	
	left = 0
	right = 2 * len(polygon) - 1
	
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

def locate_point(point: Vector2, polygon: Polygon2, cones: Callable[[int], tuple[Vector2, Vector2]], first_contact: Callable[[int], bool], *, binary_search: bool = True) -> int:
	"""
	Locates `point` in the last step map of `polygon`, defined by `cones` and `first_contact`.
	Toggles between linear search and binary search based on the `binary_search` parameter.

	:param `Vector2 (tuple[float, float])` point: The point to locate.
	:param `Polygon2 (Sequence[Vector2])` polygon: The polygon to locate the point in.
	:param `Callable[[int], tuple[Vector2, Vector2]]` cones: A function that takes a vertex index and returns the two rays defining the cone at that vertex.
	:param `Callable[[int], bool]` first_contact: A function that takes a vertex index and returns whether that vertex is in the first contact region.
	:param `bool` binary_search: Whether to use binary search or linear search for point location. Binary search is faster for large polygons, but linear search may be faster for small polygons due to lower constant factors.

	:return: The located index as follows:
	- `2n` -> cone in vertex `n`
	- `2n + 1` -> edge between vertex `n` and `n + 1`
	- `-1` -> pass through region
	"""

	if not binary_search:
		return locate_point_linear_search(point, polygon, cones, first_contact)
	
	result = locate_point_binary_search(point, polygon, cones)

	if first_contact(result // 2) or first_contact((result - 1) // 2 % len(polygon)):
		return result
	else:
		return -1


def _locate_point_binary(point: Vector2, polygon: Polygon2, cones: Callable[[int], tuple[Vector2, Vector2]], first_contact: Callable[[int], bool]) -> int:
	return locate_point(point, polygon, cones, first_contact, binary_search=True)

def _locate_point_dynamic(point: Vector2, polygon: Polygon2, cones: Callable[[int], tuple[Vector2, Vector2]], first_contact: Callable[[int], bool], *, binary_search_threshold: int = BINARY_SEARCH_THRESHOLD) -> int:
	return locate_point(point, polygon, cones, first_contact, binary_search=len(polygon) > binary_search_threshold)


def query_full(point: Vector2, i: int, start: Vector2, polygons: Sequence[Polygon2], _locate_point: Callable[[Vector2, int], int]) -> list[Vector2]:
	"""
	Returns the shortest path from `start` to `point` that visits polygons `P_1, ..., P_i`.
	
	Uses the `locate_point` function to find the last step of the path and then recursively queries for the rest of the path.
	"""

	if i == 0:
		return [start, point]
	
	polygon = polygons[i - 1]
	location = _locate_point(point, i - 1)

	if location == -1:
		return query_full(point, i - 1, start, polygons, _locate_point)

	pos = location // 2

	if location % 2 == 0:
		return query_full(polygon[pos], i - 1, start, polygons, _locate_point) + [point]

	v1, v2 = polygon[pos], polygon[(pos + 1) % len(polygon)]

	reflected = vector_reflect_segment(point, v1, v2)
	path = query_full(reflected, i - 1, start, polygons, _locate_point)
	last = path[-2]

	intersection = segment_segment_intersection(last, reflected, v1, v2)

	if intersection is None:
		raise ValueError(f"Intersection not found for point {point} in polygon {i} at edge {pos}")

	return path[:-1] + [intersection, point]

def query(point: Vector2, i: int, start: Vector2, polygons: Sequence[Polygon2], _locate_point: Callable[[Vector2, int], int]) -> Vector2:
	"""
	Returns the last step of the shortest path from `start` to `point` that visits polygons `P_1, ..., P_i`.
	
	Uses the `locate_point` function to find the last step of the path and then recursively queries for the rest of the path.
	"""

	if i == 0:
		return start
	
	polygon = polygons[i - 1]
	location = _locate_point(point, i - 1)

	if location == -1:
		return query(point, i - 1, start, polygons, _locate_point)

	pos = location // 2

	if location % 2 == 0:
		return polygon[pos]

	v1, v2 = polygon[pos], polygon[(pos + 1) % len(polygon)]

	reflected = vector_reflect_segment(point, v1, v2)
	last = query(reflected, i - 1, start, polygons, _locate_point)

	intersection = segment_segment_intersection(last, reflected, v1, v2)

	if intersection is None:
		raise ValueError(f"Intersection not found for point {point} in polygon {i} at edge {pos}")

	return intersection


type LocationFunction = Callable[[Vector2, Polygon2, Callable[[int], tuple[Vector2, Vector2]], Callable[[int], bool]], int]

def tpp_solve_normal(start: Vector2, target: Vector2, polygons: Sequence[Polygon2], location_function: LocationFunction, *, simplify: bool = False) -> list[Vector2]:

	if simplify:
		polygons = [clean_polygon(polygon) for polygon in polygons]

	def _locate_point(point: Vector2, index: int) -> int:
		return location_function(point, polygons[index], cones[index].__getitem__, first_contact[index].__getitem__)

	def _query_full(point: Vector2, i: int) -> list[Vector2]:
		return query_full(point, i, start, polygons, _locate_point)

	def _query(point: Vector2, i: int) -> Vector2:
		return query(point, i, start, polygons, _locate_point)

	def get_first_contact_region(i: int) -> list[bool]:
		"""
		Returns the first contact region of polygon `i`.
		"""

		result = []
		polygon = polygons[i - 1]

		for j in range(len(polygon)):

			v1 = polygon[j]
			v2 = polygon[(j + 1) % len(polygon)]
			last = _query(v1, i - 1)

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
			
			last = _query(vertex, i - 1)
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

	return _query_full(target, len(polygons))

def tpp_solve_jit(start: Vector2, target: Vector2, polygons: Sequence[Polygon2], location_function: LocationFunction, *, simplify: bool = False) -> list[Vector2]:

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

			last = _query(vertex, i)
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

	def _locate_point(point: Vector2, index: int) -> int:
		return location_function(point, polygons[index], lambda j: get_cone(index, j), first_contact[index].__getitem__)

	def _query_full(point: Vector2, i: int) -> list[Vector2]:
		return query_full(point, i, start, polygons, _locate_point)

	def _query(point: Vector2, i: int) -> Vector2:
		return query(point, i, start, polygons, _locate_point)

	cones: list[list[tuple[Vector2, Vector2] | None]] = [[None] * len(polygon) for polygon in polygons]
	first_contact: list[list[bool]] = [[False] * len(polygon) for polygon in polygons]

	return _query_full(target, len(polygons))


def tpp_solve_linear(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[Vector2]: 
	return tpp_solve_normal(start, target, polygons, locate_point_linear_search, simplify=simplify)

def tpp_solve_binary(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[Vector2]:
	return tpp_solve_normal(start, target, polygons, _locate_point_binary, simplify=simplify)

def tpp_solve_dynamic(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[Vector2]:
	return tpp_solve_normal(start, target, polygons, _locate_point_dynamic, simplify=simplify)

def tpp_solve_linear_jit(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[Vector2]: 
	return tpp_solve_jit(start, target, polygons, locate_point_linear_search, simplify=simplify)

def tpp_solve_binary_jit(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[Vector2]:
	return tpp_solve_jit(start, target, polygons, _locate_point_binary, simplify=simplify)

def tpp_solve_dynamic_jit(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[Vector2]:
	return tpp_solve_jit(start, target, polygons, _locate_point_dynamic, simplify=simplify)

#return tpp_solve_normal(start, target, polygons, locate_point_linear_search, simplify=simplify)

# Geometry functions

def segment_segment_intersection(start1: Vector2, end1: Vector2, start2: Vector2, end2: Vector2) -> Vector2 | None:
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

	if cross == 0:
		return None
	
	sdiff = vector_sub(start2, start1)

	rate1 = vector_cross(sdiff, diff2) / cross
	rate2 = vector_cross(sdiff, diff1) / cross

	if 0 <= rate1 <= 1 and 0 <= rate2 <= 1:
		return (start1[0] + rate1 * diff1[0], start1[1] + rate1 * diff1[1])
	
	return None

def clean_polygon(polygon: Polygon2) -> Polygon2:
	"""
	Cleans a polygon by removing collinear points and making the vertices counter-clockwise.
	"""

	cleaned: Polygon2 = []

	n = len(polygon)

	for i in range(n):

		prev = polygon[i - 1]
		curr = polygon[i]
		next = polygon[(i + 1) % n]

		v1 = vector_sub(curr, prev)
		v2 = vector_sub(next, curr)

		if vector_cross(v1, v2) == 0:
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


# Solution class
class Solution:

	start: tuple[float, float]
	target: tuple[float, float]
	polygons: Sequence[Sequence[tuple[float, float]]]

	cones: list[list[tuple[Vector2, Vector2]]]
	first_contact: list[list[bool]]
	path: list[Vector2]

	def __init__(self, start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> None:

		self.start = start
		self.target = target
		self.polygons = polygons
		self.cones = []
		self.first_contact = []

		if simplify:
			polygons = [clean_polygon(polygon) for polygon in polygons]

		def _locate_point(point: Vector2, index: int) -> int:
			return locate_point(point, polygons[index], self.cones[index].__getitem__, self.first_contact[index].__getitem__, binary_search=True)

		def _query_full(point: Vector2, i: int) -> list[Vector2]:
			return query_full(point, i, start, polygons, _locate_point)

		def _query(point: Vector2, i: int) -> Vector2:
			return query(point, i, start, polygons, _locate_point)

		def get_first_contact_region(i: int) -> list[bool]:
			"""
			Returns the first contact region of polygon `i`.
			"""

			result = []
			polygon = polygons[i - 1]

			for j in range(len(polygon)):

				v1 = polygon[j]
				v2 = polygon[(j + 1) % len(polygon)]
				last = _query(v1, i - 1)

				result.append(vector_cross(vector_sub(v2, v1), vector_sub(last, v1)) < 0)

			return result

		def get_last_step_map(i: int) -> list[tuple[Vector2, Vector2]]:
			"""
			Returns the last step map of polygon `i`.
			"""

			result = []
			polygon = polygons[i - 1]
			_first_contact = self.first_contact[i - 1]

			for j in range(len(polygon)):

				before = polygon[j - 1]
				vertex = polygon[j]
				after = polygon[(j + 1) % len(polygon)]
				
				last = _query(vertex, i - 1)
				diff = vector_sub(vertex, last)

				ray1 = vector_reflect_ray(diff, vertex, before)
				ray2 = vector_reflect_ray(diff, vertex, after)

				if not _first_contact[j - 1]:
					ray1 = diff

				if not _first_contact[j]:
					ray2 = diff

				result.append((ray1, ray2))

			return result

		for i in range(1, len(polygons) + 1):
			self.first_contact.append(get_first_contact_region(i))
			self.cones.append(get_last_step_map(i))

		self.path = _query_full(target, len(polygons))
