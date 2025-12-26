
from vector2 import Vector2
from polygon2 import Polygon2


def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2, eps: float = 1e-8) -> bool:
	"""
	Check if a point is inside the cone defined by two rays originating from a vertex.

	:param Vector2 point: The point to check.
	:param Vector2 vertex: The vertex of the cone.
	:param Vector2 ray1: The first ray direction.
	:param Vector2 ray2: The second ray direction.
	:param float eps: A small epsilon value for numerical stability. Positive values expand the cone, negative values contract it.

	:return: True if the point is inside the cone, False otherwise.
	"""

	diff = point - vertex

	if ray1.cross(ray2) >= 0:
		return ray1.cross(diff) >= -eps and ray2.cross(diff) <= eps
	else:
		return ray1.cross(diff) >= -eps or ray2.cross(diff) <= eps

def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2, eps: float = 1e-8) -> bool:

	if vertex1.is_close(vertex2):
		return point_in_cone(point, vertex1, ray1, ray2)

	p1 = point - vertex1
	p2 = point - vertex2
	dv = vertex2 - vertex1

	if ray1.is_close(ray2):
		return dv.cross(p1) >= -eps and dv.cross(p2) <= eps

	match (dv.cross(ray1) >= -eps, dv.cross(ray2) >= -eps):

		case (True, True):
			return ray2.cross(p2) < eps or ray1.cross(p1) > -eps or dv.cross(p1) < -eps

		case (False, False):
			return ray1.cross(p1) >= -eps and ray2.cross(p2) <= eps and dv.cross(p1) <= eps

		case (True, False):
			return point_in_cone(point, vertex1, ray1, vertex1 - vertex2) or point_in_cone(point, vertex2, vertex1 - vertex2, ray2, eps)

		case (False, True):
			return point_in_cone(point, vertex1, ray1, vertex2 - vertex1) or point_in_cone(point, vertex2, vertex2 - vertex1, ray2, eps)

def segment_segment_intersection(start1: Vector2, end1: Vector2, start2: Vector2, end2: Vector2, eps: float = 1e-8) -> Vector2 | None:
	"""
	Returns the intersection point of two line segments if they intersect, otherwise returns None.

	:param Vector2 start1: The start point of the first segment as a Vector2.
	:param Vector2 end1: The end point of the first segment as a Vector2.
	:param Vector2 start2: The start point of the second segment as a Vector2.
	:param Vector2 end2: The end point of the second segment as a Vector2.

	:return: The intersection point as a Vector2 if the segments intersect, otherwise None.	
	"""

	diff1 = end1 - start1
	diff2 = end2 - start2

	cross = diff1.cross(diff2)

	if abs(cross) < eps:
		return None
	
	sdiff = start2 - start1

	rate1 = sdiff.cross(diff2) / cross
	rate2 = sdiff.cross(diff1) / cross

	if -eps <= rate1 <= 1 + eps and -eps <= rate2 <= 1 + eps:
		return start1 + diff1 * rate1
	
	return None


class Solution:

	start: Vector2
	target: Vector2

	polygons: list[Polygon2]
	cones: list[list[tuple[Vector2, Vector2] | None]]

	blocked_edges: list[list[bool | None]]

	def __init__(self, start: Vector2, target: Vector2, polygons: list[Polygon2]) -> None:

		self.polygons = polygons

		self.start = start
		self.target = target

		self.cones = [[None] * len(polygon) for polygon in polygons]
		self.blocked_edges = [[None] * len(polygon) for polygon in polygons]

	def get_cone(self, i: int, j: int) -> tuple[Vector2, Vector2]:
		"""
		Returns the visibility cone at vertex j of polygon at index i.

		:param int i: The polygon index.
		:param int j: The vertex index.

		:return: A tuple containing the two ray directions defining the cone.
		"""

		j = j % len(self.polygons[i])

		if self.cones[i][j] is not None:
			return self.cones[i][j] # type: ignore
		
		polygon = self.polygons[i]

		vertex = polygon[j]
		before = polygon[j - 1]
		after = polygon[(j + 1) % len(polygon)]

		last = self.query(vertex, i)
		diff = (vertex - last).normalize()

		if diff.cross(vertex - before) > 0:
			ray1 = diff
		else:
			ray1 = diff.reflect((before - vertex).perpendicular())
		
		if diff.cross(after - vertex) > 0:
			ray2 = diff
		else:
			ray2 = diff.reflect((after - vertex).perpendicular())

		self.cones[i][j] = (ray1, ray2)

		return ray1, ray2

	def is_blocked_edge(self, i: int, j: int) -> bool:

		if self.blocked_edges[i][j] is not None:
			return True
		
		polygon = self.polygons[i]
		v1 = polygon[j]
		v2 = polygon[(j + 1) % len(polygon)]

		mid = v1.lerp(v2, 0.5)
		last, _ = self.query_full(mid, i)

		self.blocked_edges[i][j] = (v2 - v1).cross(last - mid) >= 0

		return self.blocked_edges[i][j] # type: ignore


	def locate_point(self, point: Vector2, index: int) -> int:
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
			return self.get_cone(index, i)
		
		def check(i: int) -> bool:
			return point_in_edge(point, polygon[i // 2], polygon[(i + 1) // 2], get(i // 2)[i % 2], get((i + 1) // 2)[(i + 1) % 2])

		def check2(l: int, r: int) -> bool:
			return point_in_edge(point, polygon[l // 2], polygon[r // 2], get(l // 2)[l % 2], get(r // 2)[r % 2])

		polygon = self.polygons[index - 1]
		n = len(polygon)

		if check2(0, 1):
			return 0
		
		left = 0
		right = 2 * n - 1

		while left + 1 != right:

			mid = (left + right) // 2

			if check2(left, mid + 1):
				right = mid
			else:
				left = mid

		if not check(right):
			raise ValueError("Point not located in any cone or edge.")
		
		return right

	def query_full(self, point: Vector2, index: int) -> tuple[Vector2, int]:

		if index == 0:
			return self.start, -1
		
		polygon = self.polygons[index - 1]

		location = self.locate_point(point, index - 1)
		ind = location // 2

		# Check for vertex region
		if location % 2 == 0:
			return polygon[ind], index - 1

		v1 = polygon[ind]
		v2 = polygon[(ind + 1) % len(polygon)]

		# Check if point is in pass through region
		if self.is_blocked_edge(index - 1, ind):
			return self.query_full(point, index - 1)

		reflected = point.reflect_segment(v1, v2)
		last, _ = self.query_full(reflected, index - 1)

		intersection = segment_segment_intersection(last, reflected, v1, v2)
	
		if intersection is None:
			raise ValueError("No intersection found between segments.")

		return intersection, index - 1

	def query(self, point: Vector2, index: int) -> Vector2:
		return self.query_full(point, index)[0]
	
	def solve(self) -> list[Vector2]:

		result = [self.target]
		current = self.target
		index = len(self.polygons)

		while index >= 0:
			current, index = self.query_full(current, index)
			result.append(current)

		result.reverse()

		return result
	

x = (
	Vector2(5, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(2, 4), Vector2(1, 4), Vector2(-1, 1)]),
		Polygon2([Vector2(3, 3), Vector2(4, 3), Vector2(4, 4), Vector2(3, 4)]),
		Polygon2([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)]),
	]
)
s = Solution(*x)

print(s.solve())
