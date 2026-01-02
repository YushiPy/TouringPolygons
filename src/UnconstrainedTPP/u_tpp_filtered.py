
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

	if ray1.cross(ray2) > 0:
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

	filtered: list[list[Vector2]]
	cones: list[list[tuple[Vector2, Vector2]]]
	
	def __init__(self, start: Vector2, target: Vector2, polygons: list[Polygon2]) -> None:

		self.start = start
		self.target = target
		self.polygons = polygons
	
		self.cones = []
		self.filtered = []

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

		filtered = self.filtered[index - 1]
		cones = self.cones[index - 1]

		n = len(cones)

		# Check if in the pass through region
		if point_in_edge(point, filtered[-1], filtered[0], cones[-1][1], cones[0][0]):
			return 2 * n - 1
		
		if point_in_cone(point, filtered[0], *cones[0]):
			return 0
		
		if point_in_cone(point, filtered[-1], *cones[-1]):
			return 2 * (n - 1)
		
		left = 0
		right = len(cones) - 1

		while left + 1 != right:

			mid = (left + right) // 2

			if point_in_cone(point, filtered[mid], *cones[mid]):
				return 2 * mid
			
			if point_in_edge(point, filtered[left], filtered[mid], cones[left][1], cones[mid][0]):
				right = mid
			else:
				left = mid

		if point_in_edge(point, filtered[left], filtered[right], cones[left][1], cones[right][0]):
			return 2 * left + 1
		else:
			raise ValueError("Point not located in any cone or edge, this should not happen.")

	def get_filtered(self, index: int) -> list[Vector2]:
		"""
		Filter the vertices of the polygon at the given index based on the last path segment.

		:param int index: The index of the polygon.
		
		:return: A list of filtered vertices.
		"""
		
		polygon = list(self.polygons[index - 1])

		start = -1
		end = -1

		for i in range(len(polygon)):

			if start != -1 and end != -1:
				break

			before = polygon[i - 1]
			v = polygon[i]
			after = polygon[(i + 1) % len(polygon)]
			
			last = self.query(v, index - 1)
			diff = v - last

			if start == -1 and diff.cross(after - v) < 1e-8 and diff.cross(v - before) > -1e-8:
				start = i
			
			if end == -1 and diff.cross(after - v) > -1e-8 and diff.cross(v - before) < 1e-8:
				end = i

		if start < end:
			return polygon[start:end + 1]
		else:
			return polygon[start:] + polygon[:end + 1]

	def get_cones(self, index: int) -> list[tuple[Vector2, Vector2]]:
		"""
		Compute the cones for each filtered vertex of the polygon at the given index.

		:param int index: The index of the polygon.

		:return: A list of tuples representing the cones (ray1, ray2) for each filtered vertex.
		"""

		filtered = self.filtered[index - 1]
		cones: list[tuple[Vector2, Vector2]] = []

		for i in range(len(filtered)):

			vertex = filtered[i]
			before = filtered[i - 1]
			after = filtered[(i + 1) % len(filtered)]

			last = self.query(vertex, index - 1)
			diff = vertex - last

			ray1 = diff.reflect((before - vertex).perpendicular()) if i else diff
			ray2 = diff.reflect((after - vertex).perpendicular()) if i != len(filtered) - 1 else diff

			cones.append((ray1, ray2))
		
		return cones

	def query_full(self, point: Vector2, index: int) -> tuple[Vector2, int]:

		if index == 0:
			return self.start, -1
		
		location = self.locate_point(point, index)

		if location % 2 == 0:
			return self.filtered[index - 1][location // 2], index - 1
		
		if location == 2 * len(self.cones[index - 1]) - 1:
			return self.query_full(point, index - 1)
		
		v1 = self.filtered[index - 1][location // 2]
		v2 = self.filtered[index - 1][(location // 2) + 1]

		reflected = point.reflect_segment(v1, v2)
		last, _ = self.query_full(reflected, index - 1)

		intersection = segment_segment_intersection(last, reflected, v1, v2)

		if intersection is None:
			raise ValueError("No intersection found, this should not happen.")
		
		return intersection, index - 1

	def query(self, point: Vector2, index: int) -> Vector2:
		return self.query_full(point, index)[0]

	def get_full_path(self, point: Vector2, index: int) -> list[Vector2]:

		result = [point]

		while index >= 0:
			point, index = self.query_full(point, index)
			result.append(point)
		
		result.reverse()

		return result

	def solve(self) -> list[Vector2]:
		"""
		Solve the problem and return the path from start to target.

		:return: A list of Vector2 points representing the path.
		"""

		k = len(self.polygons)

		if k == 0:
			return [self.start, self.target]
		
		self.cones = []
		self.filtered = []

		for i in range(1, k + 1):
			self.filtered.append(self.get_filtered(i))
			self.cones.append(self.get_cones(i))

		return self.get_full_path(self.target, k)


def tpp_solve(start: Vector2, target: Vector2, polygons: list[Polygon2]) -> list[Vector2]:
	return Solution(start, target, polygons).solve()
