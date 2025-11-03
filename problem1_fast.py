
from collections.abc import Iterable

from vector2 import Vector2
from polygon2 import Polygon2


type Cones = list[tuple[Vector2, Vector2]]

type _Vector2 = Iterable[float]
type _Polygon2 = Iterable[_Vector2]


def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2) -> bool:

	if vertex1.is_close(vertex2):
		return point_in_cone(point, vertex1, ray1, ray2)

	p1 = point - vertex1
	p2 = point - vertex2
	dv = vertex2 - vertex1

	match (dv.cross(ray1) >= 0, dv.cross(ray2) >= 0):

		case (True, True):
			return ray2.cross(p2) < 0 or ray1.cross(p1) > 0 or dv.cross(p1) < 0

		case (False, False):
			return ray1.cross(p1) >= 0 and ray2.cross(p2) <= 0 and dv.cross(p1) <= 0

		case (True, False):
			return point_in_cone(point, vertex1, ray1, vertex1 - vertex2) or point_in_cone(point, vertex2, vertex1 - vertex2, ray2)

		case (False, True):
			return point_in_cone(point, vertex1, ray1, vertex2 - vertex1) or point_in_cone(point, vertex2, vertex2 - vertex1, ray2)

	if ray1.cross(ray2) < 0:
		return not point_in_edge(point, vertex2, vertex1, ray2, ray1)

	return ray1.cross(point - vertex1) >= 0 and ray2.cross(point - vertex2) <= 0 and (vertex2 - vertex1).cross(point - vertex1) <= 0

def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2) -> bool:

	if ray1.cross(ray2) < 0:
		return not point_in_cone(point, vertex, ray2, ray1)

	return ray1.cross(point - vertex) >= 0 and ray2.cross(point - vertex) <= 0

def locate_point(point: Vector2, polygon: Polygon2, cones: Cones) -> int:
	"""
	Locates point in cones or edges defined by polygon and cones.
	Returns index as follows:
	
	- `2n` -> cone in vertex `n`
	- `2n + 1` -> edge between vertex `n` and `n + 1`
	"""

	def is_between(i: int, j: int) -> bool:

		ray1 = cones[i // 2][i % 2]
		ray2 = cones[j // 2][j % 2]

		v1 = polygon[i // 2]
		v2 = polygon[j // 2]

		return point_in_edge(point, v1, v2, ray1, ray2)

	left = 0
	right = 2 * len(cones) - 1

	if is_between(0, 1):
		return 0
	
	if is_between(right, 0):
		return right

	while left + 1 != right:

		mid = (left + right) // 2

		if is_between(left, mid):
			right = mid
		else:
			left = mid

	return left

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

	cones: list[Cones]
	blocked: list[list[bool]]

	def __init__(self, start: _Vector2, target: _Vector2, polygons: Iterable[_Polygon2]) -> None:

		self.start = Vector2(start)
		self.target = Vector2(target)
		self.polygons = list(map(Polygon2, polygons))

		if any(not p.is_convex() for p in self.polygons):
			raise ValueError("All polygons must be convex.")

		self.cones = [[] for _ in self.polygons]
		self.blocked = [[] for _ in self.polygons]

	def query2(self, point: Vector2, index: int) -> tuple[Vector2, int]:
		"""
		Given a point and a polygon index, returns the last step of the smallest `index`-path from `start` to `point`.

		:param Vector2 point: The point to query.
		:param int index: The index of the polygon to query.

		:return Vector2: The last step of the smallest `index`-path from `start` to `point`.
		"""

		if index == 0:
			return self.start, 0

		polygon = self.polygons[index - 1]
		cones = self.cones[index - 1]
		blocked = self.blocked[index - 1]

		location = locate_point(point, polygon, cones)

		if location % 2 == 0:
			return polygon[location // 2], index

		if blocked[location // 2]:
			return self.query2(point, index - 1)

		v1 = polygon[location // 2]
		v2 = polygon[(location // 2 + 1) % len(polygon)]

		reflected = point.reflect_segment(v1, v2)
		last, _index = self.query2(reflected, index - 1)

		intersection = segment_segment_intersection(last, reflected, v1, v2)

		if intersection is not None:
			return intersection, index
		
		# print(point, index, last, reflected, v1, v2)  # Debugging line
		raise ValueError("No intersection found, this should not happen.")

	def query(self, point: Vector2, index: int) -> Vector2:
		return self.query2(point, index)[0]

	def solve(self) -> list[Vector2]:
		"""
		Returns the shortest path from start to target touching all polygons in order.
		"""

		if not self.polygons:
			return [self.start, self.target]

		n = len(self.polygons)

		for i in range(n):

			polygon = self.polygons[i]
			m = len(polygon)

			cones = self.cones[i]
			blocked = self.blocked[i]

			fails: list[int] = [0] * m

			for j in range(m):

				vertex = polygon[j]

				before = polygon[j - 1]
				after = polygon[j + 1]

				last = self.query(vertex, i)
				diff = (vertex - last).normalize()

				ray1 = diff.reflect((vertex - before).perpendicular())
				ray2 = diff.reflect((vertex - after).perpendicular())

				if (vertex - before).cross(last - before) >= 0:
					ray1 = diff
					fails[j] |= 1
				
				if (after - vertex).cross(last - vertex) >= 0:
					ray2 = diff
					fails[j] |= 2

				cones.append((ray1, ray2))

			for j in range(m):
				blocked.append(fails[j] == 3 or fails[(j + 1) % m] == 3 or (fails[j] >= 2 and fails[(j + 1) % m] == 1))

		result = [self.target]
		current = self.target
		index = n

		while index >= 0:

			current, index = self.query2(current, index)
			index -= 1
	
			if (current - result[-1]).magnitude() > 1e-8:
				result.append(current)

		result.reverse()

		return result
