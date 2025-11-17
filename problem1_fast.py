
from collections.abc import Iterable

from vector2 import Vector2
from polygon2 import Polygon2


type Cones = list[tuple[Vector2, Vector2]]

type _Vector2 = Iterable[float]
type _Polygon2 = Iterable[_Vector2]


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

def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2, eps: float = 1e-8) -> bool:

	if ray1.cross(ray2) < -eps:
		return not point_in_cone(point, vertex, ray2, ray1, eps)

	return ray1.cross(point - vertex) >= -eps and ray2.cross(point - vertex) <= eps

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

		return point_in_edge(point, v1, v2, ray1, ray2, 0)

	left = 0
	right = 2 * len(cones) - 1

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

			fails = [0] * m

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

			"""
			for j, (a, b) in enumerate(polygon.edges()):

				mid = (a + b) / 2
				last = self.query(mid, i)

				blocked[j] = (b - a).cross(last - a) >= 0
			"""

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

"""
Complexity Analysis:

Let each polygon in the input be `P_i` with `m_i` vertices for `i = 1, 2, ..., k`, where `k` is the number of polygons.

- __init__:
	- O(m_1 + m_2 + ... + m_k) = O(n). To initialize polygons, cones, and blocked lists. As well as checking convexity.
- query(..., index):
	Worst case, `index` calls to locate_point, each `O(log m_i)`, so `O(log(m_1) + log(m_2) + ... + log(m_index))`.
	Other operations are O(1).
- solve:
	For each polygon `P_i`, we have `m_i` calls to query, each `O(log(m_1) + ... + log(m_{i-1}))`.
	Thus, total complexity is:
		O(Σ (m_i * Σ log(m_j) for j=1 to i-1) for i=1 to k)

Total Complexity:
	Since O(n) is dominated by the nested summation in solve, the overall complexity is:
		O(Σ (m_i * Σ log(m_j) for j=1 to i-1) for i=1 to k).
	This is maximum when all polygons have similar sizes, leading to
	m_i = n/k for all i, therefore Σ log(m_j) for j=1 to i-1 becomes O(i * log(n/k)) <= O(k log(n/k)).
	Thus, the overall complexity can be approximated as:
		O(n * k * log(n/k)).
	
In conclusion, the time complexity is O(n log n) for a fixed number of polygons.
Furthermore, if we say that all polygons have the same number of vertices m, then the complexity becomes O(k^2 * m * log m), 
which grows quadratically with the number of polygons.
"""
