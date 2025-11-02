
from collections.abc import Iterable
from typing import Literal

from matplotlib import pyplot as plt
from vector2 import Vector2
from polygon2 import Polygon2


type Cones = list[tuple[Vector2, Vector2]]

type _Vector2 = Iterable[float]
type _Polygon2 = Iterable[_Vector2]


def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
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

	if is_between(0, 1):
		return 0

	left = 0
	right = 2 * len(cones) - 1

	while left + 1 != right:

		mid = (left + right) // 2

		if is_between(left, mid):
			right = mid
		else:
			left = mid

	return left

def segment_segment_intersection(start1: Vector2, end1: Vector2, start2: Vector2, end2: Vector2) -> Vector2 | None:
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

	if abs(cross) < 1e-8:
		return None
	
	sdiff = start2 - start1

	rate1 = sdiff.cross(diff2) / cross
	rate2 = sdiff.cross(diff1) / cross

	if 0 <= rate1 <= 1 and 0 <= rate2 <= 1:
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
	
	def query(self, point: Vector2, index: int) -> Vector2:
		"""
		Given a point and a polygon index, returns the last step of the smallest `index`-path from `start` to `point`.

		:param Vector2 point: The point to query.
		:param int index: The index of the polygon to query.

		:return Vector2: The last step of the smallest `index`-path from `start` to `point`.
		"""

		if index == 0:
			return self.start

		polygon = self.polygons[index - 1]
		cones = self.cones[index - 1]
		blocked = self.blocked[index - 1]

		location = locate_point(point, polygon, cones)

		if location % 2 == 0:
			return polygon[location // 2]

		if blocked[location // 2]:
			return self.query(point, index - 1)

		v1 = polygon[location // 2]
		v2 = polygon[(location // 2 + 1) % len(polygon)]

		reflected = point.reflect_segment(v1, v2)
		last = self.query(reflected, index - 1)

		intersection = segment_segment_intersection(last, reflected, v1, v2)

		if intersection is not None:
			return intersection
		
		raise ValueError("No intersection found, this should not happen.")

	def solve(self) -> list[Vector2]:
		"""
		Returns the shortest path from start to target touching all polygons in order.
		"""

		if not self.polygons:
			return [self.start, self.target]

		n = len(self.polygons)

		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(3, 3, figsize=(10, 10))

		for i in range(n):

			polygon = self.polygons[i]
			m = len(polygon)

			ax[0, 0].plot(
				[v.x for v in polygon] + [polygon[0].x],
				[v.y for v in polygon] + [polygon[0].y],
			)

			ax[0, 0].fill(
				[v.x for v in polygon] + [polygon[0].x],
				[v.y for v in polygon] + [polygon[0].y],
				alpha=0.1
			)

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
					fails[j] += 1
				
				if (after - vertex).cross(last - vertex) >= 0:
					ray2 = diff
					fails[j] += 1

				cones.append((ray1, ray2))

			for j in range(m):
				blocked.append(fails[j] == 2 or fails[(j + 1) % m] == 2)

			print(blocked)

			axis = ax.flat[i + 1]

			axis.scatter([self.start.x, self.target.x], [self.start.y, self.target.y])

			# Draw polygon
			xs = [v.x for v in polygon] + [polygon[0].x]
			ys = [v.y for v in polygon] + [polygon[0].y]
			axis.plot(xs, ys)
			axis.fill(xs, ys, alpha=0.1)

			# Draw cones
			for j in range(m):
				vertex = polygon[j]
				ray1, ray2 = cones[j]

				if ray1 == ray2:
					continue

				axis.plot(
					[vertex.x, vertex.x + ray1.x * 2],
					[vertex.y, vertex.y + ray1.y * 2],
					'r--'
				)

				axis.plot(
					[vertex.x, vertex.x + ray2.x * 2],
					[vertex.y, vertex.y + ray2.y * 2],
					'r--'
				)

		return []
	

test1 = Solution(
	Vector2(5, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(2, 4), Vector2(1, 4), Vector2(-1, 1)]),
		Polygon2([Vector2(3, 3), Vector2(4, 3), Vector2(4, 4), Vector2(3, 4)]),
		Polygon2([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)]),
	]
)

from math import pi, tau

test2 = Solution(
	Vector2(-1, -1),
	Vector2(1, -1),
	[
		Polygon2([Vector2.from_polar(2, i * tau / 6 + pi * 0.35) + Vector2(4, 5) for i in range(6)]),
		Polygon2([Vector2.from_polar(2, i * tau / 3 + pi /4) + Vector2(-3, 4) for i in range(3)]),
		Polygon2([Vector2.from_polar(2, i * tau / 10) + Vector2(5, -4) for i in range(10)]),
		Polygon2([Vector2.from_polar(2, i * tau / 4 + pi / 4) + Vector2(-4, -2) for i in range(4)]),
		Polygon2([Vector2.from_polar(2, i * tau / 30 + pi / 4) + Vector2(0, -8) for i in range(30)]),
	]
)

test2.solve()
