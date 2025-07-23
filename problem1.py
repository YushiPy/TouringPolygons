"""
Implementation of the first variation of the problem. 

We will consider that:
- The polygons are convex
- There is no "fence"
- The polygons are non intersecting
- The problem is in 2D.
"""

import matplotlib.pyplot as plt

from vector2 import Vector2
from polygon2 import Polygon2


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

	det = diff1.cross(diff2)

	if abs(det) < 1e-8:
		return None

	sdiff = start2 - start1

	t = sdiff.cross(diff2) / det
	u = sdiff.cross(diff1) / det

	if 0 <= t <= 1 and 0 <= u <= 1:
		return start1 + diff1 * t

class Solution:

	start: Vector2
	end: Vector2
	polygons: list[Polygon2]

	# Indicates whether edge `j` of polygon `i` is blocked
	blocked: list[list[bool]]

	# For each polygon and vertex, we store two vectors that represent the cone of visibility.
	# They are stored in counter-clockwise order.
	cones: list[list[tuple[Vector2, Vector2]]]

	def __init__(self, start: Vector2, end: Vector2, polygons: list[Polygon2]) -> None:

		self.start = start
		self.end = end
		self.polygons = polygons

		self.blocked = []
		self.cones = []

	def get_bbox(self) -> tuple[float, float, float, float]:

		minx = min(self.start.x, self.end.x, min(v.x for p in self.polygons for v in p))
		maxx = max(self.start.x, self.end.x, max(v.x for p in self.polygons for v in p))

		miny = min(self.start.y, self.end.y, min(v.y for p in self.polygons for v in p))
		maxy = max(self.start.y, self.end.y, max(v.y for p in self.polygons for v in p))

		dx = maxx - minx
		dy = maxy - miny

		if dx > dy:
			miny -= (dx - dy) / 2
			maxy += (dx - dy) / 2
		else:
			minx -= (dy - dx) / 2
			maxx += (dy - dx) / 2

		d = max(dx, dy) * 0.1

		minx -= d
		miny -= d
		maxx += d
		maxy += d

		return minx, miny, maxx, maxy

	def draw(self) -> None:

		fig, ax = plt.subplots(1, 1, figsize=(10, 10))

		minx, miny, maxx, maxy = self.get_bbox()

		ax.set_xlim(minx, maxx)
		ax.set_ylim(miny, maxy)
		ax.set_aspect('equal', adjustable='box')

		polygon = self.polygons[0]

		ax.fill(*zip(*polygon), alpha=0.5, color='blue', label='Polygon')
		ax.plot(*zip(*polygon.edges()), color='blue', linewidth=2)
		ax.plot(self.start.x, self.start.y, 'ro', label='Start')
		ax.plot(self.end.x, self.end.y, 'go', label='End')

		for i in range(len(polygon)):

			vertex = polygon[i]
			ray1, ray2 = self.cones[0][i]

			ax.arrow(vertex.x, vertex.y, ray1.x * 0.5, ray1.y * 0.5, head_width=0.1, head_length=0.2, fc='orange', ec='orange', label='Cone' if i == 0 else "")
			ax.arrow(vertex.x, vertex.y, ray2.x * 0.5, ray2.y * 0.5, head_width=0.1, head_length=0.2, fc='orange', ec='orange')

		plt.show()

	def query(self, point: Vector2, index: int) -> Vector2:
		"""
		Query the point using the subregions up to `index`.
		Returns the point that comes before `point`.
		"""

		if index == -1:
			return self.start

	def shortest_path(self) -> list[Vector2]:
		
		if len(self.polygons) == 0:
			return [self.start, self.end]

		self.blocked.append([])

		# Determining which edges are blocked.
		for v1, v2 in self.polygons[0].edges():

			middle = v1.lerp(v2, 0.5)
			last = self.query(middle, -1)

			# Check if the segment from `last` to `middle` intersects with 
			# any edge of the polygon that is not the segment from `v1` to `v2`.

			is_blocked = any((a, b) != (v1, v2) and segment_segment_intersection(last, middle, v1, v2) is not None for a, b in self.polygons[0].edges())
			self.blocked[0].append(is_blocked)

		self.cones.append([])

		# Determining the cones of visibility for each vertex.
		for i in range(len(self.polygons[0])):

			vertex = self.polygons[0][i]

			before = self.polygons[0][i - 1]
			after = self.polygons[0][i + 1]

			last = self.query(vertex, -1)
			diff = vertex - last

			ray1 = diff.reflect((before - vertex).perpendicular()).normalize()
			ray2 = diff.reflect((after - vertex).perpendicular()).normalize()

			if self.blocked[0][i - 1]:
				ray1 = diff.normalize()
			if self.blocked[0][i]:
				ray2 = diff.normalize()

			self.cones[0].append((ray1, ray2))

		self.draw()

sol = Solution(Vector2(-3, 0), Vector2(3, 0), [
	Polygon2([Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)])
])

path = sol.shortest_path()
