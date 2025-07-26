"""
Implementation of the first variation of the problem. 

We will consider that:
- The polygons are convex
- There is no "fence"
- The polygons are non intersecting
- The problem is in 2D.
"""

from math import inf, tau
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

def line_line_intersection(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2) -> Vector2:
	"""
	Returns the intersection point of two rays.
	If the rays are parallel, returns infinity vector.

	:param Vector2 start1: The start point of the first ray as a Vector2.
	:param Vector2 direction1: The direction vector of the first ray as a Vector2.
	:param Vector2 start2: The start point of the second ray as a Vector2.
	:param Vector2 direction2: The direction vector of the second ray as a Vector2.

	:return: The intersection point as a Vector2 if the rays intersect, otherwise None.	
	"""

	cross = direction1.cross(direction2)

	if abs(cross) < 1e-8:
		return Vector2(inf, inf)

	rate = (start2 - start1).cross(direction1) / cross

	return start1 + direction1 * rate


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

		xpoints = [v.x for v in polygon] + [polygon[0].x]
		ypoints = [v.y for v in polygon] + [polygon[0].y]

		ax.fill(*zip(*polygon), alpha=0.5, color='blue', label='Polygon')
		ax.plot(xpoints, ypoints, color='blue', linewidth=2)

		ax.plot(self.start.x, self.start.y, 'ro', label='Start')
		ax.plot(self.end.x, self.end.y, 'go', label='End')

		for i in range(len(polygon)):

			vertex = polygon[i]

			ray1, ray2 = self.cones[0][i]

			if ray1 == ray2:
				continue

			mid = ray1 + ray2

			ray1.scale_to_length_ip(2 * (maxx - minx))
			ray2.scale_to_length_ip(4 * (maxx - minx))
			mid.scale_to_length_ip(2 * (maxx - minx))


			xpoints = [vertex.x, vertex.x + ray1.x, vertex.x + mid.x, vertex.x + ray2.x]
			ypoints = [vertex.y, vertex.y + ray1.y, vertex.y + mid.y, vertex.y + ray2.y]

			ax.fill(xpoints, ypoints, alpha=0.45, color="red")
			ax.plot(*zip(vertex, vertex + ray1), color="red", linewidth=2, linestyle='--')
			ax.plot(*zip(vertex, vertex + ray2), color="red", linewidth=2, linestyle='--')

		for i in range(len(polygon)):

			v1 = polygon[i]
			v2 = polygon[i + 1]

			ray1 = self.cones[0][i][1]
			ray2 = self.cones[0][i][0]
			mid = ray1 + ray2

			ray1.scale_to_length_ip(2 * (maxx - minx))
			ray2.scale_to_length_ip(2 * (maxx - minx))
			mid.scale_to_length_ip(4 * (maxx - minx))

			xpoints = [v1.x, v1.x + ray1.x, v1.x + mid.x, v2.x + ray2.x, v2.x]
			ypoints = [v1.y, v1.y + ray1.y, v1.y + mid.y, v2.y + ray2.y, v2.y]

			# ax.fill(xpoints, ypoints, alpha=0.45, color="red")

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

			is_blocked = any((a, b) != (v1, v2) and segment_segment_intersection(last, middle, a, b) is not None for a, b in self.polygons[0].edges())
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

def regular(n: int, r: float, start: Vector2 = Vector2(), angle: float = 0) -> Polygon2:
	"""
	Create a regular polygon with `n` vertices and radius `r`.

	:param int n: The number of vertices.
	:param float r: The radius of the polygon.

	:return: A Polygon2 object representing the regular polygon.
	"""
	return Polygon2(start + Vector2.from_spherical(r, i * tau / n + angle) for i in range(n))

sol = Solution(Vector2(-3, 0), Vector2(3, 0), [
	Polygon2([Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)])
])

sol2 = Solution(Vector2(-3, 0), Vector2(3, 0), [regular(6, 1, angle=tau / 2)])

path = sol2.shortest_path()
