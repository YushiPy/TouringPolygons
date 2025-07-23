"""
Implementation of the first variation of the problem. 

We will consider that:
- The polygons are convex
- There is no "fence"
- The polygons are non intersecting
- The problem is in 2D.
"""

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

		self.blocked = [[False] * len(polygon) for polygon in polygons]
		self.cones = [[(Vector2(), Vector2()) for _ in range(len(polygon))] for polygon in polygons]

	def draw(self) -> None:
		pass

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

		for v1, v2 in self.polygons[0].edges():

			middle = v1.lerp(v2, 0.5)
			last = self.query(middle, -1)



		pass
