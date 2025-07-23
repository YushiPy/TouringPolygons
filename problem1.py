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

class Solution:

	start: Vector2
	end: Vector2
	polygons: list[Polygon2]

	# Indicates whether edge `j` of polygon `i` is blocked
	blocked: list[list[bool]]

	# Indicates, cone vectors for each polygon vertex
	# Each cone is represented by a pair of vectors (start, end)
	cones: list[list[tuple[Vector2, Vector2]]]

	def __init__(self, start: Vector2, end: Vector2, polygons: list[Polygon2]) -> None:

		self.start = start
		self.end = end
		self.polygons = polygons

	def draw(self) -> None:
		pass

	def query(self, point: Vector2, index: int) -> Vector2:
		"""
		Query the point using the subregions up to `index`.
		Returns the point that comes before `point`.
		"""

	def shortest_path(self) -> list[Polygon2]:
		


		pass


