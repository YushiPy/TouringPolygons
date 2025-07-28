
from typing import Iterable, SupportsIndex
from vector2 import Vector2

class Polygon2(tuple[Vector2, ...]):

	def __new__(cls, points: Iterable[Iterable[float]]) -> 'Polygon2':
		"""
		Create a new Polygon2 instance from a list of Vector2 points.
		The points should be in order (either clockwise or counter-clockwise).
		"""

		result: list[Vector2] = []

		for point in points:

			values = tuple(point)

			if len(values) != 2:
				raise ValueError("Each point must be a tuple of two coordinates (x, y).")
			
			result.append(Vector2(*values))

		if len(result) < 3:
			raise ValueError("A polygon must have at least 3 points.")
		
		seg1 = result[1] - result[0]
		seg2 = result[2] - result[1]

		# Ensure the points are in counter-clockwise order
		if seg1.cross(seg2) < 0:
			result.reverse()

		return super().__new__(cls, result)

	def is_convex(self) -> bool:
		"""
		Check if the polygon is convex.
		A polygon is convex if all its internal angles are less than 180 degrees.
		"""

		if len(self) < 3:
			return False

		segments = [b - a for a, b in self.edges()]

		sign = segments[0].cross(segments[1])

		for i in range(1, len(segments)):

			if segments[i].cross(segments[(i + 1) % len(segments)]) * sign < 0:
				return False

		return True

	def edges(self) -> list[tuple[Vector2, Vector2]]:
		"""
		Iterate over the edges of the polygon.
		Each edge is represented as a tuple of two Vector2 points.
		"""
		return [(self[i], self[(i + 1) % len(self)]) for i in range(len(self))]

	def __getitem__(self, index: SupportsIndex) -> Vector2: # type: ignore
		"""
		Allow indexing into the polygon like a list.
		"""

		if hasattr(index, '__index__'):
			return super().__getitem__(index.__index__() % len(self))

		return super().__getitem__(index)
