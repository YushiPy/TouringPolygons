
from typing import Iterator
from vector2 import Vector2

class Polygon2(tuple[Vector2, ...]):

	def edges(self) -> Iterator[tuple[Vector2, Vector2]]:
		"""
		Iterate over the edges of the polygon.
		Each edge is represented as a tuple of two Vector2 points.
		"""

		for i in range(len(self) - 1):
			yield (self[i], self[i + 1])
		
		if self:
			yield (self[-1], self[0])
