
from typing import SupportsIndex
from vector2 import Vector2

class Polygon2(tuple[Vector2, ...]):

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
