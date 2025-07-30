
from typing import Iterable, Literal, SupportsIndex
from vector2 import Vector2


def _orient(a: Vector2, b: Vector2, c: Vector2) -> float:
	return (b - a).cross(c - a)

def _on_segment(a: Vector2, b: Vector2, p: Vector2, eps: float = 1e-12) -> bool:
	return min(a.x, b.x) - eps <= p.x <= max(a.x, b.x) + eps and \
			min(a.y, b.y) - eps <= p.y <= max(a.y, b.y) + eps

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

	def contains_point(self, point: Vector2, eps: float = 1e-12) -> Literal[-1, 0, 1]:
		"""
		Check if the polygon contains a point.
		Returns 1 if the point is inside, 0 if on the boundary, and -1 if outside.
		"""
		
		inside = False

		for a, b in self.edges():

			# Boundary: collinear with edge and within its box
			if abs(_orient(a, b, point)) <= eps and _on_segment(a, b, point, eps):
				return 0

			# Half-open rule to avoid double counting at vertices
			if (a.y > point.y) != (b.y > point.y):

				x_int = a.x + (b.x - a.x) * (point.y - a.y) / (b.y - a.y)

				if x_int > point.x + eps:
					inside = not inside

		return 1 if inside else -1

	def bbox(self, extra: float = 0, square: bool = False) -> tuple[Vector2, Vector2]:

		xmin = min(vertex.x for vertex in self)
		xmax = max(vertex.x for vertex in self)
		ymin = min(vertex.y for vertex in self)
		ymax = max(vertex.y for vertex in self)

		center = Vector2((xmin + xmax) / 2, (ymin + ymax) / 2)

		dx = (xmax - xmin) * (1 + extra)
		dy = (ymax - ymin) * (1 + extra)

		if square:
			dx, dy = max(dx, dy), max(dx, dy)

		xmin = center.x - dx / 2
		xmax = center.x + dx / 2
		ymin = center.y - dy / 2
		ymax = center.y + dy / 2

		return Vector2(xmin, ymin), Vector2(xmax, ymax)

	def __getitem__(self, index: SupportsIndex) -> Vector2: # type: ignore
		"""
		Allow indexing into the polygon like a list.
		"""

		if hasattr(index, '__index__'):
			return super().__getitem__(index.__index__() % len(self))

		return super().__getitem__(index)
