
from typing import Iterable, Literal, SupportsIndex
from vector2 import Vector2


def _orient(a: Vector2, b: Vector2, c: Vector2) -> float:
	return (b - a).cross(c - a)

def _on_segment(a: Vector2, b: Vector2, p: Vector2, eps: float = 1e-12) -> bool:
	return min(a.x, b.x) - eps <= p.x <= max(a.x, b.x) + eps and \
			min(a.y, b.y) - eps <= p.y <= max(a.y, b.y) + eps

def _intersection_rates(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2) -> tuple[float, float] | None:

	cross = direction1.cross(direction2)

	if abs(cross) < 1e-8:
		return None
	
	sdiff = start2 - start1

	rate1 = sdiff.cross(direction2) / cross
	rate2 = sdiff.cross(direction1) / cross

	return rate1, rate2

def _segment_segment_intersection(start1: Vector2, end1: Vector2, start2: Vector2, end2: Vector2) -> Vector2 | None:
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

	rates = _intersection_rates(start1, diff1, start2, diff2)

	if rates is not None and 0 <= rates[0] <= 1 and 0 <= rates[1] <= 1:
		return start1 + diff1 * rates[0]


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

	def contains_segment(self, start: Vector2, end: Vector2, eps: float = 1e-12) -> bool:
		"""
		Check if the polygon contains a line segment.
		The segment is considered contained if it does not intersect any edges of the polygon,
		except at its endpoints.
		"""

		def is_near(v1: Vector2, v2: Vector2) -> bool:
			return (v1 - v2).length() < eps

		for a, b in self.edges():

			if is_near(start, a) or is_near(start, b) or is_near(end, a) or is_near(end, b):
				continue

			intersection = _segment_segment_intersection(start, end, a, b)

			if intersection is not None and not is_near(intersection, start) and not is_near(intersection, end):
				return False

		# Also check if the midpoint is inside the polygon for robustness
		return self.contains_point((start + end) / 2, eps) >= 0

	def bbox(self, extra: float = 0, square: bool = False) -> tuple[Vector2, Vector2]:
		"""
		Calculate the bounding box of the polygon.
		The bounding box is defined by two points: the bottom-left and top-right corners.
		An optional `extra` parameter can be provided to expand the bounding box by a certain factor.
		If `square` is True, the bounding box will be square, expanding the smaller side to match the larger one.
		"""

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
