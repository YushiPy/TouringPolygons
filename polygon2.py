
from functools import cached_property
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

	@staticmethod
	def bbox(points: Iterable[Vector2], extra: float = 0.1, square: bool = True) -> tuple[Vector2, Vector2]:
		"""
		Calculate the bounding box of a set of points.
		The bounding box is defined by two points: the bottom-left and top-right corners.

		:param Iterable[Vector2] points: An iterable of Vector2 points.

		:return: A tuple (minx, miny, maxx, maxy) representing the bounding box.
		"""

		points = list(points)

		xmin = min(point.x for point in points)
		xmax = max(point.x for point in points)
		ymin = min(point.y for point in points)
		ymax = max(point.y for point in points)

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
	
	def __new__(cls, points: Iterable[Iterable[float]]) -> "Polygon2":
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

		signed_area = 0

		for i in range(len(result)):
			a = result[i]
			b = result[(i + 1) % len(result)]
			signed_area += a.cross(b)

		# Reverse the order of points to ensure counter-clockwise orientation
		if signed_area < 0:
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

	def far_edges(self, *avoid: Vector2, eps: float = 1e-12) -> list[tuple[Vector2, Vector2]]:
		"""
		Iterate over the edges of the polygon that aren't included 
		in the avoid list.
		Each edge is represented as a tuple of two Vector2 points.
		"""

		def is_far(v: Vector2) -> bool:
			return all((v - a).length() > eps for a in avoid)

		return [(a, b) for a, b in self.edges() if is_far(a) and is_far(b)]

	@cached_property
	def reflex_vertices(self) -> list[Vector2]:
		"""
		Find the reflex vertices in the polygon.
		A reflex vertex is one where the internal angle is greater than 180 degrees.
		"""
		return [self[i] for i in self.reflex_vertices_indices]

	@cached_property
	def reflex_vertices_indices(self) -> list[int]:
		"""
		Get the indices of the reflex vertices in the polygon.
		A reflex vertex is one where the internal angle is greater than 180 degrees.
		"""

		return [i for i in range(len(self)) if _orient(self[i - 1], self[i], self[(i + 1) % len(self)]) < 0]

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

	def intersects_segment(self, start: Vector2, end: Vector2, eps: float = 1e-12) -> bool:
		"""
		Check if the polygon intersects a line segment.
		The segment is considered intersecting if it crosses any edge of the polygon.
		"""

		for a, b in self.far_edges(start, end, eps=eps):

			intersection = _segment_segment_intersection(start, end, a, b)

			if intersection is not None and not intersection.is_close(start, eps) and not intersection.is_close(end, eps):
				return True

		return False

	def contains_segment(self, start: Vector2, end: Vector2, eps: float = 1e-12) -> bool:
		"""
		Check if the polygon contains a line segment.
		The segment is considered contained if it does not intersect any edges of the polygon,
		except at its endpoints.
		"""

		if self.intersects_segment(start, end, eps):
			return False

		# Also check if the midpoint is inside the polygon for robustness
		return self.contains_point((start + end) / 2, eps) >= 0

	def __getitem__(self, index: SupportsIndex) -> Vector2: # type: ignore
		"""
		Allow indexing into the polygon like a list.
		"""

		if hasattr(index, '__index__'):
			return super().__getitem__(index.__index__() % len(self))

		return super().__getitem__(index)
