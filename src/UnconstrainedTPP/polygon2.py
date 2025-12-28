
from collections.abc import Iterable
from vector2 import Vector2


def intersection_rates(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2) -> tuple[float, float] | None:
	"""
	Calculate the intersection rates of two lines defined by a point and a direction vector.
	Returns None if the lines are parallel.
	"""

	cross = direction1.cross(direction2)

	if abs(cross) < 1e-8:
		return None
	
	sdiff = start2 - start1

	rate1 = sdiff.cross(direction2) / cross
	rate2 = sdiff.cross(direction1) / cross

	return rate1, rate2

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

	rates = intersection_rates(start1, diff1, start2, diff2)

	if rates is not None and 0 <= rates[0] <= 1 and 0 <= rates[1] <= 1:
		return start1 + diff1 * rates[0]

	return None

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

		signed_area = 0.0

		for i in range(len(result) - 1):
			a = result[(i + 1) % len(result)] - result[i]
			b = result[(i + 2) % len(result)] - result[(i + 1) % len(result)]
			signed_area += a.cross(b)

		# Reverse the order of points to ensure counter-clockwise orientation
		if signed_area < 0:
			result = [result[0]] + result[:0:-1]

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
