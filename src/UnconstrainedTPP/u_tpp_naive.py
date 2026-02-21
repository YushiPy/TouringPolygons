"""
First approach to solve the unconstrained TPP. 
This implementation is not optimized and may not be efficient for large inputs, 
but it serves as a proof of concept and a baseline for further improvements.

See report for details on the algorithm and its complexity analysis.
"""

from collections.abc import Iterable
from vector2 import Vector2
from polygon2 import Polygon2


def segment_segment_intersection(start1: Vector2, end1: Vector2, start2: Vector2, end2: Vector2) -> Vector2 | None:
	"""
	Returns the intersection point of segments (start1, end1) and (start2, end2) if they intersect, otherwise returns None.
	"""

	direction1 = end1 - start1
	direction2 = end2 - start2

	cross = direction1.cross(direction2)

	if cross == 0:
		return None

	sdiff = start2 - start1
	rate1 = sdiff.cross(direction2) / cross
	rate2 = sdiff.cross(direction1) / cross

	if 0 <= rate1 <= 1 and 0 <= rate2 <= 1:
		return start1 + direction1 * rate1
	
	return None

def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
	"""
	Check if `point` is inside the cone defined by `vertex` and rays `ray1` and `ray2`. 
	The rays are in counter-clockwise order.
	"""

	if ray1.cross(ray2) == 0 and ray1.dot(ray2) >= 0:
		return ray1.cross(point - vertex) == 0 and ray1.dot(point - vertex) >= 0

	if ray1.cross(ray2) >= 0:
		return ray1.cross(point - vertex) >= 0 and ray2.cross(point - vertex) <= 0
	else:
		return ray1.cross(point - vertex) >= 0 or ray2.cross(point - vertex) <= 0

def point_in_edge(point: Vector2, vertex1: Vector2, ray1: Vector2, vertex2: Vector2, ray2: Vector2) -> bool:
	"""
	Check if `point` is inside the edge region defined by `ray1` coming from `vertex1` and `ray2` coming from `vertex2`.
	The rays and edge are in counter-clockwise order.
	"""

	return ray1.cross(point - vertex1) > 0 and ray2.cross(point - vertex2) < 0 and (vertex2 - vertex1).cross(point - vertex1) <= 0


class Solution:

	start: Vector2
	target: Vector2
	polygons: list[Polygon2]

	# `first_contact[i][j]` is True if the first contact region of polygon `i` contains edge `j`.
	first_contact: list[list[bool]]

	# `cones[i][j]` is the cone of visibility of vertex `j` of polygon `i`.
	# The rays are stored in counter-clockwise order.
	cones: list[list[tuple[Vector2, Vector2]]]

	def __init__(self, start: Vector2, target: Vector2, polygons: list[Polygon2] | Iterable[Iterable[Iterable[float]]]) -> None:

		self.start = Vector2(start)
		self.target = Vector2(target)

		self.polygons = list(map(Polygon2, polygons))

	def locate_point(self, point: Vector2, i: int) -> int:
		"""
		Locates `point` in the shortest last step map of `i` and returns the index of the region as follows:
		- `2n` if the point is in the region of vertex `n`
		- `2n + 1` if the point is between vertices `n` and `n + 1`.
		"""

		polygon = self.polygons[i - 1]
		cones = self.cones[i - 1]

		for j in range(len(polygon)):
			v = polygon[j]
			ray1, ray2 = cones[j]
			if point_in_cone(point, v, ray1, ray2):
				return 2 * j

		for j in range(len(polygon)):

			v1 = polygon[j]
			v2 = polygon[(j + 1) % len(polygon)]

			ray1 = cones[j][1]
			ray2 = cones[(j + 1) % len(cones)][0]

			if point_in_edge(point, v1, ray1, v2, ray2):
				return 2 * j + 1

		raise ValueError(f"Point {point} is not located in any region of polygon {i}.")

	def query(self, point: Vector2, i: int) -> Vector2:
		"""
		Returns the last step of the `i`-path to `point`.
		"""

		if i == 0:
			return self.start
		
		polygon = self.polygons[i - 1]
		location = self.locate_point(point, i)
		pos = location // 2

		if location % 2 == 0:
			return polygon[pos]

		if not self.first_contact[i - 1][pos]:
			return self.query(point, i - 1)

		v1, v2 = polygon[pos], polygon[(pos + 1) % len(polygon)]

		reflected = point.reflect_segment(v1, v2)
		last = self.query(reflected, i - 1)

		intersection = segment_segment_intersection(last, reflected, v1, v2)

		if intersection is None:
			raise ValueError(f"Intersection not found for point {point} in polygon {i} at edge {pos}")

		return intersection
	
	def query_full(self, point: Vector2, i: int) -> list[Vector2]:
		"""
		Returns the `i`-path to `point`.
		"""

		if i == 0:
			return [self.start, point]
		
		polygon = self.polygons[i - 1]
		location = self.locate_point(point, i)
		pos = location // 2

		if location % 2 == 0:
			return self.query_full(polygon[pos], i - 1) + [point]

		if not self.first_contact[i - 1][pos]:
			return self.query_full(point, i - 1)

		v1, v2 = polygon[pos], polygon[(pos + 1) % len(polygon)]

		reflected = point.reflect_segment(v1, v2)
		path = self.query_full(reflected, i - 1)
		last = path[-2]

		intersection = segment_segment_intersection(last, reflected, v1, v2)

		if intersection is None:
			raise ValueError(f"Intersection not found for point {point} in polygon {i} at edge {pos}")

		return path[:-1] + [intersection, point]

	def get_first_contact_region(self, i: int) -> list[bool]:
		"""
		Returns the first contact region of polygon `i`.
		"""

		result = []
		polygon = self.polygons[i - 1]

		for j in range(len(polygon)):

			v1 = polygon[j]
			v2 = polygon[(j + 1) % len(polygon)]
			last = self.query(v1, i - 1)

			result.append((v2 - v1).cross(last - v1) < 0)

		return result

	def get_last_step_map(self, i: int) -> list[tuple[Vector2, Vector2]]:
		"""
		Returns the last step map of polygon `i`.
		"""

		result = []
		polygon = self.polygons[i - 1]
		first_contact = self.first_contact[i - 1]

		for j in range(len(polygon)):

			before = polygon[j - 1]
			vertex = polygon[j]
			after = polygon[(j + 1) % len(polygon)]
			
			last = self.query(vertex, i - 1)
			diff = vertex - last

			ray1 = diff.reflect((vertex - before).perpendicular()).normalize()
			ray2 = diff.reflect((vertex - after).perpendicular()).normalize()

			if not first_contact[j - 1]:
				ray1 = diff.normalize()

			if not first_contact[j]:
				ray2 = diff.normalize()

			result.append((ray1, ray2))

		return result

	def solve(self) -> list[Vector2]:
		"""
		Returns the shortest path from `start` to `target` that visits all polygons in order.
		"""

		self.first_contact = []
		self.cones = []

		for i in range(1, len(self.polygons) + 1):
			self.first_contact.append(self.get_first_contact_region(i))
			self.cones.append(self.get_last_step_map(i))
	
		return self.query_full(self.target, len(self.polygons))

def tpp_solve(start: Vector2, target: Vector2, polygons: list[Polygon2]) -> list[Vector2]:
	"""
	Returns the shortest path from `start` to `target` that visits all polygons in order.
	"""
	return Solution(Vector2(start), Vector2(target), list(map(Polygon2, polygons))).solve()
