"""
Implementation of the first variation of the problem. 

We will consider that:
- The polygons are convex
- There is no "fence"
- The polygons are non intersecting
- The problem is in 2D.
"""

from typing import Iterable
from vector2 import Vector2
from polygon2 import Polygon2


def intersection_rates(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2) -> tuple[float, float] | None:

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


def point_in_cone(point: Vector2, start: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
	"""
	Check if a point is inside the cone defined by two rays starting from `start`.
	The rays are in clockwise order.

	:param Vector2 point: The point to check.
	:param Vector2 start: The starting point of the cone.
	:param Vector2 ray1: The first ray direction.
	:param Vector2 ray2: The second ray direction.

	:return: True if the point is inside the cone, False otherwise.
	"""

	vector = point - start

	cross1 = ray1.cross(vector)
	cross2 = ray2.cross(vector)

	# Is to the counter-clockwise side of the first ray and 
	# clockwise side of the second ray.
	return (cross1 >= 0 and cross2 <= 0)

def point_in_edge(point: Vector2, start1: Vector2, ray1: Vector2, start2: Vector2, ray2: Vector2) -> bool:
	"""
	Check if a point is inside the edge defined by two rays starting from `start1` and `start2`.

	:param Vector2 point: The point to check.
	:param Vector2 start1: The starting point of the first ray.
	:param Vector2 ray1: The direction vector of the first ray.
	:param Vector2 start2: The starting point of the second ray.
	:param Vector2 ray2: The direction vector of the second ray.

	:return: True if the point is inside the edge, False otherwise.
	"""

	vector1 = point - start1
	vector2 = point - start2

	ray3 = start2 - start1

	cross1 = ray1.cross(vector1)
	cross2 = ray2.cross(vector2)
	cross3 = ray3.cross(vector1)

	# Is to the counter-clockwise side of the first ray and 
	# clockwise side of the second ray.
	# Also checks if the point is clockwise to the segment from `start1` to `start2`.
	return (cross1 >= 0 and cross2 <= 0 and cross3 <= 0)


class Solution:

	start: Vector2
	end: Vector2
	polygons: list[Polygon2]

	# Indicates whether edge `j` of polygon `i` is blocked
	blocked: list[list[bool]]

	# For each polygon and vertex, we store two vectors that represent the cone of visibility.
	# They are stored in counter-clockwise order.
	cones: list[list[tuple[Vector2, Vector2]]]

	final_path: list[Vector2]

	def __init__(self, start: Vector2, end: Vector2, polygons: list[Polygon2] | Iterable[Iterable[Iterable[float]]]) -> None:

		self.polygons = []

		for polygon in polygons:

			polygon = Polygon2(polygon)

			if not polygon.is_convex():
				raise ValueError("All polygons must be convex.")

			self.polygons.append(polygon)

		self.start = start
		self.end = end

		self.blocked = []
		self.cones = []

		self.final_path = []

	def query(self, point: Vector2, index: int) -> Vector2:
		"""
		Query the point using the subregions up to `index`.
		Returns the point that comes before `point`.
		"""

		if index == -1:
			return self.start

		polygon = self.polygons[index]
		cones = self.cones[index]
		blocked = self.blocked[index]

		# Check if the point is inside a cone region.
		for vertex, (ray1, ray2) in zip(polygon, cones):

			if ray1 == ray2:
				continue
			
			if point_in_cone(point, vertex, ray1, ray2):
				return vertex

		m: int = len(polygon)

		# Check if the point is inside an edge region.
		for i in range(m):

			if blocked[i]:
				continue

			v1 = polygon[i]
			v2 = polygon[(i + 1) % m]

			ray1 = cones[i][1]
			ray2 = cones[(i + 1) % len(cones)][0]

			if not point_in_edge(point, v1, ray1, v2, ray2):
				continue

			# If the point is inside the edge, we reflect the 
			# point across the segment and recursively repeat 
			# the process with the previous polygon

			reflected = point.reflect_segment(v1, v2)
			
			# Path comes from `last`, reflects on segment
			# and reaches `point`.
			last = self.query(reflected, index - 1)

			# We must find the point on the segment where the 
			# path reflects.

			result = segment_segment_intersection(last, reflected, v1, v2)

			if result is None:
				# This should not happen, but if it does, we raise an error.
				raise ValueError(f"Unexpected result: {result} for point {point} in polygon {index} at edge {i}")

			return result

		# The point must be inside a pass through region
		return self.query(point, index - 1)

	def shortest_path(self) -> list[Vector2]:
		
		if len(self.polygons) == 0:
			return [self.start, self.end]

		for i in range(len(self.polygons)):

			polygon = self.polygons[i]
			blocked: list[bool] = []
			cones: list[tuple[Vector2, Vector2]] = []

			self.blocked.append(blocked)
			self.cones.append(cones)

			# Determining which edges are blocked.
			for v1, v2 in polygon.edges():

				middle = v1.lerp(v2, 0.5)
				last = self.query(middle, i - 1)

				# Check if the segment from `last` to `middle` intersects with 
				# any edge of the polygon that is not the segment from `v1` to `v2`.
				is_blocked = any((a, b) != (v1, v2) and segment_segment_intersection(last, middle, a, b) is not None for a, b in polygon.edges())
				blocked.append(is_blocked)

			# Determining the cones of visibility for each vertex.
			for j in range(len(polygon)):

				vertex = polygon[j]

				before = polygon[j - 1]
				after = polygon[j + 1]

				last = self.query(vertex, i - 1)

				diff = vertex - last

				ray1 = diff.reflect((vertex - before).perpendicular()).normalize()
				ray2 = diff.reflect((vertex - after).perpendicular()).normalize()

				if blocked[j - 1]:
					ray1 = diff.normalize()
				if blocked[j]:
					ray2 = diff.normalize()

				cones.append((ray1, ray2))

		result: list[Vector2] = [self.end]
		current: Vector2 = self.end

		for i in range(len(self.polygons) - 1, -1, -1):

			current = self.query(current, i)

			# If the new point is not the same as the last
			if (current - result[-1]).magnitude() > 1e-8:
				result.append(current)

		result.append(self.start)
		result.reverse()

		self.final_path = result

		#print("\n\n".join(
		#	"\n".join(str(x) for x in cones) for cones in self.cones
		#))

		return result
