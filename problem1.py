"""
Implementation of the first variation of the problem. 

We will consider that:
- The polygons are convex
- There is no "fence"
- The polygons are non intersecting
- The problem is in 2D.
"""

from math import inf, isclose, isqrt
from typing import Any

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

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


def locate_ray(start: Vector2, direction: Vector2, bbox: tuple[float, float, float, float]) -> Vector2:
	"""
	Locate the ray starting from `start` in the direction of `direction` within the bounding box `bbox`.
	
	:param Vector2 start: The starting point of the ray.
	:param Vector2 direction: The direction vector of the ray.
	:param tuple bbox: The bounding box defined as (minx, miny, maxx, maxy).

	:return: The point where the ray intersects with the bounding box.
	"""

	minx, miny, maxx, maxy = bbox
	dx = maxx - minx
	dy = maxy - miny

	walls = [
		(Vector2(minx, miny), Vector2(dx, 0)),
		(Vector2(maxx, miny), Vector2(0, dy)),
		(Vector2(maxx, maxy), Vector2(-dx, 0)),
		(Vector2(minx, maxy), Vector2(0, -dy))
	]

	for wall_start, wall_dir in walls:

		rates = intersection_rates(start, direction, wall_start, wall_dir)

		if rates is not None and rates[0] >= 0 and 0 <= rates[1] <= 1:
			return start + direction * rates[0]

	# Should be unreachable if the ray is inside the bounding box.
	return Vector2(inf, inf)

def locate_edge(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2, bbox: tuple[float, float, float, float]) -> list[Vector2]:
	"""
	Locate the edge defined by two directions starting from `start1` and `start2` within the bounding box `bbox`.

	:param Vector2 start1: The starting point of the first edge.
	:param Vector2 direction1: The direction vector of the first edge.
	:param Vector2 start2: The starting point of the second edge.
	:param Vector2 direction2: The direction vector of the second edge.
	:param tuple bbox: The bounding box defined as (minx, miny, maxx, maxy).

	:return: A list of Vector2 points representing the edge's vertices.
	"""

	def get_wall(point: Vector2) -> int:
		"""
		Determine which wall of the bounding box the point is closest to.
		Returns an index corresponding to the wall:
		0: bottom, 1: right, 2: top, 3: left.
		"""

		if isclose(point.y, miny):
			return 0
		elif isclose(point.x, maxx):
			return 1
		elif isclose(point.y, maxy):
			return 2
		elif isclose(point.x, minx):
			return 3

		# Should not happen if the point is within the bounding box
		return -1 

	minx, miny, maxx, maxy = bbox

	p1 = locate_ray(start1, direction1, bbox)
	p2 = locate_ray(start2, direction2, bbox)

	w1 = get_wall(p1)
	w2 = get_wall(p2)

	corners = [
		Vector2(maxx, miny),
		Vector2(maxx, maxy),
		Vector2(minx, maxy),
		Vector2(minx, miny)
	]
	
	result = [start1, p1]

	while w1 != w2:
		result.append(corners[w1])
		w1 = (w1 + 1) % 4
	
	result.append(p2)
	result.append(start2)
	result.append(start1)

	return result

def locate_cone(start: Vector2, direction1: Vector2, direction2: Vector2, bbox: tuple[float, float, float, float]) -> list[Vector2]:
	"""
	Locate the cone defined by two directions starting from `start` within the bounding box `bbox`.

	:param Vector2 start: The starting point of the cone.
	:param Vector2 direction1: The first direction vector of the cone.
	:param Vector2 direction2: The second direction vector of the cone.
	:param tuple bbox: The bounding box defined as (minx, miny, maxx, maxy).

	:return: A Polygon2 object representing the cone's vertices.
	"""

	return locate_edge(start, direction1, start, direction2, bbox)


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

	def __init__(self, start: Vector2, end: Vector2, polygons: list[Polygon2]) -> None:

		self.start = start
		self.end = end
		self.polygons = polygons

		self.blocked = []
		self.cones = []

		self.final_path = []

	def get_bbox(self) -> tuple[float, float, float, float]:

		minx = min(self.start.x, self.end.x, min(v.x for p in self.polygons for v in p))
		maxx = max(self.start.x, self.end.x, max(v.x for p in self.polygons for v in p))

		miny = min(self.start.y, self.end.y, min(v.y for p in self.polygons for v in p))
		maxy = max(self.start.y, self.end.y, max(v.y for p in self.polygons for v in p))

		dx = maxx - minx
		dy = maxy - miny

		if dx > dy:
			miny -= (dx - dy) / 2
			maxy += (dx - dy) / 2
		else:
			minx -= (dy - dx) / 2
			maxx += (dy - dx) / 2

		d = max(dx, dy) * 0.1

		minx -= d
		miny -= d
		maxx += d
		maxy += d

		return minx, miny, maxx, maxy

	def draw(self, scenes: list[int] | None = None) -> None:

		if not self.final_path:
			self.shortest_path()

		n: int = len(self.polygons)

		if scenes is None:
			scenes = list(range(n))

		count = len(scenes) + 1

		height = isqrt(count)
		width = (count + height - 1) // height

		fig, axs = plt.subplots(height, width, figsize=(width * 5, height * 5), constrained_layout=True) # type: ignore
		flat = axs.flatten() if count > 1 else [axs]

		for i, a in enumerate(scenes):
			self.draw_scene(flat[i + 1], a)
			flat[i + 1].set_title(f"Regions for polygon {a + 1}", fontsize=14)

		for i in range(count, len(flat)):
			flat[i].set_axis_off()
		
		self.draw_scene(flat[0], -1)

		# Set title for the whole figure
		fig.suptitle("Shortest path from Start to End passing touching every polygon", fontsize=16) # type: ignore

		flat[0].set_title("Final Path", fontsize=14)
		flat[0].legend()

		plt.show() # type: ignore

	def draw_scene(self, ax: Axes, index: int = 0) -> None:

		def fill(*args: Any, **kwargs: Any) -> None:

			original = kwargs.copy()

			kwargs.pop("label", None)
			kwargs["color"] = "white"
			kwargs["alpha"] = 1

			ax.fill(*args, **kwargs) # type: ignore
			ax.fill(*args, **original) # type: ignore
		
		def plot(*args: Any, **kwargs: Any) -> None:
			"""
			Plot a line with the given arguments.
			"""

			original = kwargs.copy()

			kwargs["color"] = "white"
			kwargs["linewidth"] = 4
			kwargs["linestyle"] = "solid"
			kwargs["markersize"] = 7
			kwargs.pop("label", None)

			ax.plot(*args, **kwargs) # type: ignore
			ax.plot(*args, **original) # type: ignore

		def draw_cones() -> None:

			for i in range(len(polygon)):

				vertex = polygon[i]

				ray1, ray2 = self.cones[index][i]

				if ray1 == ray2:
					continue

				points = locate_cone(vertex, ray1, ray2, bbox)

				fill(*zip(*points), alpha=0.45, color="red")
				
				p1 = locate_ray(vertex, ray1, bbox)
				p2 = locate_ray(vertex, ray2, bbox)

				plot(*zip(vertex, p1), color="red", linewidth=2, linestyle='--')
				plot(*zip(vertex, p2), color="red", linewidth=2, linestyle='--')

		def draw_edges() -> None:

			for i in range(len(polygon)):

				if self.blocked[index][i]:
					continue

				v1 = polygon[i]
				v2 = polygon[(i + 1) % len(polygon)]

				ray1 = self.cones[index][i][1]
				ray2 = self.cones[index][(i + 1) % len(self.cones[index])][0]

				points = locate_edge(v1, ray1, v2, ray2, bbox)

				fill(*zip(*points), alpha=0.45, color="green")

		bbox = self.get_bbox()
		minx, miny, maxx, maxy = bbox

		ax.set_xlim(minx, maxx)
		ax.set_ylim(miny, maxy)
		ax.set_aspect('equal', adjustable='box')

		# Fill the background with a cyan color
		fill([minx, minx, maxx, maxx], [miny, maxy, maxy, miny], color="#6abdbe", alpha=0.7)

		if 0 <= index < len(self.polygons):
			polygon = self.polygons[index]
			draw_cones()
			draw_edges()

		for i, polygon in enumerate(self.polygons):
			fill(*zip(*polygon), alpha=0.8, label=f'Polygon {i + 1}')
			plot(*zip(*polygon, polygon[0]), linewidth=2)

		# Plot the final path
		plot(*zip(*self.final_path), color="purple")

		# Plot the start and end points
		plot(*zip(self.start), "o", color="green", label='Start', markersize=4)
		plot(*zip(self.end), "o", color="red", label='End', markersize=4)

		# ax.legend() # type: ignore
		ax.grid() # type: ignore

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
