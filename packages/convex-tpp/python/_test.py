
from itertools import chain
from math import inf, isclose
from typing import Any

import matplotlib.pyplot as plt

from problem1 import segment_segment_intersection
from vector2 import Vector2
from polygon2 import Polygon2
from problem1_fast import Solution


def intersection_rates(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2) -> tuple[float, float] | None:

	cross = direction1.cross(direction2)

	if abs(cross) < 1e-8:
		return None
	
	sdiff = start2 - start1

	rate1 = sdiff.cross(direction2) / cross
	rate2 = sdiff.cross(direction1) / cross

	return rate1, rate2

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


class Drawing:

	def __init__(self, start: Vector2, polygon: Polygon2) -> None:

		self.start = start
		self.polygon = polygon

		sol = Solution(start, Vector2(0, 0), [polygon])
		sol.solve()

		self.cones = sol.cones[0]
		self.blocked = sol.blocked[0]

	def get_bbox(self, extra: float = 0.1, square: bool = True) -> tuple[float, float, float, float]:
		"""
		Returns the bounding box of the drawing, which is the smallest rectangle
		that contains the start and end points, as well as all polygons.
		
		:param float extra: An optional parameter to expand the bounding box by a certain factor.
		:param bool square: If True, the bounding box will be square, expanding the smaller 
		side to match the larger one.
		
		:return: A tuple (minx, miny, maxx, maxy) representing the bounding box.
		"""

		points = list(chain([self.start], self.polygon))
		bleft, tright = Polygon2.bbox(points, extra, square)

		minx, miny = bleft.x, bleft.y
		maxx, maxy = tright.x, tright.y

		return minx, miny, maxx, maxy

	def draw(self) -> None:

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
			kwargs["markersize"] = 5
			kwargs.pop("label", None)

			ax.plot(*args, **kwargs) # type: ignore
			ax.plot(*args, **original) # type: ignore

		def draw_cones() -> None:

			for i in range(len(self.polygon)):

				vertex = self.polygon[i]

				ray1, ray2 = self.cones[i]

				if ray1 == ray2:
					continue

				points = locate_cone(vertex, ray1, ray2, bbox)

				fill(*zip(*points), alpha=0.45, color="red")
				
				p1 = locate_ray(vertex, ray1, bbox)
				p2 = locate_ray(vertex, ray2, bbox)

				color1 = "red" if not self.blocked[i - 1] else "blue"
				color2 = "red" if not self.blocked[(i) % len(self.blocked)] else "blue"

				plot(*zip(vertex, p1), color=color1, linewidth=2, linestyle='--')
				plot(*zip(vertex, p2), color=color2, linewidth=2, linestyle='--')

				mid = vertex + (ray1 + ray2).normalize() * 1
				ax.text(mid.x - 0.1, mid.y, f"$v_{{{i + 1}}}$", fontsize=12, color="black")

			fill([minx], [miny], alpha=0.45, color="red", label="Cone Region")

		def draw_edges() -> None:

			for i in range(len(self.polygon)):

				if self.blocked[i]:
					continue

				v1 = self.polygon[i]
				v2 = self.polygon[(i + 1) % len(self.polygon)]

				ray1 = self.cones[i][1]
				ray2 = self.cones[(i + 1) % len(self.cones)][0]

				points = locate_edge(v1, ray1, v2, ray2, bbox)

				fill(*zip(*points), alpha=1, color="#90dc8e")

				mid = (v1 + v2) * 0.5 + (ray1 + ray2).normalize() * 1
				ax.text(mid.x - 0.1, mid.y, f"$e_{{{i + 1}}}$", fontsize=12, color="black")
			
			fill([minx], [miny], alpha=0.45, color="#a0fc8e", label="Edge Region")

		fig, ax = plt.subplots(1, 1, figsize=(8, 8))

		bbox = self.get_bbox(0.5, False)
		minx, miny, maxx, maxy = bbox

		ax.set_xlim(minx, maxx)
		ax.set_ylim(miny, maxy)
		ax.set_aspect('equal', adjustable='box')

		# Setting labels
		plot(*zip(self.start), "o", color="green", label='Start', markersize=2)

		# Fill the background with a cyan color
		fill([minx, minx, maxx, maxx], [miny, maxy, maxy, miny], color="#6abdbe", alpha=0.7)
		fill([minx], [miny], color="#6abdbe", alpha=0.7)

		fill(*zip(*polygon), alpha=0.8, color="orange")
		plot(*zip(*polygon, polygon[0]), linewidth=1.5, color="orange")

		mid_point = sum(polygon, Vector2()) / len(polygon)
		ax.text(mid_point.x - 0.0, mid_point.y + 0.1, "$P_1$", fontsize=20, color="black")

		for i in range(len(polygon)):
			a = polygon[i]
			b = polygon[(i + 1) % len(polygon)]

			if not self.blocked[i]:
				plot(*zip(a, b), color="blue", linewidth=3, linestyle='-')
			else:
				plot(*zip(a, b), color="orange", linewidth=3, linestyle='-')

		mid_blocked = sum((a + b for i, (a, b) in enumerate(polygon.edges()) if self.blocked[i]), Vector2()) / max(1, sum(1 for b in self.blocked if b))
		ax.text(mid_blocked.x - 2.2, mid_blocked.y + 0.7, "$T_1$", fontsize=20, color="blue")

		ax.scatter(*zip(*polygon), color="white", s=50, zorder=5)	
		ax.scatter(*zip(*polygon), color="orange", s=30, zorder=5)

		for i, v in enumerate(polygon):

			inters = [segment_segment_intersection(self.start, v, a, b) for a, b in polygon.far_edges(v)]
			point = min((p for p in inters if p is not None), key=lambda p: (p - v).length(), default=v)
			straight = v + (v - point) * 100

			ax.plot([self.start.x, point.x], [self.start.y, point.y], color="white", linewidth=3, alpha=0.7)
			ax.plot([self.start.x, point.x], [self.start.y, point.y], color="black", linewidth=1.5, linestyle='--', alpha=0.7)
			ax.plot([point.x, straight.x], [point.y, straight.y], color="white", linewidth=3)
			ax.plot([point.x, straight.x], [point.y, straight.y], color="blue", linewidth=1.5, linestyle='--')

		draw_cones()
		draw_edges()

		# Plot the start and end points
		ax.scatter([self.start.x], [self.start.y], color="white", s=60, zorder=5)
		ax.scatter([self.start.x], [self.start.y], color="green", s=20, zorder=5)
		ax.text(self.start.x - 0.2, self.start.y + 0.1, "$s$", fontsize=12, color="black")

		# ax.legend() # type: ignore

		ax.set_xticks([])
		ax.set_yticks([])

		fig.tight_layout()

		plt.show()

polygon = Polygon2([
	Vector2(-2, -0.5), Vector2(-1, -1.6), Vector2(0.5, -2.4), Vector2(3, -2.5),
	Vector2(2, -1.5), Vector2(0, -0.4)
])

d = Drawing(Vector2(-4, -4), polygon)
d.draw()
