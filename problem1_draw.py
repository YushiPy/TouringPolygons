"""
Implementation of the first variation of the problem. 

We will consider that:
- The polygons are convex
- There is no "fence"
- The polygons are non intersecting
- The problem is in 2D.
"""

from itertools import chain
from math import inf, isclose, isqrt
from typing import Any

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from vector2 import Vector2
from polygon2 import bbox
from problem1 import Solution


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


class Drawing(Solution):

	def get_bbox(self, extra: float = 0.1) -> tuple[float, float, float, float]:
		"""
		Returns the bounding box of the drawing, which is the smallest rectangle
		that contains the start and end points, as well as all polygons.
		
		:param float extra: An optional parameter to expand the bounding box by a certain factor.
		:param bool square: If True, the bounding box will be square, expanding the smaller 
		side to match the larger one.
		
		:return: A tuple (minx, miny, maxx, maxy) representing the bounding box.
		"""

		points = list(chain([self.start, self.end], *self.polygons))
		bleft, tright = bbox(points, extra, True)

		minx, miny = bleft.x, bleft.y
		maxx, maxy = tright.x, tright.y

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
		fig.suptitle("Shortest path from Start to End touching every polygon", fontsize=16) # type: ignore

		flat[0].set_title("Final Path", fontsize=14)
		flat[0].legend()

		for i in range(1, len(scenes) + 1):
			flat[i].legend()

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

			fill([minx], [miny], alpha=0.45, color="red", label="Cone Region")

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
			
			fill([minx], [miny], alpha=0.45, color="green", label="Edge Region")

		bbox = self.get_bbox()
		minx, miny, maxx, maxy = bbox

		ax.set_xlim(minx, maxx)
		ax.set_ylim(miny, maxy)
		ax.set_aspect('equal', adjustable='box')

		# Fill the background with a cyan color
		fill([minx, minx, maxx, maxx], [miny, maxy, maxy, miny], color="#6abdbe", alpha=0.7)
		
		fill([minx], [miny], color="#6abdbe", alpha=0.7, label="Pass Through Region" * (index != -1))

		if 0 <= index < len(self.polygons):
			polygon = self.polygons[index]
			draw_cones()
			draw_edges()

		for i, polygon in enumerate(self.polygons):
			fill(*zip(*polygon), alpha=0.8, label=f'Polygon {i + 1}' * (index == -1 or index == i))
			plot(*zip(*polygon, polygon[0]), linewidth=2)

		# Plot the final path
		plot(*zip(*self.final_path), color="purple")

		# Plot the start and end points
		plot(*zip(self.start), "o", color="green", label='Start' * (index == -1), markersize=4)
		plot(*zip(self.end), "o", color="red", label='End' * (index == -1), markersize=4)

		# ax.legend() # type: ignore
		ax.grid() # type: ignore
