

from itertools import chain
from math import isclose, isqrt
from typing import Any

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from polygon2 import Polygon2
from problem2 import Solution
from vector2 import Vector2


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
	return Vector2.inf()

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


def plot(ax: Axes, *args: Any, **kwargs: Any) -> None:
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

def fill(ax: Axes, *args: Any, **kwargs: Any) -> None:

	original = kwargs.copy()

	kwargs.pop("label", None)
	kwargs["color"] = "white"
	kwargs["alpha"] = 1

	ax.fill(*args, **kwargs) # type: ignore
	ax.fill(*args, **original) # type: ignore

def draw_polygon(ax: Axes, polygon: Polygon2, **kwargs: Any) -> None:
	"""
	Draw a polygon on the given Axes.

	:param Axes ax: The Axes to draw on.
	:param Polygon2 polygon: The polygon to draw.
	:param kwargs: Additional keyword arguments to pass to the fill function.
	"""

	fill(ax, *zip(*polygon), **kwargs)

	if "label" in kwargs:
		kwargs = kwargs.copy()
		kwargs.pop("label")
	
	plot(ax, *zip(*(polygon + (polygon[0],))), **kwargs)



class Drawing(Solution):

	bboxes: list[tuple[Vector2, Vector2]]

	def draw(self) -> None:

		self.bboxes = []

		scenes = len(self.polygons) + 1
		scenes = 1
		height = isqrt(scenes - 1) + 1
		width = (scenes + height - 1) // height

		_, ax = plt.subplots(height, width, figsize=(6 * width, 6 * height)) # type: ignore

		flat: list[Axes] = ax.flatten() if scenes > 1 else [ax]

		self.draw0(flat[0])

		for i in range(1, scenes):
			self.draw_scene(flat[i], i - 1)
		
		for i in range(scenes, len(flat)):
			flat[i].axis("off")

		plt.tight_layout()
		#plt.show() # type: ignore

	def draw0(self, ax: Axes) -> None:

		path = self.solve()

		bbox = Polygon2.bbox(chain(*self.fences, *self.polygons, [self.start, self.target]))
		minx, miny = bbox[0].x, bbox[0].y
		maxx, maxy = bbox[1].x, bbox[1].y

		ax.set_xlim(minx, maxx)
		ax.set_ylim(miny, maxy)
		ax.set_aspect('equal', adjustable='box')

		ax.scatter(self.start.x, self.start.y, color="green", marker="o", label="Start") # type: ignore
		ax.scatter(self.target.x, self.target.y, color="red", marker="o", label="Target") # type: ignore

		plot(ax, *zip(*path), color="purple", linewidth=2, linestyle="dashed", marker="o", markersize=3, label="Path") # type: ignore
		
		fill(ax, [minx, minx, maxx, maxx], [miny, maxy, maxy, miny], color="#ccd7d8", alpha=0.7)

		for i, p in enumerate(self.polygons, 1):
			draw_polygon(ax, p, label=f"Polygon {i}")

		plt.gca().set_prop_cycle(None) # type: ignore

		for i, f in enumerate(self.fences, 1):
			plot(ax, *zip(*(f + (f[0],))), alpha=0.8, label=f"Fence {i}")

		plot(ax, *zip(*path), color="purple", linewidth=2, linestyle="dashed", marker="o", markersize=3) # type: ignore

		plot(ax, self.start.x, self.start.y, color="green", marker="o", markersize=4)
		plot(ax, self.target.x, self.target.y, color="red", marker="o", markersize=4)

		ax.grid() # type: ignore
		ax.legend() # type: ignore

		plt.tight_layout()
		plt.savefig("output.png", dpi=300) # type: ignore

	def draw_cones(self, ax: Axes, index: int) -> None:

		polygon = self.polygons[index]
		bbox = self.bboxes[index]
		_bbox = (bbox[0].x, bbox[0].y, bbox[1].x, bbox[1].y)

		for i in range(len(polygon)):

			vertex = polygon[i]

			ray1, ray2 = self.cones[index][i]

			if ray1 == ray2:
				continue

			print(ray1, ray2, index, i)

			points = locate_cone(vertex, ray1, ray2, _bbox)

			fill(ax, *zip(*points), alpha=0.45, color="red")
			
			p1 = locate_ray(vertex, ray1, _bbox)
			p2 = locate_ray(vertex, ray2, _bbox)

			plot(ax, *zip(vertex, p1), color="red", linewidth=2, linestyle='--')
			plot(ax, *zip(vertex, p2), color="red", linewidth=2, linestyle='--')

		fill(ax, [0], [0], alpha=0.45, color="red", label="Cone Region")

	def draw_edges(self, ax: Axes, index: int) -> None:

		polygon = self.polygons[index]
		bbox = self.bboxes[index]
		_bbox = (bbox[0].x, bbox[0].y, bbox[1].x, bbox[1].y)

		for i in range(len(polygon)):

			if self.blocked[index][i]:
				continue

			v1 = polygon[i]
			v2 = polygon[(i + 1) % len(polygon)]

			ray1 = self.cones[index][i][1]
			ray2 = self.cones[index][(i + 1) % len(self.cones[index])][0]

			points = locate_edge(v1, ray1, v2, ray2, _bbox)

			fill(ax, *zip(*points), alpha=0.45, color="green")
		
		fill(ax, [0], [0], alpha=0.45, color="green", label="Edge Region")

	def draw_scene(self, ax: Axes, index: int) -> None:

		polygon = self.polygons[index]
		fence1 = self.fences[index]
		fence2 = self.fences[(index + 1) % len(self.fences)]

		bbox = Polygon2.bbox(chain(polygon, fence1, fence2, [self.start, self.target]))
		minx, miny = bbox[0].x, bbox[0].y
		maxx, maxy = bbox[1].x, bbox[1].y

		self.bboxes.append((Vector2(minx, miny), Vector2(maxx, maxy)))

		ax.set_title(f"Polygon {index}") # type: ignore
		ax.set_xlim(minx, maxx)
		ax.set_ylim(miny, maxy)
		ax.set_aspect('equal', adjustable='box')

		ax.grid() # type: ignore

		fill(ax, [minx, minx, maxx, maxx], [miny, maxy, maxy, miny], color="#6abdbe", alpha=0.7)

		ax.fill(*zip(*(fence1 + (fence1[0],))), color="#ffffff", alpha=1) # type: ignore
		ax.fill(*zip(*(fence2 + (fence2[0],))), color="#ffffff", alpha=1) # type: ignore


		if len(self.cones) > index and self.cones[index]:
			self.draw_cones(ax, index)
			self.draw_edges(ax, index)


		ax.fill(*zip(*(fence1 + (fence1[0],))), color="#8c00ff", alpha=0.2) # type: ignore
		ax.fill(*zip(*(fence2 + (fence2[0],))), color="#00a6ff", alpha=0.2) # type: ignore

		ax.plot(*zip(*(fence1 + (fence1[0],))), color="#ffffff", alpha=0.8, linewidth=4) # type: ignore
		ax.plot(*zip(*(fence2 + (fence2[0],))), color="#ffffff", alpha=0.8, linewidth=4) # type: ignore

		ax.plot(*zip(*(fence1 + (fence1[0],))), color="#8c00ff", alpha=0.8) # type: ignore
		ax.plot(*zip(*(fence2 + (fence2[0],))), color="#00a6ff", alpha=0.8) # type: ignore

		# draw_polygon(ax, polygon, color="#ffcc00", alpha=0.7, label="Polygon")

		for p in self.polygons:
			draw_polygon(ax, p)

		plot(ax, self.start.x, self.start.y, color="#00ff00", marker="o", label="Start")
		plot(ax, self.target.x, self.target.y, color="#ff00ff", marker="o", label="Target")

x = [
	[(-2.04, 5.06)],
	[(-3.7, 5.14)],
	[(-5.74, 2.0), (-4.5, -0.58), (1.94, -6.44), (5.86, -0.36), (0.0, -0.54), (-4.58, 1.74), (0.62, 3.88), (-2.02, 7.38), (-3.76, 7.02), (-2.08, 4.2)],
	[(5.08, -0.74), (0.08, -3.1), (1.76, -5.34)],
	[(-0.48, 6.8), (-1.78, 0.74), (-1.58, -5.62), (5.34, -6.52), (8.98, -0.38), (2.88, 0.3), (-0.84, 1.46), (7.38, 0.74), (7.84, 5.08)],
	[(2.4, 3.22), (4.86, 1.46), (5.8, 2.94), (1.16, 6.0), (-0.16, 6.0)],
	[(5.36, 7.12), (8.98, 0.68), (-6.54, -0.06), (0.06, 4.6), (-3.7, 6.34), (-3.18, 4.46), (-7.08, 5.52), (-4.54, 8.08)],
]

start = x[0]
target = x[1]
polygons = x[3:7:2]
fences = x[2:7:2]

print(fences)
print(polygons)

d = Drawing(Vector2(*start[0]), Vector2(*target[0]), [Polygon2([Vector2(*v) for v in p]) for p in polygons], [[Vector2(*v) for v in f] for f in fences])

d.draw()

