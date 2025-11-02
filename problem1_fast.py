
from collections.abc import Iterable
from itertools import cycle, islice
import math
from typing import Literal

from matplotlib import pyplot as plt
from vector2 import Vector2
from polygon2 import Polygon2


type Cones = list[tuple[Vector2, Vector2]]

type _Vector2 = Iterable[float]
type _Polygon2 = Iterable[_Vector2]


def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
	return ray1.cross(point - vertex1) >= 0 and ray2.cross(point - vertex2) <= 0 and (vertex2 - vertex1).cross(point - vertex1) <= 0

def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2) -> bool:

	if ray1.cross(ray2) < 0:
		return not point_in_cone(point, vertex, ray2, ray1)

	return ray1.cross(point - vertex) >= 0 and ray2.cross(point - vertex) <= 0

def locate_point(point: Vector2, polygon: Polygon2, cones: Cones) -> int:
	"""
	Locates point in cones or edges defined by polygon and cones.
	Returns index as follows:
	
	- `2n` -> cone in vertex `n`
	- `2n + 1` -> edge between vertex `n` and `n + 1`
	"""

	def is_between(i: int, j: int) -> bool:

		ray1 = cones[i // 2][i % 2]
		ray2 = cones[j // 2][j % 2]

		v1 = polygon[i // 2]
		v2 = polygon[j // 2]

		return point_in_edge(point, v1, v2, ray1, ray2)

	if is_between(0, 1):
		return 0

	left = 0
	right = 2 * len(cones) - 1

	while left + 1 != right:

		mid = (left + right) // 2

		if is_between(left, mid):
			right = mid
		else:
			left = mid

	return left

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

	cross = diff1.cross(diff2)

	if abs(cross) < 1e-8:
		return None
	
	sdiff = start2 - start1

	rate1 = sdiff.cross(diff2) / cross
	rate2 = sdiff.cross(diff1) / cross

	if 0 <= rate1 <= 1 and 0 <= rate2 <= 1:
		return start1 + diff1 * rate1
	
	return None


class Solution:

	start: Vector2
	target: Vector2
	polygons: list[Polygon2]

	cones: list[Cones]
	blocked: list[list[bool]]

	def __init__(self, start: _Vector2, target: _Vector2, polygons: Iterable[_Polygon2]) -> None:

		self.start = Vector2(start)
		self.target = Vector2(target)
		self.polygons = list(map(Polygon2, polygons))

		if any(not p.is_convex() for p in self.polygons):
			raise ValueError("All polygons must be convex.")

		self.cones = [[] for _ in self.polygons]
		self.blocked = [[] for _ in self.polygons]
	
	def query(self, point: Vector2, index: int) -> Vector2:
		"""
		Given a point and a polygon index, returns the last step of the smallest `index`-path from `start` to `point`.

		:param Vector2 point: The point to query.
		:param int index: The index of the polygon to query.

		:return Vector2: The last step of the smallest `index`-path from `start` to `point`.
		"""

		if index == 0:
			return self.start

		polygon = self.polygons[index - 1]
		cones = self.cones[index - 1]
		blocked = self.blocked[index - 1]

		location = locate_point(point, polygon, cones)

		if location % 2 == 0:
			return polygon[location // 2]

		if blocked[location // 2]:
			return self.query(point, index - 1)

		v1 = polygon[location // 2]
		v2 = polygon[(location // 2 + 1) % len(polygon)]

		reflected = point.reflect_segment(v1, v2)
		last = self.query(reflected, index - 1)

		intersection = segment_segment_intersection(last, reflected, v1, v2)

		if intersection is not None:
			return intersection
		
		raise ValueError("No intersection found, this should not happen.")

	def solve(self) -> list[Vector2]:
		"""
		Returns the shortest path from start to target touching all polygons in order.
		"""

		if not self.polygons:
			return [self.start, self.target]

		n = len(self.polygons)

		for i in range(n):

			polygon = self.polygons[i]
			m = len(polygon)

			cones = self.cones[i]
			blocked = self.blocked[i]
			
			fails: list[int] = [0] * m

			for j in range(m):

				vertex = polygon[j]

				before = polygon[j - 1]
				after = polygon[j + 1]

				last = self.query(vertex, i)
				diff = (vertex - last).normalize()

				ray1 = diff.reflect((vertex - before).perpendicular())
				ray2 = diff.reflect((vertex - after).perpendicular())

				if (vertex - before).cross(last - before) >= 0:
					ray1 = diff
					fails[j] += 1
				
				if (after - vertex).cross(last - vertex) >= 0:
					ray2 = diff
					fails[j] += 1

				cones.append((ray1, ray2))

			for j in range(m):
				blocked.append(fails[j] == 2 or fails[(j + 1) % m] == 2)

		return []

def get_bbox(*points: Vector2, square: bool = False, scale: float = 1.0) -> tuple[float, float, float, float]:
	"""
	Given an iterable of Vector2 points, return the bounding box as (min_x, max_x, min_y, max_y).	
	"""

	min_x = min(point.x for point in points)
	max_x = max(point.x for point in points)
	min_y = min(point.y for point in points)
	max_y = max(point.y for point in points)

	center_x = (min_x + max_x) / 2
	center_y = (min_y + max_y) / 2

	if square:
		half_size = max((max_x - min_x), (max_y - min_y)) / 2
		min_x = center_x - half_size
		max_x = center_x + half_size
		min_y = center_y - half_size
		max_y = center_y + half_size
	
	min_x = center_x + (min_x - center_x) * scale
	max_x = center_x + (max_x - center_x) * scale
	min_y = center_y + (min_y - center_y) * scale
	max_y = center_y + (max_y - center_y) * scale

	return min_x, max_x, min_y, max_y

def draw(polygon: Polygon2, cones: Cones, point: Vector2) -> None:
	
	import matplotlib.pyplot as plt

	def plot(*args: Any, **kwargs: Any) -> None:

		kwargs2 = kwargs.copy()
		kwargs2["color"] = "white"
		kwargs2["alpha"] = 1

		ax.plot(*args, **kwargs2) # type: ignore
		ax.plot(*args, **kwargs) # type: ignore

	def fill(*args: Any, **kwargs: Any) -> None:

		kwargs2 = kwargs.copy()
		kwargs2["color"] = "white"
		kwargs2["alpha"] = 1

		ax.fill(*args, **kwargs2) # type: ignore
		ax.fill(*args, **kwargs) # type: ignore

	def locate_ray(vertex: Vector2, ray: Vector2) -> Literal[0, 1, 2, 3]:

		match ray.quadrant():
			case 0: return 0 if (max_x - vertex.x) * ray.y <= (max_y - vertex.y) * ray.x else 1
			case 1: return 2 if (vertex.x - min_x) * ray.y <= (max_y - vertex.y) * -ray.x else 1
			case 2: return 2 if (vertex.x - min_x) * -ray.y <= (vertex.y - min_y) * -ray.x else 3
			case 3: return 0 if (max_x - vertex.x) * -ray.y <= (vertex.y - min_y) * ray.x else 3
		
	def extend_ray(vertex: Vector2, ray: Vector2) -> Vector2:
		
		match locate_ray(vertex, ray):
			case 0: return ray * (max_x - vertex.x) / ray.x
			case 1: return ray * (max_y - vertex.y) / ray.y
			case 2: return ray * (vertex.x - min_x) / -ray.x
			case 3: return ray * (vertex.y - min_y) / -ray.y

	def get_points(vertex: Vector2, ray1: Vector2, ray2: Vector2) -> list[Vector2]:

		ray1 = extend_ray(vertex, ray1)
		ray2 = extend_ray(vertex, ray2)

		side1: int = locate_ray(vertex, ray1)
		side2 = locate_ray(vertex, ray2)

		rotated_corners = islice(cycle(corners), side1, side1 + 4)
		points = [vertex, vertex + ray1]

		if side1 == side2 and ray1.cross(ray2) < 0:
			return points + list(rotated_corners) + [vertex + ray2]

		while side1 != side2:
			points.append(next(rotated_corners))
			side1 = (side1 + 1) % 4

		points.append(vertex + ray2)
	
		return points

	def get_points2(vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2) -> list[Vector2]:

		ray1 = extend_ray(vertex1, ray1)
		ray2 = extend_ray(vertex2, ray2)

		side1: int = locate_ray(vertex1, ray1)
		side2 = locate_ray(vertex2, ray2)

		rotated = islice(cycle(corners), side1, side1 + 4)
		points = [vertex1, vertex1 + ray1]

		if side1 == side2 and ray1.cross(ray2) < 0:
			return points + list(rotated) + [vertex2 + ray2, vertex2]

		while side1 != side2:
			points.append(next(rotated))
			side1 = (side1 + 1) % 4

		points.append(vertex2 + ray2)
		points.append(vertex2)

		return points

	def draw_cone(vertex: Vector2, ray1: Vector2, ray2: Vector2, *args: Any, **kwargs: Any) -> None:

		points = get_points(vertex, ray1, ray2)

		fill(*zip(*points), *args, **kwargs)

		p1 = vertex + extend_ray(vertex, ray1)
		p2 = vertex + extend_ray(vertex, ray2)

		plot([vertex.x, p1.x], [vertex.y, p1.y], color="black")
		plot([vertex.x, p2.x], [vertex.y, p2.y], color="black")

	def draw_edge(vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2, *args: Any, **kwargs: Any) -> None:

		points = get_points2(vertex1, vertex2, ray1, ray2)

		fill(*zip(*points), *args, **kwargs)

		p1 = vertex1 + extend_ray(vertex1, ray1)
		p2 = vertex2 + extend_ray(vertex2, ray2)

		plot([vertex1.x, p1.x], [vertex1.y, p1.y], color="black")
		plot([vertex2.x, p2.x], [vertex2.y, p2.y], color="black")

	min_x, max_x, min_y, max_y = get_bbox(*polygon, point, square=True, scale=1.2)
	corners = [Vector2(max_x, max_y), Vector2(min_x, max_y), Vector2(min_x, min_y), Vector2(max_x, min_y)]

	fig, ax = plt.subplots() # type: ignore

	ax.set_xlim(min_x, max_x)
	ax.set_ylim(min_y, max_y)

	fill(*zip(*corners), color="#afbebe", alpha=1)

	for vertex, (ray1, ray2) in zip(polygon, cones):
		draw_cone(vertex, ray1, ray2, alpha=0.5, color="blue")

	for i in range(len(polygon)):

		vertex1 = polygon[i]
		vertex2 = polygon[(i + 1) % len(polygon)]

		ray1 = cones[i][1]
		ray2 = cones[(i + 1) % len(polygon)][0]

		draw_edge(vertex1, vertex2, ray1, ray2, alpha=0.5, color="green")

	index = locate_point(point, polygon, cones)

	if index % 2 == 0:
		vertex = polygon[index // 2]
		ray1, ray2 = cones[index // 2]
		draw_cone(vertex, ray1, ray2, alpha=0.8, color="red")
	else:
		i = index // 2
		vertex1 = polygon[i]
		vertex2 = polygon[(i + 1) % len(polygon)]
		ray1 = cones[i][1]
		ray2 = cones[(i + 1) % len(polygon)][0]
		draw_edge(vertex1, vertex2, ray1, ray2, alpha=0.8, color="red")

	fill(*zip(*polygon), color="red", alpha=0.3)
	plot(*zip(*(polygon + (polygon[0],))), color="black", linewidth=1.2)

	plot(*point, marker="o", color="white", markersize=8, markeredgecolor="black", zorder=5)

	ax.grid(True, which='both', linestyle='--', linewidth=1) # type: ignore
	fig.tight_layout()

	plt.show() # type: ignore


test1 = Solution(
	Vector2(5, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(2, 4), Vector2(1, 4), Vector2(-1, 1)]),
		Polygon2([Vector2(3, 3), Vector2(4, 3), Vector2(4, 4), Vector2(3, 4)]),
		Polygon2([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)]),
	]
)

from math import pi, tau

test2 = (
	Vector2(-1, -1),
	Vector2(1, -1),
	[
		Polygon2([Vector2.from_polar(2, i * tau / 6 + pi * 0.35) + Vector2(4, 5) for i in range(6)]),
		Polygon2([Vector2.from_polar(2, i * tau / 3 + pi /4) + Vector2(-3, 4) for i in range(3)]),
		Polygon2([Vector2.from_polar(2, i * tau / 10) + Vector2(5, -4) for i in range(10)]),
		Polygon2([Vector2.from_polar(2, i * tau / 4 + pi / 4) + Vector2(-4, -2) for i in range(4)]),
		Polygon2([Vector2.from_polar(2, i * tau / 30 + pi / 4) + Vector2(0, -8) for i in range(30)]),
	]
)

from problem1_draw import Drawing

sol = Solution(*test2)
other = Drawing(*test2)

other.shortest_path()
sol.solve()

other.draw()

other.cones = sol.cones
other.draw()
