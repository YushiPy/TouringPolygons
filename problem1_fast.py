
from itertools import cycle, islice
import math

from typing import Any, Literal


from vector2 import Vector2
from polygon2 import Polygon2


type Cone = tuple[Vector2, Vector2]
type Cones = list[Cone]

def locate_point(point: Vector2, directions: list[Vector2]) -> int:

	if len(directions) < 2:
		raise ValueError("Should have >= 2 directions")

	def locate_ray(ray: Vector2) -> Literal[0, 1, 2, 3]:
		return ray.rotate(math.pi / 4).quadrant()

	def extend_ray(ray: Vector2) -> Vector2:

		match locate_ray(ray):
			case 0: return ray * side_length / ray.x
			case 1: return ray * side_length / ray.y
			case 2: return ray * -side_length / ray.x
			case 3: return ray * -side_length / ray.y

	import matplotlib.pyplot as plt

	def get_points(ray1: Vector2, ray2: Vector2) -> list[Vector2]:

		ray1 = extend_ray(ray1)
		ray2 = extend_ray(ray2)

		side1 = locate_ray(ray1)
		side2 = locate_ray(ray2)

		if side1 == side2 and ray1.cross(ray2) > 0:
			return [Vector2(), ray1, ray2]

		points = [Vector2(), ray1]
		flag = True

		while flag or side1 != side2:
			flag = False
			points.append(Vector2.from_spherical(side_length * 2 ** .5, math.pi / 4 + side1 * math.pi / 2))
			side1 = (side1 + 1) % 4

		points.append(ray2)
	
		return points

	def draw_cone(ray1: Vector2, ray2: Vector2, *args: Any, **kwargs: Any) -> None:

		ray1 = extend_ray(ray1)
		ray2 = extend_ray(ray2)

		points = get_points(ray1, ray2)

		# plt.fill(*zip(*points), color="white", alpha=1)
		plt.fill(*zip(*points), *args, **kwargs) # type: ignore
		plt.plot([0.0, ray1.x], [0.0, ray1.y], color="black") # type: ignore

	side_length = point.magnitude() * 1.5

	fig, ax = plt.subplots() # type: ignore

	ax.set_xlim(-side_length, side_length)
	ax.set_ylim(-side_length, side_length)

	# directions.sort(key=Vector2.angle)

	for i in range(len(directions)):

		ray1 = directions[i]
		ray2 = directions[(i + 1) % len(directions)]

		color = ["blue", "red"][i % 2]

		draw_cone(ray1, ray2, alpha=0.5, color=color)

	# cone = next(i for i in range(len(directions)) if point_in_cone(directions[i], directions[(i + 1) % len(directions)], point))
	cone = find_point(point, directions)

	ray1 = directions[cone]
	ray2 = directions[(cone + 1) % len(directions)]

	draw_cone(ray1, ray2, alpha=0.8, color="green")

	ax.axhline(0, color='black', linewidth=1, alpha=0.3)  # x-axis # type: ignore
	ax.axvline(0, color='black', linewidth=1, alpha=0.3)  # y-axis # type: ignore

	ax.grid(True, which='both', linestyle='--', linewidth=0.5) # type: ignore

	ax.scatter(*point, alpha=1, color="white", edgecolor="black", zorder=5) # type: ignore

	plt.show() # type: ignore

	return cone

def point_in_cone(ray1: Vector2, ray2: Vector2, point: Vector2) -> bool:

	if ray1.cross(ray2) < 0:
		return not point_in_cone(ray2, ray1, point)

	return ray1.cross(point) >= 0 and ray2.cross(point) <= 0

def find_point(point: Vector2, directions: list[Vector2]) -> int:

	def found(i: int, j: int) -> bool:
		return point_in_cone(directions[i], directions[j], point)

	if found(-1, 0):
		return len(directions) - 1

	left = 0
	right = len(directions) - 1

	while left + 1 != right:

		mid = (left + right) // 2

		if found(left, mid):
			right = mid
		else:
			left = mid
	
	return left


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
			case 0: return vertex + ray * (max_x - vertex.x) / ray.x
			case 1: return vertex + ray * (max_y - vertex.y) / ray.y
			case 2: return vertex + ray * (vertex.x - min_x) / -ray.x
			case 3: return vertex + ray * (vertex.y - min_y) / -ray.y

	def get_points(vertex: Vector2, ray1: Vector2, ray2: Vector2) -> list[Vector2]:

		side1 = locate_ray(vertex, ray1)
		side2 = locate_ray(vertex, ray2)

		rotated_corners = islice(cycle(corners), side1, side1 + 4)

		if side1 == side2 and ray1.cross(ray2) < 0:
			return [vertex, ray1] + list(rotated_corners) + [ray2]

		points = [vertex, extend_ray(vertex, ray1)]

		while side1 != side2:
			points.append(next(rotated_corners))
			side1 = (side1 + 1) % 4

		points.append(extend_ray(vertex, ray2))
	
		return points

	def draw_cone(vertex: Vector2, ray1: Vector2, ray2: Vector2, *args: Any, **kwargs: Any) -> None:

		points = get_points(vertex, ray1, ray2)

		print(points)

		fill(*zip(*points), *args, **kwargs)

		p1 = vertex + ray1.scale_to_length(base_length)
		p2 = vertex + ray2.scale_to_length(base_length)

		plot([vertex.x, p1.x], [vertex.y, p1.y], color="black")
		plot([vertex.x, p2.x], [vertex.y, p2.y], color="black")

	min_x, max_x, min_y, max_y = get_bbox(*polygon, point, square=True, scale=1.2)
	base_length = max(max_x - min_x, max_y - min_y) * 2 ** 0.5

	corners = [Vector2(max_x, max_y), Vector2(min_x, max_y), Vector2(min_x, min_y), Vector2(max_x, min_y)]

	fig, ax = plt.subplots() # type: ignore

	ax.set_xlim(min_x, max_x)
	ax.set_ylim(min_y, max_y)

	fill(*zip(*corners), color="#afbebe", alpha=1)

	for vertex, (ray1, ray2) in zip(polygon, cones):
		draw_cone(vertex, ray1, ray2, alpha=0.5, color="blue")

	fill(*zip(*polygon), color="red", alpha=0.3)
	plot(*zip(*(polygon + (polygon[0],))), color="black", linewidth=1.2)

	plot(*point, marker="o", color="white", markersize=8, markeredgecolor="black", zorder=5)

	ax.grid(True, which='both', linestyle='--', linewidth=1) # type: ignore
	fig.tight_layout()


def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2) -> bool:

	if ray1.cross(ray2) < 0:
		return not point_in_edge(point, vertex2, vertex1, ray2, ray1)

	return ray1.cross(point - vertex1) >= 0 and ray2.cross(point - vertex2) <= 0 and (vertex2 - vertex1).cross(point) <= 0

def find_point2(point: Vector2, polygon: Polygon2, cones: Cones) -> int:

	pass

def generate(center: Vector2, radius: float, num_sides: int, opening: float) -> tuple[Polygon2, Cones]:

	vertices: list[Vector2] = []
	cones: Cones = []

	for i in range(num_sides):

		angle = math.tau * i / num_sides

		vertex = center + Vector2.from_spherical(radius, angle)
		vertices.append(vertex)

		ray1 = Vector2.from_spherical(1.0, angle - opening / 2)
		ray2 = Vector2.from_spherical(1.0, angle + opening / 2)

		cones.append((ray1, ray2))

	return Polygon2(vertices), cones

#polygon = Polygon2([Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)])
#cones = [(Vector2(-1, 0), Vector2(0, -1)), (Vector2(0, -1), Vector2(1, 0)), (Vector2(1, 0), Vector2(0, 1)), (Vector2(0, 1), Vector2(-1, 0))]
#point = Vector2(-3, 1.5)

#polygon, cones = generate(Vector2(), 5.0, 12, math.pi / 10)
#point = Vector2(-10.0, 4.0)

#draw(polygon, cones, point)

vertex1 = Vector2(1, 0)
vertex2 = Vector2(-1, 0)
ray1 = Vector2(1, 0).normalize()
ray2 = Vector2(-1, 0).normalize()

left = min(vertex1.x, vertex2.x) - 1
right = max(vertex1.x, vertex2.x) + 1
bottom = min(vertex1.y, vertex2.y) - 1
top = max(vertex1.y, vertex2.y) + 1

import matplotlib.pyplot as plt
import random

points = [Vector2(random.uniform(left, right), random.uniform(bottom, top)) for _ in range(1000)]
inside = [point_in_edge(point, vertex1, vertex2, ray1, ray2) for point in points]

plt.fill([left, right, right, left], [bottom, bottom, top, top], color="#afbebe", alpha=1)
plt.scatter([p.x for p in points], [p.y for p in points], c=["red" if inc else "blue" for inc in inside])

plt.plot(vertex1.x, vertex1.y, marker="o", color="white", markersize=8, markeredgecolor="black", zorder=5) # type: ignore
plt.plot(vertex2.x, vertex2.y, marker="o", color="white", markersize=8, markeredgecolor="black", zorder=5) # type: ignore

plt.plot([vertex1.x, vertex1.x + ray1.x * 1], [vertex1.y, vertex1.y + ray1.y * 1], color="black") # type: ignore
plt.plot([vertex2.x, vertex2.x + ray2.x * 1], [vertex2.y, vertex2.y + ray2.y * 1], color="black") #
