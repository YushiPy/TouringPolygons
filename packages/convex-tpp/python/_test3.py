
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

		side1: int = locate_ray(ray1)
		side2 = locate_ray(ray2)

		if side1 == side2 and ray1.cross(ray2) > 0:
			return [Vector2(), ray1, ray2]

		points = [Vector2(), ray1]
		flag = True

		while flag or side1 != side2:
			flag = False
			points.append(Vector2.from_polar(side_length * 2 ** .5, math.pi / 4 + side1 * math.pi / 2))
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

def find_point(point: Vector2, directions: list[Vector2]) -> int:

	def found(i: int, j: int) -> bool:
		return point_in_cone(point, Vector2(), directions[i], directions[j])

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

	width = max_x - min_x
	height = max_y - min_y
	
	min_x = center_x - width * scale / 2
	min_y = center_y - height * scale / 2
	max_x = center_x + width * scale / 2
	max_y = center_y + height * scale / 2

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

	index = find_point2(point, polygon, cones)

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

def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2) -> bool:

	p1 = point - vertex1
	p2 = point - vertex2
	dv = vertex2 - vertex1

	match (dv.cross(ray1) > 0, dv.cross(ray2) > 0):

		case (True, True):
			return ray2.cross(p2) < 0 or ray1.cross(p1) > 0 or dv.cross(p1) < 0

		case (False, False):
			return ray1.cross(p1) >= 0 and ray2.cross(p2) <= 0 and dv.cross(p1) <= 0

		case (True, False):
			return point_in_cone(point, vertex1, ray1, vertex1 - vertex2) or point_in_cone(point, vertex2, vertex1 - vertex2, ray2)

		case (False, True):
			return point_in_cone(point, vertex1, ray1, vertex2 - vertex1) or point_in_cone(point, vertex2, vertex2 - vertex1, ray2)

	if ray1.cross(ray2) < 0:
		return not point_in_edge(point, vertex2, vertex1, ray2, ray1)

	return ray1.cross(point - vertex1) >= 0 and ray2.cross(point - vertex2) <= 0 and (vertex2 - vertex1).cross(point - vertex1) <= 0

def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2) -> bool:

	if ray1.cross(ray2) < 0:
		return not point_in_cone(point, vertex, ray2, ray1)

	return ray1.cross(point - vertex) >= 0 and ray2.cross(point - vertex) <= 0

def find_point2(point: Vector2, polygon: Polygon2, cones: Cones) -> int:
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

def generate(center: Vector2, radius: float, num_sides: int, opening: float) -> tuple[Polygon2, Cones]:

	vertices: list[Vector2] = []
	cones: Cones = []

	for i in range(num_sides):

		angle = math.tau * i / num_sides

		vertex = center + Vector2.from_polar(radius, angle)
		vertices.append(vertex)

		ray1 = Vector2.from_polar(1.0, angle - opening / 2)
		ray2 = Vector2.from_polar(1.0, angle + opening / 2)

		cones.append((ray1, ray2))

	return Polygon2(vertices), cones

from matplotlib.axis import Axis
import matplotlib.pyplot as plt

def draw_edge(ax: Axis, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2) -> None:
	
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

	min_x, max_x, min_y, max_y = get_bbox(vertex1, vertex2, scale=2)
	corners = [Vector2(max_x, max_y), Vector2(min_x, max_y), Vector2(min_x, min_y), Vector2(max_x, min_y)]

	def plot(*args: Any, **kwargs: Any) -> None:

		kwargs2 = kwargs.copy()
		kwargs2["color"] = "white"
		kwargs2["linewidth"] = 4
		kwargs2["alpha"] = 1

		ax.plot(*args, **kwargs2) # type: ignore
		ax.plot(*args, **kwargs) # type: ignore
	
	def fill(*args: Any, **kwargs: Any) -> None:

		kwargs2 = kwargs.copy()
		kwargs2["color"] = "white"
		kwargs2["alpha"] = 1
		kwargs2.pop("label", None)

		ax.fill(*args, **kwargs2) # type: ignore
		ax.fill(*args, **kwargs) # type: ignore

	# set ax limits
	ax.set_xlim(min_x, max_x)
	ax.set_ylim(min_y, max_y)

	fill([min_x, max_x, max_x, min_x], [min_y, min_y, max_y, max_y], color="#c2eded", alpha=1)


	ray1 = extend_ray(vertex1, ray1)
	ray2 = extend_ray(vertex2, ray2)

	side1: int = locate_ray(vertex1, ray1)
	side2 = locate_ray(vertex2, ray2)

	rotated_corners = islice(cycle(corners), side1, side1 + 4)
	points = [vertex1, vertex1 + ray1]

	if side1 == side2 and ray1.cross(ray2) < 0:
		points = points + list(rotated_corners) + [vertex2 + ray2]
	else:

		while side1 != side2:
			points.append(next(rotated_corners))
			side1 = (side1 + 1) % 4

		points.append(vertex2 + ray2)

	points.append(vertex2)

	# ax.fill(*zip(*points), color="white", alpha=1)
	fill(*zip(*points), color="green", alpha=0.45, label="Região de Aresta")

	#ax.plot(*zip(vertex1, vertex1 + ray1), color="white", linewidth=3)
	plot(*zip(vertex1, vertex1 + ray1), color="black", linewidth=1.2)
	#ax.plot(*zip(vertex2, vertex2 + ray2), color="white", linewidth=3)
	plot(*zip(vertex2, vertex2 + ray2), color="black", linewidth=1.2)

	ax.scatter(*vertex1, color="white", edgecolor="white", zorder=5, s=100) # type: ignore
	ax.scatter(*vertex1, color="yellow", edgecolor="black", zorder=5) # type: ignore
	ax.scatter(*vertex2, color="yellow", edgecolor="black", zorder=5) # type: ignore

	# Draw segmented line
	ax.plot([vertex1.x, vertex2.x], [vertex1.y, vertex2.y], color="white", linestyle="-", linewidth=5) # type: ignore
	ax.plot([vertex1.x, vertex2.x], [vertex1.y, vertex2.y], color="black", linestyle="--", linewidth=1) # type: ignore

	# Draw arrow to represent ray1
	arrow1_start = vertex1 + ray1 * 0.1
	arrow1_end = vertex1 + ray1 * 0.5
	ax.arrow(arrow1_start.x, arrow1_start.y, arrow1_end.x - arrow1_start.x, arrow1_end.y - arrow1_start.y,
			  head_width=0.1, head_length=0.2, fc='green', ec='green') # type: ignore

	# Draw arrow to represent ray2
	arrow2_start = vertex2 + ray2 * 0.1
	arrow2_end = vertex2 + ray2 * 0.5
	ax.arrow(arrow2_start.x, arrow2_start.y, arrow2_end.x - arrow2_start.x, arrow2_end.y - arrow2_start.y,
			  head_width=0.1, head_length=0.2, fc='green', ec='green') # type: ignore

	# Draw small arrow from vertex1 to vertex2
	arrow_length = (vertex2 - vertex1).magnitude() * 0.7

	arrow_start = vertex1
	arrow_end = vertex1 + (vertex2 - vertex1).scale_to_length(0.2 + arrow_length)
	# Draw arrow on top of all else
	ax.arrow(arrow_start.x, arrow_start.y, arrow_end.x - arrow_start.x, arrow_end.y - arrow_start.y,
			  head_width=0.1, head_length=0.2, fc='orange', ec='orange', zorder=1) # type: ignore

	# Add label for vertex1
	pos = vertex1 + Vector2(-1, 1).scale_to_length(0.1)
	ax.text(pos.x, pos.y, "u", fontsize=10, ha='right', va='bottom', color='black')

	pos = vertex2 + Vector2(-1, 1).scale_to_length(0.1)
	ax.text(pos.x, pos.y, "v", fontsize=10, ha='right', va='bottom')

	match ((vertex2 - vertex1).cross(ray1) > 0, (vertex2 - vertex1).cross(ray2) > 0):
		case (True, True):
			ax.set_title("Ambos Raios no Lado Positivo")
			pass
		case (False, False):
			ax.set_title("Ambos Raios no Lado Negativo")
			pass
		case (True, False):
			ax.set_title("Raios em Lados Opostos (Caso 1)")
			r = extend_ray(vertex1, vertex1 - vertex2)
			ax.plot([vertex1.x, vertex1.x + r.x], [vertex1.y, vertex1.y + r.y], color="white", linestyle="-", linewidth=4) # type: ignore
			ax.plot([vertex1.x, vertex1.x + r.x], [vertex1.y, vertex1.y + r.y], color="green", linestyle="-", linewidth=4, alpha=0.1) # type: ignore
			ax.plot([vertex1.x, vertex1.x + r.x], [vertex1.y, vertex1.y + r.y], color="orange", linestyle="--", linewidth=2, label="Reta vu") # type: ignore

		case (False, True):
			ax.set_title("Raios em Lados Opostos (Caso 2)")
			r = extend_ray(vertex2, vertex2 - vertex1)
			ax.plot([vertex2.x, vertex2.x + r.x], [vertex2.y, vertex2.y + r.y], color="white", linestyle="-", linewidth=4) # type: ignore
			ax.plot([vertex2.x, vertex2.x + r.x], [vertex2.y, vertex2.y + r.y], color="green", linestyle="-", linewidth=4, alpha=0.1) # type: ignore
			ax.plot([vertex2.x, vertex2.x + r.x], [vertex2.y, vertex2.y + r.y], color="orange", linestyle="--", linewidth=2, label="Reta uv") # type: ignore

	
	# Increase legend font size
	ax.legend()
	
	# Remove ticks
	#plt.xticks([])
	#plt.yticks([])

	xticks = ax.get_xticks()
	yticks = ax.get_yticks()

	ax.set_xticks(xticks, labels=[""] * len(xticks))
	ax.set_yticks(yticks, labels=[""] * len(yticks))

	ax.grid()

	import os
	i = next(i for i in range(1, 10 ** 10) if not os.path.exists(f"Relatório/Problem1Images/edge_region_{i}.png"))

	plt.savefig("out.png")

	return


v1 = Vector2(2, 2)
v2 = Vector2(1, 1)

case1 = (Vector2(-1, 0.2), Vector2(-1, -0.3))
case2 = (Vector2(-1, -0.2), Vector2(2, 0))
case3 = (Vector2(2, 0), Vector2(-1, -0.2))
case4 = (-Vector2(-1, -0.2), -Vector2(-1, 0.3))

fig, ax = plt.subplots(2, 2, figsize=(10, 8.5)) # type: ignore
fig.tight_layout()

draw_edge(ax[0, 0], v1, v2, *case1)
draw_edge(ax[1, 0], v1, v2, *case2)
draw_edge(ax[0, 1], v1, v2, *case3)
draw_edge(ax[1, 1], v1, v2, *case4)


plt.show()
