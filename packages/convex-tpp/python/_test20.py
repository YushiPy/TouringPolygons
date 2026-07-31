
from collections.abc import Sequence
from math import isclose
from LegacySolutions.vector2 import Vector2

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
	raise ValueError("Ray does not intersect with bounding box.")

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

	if w1 == w2 and direction1.cross(direction2) < 0:
		return [start1, p1] + corners[w1:] + corners[:w1] + [p2, start2]
	
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


def get_bbox(points: Sequence[Vector2 | tuple[float, float]], square: bool = False, scale: float = 1.0) -> tuple[float, float, float, float]:

	minx = min(p[0] for p in points)
	maxx = max(p[0] for p in points)
	miny = min(p[1] for p in points)
	maxy = max(p[1] for p in points)

	center = (minx + maxx) / 2, (miny + maxy) / 2
	width = (maxx - minx) * scale
	height = (maxy - miny) * scale

	if square:
		width = max(width, height)	
		height = width

	minx = center[0] - width / 2
	maxx = center[0] + width / 2
	miny = center[1] - height / 2
	maxy = center[1] + height / 2

	return minx, miny, maxx, maxy

def regular_polygon(center: Vector2, radius: float, sides: int, rotation: float = 0.0) -> list[Vector2]:
	from math import cos, sin, pi
	return [
		Vector2(
			center[0] + radius * cos(2 * pi * i / sides + rotation * 2 * pi + pi / 2),
			center[1] + radius * sin(2 * pi * i / sides + rotation * 2 * pi + pi / 2)
		)
		for i in range(sides)
	]


import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


from common import Solution

start = Vector2(1.0, -3.0)
polygons = [
	regular_polygon(Vector2(-3, -2), 1, 4, rotation=0.15),
	regular_polygon(Vector2(2, 1), 2, 3, rotation=0.05)
]

polygon_index = 1

solution = Solution(start, start, polygons[:polygon_index + 1]) # type: ignore
polygon = list(map(Vector2, solution.polygons[polygon_index]))

bbox = get_bbox([start] + [v for poly in polygons for v in poly], square=True, scale=1.5)
bbox_width = bbox[2] - bbox[0]

fig: Figure
ax: Axes

for i in range(len(polygon)):

	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	ax.set_aspect('equal')
	ax.set_xlim(bbox[0], bbox[2])
	ax.set_ylim(bbox[1], bbox[3])
	ax.set_xticks([])
	ax.set_yticks([])

	ax.scatter(*start, color='green', edgecolor='white', s=80, zorder=5, linewidth=2)
	ax.text(start[0] - bbox_width * 0.03, start[1], '$s$', color='black', fontsize=18, ha='center', va='center', zorder=6)

	u = polygon[i]
	v = polygon[(i + 1) % len(polygon)]

	m = (u + v) * 0.5
	perp = -(v - u).perpendicular().normalize()

	p = m + perp * bbox_width * 0.04

	ax.text(p[0], p[1], f'$e_{i + 1}$', color='black', fontsize=18, ha='center', va='center', zorder=6)

	ax.fill([bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3]], color='lightgray', alpha=0.5, zorder=0)

	for j, poly in enumerate(polygons):

		color = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow'][j % 6]
		ax.fill(*zip(*poly), alpha=0.7, edgecolor='black')

		centerx = sum(p[0] for p in poly) / len(poly)
		centery = sum(p[1] for p in poly) / len(poly)
		center = (centerx, centery)

		ax.text(*center, f'$P_{j + 1}$', color='black', fontsize=18, ha='center', va='center', zorder=6)
	
	color = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow'][polygon_index % 6]

	ax.plot(*zip(u.lerp(v, -100), v.lerp(u, -100)), color="white", linestyle='-', linewidth=3, zorder=4)
	ax.plot(*zip(u.lerp(v, -100), v.lerp(u, -100)), color=color, linestyle='--', linewidth=2, zorder=4)

	path = list(map(Vector2, solution.query_full(u, polygon_index)))
	q = path[-2]

	edge_color = "green" if (v - u).cross(q - u) < 0 else "red"
	ax.plot(*zip(u, v), color="white", linestyle='-', linewidth=3, zorder=4)
	ax.plot(*zip(u, v), color=edge_color, linestyle='--', linewidth=2, zorder=4)

	ax.scatter(*u, color=edge_color, edgecolor='white', s=80, zorder=5, linewidth=2)
	ax.scatter(*v, color=edge_color, edgecolor='white', s=80, zorder=5, linewidth=2)

	r1 = (polygon[i] - polygon[i - 1]).normalize() + (polygon[i] - polygon[(i + 1) % len(polygon)]).normalize()
	ax.text(*(u + r1.scale_to_length(bbox_width * 0.05)), f'$u$', color='black', fontsize=18, ha='center', va='center', zorder=6)

	r2 = (polygon[(i + 1) % len(polygon)] - polygon[i]).normalize() + (polygon[(i + 1) % len(polygon)] - polygon[(i + 2) % len(polygon)]).normalize()
	ax.text(*(v + r2.scale_to_length(bbox_width * 0.05)), f'$v$', color='black', fontsize=18, ha='center', va='center', zorder=6)


	ax.scatter(*q, color=edge_color, edgecolor='white', s=80, zorder=5, linewidth=2)
	ax.text(q[0] + bbox_width * 0.05, q[1], f'$q$', color='black', fontsize=18, ha='center', va='center', zorder=6)

	points1 = locate_edge(u, u - v, v, v - u, bbox)
	ax.fill(*zip(*points1), color='green', alpha=0.5, label='Externo', zorder=0)

	points2 = locate_edge(v, v - u, u, u - v, bbox)
	ax.fill(*zip(*points2), color='red', alpha=0.4, label='Interno', zorder=0)

	ax.plot(*zip(*path), color='white', linestyle='-', linewidth=3, zorder=4)
	ax.plot(*zip(*path), color='purple', linestyle='--', linewidth=2, zorder=4, label=f'{polygon_index}-path até $u$')

	ax.legend(loc='upper left', fontsize=12)

	plt.savefig(f"aresta-{i}.png", dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(bbox[0], bbox[2])
ax.set_ylim(bbox[1], bbox[3])
ax.set_xticks([])
ax.set_yticks([])

ax.scatter(*start, color='green', edgecolor='white', s=80, zorder=5, linewidth=2)
ax.text(start[0] - bbox_width * 0.03, start[1], '$s$', color='black', fontsize=18, ha='center', va='center', zorder=6)

ax.fill([bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3]], color='lightgray', alpha=0.5, zorder=0)

for j, poly in enumerate(polygons):

	color = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow'][j % 6]
	ax.fill(*zip(*poly), alpha=0.7, edgecolor='black')

	centerx = sum(p[0] for p in poly) / len(poly)
	centery = sum(p[1] for p in poly) / len(poly)
	center = (centerx, centery)

	ax.text(*center, f'$P_{j + 1}$', color='black', fontsize=18, ha='center', va='center', zorder=6)

for j in range(len(polygon)):
	u = polygon[j]
	v = polygon[(j + 1) % len(polygon)]

	q = Vector2(solution.query(u, polygon_index))
	edge_color = "green" if (v - u).cross(q - u) < 0 else "red"

	ax.plot(*zip(u, v), color="white", linestyle='-', linewidth=3, zorder=4)
	ax.plot(*zip(u, v), color=edge_color, linestyle='--', linewidth=2, zorder=4)

	m = (u + v) * 0.5
	perp = -(v - u).perpendicular().normalize()

	p = m + perp * bbox_width * 0.04
	ax.text(p[0], p[1], f'$e_{j + 1}$', color='black', fontsize=18, ha='center', va='center', zorder=6)

ind = 2
pos = polygon[ind] - bbox_width * 0.04 * ((polygon[(ind + 1) % len(polygon)] - polygon[ind]).normalize() + (polygon[(ind - 1) % len(polygon)] - polygon[ind]).normalize())
ax.text(*pos, f'$T_{polygon_index + 1}$', color='black', fontsize=18, ha='center', va='center', zorder=6)

plt.savefig('arestas-resultado.png', dpi=300, bbox_inches='tight')
