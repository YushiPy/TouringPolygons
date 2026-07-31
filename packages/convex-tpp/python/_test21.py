
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

start = Vector2(-1.7, -0.0)
polygons = [
	regular_polygon(Vector2(-0, -0), 1, 5, rotation=0.05),
	#regular_polygon(Vector2(2, 1), 2, 3, rotation=0.05)
]

polygon_index = 0

solution = Solution(start, start, polygons[:polygon_index + 1]) # type: ignore
polygon = list(map(Vector2, solution.polygons[polygon_index]))
cones = [(Vector2(a), Vector2(b)) for a, b in solution.cones[polygon_index]]

bbox = get_bbox([start] + [v for poly in polygons for v in poly], square=True, scale=2.0)
bbox_width = bbox[2] - bbox[0]

fig: Figure
ax: Axes

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(bbox[0], bbox[2])
ax.set_ylim(bbox[1], bbox[3])
ax.set_xticks([])
ax.set_yticks([])

ax.scatter(*start, color='green', edgecolor='white', s=80, zorder=5, linewidth=2)
ax.text(start[0] - bbox_width * 0.04, start[1], '$s$', color='black', fontsize=18, ha='center', va='center', zorder=6)

ax.fill([bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3]], color='lightgray', alpha=0.5, zorder=0)

for j, poly in enumerate(polygons):

	color = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow'][j % 6]
	ax.fill(*zip(*poly), alpha=0.7, edgecolor='black')

	centerx = sum(p[0] for p in poly) / len(poly)
	centery = sum(p[1] for p in poly) / len(poly)
	center = (centerx, centery)

	ax.text(*center, f'$P_{j + 1}$', color='black', fontsize=18, ha='center', va='center', zorder=6)

for j in range(len(polygon)):

	v = polygon[j]
	ray1, ray2 = cones[j]

	points = locate_cone(v, ray1, ray2, bbox)

	ax.fill(*zip(*points), color='red', alpha=0.5, edgecolor='black')
	ax.plot(*zip(*(points + [points[0]])), color='red', zorder=4)

	scale = 0.2

	if j == 1:
		scale = 0.06
	elif j in {3, 4}:
		scale = 0.15

	p = v + (ray2.normalize() + ray1.normalize()).normalize() * bbox_width * scale

	if ray1 == ray2:
		ax.text(p[0], p[1] - bbox_width * 0.00, f'$v^{j + 1}$', color='black', fontsize=16, ha='center', va='center', zorder=6, backgroundcolor='white')
	else:
		ax.text(p[0], p[1] - bbox_width * 0.00, f'$v^{j + 1}$', color='black', fontsize=16, ha='center', va='center', zorder=6)

for j in range(len(polygon)):

	u = polygon[j]
	v = polygon[(j + 1) % len(polygon)]

	ray1 = cones[j][1]
	ray2 = cones[(j + 1) % len(polygon)][0]

	points = locate_edge(u, ray1, v, ray2, bbox)
	ax.fill(*zip(*points), color='green', alpha=0.5)

	p = (u + v) / 2 + (ray1.normalize() + ray2.normalize()).normalize() * bbox_width * 0.2
	ax.text(p[0], p[1] - bbox_width * 0.00, f'$e^{j + 1}$', color='black', fontsize=16, ha='center', va='center', zorder=6)

plt.savefig('last_step_map_new.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(bbox[0], bbox[2])
ax.set_ylim(bbox[1], bbox[3])
ax.set_xticks([])
ax.set_yticks([])

ax.scatter(*start, color='green', edgecolor='white', s=80, zorder=5, linewidth=2)
ax.text(start[0] - bbox_width * 0.04, start[1], '$s$', color='black', fontsize=18, ha='center', va='center', zorder=6)

ax.fill([bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3]], color='lightgray', alpha=0.5, zorder=0)

for j, poly in enumerate(polygons):

	color = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow'][j % 6]
	ax.fill(*zip(*poly), alpha=0.7, edgecolor='black')

	centerx = sum(p[0] for p in poly) / len(poly)
	centery = sum(p[1] for p in poly) / len(poly)
	center = (centerx, centery)

	ax.text(*center, f'$P_{j + 1}$', color='black', fontsize=18, ha='center', va='center', zorder=6)

for j in range(len(polygon)):

	v = polygon[j]
	ray1, ray2 = cones[j]

	points = locate_cone(v, ray1, ray2, bbox)

	ax.fill(*zip(*points), color='red', alpha=0.5, edgecolor='black')
	ax.plot(*zip(*(points + [points[0]])), color='red', zorder=4)

for j in range(len(polygon)):

	u = polygon[j]
	v = polygon[(j + 1) % len(polygon)]

	ray1 = cones[j][1]
	ray2 = cones[(j + 1) % len(polygon)][0]

	points = locate_edge(u, ray1, v, ray2, bbox)
	ax.fill(*zip(*points), color='green', alpha=0.5)

l = 2
r = 5

u = polygon[l]
v = polygon[r % len(polygon)]

ray1 = cones[l][0]
ray2 = cones[r % len(polygon)][0]
points = locate_edge(u, ray1, v, ray2, bbox)
ax.fill(*zip(*points), color='purple', alpha=0.5)

ax.scatter(*u, color='black', edgecolor='white', s=80, zorder=5, linewidth=2)
ax.scatter(*v, color='black', edgecolor='white', s=80, zorder=5, linewidth=2)

ax.text(u[0] - bbox_width * 0.02, u[1] - bbox_width * 0.06, f'$v^{l + 1}$', color='black', fontsize=16, ha='center', va='center', zorder=6)
ax.text(v[0] - bbox_width * 0.02, v[1] + bbox_width * 0.06, f'$v^{r + 1}$', color='black', fontsize=16, ha='center', va='center', zorder=6)

ax.plot(*zip(u, u + ray1.normalize() * bbox_width * 1), color='white', linestyle='-', zorder=4, linewidth=4)
ax.plot(*zip(u, u + ray1.normalize() * bbox_width * 1), color='purple', linestyle='--', zorder=4, linewidth=2)

ax.plot(*zip(v, v + ray2.normalize() * bbox_width * 1), color='white', linestyle='-', zorder=4, linewidth=4)
ax.plot(*zip(v, v + ray2.normalize() * bbox_width * 1), color='purple', linestyle='--', zorder=4, linewidth=2)

ax.plot(*zip(u, v), color='white', linestyle='-', zorder=4, linewidth=4)
ax.plot(*zip(u, v), color='purple', linestyle='--', zorder=4, linewidth=2)

p = (u + v) / 2 + (ray1.normalize() + ray2.normalize()).normalize() * bbox_width * 0.35

ax.text(p[0], p[1] - bbox_width * 0.00, '$R$', color='purple', fontsize=24, ha='center', va='center', zorder=6, backgroundcolor='pink')

plt.savefig('fake_edge.png', dpi=300, bbox_inches='tight')
