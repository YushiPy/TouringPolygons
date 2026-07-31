
from math import isclose
from vector2 import Vector2
from u_tpp_fast_locate import Solution, segment_segment_intersection


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



def point_in_cone(point: Vector2, vertex: Vector2, ray1: Vector2, ray2: Vector2, eps: float = 1e-8) -> bool:
	"""
	Check if a point is inside the cone defined by two rays originating from a vertex.

	:param Vector2 point: The point to check.
	:param Vector2 vertex: The vertex of the cone.
	:param Vector2 ray1: The first ray direction.
	:param Vector2 ray2: The second ray direction.
	:param float eps: A small epsilon value for numerical stability. Positive values expand the cone, negative values contract it.

	:return: True if the point is inside the cone, False otherwise.
	"""

	diff = point - vertex

	if ray1.cross(ray2) >= 0:
		return ray1.cross(diff) >= -eps and ray2.cross(diff) <= eps
	else:
		return ray1.cross(diff) >= -eps or ray2.cross(diff) <= eps

def point_in_edge(point: Vector2, vertex1: Vector2, vertex2: Vector2, ray1: Vector2, ray2: Vector2, eps: float = 1e-8) -> bool:

	if vertex1.is_close(vertex2):
		return point_in_cone(point, vertex1, ray1, ray2)

	p1 = point - vertex1
	p2 = point - vertex2
	dv = vertex2 - vertex1

	if ray1.is_close(ray2):
		return dv.cross(p1) >= -eps and dv.cross(p2) <= eps

	match (dv.cross(ray1) >= -eps, dv.cross(ray2) >= -eps):

		case (True, True):
			return ray2.cross(p2) < eps or ray1.cross(p1) > -eps or dv.cross(p1) < -eps

		case (False, False):
			return ray1.cross(p1) >= -eps and ray2.cross(p2) <= eps and dv.cross(p1) <= eps

		case (True, False):
			return point_in_cone(point, vertex1, ray1, vertex1 - vertex2) or point_in_cone(point, vertex2, vertex1 - vertex2, ray2, eps)

		case (False, True):
			return point_in_cone(point, vertex1, ray1, vertex2 - vertex1) or point_in_cone(point, vertex2, vertex2 - vertex1, ray2, eps)



def get_bbox(points: list[Vector2], square: bool = False, scale: float = 1.0) -> tuple[float, float, float, float]:

	minx = min(p.x for p in points)
	maxx = max(p.x for p in points)
	miny = min(p.y for p in points)
	maxy = max(p.y for p in points)

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
			center.x + radius * cos(2 * pi * i / sides + rotation + pi / 2),
			center.y + radius * sin(2 * pi * i / sides + rotation + pi / 2)
		)
		for i in range(sides)
	]


start = Vector2(-1, 4)
target = Vector2(2, 3)

polygon = regular_polygon(Vector2(0, 0), 2, 6, rotation=0.2)

targets = [
	Vector2(4, 1),
	Vector2(4, 3),
	Vector2(2.2, -3),
]

sol = Solution(start, target, [polygon])
sol.solve()

bbox = get_bbox([start] + targets + polygon, square=True, scale=1.2)
minx, miny, maxx, maxy = bbox

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

for i, axis in enumerate(ax):

	axis.set_aspect('equal')

	axis.set_xlim(minx, maxx)
	axis.set_ylim(miny, maxy)
	
	axis.set_xticks([])
	axis.set_yticks([])

	axis.fill(*zip(*polygon), alpha=1.0, edgecolor='black', facecolor="#b2b2b2", linewidth=2.5)
	axis.plot(*zip(*(polygon + [polygon[0]])), color='black', linewidth=1.5)

	axis.scatter(*start, color='white', edgecolor='green', s=140, linewidth=2.5, zorder=5, label='Início' if i == 0 else "")
	axis.scatter(*targets[i], color='white', edgecolor='red', s=140, linewidth=2.5, zorder=5, label='Fim' if i == 0 else "")

	for j in range(len(sol.polygons[0])):

		if sol.blocked[0][j] and sol.blocked[0][j - 1]:
			continue

		v = sol.polygons[0][j]
		r1, r2 = sol.cones[0][j]

		points = locate_cone(v, r1, r2, bbox)
		alpha = 0.2

		if point_in_cone(targets[i], v, r1, r2):
			alpha += 0.3

			axis.plot(*zip(start, v, targets[i]), color='white', linewidth=4.4, zorder=4)
			axis.plot(*zip(start, v, targets[i]), color='purple', linewidth=2.5, linestyle='--', zorder=4, label='Caminho')


		axis.fill(*zip(*points), alpha=alpha, color='red')
		axis.plot(*zip(*points), color='black', linewidth=1.5)

	for j in range(len(sol.polygons[0])):

		if sol.blocked[0][j]:
			continue

		v1 = sol.polygons[0][j]
		v2 = sol.polygons[0][(j + 1) % len(sol.polygons[0])]
		r1 = sol.cones[0][j][1]
		r2 = sol.cones[0][(j + 1) % len(sol.polygons[0])][0]

		points = locate_edge(v1, r1, v2, r2, bbox)

		color = 'blue' if sol.blocked[0][j] else 'green'
		edgecolor = 'black' if sol.blocked[0][j] else 'white'
		alpha = 0.2
		
		if point_in_edge(targets[i], v1, v2, r1, r2):
			alpha += 0.3

			reflected = targets[i].reflect_segment(v1, v2)
			intersection = segment_segment_intersection(start, reflected, v1, v2)

			axis.plot(*zip(v1.lerp(v2, -10), v2.lerp(v1, -10)), color='green', linewidth=3.0, zorder=4)
			axis.plot(*zip(v1.lerp(v2, -10), v2.lerp(v1, -10)), color='white', linestyle="--", linewidth=2.4, zorder=4)

			axis.plot(*zip(start, intersection, targets[i]), color='white', linewidth=4.4, zorder=4)
			axis.plot(*zip(start, intersection, targets[i]), color='purple', linewidth=2.5, linestyle='--', zorder=4)

			axis.plot(*zip(intersection, reflected), color='white', linewidth=3.0, zorder=4)
			axis.plot(*zip(intersection, reflected), color='orange', linestyle="--", linewidth=2.0, zorder=4)

			axis.plot(*zip(reflected, targets[i]), color='white', linewidth=3.0, zorder=4)
			axis.plot(*zip(reflected, targets[i]), color='orange', linestyle="--", linewidth=2.0, zorder=4)

			axis.scatter(*reflected, color='white', edgecolor='orange', s=150, linewidth=2.5, zorder=5, label='Reflexão')

		axis.fill(*zip(*points), alpha=alpha, color=color, linewidth=2.5, edgecolor=edgecolor)

	vstart = next(j for j in range(len(sol.polygons[0])) if sol.blocked[0][j] and not sol.blocked[0][j - 1])
	vend = next(j for j in range(len(sol.polygons[0])) if not sol.blocked[0][j] and sol.blocked[0][j - 1])

	v1 = sol.polygons[0][vstart]
	v2 = sol.polygons[0][(vend) % len(sol.polygons[0])]
	r1 = sol.cones[0][vstart][1]
	r2 = sol.cones[0][(vend) % len(sol.polygons[0])][0]
	points = locate_edge(v1, r1, v2, r2, bbox)

	alpha = 0.2
	
	if point_in_edge(targets[i], v1, v2, r1, r2):
		alpha += 0.3

		axis.plot(*zip(start, targets[i]), color='white', linewidth=4.4, zorder=4)
		axis.plot(*zip(start, targets[i]), color='purple', linewidth=2.5, linestyle='--', zorder=4)

	axis.fill(*zip(*points), alpha=alpha, color='blue', linewidth=2.5, edgecolor='black', zorder=0)

	axis.legend(loc='upper right', fontsize=12)
