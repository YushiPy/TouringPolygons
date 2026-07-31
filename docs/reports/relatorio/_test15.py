
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


import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def draw(ax: Axes, start: Vector2, target: Vector2, polygon: list[Vector2]) -> None:

	solution = Solution(start, target, [polygon])
	solution.solve()

	bbox = get_bbox([start] + targets + polygon, square=True, scale=1.2)
	minx, miny, maxx, maxy = bbox

	ax.set_xlim(minx, maxx)
	ax.set_ylim(miny, maxy)
	ax.set_aspect('equal', adjustable='box')

	ax.set_xticks([])
	ax.set_yticks([])

	ax.fill(*zip(*polygon), alpha=1.0, edgecolor='black', facecolor="#b2b2b2", linewidth=2.5)
	ax.plot(*zip(*(polygon + [polygon[0]])), color='black', linewidth=1.5)

	ax.scatter(*start, color='white', edgecolor='green', s=140, linewidth=2.5, zorder=5, label='Início')
	ax.scatter(*target, color='white', edgecolor='red', s=140, linewidth=2.5, zorder=5, label='Fim')

	blocked = solution.blocked[0]
	poly = solution.polygons[0]
	cones = solution.cones[0]

	for j in range(len(poly)):

		if blocked[j] and blocked[j - 1]:
			continue

		v = poly[j]
		r1, r2 = cones[j]

		points = locate_cone(v, r1, r2, bbox)
		ax.fill(*zip(*points), alpha=0.2, color='red')
		ax.plot(*zip(*points), color='black', linewidth=1.5)
	
	for j in range(len(poly)):
		
		if blocked[j]:
			continue

		v1 = poly[j]
		v2 = poly[(j + 1) % len(poly)]
		r1 = cones[j][1]
		r2 = cones[(j + 1) % len(poly)][0]

		points = locate_edge(v1, r1, v2, r2, bbox)

		ax.fill(*zip(*points), alpha=0.2, color='green', edgecolor='white', linewidth=2.5)
		ax.plot(*zip(*points), color='black', linewidth=1.5)

	vstart = next(j for j in range(len(poly)) if blocked[j] and not blocked[j - 1])
	vend = next(j for j in range(len(poly)) if not blocked[j] and blocked[j - 1])

	v1 = poly[vstart]
	v2 = poly[(vend) % len(poly)]
	r1 = cones[vstart][1]
	r2 = cones[(vend) % len(poly)][0]

	points = locate_edge(v1, r1, v2, r2, bbox)
	ax.fill(*zip(*points), alpha=0.2, color='blue', edgecolor='black', linewidth=2.5, zorder=0)


	location = solution.locate_point(target, 0)
	ind = location // 2

	if location % 2 == 0:
		v = poly[ind]
		ax.plot(*zip(start, v, target), color='purple', linewidth=4.5, zorder=4)
		ax.plot(*zip(start, v, target), color='white', linewidth=2.0, zorder=4, linestyle='--')
	elif blocked[ind]:
		ax.plot(*zip(start, target), color='purple', linewidth=4.5, zorder=4)
		ax.plot(*zip(start, target), color='white', linewidth=2.0, zorder=4, linestyle='--')
	else:

		v1 = poly[ind]
		v2 = poly[(ind + 1) % len(poly)]

		reflected = target.reflect_segment(v1, v2)
		last = solution.query(reflected, 0)

		intersection = segment_segment_intersection(last, reflected, v1, v2)

		ax.plot(*zip(v1.lerp(v2, -10), v2.lerp(v1, -10)), color='green', linewidth=4.5, zorder=3)
		ax.plot(*zip(v1.lerp(v2, -10), v2.lerp(v1, -10)), color='white', linewidth=2.0, zorder=3, linestyle='--')

		if intersection is not None:
		
			ax.plot(*zip(start, intersection, target), color='purple', linewidth=4.5, zorder=4)
			ax.plot(*zip(start, intersection, target), color='white', linewidth=2.0, zorder=4, linestyle='--')

			ax.plot(*zip(intersection, reflected), color='orange', linewidth=4.5, zorder=4)
			ax.plot(*zip(intersection, reflected), color='white', linewidth=2.0, zorder=4, linestyle='--')

		ax.scatter(*reflected, color='white', edgecolor='orange', s=140, linewidth=2.5, zorder=5, label='Ponto de Reflexão')

start = Vector2(-1, 4)
polygon = regular_polygon(Vector2(0, 0), 2, 6, rotation=0.2)

targets = [
	Vector2(4, 1),
	Vector2(4, 3),
	Vector2(2.2, -3),
]

for i, target in enumerate(targets):

	fig, ax = plt.subplots(figsize=(6, 6))
	draw(ax, start, target, polygon)
	plt.savefig(f'_test15-{i + 1}.png', dpi=300, bbox_inches='tight')
