

from math import isclose
from vector2 import Vector2
from u_tpp_filtered import Solution


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


start = Vector2(0, 10)
target = Vector2(2, 3)

polygon = regular_polygon(Vector2(0, 0), 2.5, 7, rotation=0.2)

polygon = list(map(Vector2, polygon))

sol = Solution(start, target, [polygon])
sol.solve()

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

pairs = [plt.subplots(1, 1, figsize=(6, 6)) for _ in range(4)]
figures, flat = zip(*pairs)

bbox = (-5, -5, 5, 5)
minx, miny, maxx, maxy = bbox

def draw(axis: Axes, left: int, right: int) -> None:
	
	v1 = sol.polygons[0][left]
	v2 = sol.polygons[0][right]

	r1 = v1 - sol.start if v1 not in sol.filtered[0] else sol.cones[0][sol.filtered[0].index(v1)][0]
	r2 = v2 - sol.start if v2 not in sol.filtered[0] else sol.cones[0][sol.filtered[0].index(v2)][1]

	axis.fill([minx, maxx, maxx, minx], [miny, miny, maxy, maxy], alpha=0.1, edgecolor='black', facecolor='black', linewidth=1)

	axis.fill(*zip(*polygon), alpha=0.5, edgecolor='blue', facecolor='blue', linewidth=2)
	axis.scatter(*zip(v1, v2), color='white', edgecolor="red", s=150, zorder=20, linewidth=3)

	axis.text(v1.x, v1.y + 0.5, f"$u$", fontsize=20, ha='center', color='black', backgroundcolor='white', zorder=30)
	axis.text(v2.x, v2.y + 0.5, f"$v$", fontsize=20, ha='center', color='black', backgroundcolor='white', zorder=30)

	r1.scale_to_length_ip(2.1)
	axis.arrow(v1.x, v1.y, r1.x, r1.y, head_width=0.3, head_length=0.2, fc='white', ec='blue', linewidth=6, alpha=1, zorder=10)
	axis.arrow(v1.x, v1.y, r1.x, r1.y, head_width=0.3, head_length=0.2, fc='white', ec='white', linewidth=2, alpha=1, zorder=11)

	axis.text(v1.x + r1.x * 0.8 + 0.5, v1.y + r1.y * 0.8, f"$r^1$", fontsize=20, ha='center', color='black', backgroundcolor='white', zorder=30, va='center')

	r2.scale_to_length_ip(2.1)
	axis.arrow(v2.x, v2.y, r2.x, r2.y, head_width=0.3, head_length=0.2, fc='white', ec='blue', linewidth=6, alpha=1, zorder=10)
	axis.arrow(v2.x, v2.y, r2.x, r2.y, head_width=0.3, head_length=0.2, fc='white', ec='white', linewidth=2, alpha=1, zorder=11)

	axis.text(v2.x + r2.x * 0.8, v2.y + r2.y * 0.8 + 0.5, f"$r^2$", fontsize=20, ha='center', color='black', backgroundcolor='white', zorder=30, va='center')

	axis.arrow(*v1, *((v2 - v1) * 0.95), head_width=0.3, head_length=0.2, fc='white', ec='blue', linewidth=6, alpha=1, zorder=10, length_includes_head=True)
	axis.arrow(*v1, *((v2 - v1) * 0.95), head_width=0.3, head_length=0.2, fc='white', ec='white', linewidth=2, alpha=1, zorder=11, length_includes_head=True)

	axis.plot(*zip(v1.lerp(v2, -10), v2.lerp(v1, -10)), linestyle='--', color='blue', linewidth=2.5, alpha=0.7)


	points = locate_edge(v1, r1, v2, r2, bbox)
	axis.fill(*zip(*points), alpha=0.45, edgecolor='green', facecolor='green', linewidth=4.5)

	points = locate_edge(v2, r2, v1, r1, bbox)
	#axis.fill(*zip(*points), alpha=0.25, edgecolor='red', facecolor='red', linewidth=2.5)

	for v, (r1, r2) in zip(sol.filtered[0], sol.cones[0]):
		points = locate_cone(v, r1, r2, bbox)
		#axis.fill(*zip(*points), alpha=0.15, edgecolor='red', facecolor='red', linewidth=1.5)
	
	for i in range(len(sol.filtered[0]) - 1):
		v1 = sol.filtered[0][i]
		v2 = sol.filtered[0][(i + 1) % len(sol.filtered[0])]
		r1 = sol.cones[0][i][1]
		r2 = sol.cones[0][(i + 1) % len(sol.cones[0])][0]
		points = locate_edge(v1, r1, v2, r2, bbox)
		#axis.fill(*zip(*points), alpha=0.1, edgecolor='green', facecolor='green', linewidth=1.5)

	for i in range(len(sol.polygons[0])):
		v1 = sol.polygons[0][i % len(polygon)]
		v2 = sol.polygons[0][(i + 1) % len(polygon)]

		if v2 in sol.filtered[0] and v1 in sol.filtered[0]:
			continue

		r1 = (v1 - start)
		r2 = (v2 - start)
		points = locate_edge(v1, r1, v2, r2, bbox)
		#axis.fill(*zip(*points), alpha=0.2, facecolor='blue', linewidth=1.5)


for axis in flat:
	
	axis: Axes

	axis.set_xlim(minx, maxx)
	axis.set_ylim(miny, maxy)

	axis.set_xticks([])
	axis.set_yticks([])

	axis.set_aspect('equal')

draw(flat[0], 1, 4)
draw(flat[1], 4, 1)
draw(flat[2], 2, 0)
draw(flat[3], 0, 5)

for i, fig in enumerate(figures, 1):
	fig.tight_layout()
	fig.savefig(f"binary_search_cases-{i}.png", dpi=300, bbox_inches='tight')
