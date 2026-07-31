
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
			center.x + radius * cos(2 * pi * i / sides + rotation * 2 * pi + pi / 2),
			center.y + radius * sin(2 * pi * i / sides + rotation * 2 * pi + pi / 2)
		)
		for i in range(sides)
	]


import matplotlib.pyplot as plt
from matplotlib.axes import Axes

start = Vector2(0, 0)
polygons = [
	regular_polygon(Vector2(6, 0), 2, 6, rotation=1/12),
	regular_polygon(Vector2(1, 5), 1.5, 5, rotation=0),
]

from u_tpp_filtered import Solution
sol = Solution(start, start, polygons)
sol.solve()

bbox = get_bbox([start] + [v for poly in polygons for v in poly], square=True, scale=1.5)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(bbox[0], bbox[2])
ax.set_ylim(bbox[1], bbox[3])

ax.set_xticks([])
ax.set_yticks([])

ax.fill([bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3]], color='lightgray', alpha=0.5, label='Bounding Box')
ax.scatter(*start, color='green', label='Start', edgecolor='white', s=80, zorder=5, linewidth=2)
ax.text(*(start + Vector2(-0.7, 0)), '$s$', color='black', fontsize=18, ha='center', va='center', zorder=6)

for i, polygon in enumerate(polygons):
	
	color = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow'][i % 6]

	ax.fill(*zip(*polygon), alpha=0.7, label=f'Polygon {1}', edgecolor='black', )
	center = sum(polygon, Vector2(0, 0)) * (1 / len(polygon))
	ax.text(*center, f'$P_{i + 1}$', color='black', fontsize=18, ha='center', va='center', zorder=6)

	filtered = sol.filtered[i]

	for v1, v2 in zip(filtered, filtered[1:]):
		ax.plot([v1.x, v2.x], [v1.y, v2.y], color='white', linewidth=4, zorder=4)
		ax.plot([v1.x, v2.x], [v1.y, v2.y], color=color, linewidth=2, zorder=4)
		ax.scatter([v1.x, v2.x], [v1.y, v2.y], color=color, edgecolor='white', s=80, zorder=5, linewidth=2)
	
	center = sum(filtered, Vector2(0, 0)) * (1 / len(filtered))
	ax.text(*center, f'$T_{i + 1}$', color=color, fontsize=15, ha='center', va='center', zorder=3, backgroundcolor='white', alpha=1)

	v1 = filtered[0]
	points1 = Solution(start, v1, polygons[:i]).solve()

	v2 = filtered[-1]
	points2 = Solution(start, v2, polygons[:i]).solve()

	ax.plot(*zip(*points1), color=color, linestyle='--', linewidth=2, zorder=4)
	ax.plot(*zip(points1[-1], points1[-2].lerp(points1[-1], 100)), color=color, linestyle='--', linewidth=2, zorder=4)

	ax.plot(*zip(*points2), color=color, linestyle='--', linewidth=2, zorder=4)
	ax.plot(*zip(points2[-1], points2[-2].lerp(points2[-1], 100)), color=color, linestyle='--', linewidth=2, zorder=4)

plt.savefig('first_contact.png', dpi=300, bbox_inches='tight')
