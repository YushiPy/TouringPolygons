

from math import inf, isclose
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


def get_points(point: Vector2, polygon: list[Vector2], cones: list[tuple[Vector2, Vector2]]) -> list[tuple[int, int]]:

	def check_cone(i: int) -> bool:
		return point_in_cone(point, polygon[i], *cones[i])

	def check_edge(i: int, j: int) -> bool:

		v1 = polygon[i]
		v2 = polygon[j]

		r1 = cones[i][1]
		r2 = cones[j][0]

		return point_in_edge(point, v1, v2, r1, r2)

	result: list[tuple[int, int]] = []

	if check_cone(0):
		return result
	
	left = 0
	right = len(polygon)

	while right - left > 1:

		result.append((left, right))
		mid = (left + right) // 2

		if check_cone(mid):
			return result

		if check_edge(left, mid):
			right = mid
		else:
			left = mid
	
	result.append((left, right))

	return result


start = Vector2(0, 10)
target = Vector2(2, 3)

polygon = [
	(6, -0.5),
	#(5, 1),
	(4.0, 2),
	(0, 3),
	(-4.0 , 2),
	#(-5, 1),
	(-6, -0.5),
	(-4, -2),
	(0, -3),
	(4, -2),
]

polygon = list(map(Vector2, polygon))

sol = Solution(start, target, [polygon])
sol.solve()

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Any

pairs = [plt.subplots(1, 1, figsize=(6, 6)) for _ in range(4)]
figures, flat = zip(*pairs)

#flat: list[Axes] = ax.flatten()

bbox = get_bbox([start, target] + polygon, square=True, scale=1.35)
minx, miny, maxx, maxy = bbox

for axis in flat:
	
	axis: Axes

	axis.set_xlim(minx, maxx)
	axis.set_ylim(miny, maxy)

	axis.set_xticks([])
	axis.set_yticks([])

	axis.set_aspect('equal')

	#axis.fill([minx, maxx, maxx, minx], [miny, miny, maxy, maxy], color='lightgray', zorder=0)
	#draw_point(axis, start, color='green', s=100)
	#draw_point(axis, target, color='red', s=100)

ax1 = flat[0]

# Draw first scene

m, n = len(sol.filtered[0]), len(sol.polygons[0])

for i in range(m):
	v = sol.filtered[0][i]
	r1, r2 = sol.cones[0][i]
	points = locate_cone(v, r1, r2, bbox)
	ax1.fill(*zip(*points), alpha=0.2, edgecolor='red', facecolor='red', linewidth=1.5)

for i in range(m - 1):
	v1 = sol.filtered[0][i]
	v2 = sol.filtered[0][i + 1]
	r1 = sol.cones[0][i][1]
	r2 = sol.cones[0][i + 1][0]
	points = locate_edge(v1, r1, v2, r2, bbox)
	ax1.fill(*zip(*points), alpha=0.2, edgecolor='green', facecolor='green', linewidth=1.5)

for i in range(m - 1, n):
	v1 = sol.polygons[0][i % n]
	v2 = sol.polygons[0][(i + 1) % n]
	r1 = v1 - start
	r2 = v2 - start
	points = locate_edge(v1, r1, v2, r2, bbox)
	ax1.fill(*zip(*points), alpha=0.3, facecolor='blue', linewidth=1.5)

ax1.fill(*zip(*polygon), alpha=0.5, edgecolor='black', facecolor='gray', linewidth=2)

for i in range(m):
	v = sol.filtered[0][i]
	pos = v + Vector2(0.0, 1.2)
	ax1.text(*pos, f"$u^{i + 1}$", color='black', fontsize=12, ha='center', va='center', backgroundcolor='white', alpha=0.7)

for i in range(m):
	v = sol.filtered[0][i]
	ax1.scatter(*v, color="#67E185", edgecolor='black', linewidth=1.5, s=50, zorder=5)

for i in range(m, n):
	v = sol.polygons[0][i % n]
	ax1.scatter(*v, color="#4568DA", edgecolor='black', linewidth=1.5, s=50, zorder=5)

ax2 = flat[1]

for i in range(m):
	v = sol.filtered[0][i]
	r1, r2 = sol.cones[0][i]
	points = locate_cone(v, r1, r2, bbox)
	ax2.fill(*zip(*points), alpha=0.2, edgecolor='red', facecolor='red', linewidth=1.5)

for i in range(m - 1):
	v1 = sol.filtered[0][i]
	v2 = sol.filtered[0][i + 1]
	r1 = sol.cones[0][i][1]
	r2 = sol.cones[0][i + 1][0]
	points = locate_edge(v1, r1, v2, r2, bbox)
	ax2.fill(*zip(*points), alpha=0.2, edgecolor='green', facecolor='green', linewidth=1.5)

ax2.fill(*zip(*polygon), alpha=0.5, edgecolor='black', facecolor='gray', linewidth=2)

v1 = sol.filtered[0][-1]
v2 = sol.filtered[0][0]
r1 = sol.cones[0][-1][1]
r2 = sol.cones[0][0][0]
points = locate_edge(v1, r1, v2, r2, bbox)
ax2.fill(*zip(*points), alpha=0.2, edgecolor='cyan', facecolor='cyan', linewidth=1.5)

ax3 = flat[2]

for i in range(m):
	v = sol.filtered[0][i]
	r1, r2 = sol.cones[0][i]
	points = locate_cone(v, r1, r2, bbox)
	ax3.fill(*zip(*points), alpha=0.2, edgecolor='red', facecolor='red', linewidth=1.5)

for i in range(m - 1):
	v1 = sol.filtered[0][i]
	v2 = sol.filtered[0][i + 1]
	r1 = sol.cones[0][i][1]
	r2 = sol.cones[0][i + 1][0]
	points = locate_edge(v1, r1, v2, r2, bbox)
	ax3.fill(*zip(*points), alpha=0.2, edgecolor='green', facecolor='green', linewidth=1.5)

v = sol.filtered[0][0]
r1, r2 = sol.cones[0][0]
points = locate_cone(v, r1, r2, bbox)
ax3.fill(*zip(*points), alpha=0.35, edgecolor='red', facecolor='red', linewidth=2.5)

v = sol.filtered[0][-1]
r1, r2 = sol.cones[0][-1]
points = locate_cone(v, r1, r2, bbox)
ax3.fill(*zip(*points), alpha=0.35, edgecolor='red', facecolor='red', linewidth=2.5)

ax3.fill(*zip(*polygon), alpha=0.5, edgecolor='black', facecolor='gray', linewidth=2)


ax4 = flat[3]

v = sol.filtered[0][0]
r = sol.cones[0][0][0]
points = locate_cone(v, r, r, bbox)
ax4.plot(*zip(*points), alpha=0.35, color='red', linestyle='--', linewidth=3.5)

v = sol.filtered[0][-1]
r = sol.cones[0][-1][1]
points = locate_cone(v, r, r, bbox)
ax4.plot(*zip(*points), alpha=0.35, color='red', linestyle='--', linewidth=3.5)

v = sol.filtered[0][m // 2]
r1, r2 = sol.cones[0][m // 2]
points = locate_cone(v, r1, r2, bbox)
ax4.fill(*zip(*points), alpha=0.35, edgecolor='red', facecolor='red', linewidth=2.5)

v1 = sol.filtered[0][0]
v2 = sol.filtered[0][m // 2]
r1 = sol.cones[0][0][1]
r2 = sol.cones[0][m // 2][0]
points = locate_edge(v1, r1, v2, r2, bbox)
ax4.fill(*zip(*points), alpha=0.35, edgecolor='purple', facecolor='purple', linewidth=2.5)

v1 = sol.filtered[0][m // 2]
v2 = sol.filtered[0][-1]
r1 = sol.cones[0][m // 2][1]
r2 = sol.cones[0][-1][0]
points = locate_edge(v1, r1, v2, r2, bbox)
ax4.fill(*zip(*points), alpha=0.35, edgecolor='orange', facecolor='orange', linewidth=2.5)

ax4.fill(*zip(*polygon), alpha=0.5, edgecolor='black', facecolor='gray', linewidth=2)

#fig.tight_layout()

for fig in figures:
	fig.tight_layout()
	fig.savefig("binary_search_idea-" + str(figures.index(fig) + 1) + ".png", dpi=300, bbox_inches='tight')

#plt.savefig("binary_search-1.png", dpi=300, bbox_inches='tight')
#plt.show()
