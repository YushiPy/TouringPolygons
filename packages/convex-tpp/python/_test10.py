

from math import ceil, isclose, tau
from vector2 import Vector2
from u_tpp_fast_locate import Solution


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
	return Vector2(inf, inf)

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

	return minx, maxx, miny, maxy

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

		return point_in_edge(target, v1, v2, r1, r2)

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

polygon = regular_polygon(Vector2(0, 0), 1, 12)

start = Vector2(0.0, 2.0)
target = start

sol = Solution(start, target, [polygon])
sol.solve()

index = -2
v1 = polygon[index]
v2 = polygon[(index + 1) % len(polygon)]
r1 = sol.cones[0][index][1]
r2 = sol.cones[0][(index + 1) % len(polygon)][0]

target = v1.lerp(v2, 0.5) + r1

points = get_points(target, sol.polygons[0], sol.cones[0])

print(points)

sol = Solution(start, target, [polygon])
sol.solve()

minx, maxx, miny, maxy = get_bbox(polygon + [start], square=True, scale=1.35)

import matplotlib.pyplot as plt

count = 1 + len(points)
width = ceil(count ** .5)
height = ceil(count / width)

fig, ax = plt.subplots(height, width, figsize=(16, 16))

for axis in ax.flatten():

	axis.set_xlim(minx, maxx)
	axis.set_ylim(miny, maxy)

	axis.set_xticks([])
	axis.set_yticks([])

	axis.set_aspect('equal', adjustable='box')

	#axis.fill([minx, maxx, maxx, minx], [miny, miny, maxy, maxy], color='lightgray', zorder=0)
	#ax.scatter([start.x], [start.y], color='green', s=100, label='Start', zorder=3)

	axis.fill(*zip(*polygon), color='white', alpha=1)
	axis.fill(*zip(*polygon), color='blue', alpha=0.4)
	axis.plot(*zip(*(polygon + [polygon[0]])), color='blue', linewidth=2, label='Path', zorder=4)

	for v, (r1, r2) in zip(sol.polygons[0], sol.cones[0]):

		r1.scale_to_length_ip(10)
		r2.scale_to_length_ip(10)

		axis.plot([v.x, v.x + r1.x], [v.y, v.y + r1.y], color='red', linewidth=1, zorder=5)
		axis.plot([v.x, v.x + r2.x], [v.y, v.y + r2.y], color='red', linewidth=1, zorder=5)

		axis.fill(*zip(v, v + r1, v + (r1 + r2), v + r2), color='red', alpha=0.2, zorder=4)

	for i in range(len(polygon)):

		v1 = sol.polygons[0][i]
		v2 = sol.polygons[0][(i + 1) % len(sol.polygons[0])]

		r1 = sol.cones[0][i][1]
		r2 = sol.cones[0][(i + 1) % len(sol.polygons[0])][0]

		color = "#62E1DA" if sol.blocked[0][i] else 'green'
		axis.fill(*zip(v1, v1 + r1, v2 + r2, v2), color=color, alpha=0.3, zorder=4)

	for i in range(len(polygon)):

		v = sol.polygons[0][i]

		v_prev = sol.polygons[0][i - 1]
		v_next = sol.polygons[0][(i + 1) % len(sol.polygons[0])]

		diff1 = (v - v_prev).normalize()
		diff2 = (v - v_next).normalize()

		pos = v + (diff1 + diff2).normalize() * 0.20
		#pos += Vector2(0.05, 0.0)

		#axis.scatter([v.x], [v.y], color='blue', s=150, zorder=7)
		#axis.scatter([v.x], [v.y], color='white', s=60, zorder=7)

		#axis.text(pos.x, pos.y, str(i + 1), color='black', fontsize=20, ha='center', va='center', zorder=6, backgroundcolor='white')

	#axis.scatter([start.x], [start.y], color='green', s=150, label='start', zorder=5, alpha=1)
	#axis.scatter([start.x], [start.y], color='white', s=60, label='Target', zorder=5)

	axis.scatter([target.x], [target.y], color='red', s=150, label='Target', zorder=5, alpha=1)
	axis.scatter([target.x], [target.y], color='white', s=60, label='Target', zorder=5)

for i in range(len(points)):

	axis = ax.flatten()[i + 1]

	l, r = points[i]
	mid = (l + r) // 2

	for i in range(l, r):

		v1 = polygon[i]
		v2 = polygon[(i + 1) % len(polygon)]
		r1 = sol.cones[0][i][1]
		r2 = sol.cones[0][(i + 1) % len(polygon)][0]

		ps = locate_edge(v1, r1, v2, r2, (minx, miny, maxx, maxy))

		color = "orange" if i < mid else "purple"
		axis.fill(*zip(*ps), color="white", alpha=1, zorder=4)
		axis.fill(*zip(*ps), color=color, alpha=0.5, zorder=4)
	
	for i in range(l + 1, r):

		v = sol.polygons[0][i]
		r1, r2 = sol.cones[0][i]

		ps = locate_cone(v, r1, r2, (minx, miny, maxx, maxy))

		color = "orange" if i < mid else "purple"
		axis.fill(*zip(*ps), color="white", alpha=1, zorder=4)
		axis.fill(*zip(*ps), color=color, alpha=0.3, zorder=4)




fig.tight_layout()

plt.show()
