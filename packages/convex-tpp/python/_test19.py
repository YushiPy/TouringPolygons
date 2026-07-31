
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
from matplotlib.figure import Figure

from u_tpp_fast_locate import Solution

def draw_map(solution: Solution, index: int | None = None, targets: list[Vector2] = []) -> tuple[Figure, Axes]:
	
	ax: Axes
	fig, ax = plt.subplots(1, 1, figsize=(6, 6)) # type: ignore
	bbox = get_bbox([solution.start] + [v for poly in solution.polygons for v in poly], square=True, scale=1.5)
	minx, miny, maxx, maxy = bbox

	ax.set_aspect('equal')

	ax.set_xlim(minx, maxx)
	ax.set_ylim(miny, maxy)
	ax.set_xticks([])
	ax.set_yticks([])

	ax.scatter(*solution.start, color='black', edgecolor='white', s=80, zorder=8, linewidth=2)
	ax.text(*(solution.start + Vector2(0, -3)), '$s$', color='black', fontsize=18, ha='center', va='center', zorder=6)

	if index is None:
		index = len(solution.polygons) - 1

	if not solution.cones:
		solution.solve()

	polygon = solution.polygons[index]
	cones = solution.cones[index]
	blocked = solution.blocked[index]

	for i in range(len(polygon)):

		if blocked[i] and blocked[(i - 1) % len(polygon)]:
			continue

		v = polygon[i]
		r1, r2 = cones[i]

		points = locate_cone(v, r1, r2, bbox)
		ax.fill(*zip(*points), color='red', alpha=0.5)
		ax.plot(*zip(*points), color='red', linewidth=2, zorder=4)

		v1 = polygon[i - 1]
		v2 = polygon[(i + 1) % len(polygon)]

	for i in range(len(polygon)):

		if blocked[i]:
			continue

		v1 = polygon[i]
		v2 = polygon[(i + 1) % len(polygon)]

		r1 = cones[i][1]
		r2 = cones[(i + 1) % len(polygon)][0]

		points = locate_edge(v1, r1, v2, r2, bbox)
		ax.fill(*zip(*points), color='green', alpha=0.5)
		ax.plot(*zip(*points), color='green', linewidth=2, zorder=3)

	start = next(i for i in range(len(polygon)) if blocked[i] and not blocked[(i - 1) % len(polygon)])
	end = next(i for i in range(len(polygon)) if blocked[i] and not blocked[(i + 1) % len(polygon)]) + 1

	v1 = polygon[start]
	r1 = cones[start][1]
	v2 = polygon[end]
	r2 = cones[end][0]

	points = locate_edge(v1, r1, v2, r2, bbox)
	ax.fill(*zip(*points), color='blue', alpha=0.5)

	ax.fill([minx - 1], [miny - 1], color='red', alpha=0.5, label='Vértice')
	ax.fill([minx - 1], [miny - 1], color='green', alpha=0.5, label='Aresta')
	ax.fill([minx - 1], [miny - 1], color='blue', alpha=0.5, label='Atravessar')

	for i, polygon in enumerate(solution.polygons):
		ax.fill(*zip(*polygon), alpha=1.0, zorder=2, color="white", edgecolor='white', linewidth=2)
		ax.fill(*zip(*polygon), alpha=0.5, zorder=2, edgecolor='blue', linewidth=2)

		center = sum(polygon, Vector2(0, 0)) / len(polygon)
		ax.text(*center, f"$P_{{{i + 1}}}$", color='black', fontsize=18, ha='center', va='center', zorder=6)

	for i, p in enumerate(targets):

		path = solution.full_query(p, index + 1)
		location = solution.locate_point(p, index + 1)
		last = solution.query(p, index + 1)

		if location % 2 == 0:
			color = 'red'
		elif blocked[location // 2]:
			color = 'blue'
		else:
			color = 'green'

		if color == 'red':

			r1, r2 = cones[location // 2]
			offset = ((r1.normalize() + r2.normalize()) * (-1 if r1.cross(r2) < 0 else 1)).scale_to_length(4)

			name1 = f"$p^{i + 1}$" if len(targets) > 1 else "$p$"
			name2 = f"$q^{i + 1}$" if len(targets) > 1 else "$q$"

			ax.plot(*zip(*path), color='white', linewidth=3, zorder=5)
			ax.plot(*zip(*path), color=color, linewidth=2, zorder=5, linestyle='--')

			ax.scatter(*p, color=color, edgecolor='white', s=80, zorder=5, linewidth=2)
			ax.scatter(*last, color=color, edgecolor='white', s=80, zorder=5, linewidth=2)

			ax.text(*(p + offset), name1, color='black', fontsize=16, ha='center', va='center', zorder=10)
			ax.text(*(last + offset), name2, color='black', fontsize=16, ha='center', va='center', zorder=10)

		elif color == 'green':
			
			r1, r2 = cones[location // 2][1], cones[(location // 2 + 1) % len(polygon)][0]
			offset = ((r1.normalize() + r2.normalize()) * (-1 if r1.cross(r2) < 0 else 1)).scale_to_length(4)
			offset = Vector2(3, 3)

			v1, v2 = polygon[location // 2], polygon[(location // 2 + 1) % len(polygon)]
			ax.plot(*zip(v1.lerp(v2, -10), v2.lerp(v1, -10)), color='blue', linewidth=2, zorder=5, linestyle='--')

			reflected = p.reflect_segment(v1, v2)

			name1 = f"$p^{i + 1}$" if len(targets) > 1 else "$p$"
			name2 = f"$q^{i + 1}$" if len(targets) > 1 else "$q$"
			name3 = f"$p'^{i + 1}$" if len(targets) > 1 else "$p'$"

			ax.plot(*zip(*path), color='white', linewidth=3, zorder=5)
			ax.plot(*zip(*path), color=color, linewidth=2, zorder=5, linestyle='--')
			ax.plot(*zip(last, reflected), color='white', linewidth=3, zorder=5, linestyle='-')
			ax.plot(*zip(last, reflected), color='purple', linewidth=2, zorder=5, linestyle='--')

			ax.scatter(*p, color=color, edgecolor='white', s=80, zorder=5, linewidth=2)
			ax.scatter(*last, color=color, edgecolor='white', s=80, zorder=5, linewidth=2)
			ax.scatter(*reflected, color='purple', edgecolor='white', s=80, zorder=5, linewidth=2)
			
			ax.scatter(*solution.query(reflected, index), color='purple', edgecolor='white', s=80, zorder=10, linewidth=2)
			ax.text(*(solution.query(reflected, index) + Vector2(-2, 4)), f"$q'$" if len(targets) > 1 else "$q'$", color='black', fontsize=16, ha='center', va='center', zorder=10)

			ax.text(*(p + offset), name1, color='black', fontsize=16, ha='center', va='center', zorder=10)
			ax.text(*(last + offset), name2, color='black', fontsize=16, ha='center', va='center', zorder=10)
			ax.text(*(reflected + offset), name3, color='black', fontsize=16, ha='center', va='center', zorder=10)


			ax.scatter(*v1, color='blue', edgecolor='white', s=80, zorder=5, linewidth=2)
			ax.scatter(*v2, color='blue', edgecolor='white', s=80, zorder=5, linewidth=2)
			ax.text(*(v1 + offset), f"$u$", color='black', fontsize=16, ha='center', va='center', zorder=10)
			ax.text(*(v2 + offset), f"$v$", color='black', fontsize=16, ha='center', va='center', zorder=10)

		else:
			ax.scatter(*p, color=color, edgecolor='white', s=80, zorder=6, linewidth=2)
			ax.plot(*zip(*path), color='white', linewidth=3, zorder=5)
			ax.plot(*zip(*path), color=color, linewidth=2, zorder=5, linestyle='--')
			ax.text(*(p + Vector2(2.5, 2)), f"$p^{i + 1}$" if len(targets) > 1 else "$p$", color='black', fontsize=16, ha='center', va='center', zorder=10)

	return fig, ax

start = Vector2(-5.0, -25.0)
polygons = [
	regular_polygon(Vector2(25, -6), 8, 4, rotation=0.13),
	regular_polygon(Vector2(-3, 0), 10, 3, rotation=0.23)
]

solution = Solution(start, start, polygons)
solution.solve()

vertices_points = [Vector2(-10, -20), Vector2(10, 10)]
edge_points = [Vector2(20, 5)]
pass_points = [Vector2(-10, 10)]

fig, ax = draw_map(solution, 1, pass_points)




ax.legend(loc='upper right', fontsize=14)

plt.savefig('last_step_map.png', dpi=300, bbox_inches='tight')
plt.show()