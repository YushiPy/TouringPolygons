
from math import inf, isclose, tau
from vector2 import Vector2
from polygon2 import Polygon2
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


def regular_polygon(n: int, center: Vector2 = Vector2(0, 0), radius: float = 1.0, rotation: float = 0.0) -> Polygon2:
	points: list[Vector2] = []
	for i in range(n):
		angle = rotation + (i / n) * tau
		point = Vector2.from_polar(radius, angle) + center
		points.append(point)
	return Polygon2(points)


start = Vector2(0, 4)
target = Vector2(0, -2)
point = Vector2(0.5, 2)

n_sides = 20
polygon = regular_polygon(n_sides, Vector2(0, 0), 1.0, tau / 4)
polygon = Polygon2(polygon[:n_sides // 4 + 1] + polygon[-(n_sides // 4):])

sol = Solution(start, target, [polygon])
sol.solve()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

xlim = -1.5, 1.5
ylim = -0.5, 2.5

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect('equal', adjustable='box')

ax.fill([xlim[0], xlim[1], xlim[1], xlim[0]], [ylim[0], ylim[0], ylim[1], ylim[1]], color='lightgray', zorder=-1)


ax.fill(*zip(*polygon), color='blue', alpha=0.5, zorder=0)
ax.plot(*zip(*polygon, polygon[0]), color='black', zorder=1)

for i in range(len(sol.filtered[0])):
	
	v = sol.filtered[0][i]
	ray1 = sol.cones[0][i][0]
	ray2 = sol.cones[0][i][1]

	ax.plot(*zip(*[v, v + 4.0 * ray1]), color='white', zorder=2)
	ax.plot(*zip(*[v, v + 1.0 * ray1]), color='red', linestyle='--', zorder=2)

	ax.plot(*zip(*[v, v + 4.0 * ray2]), color='white', zorder=2)
	ax.plot(*zip(*[v, v + 1.0 * ray2]), color='red', linestyle='--', zorder=2)


	ax.fill([v.x, v.x + ray1.x * 10, v.x + ray2.x * 10], [v.y, v.y + ray1.y * 10, v.y + ray2.y * 10], color='red', alpha=0.2, zorder=1)

for i in range(len(sol.filtered[0]) - 1):

	v1 = sol.filtered[0][i]
	v2 = sol.filtered[0][i + 1]

	ray1 = sol.cones[0][i][1]
	ray2 = sol.cones[0][i + 1][0]
	
	ax.fill([v1.x, v1.x + ray1.x * 10, v2.x + ray2.x * 10, v2.x], [v1.y, v1.y + ray1.y * 10, v2.y + ray2.y * 10, v2.y], color='green', alpha=0.2, zorder=1)

#start_index = len(sol.filtered[0]) // 4
#end_index = 3 * len(sol.filtered[0]) // 4

positions = [
	(0, 8),
	(0, 5),
	(2, 6),
	(3, 8),
]

start_index, end_index = positions[3]

v1 = sol.filtered[0][start_index]
v2 = sol.filtered[0][end_index]

ray1 = sol.cones[0][start_index][1]
ray2 = sol.cones[0][end_index][0]

ax.plot(*zip(*[v1, v1 + 4.0 * ray1]), color='purple', zorder=4, linewidth=4)
ax.plot(*zip(*[v2, v2 + 4.0 * ray2]), color='purple', zorder=4, linewidth=4)
ax.plot(*zip(*[v1, v2]), color='purple', zorder=4, linewidth=4)

ax.plot(*zip(*[v1.lerp(v2, -1), v2.lerp(v1, -1)]), color='white', zorder=4, linewidth=2, linestyle="--")
ax.arrow(*v1, *(v2 - v1), color='pink', zorder=4, width=0.008, head_width=0.08, length_includes_head=True)

points = locate_edge(v1, ray1, v2, ray2, (xlim[0], ylim[0], xlim[1], ylim[1]))
ax.fill(*zip(*points), color='purple', alpha=0.4, zorder=1)
# ax.fill([v1.x, v1.x + ray1.x * 10, v2.x + ray2.x * 10, v2.x], [v1.y, v1.y + ray1.y * 10, v2.y + ray2.y * 10, v2.y], color='purple', alpha=0.4, zorder=1)

ax.scatter(*v1, color='black', zorder=5, s=60)
ax.scatter(*v1, color='white', zorder=5, s=20)
ax.text(*(v1 - Vector2(0.05, 0.20)), "$u$", color='black', fontsize=16, zorder=5, ha='right', va='bottom')

ax.scatter(*v2, color='black', zorder=5, s=60)
ax.scatter(*v2, color='white', zorder=5, s=20)
ax.text(*(v2 + Vector2(0.15, -0.20)), "$v$", color='black', fontsize=16, zorder=5, ha='right', va='bottom')

ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()

# plt.savefig(f"out{start_index}-{end_index}.png", dpi=300)
plt.show()