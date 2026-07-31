
from math import isclose, isqrt
from vector2 import Vector2

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

	return minx, maxx, miny, maxy

start1 = Vector2(0.8, -0.5)


indeces = [0, 1, 2, 3]
starts = [start1] * len(indeces)

polygon = [
	Vector2(0, 0.0),
	Vector2(0, 1.5),
	Vector2(-1.5, 0.5),
	Vector2(-1.5, -1),
]

minx, maxx, miny, maxy = get_bbox(starts + polygon, square=True, scale=1.3)

import matplotlib.pyplot as plt

height = isqrt(len(indeces))
width = (len(indeces) + height - 1) // height

fig, ax = plt.subplots(height, width, figsize=(6 * width, 6 * height))
flat = ax.flatten()

for i, axis in enumerate(flat):

	axis.set_xlim(minx, maxx)
	axis.set_ylim(miny, maxy)
	axis.set_aspect('equal', adjustable='box')
	axis.set_xticks([])
	axis.set_yticks([])
	
	index = indeces[i]

	v1 = Vector2(polygon[index])
	v2 = Vector2(polygon[(index + 1) % len(polygon)])

	points1 = locate_cone(v1, (v2 - v1), (v1 - v2), (minx, miny, maxx, maxy))
	points2 = locate_cone(v1, (v1 - v2), (v2 - v1), (minx, miny, maxx, maxy))

	axis.fill(*zip(*points1), color='red', alpha=0.4)
	axis.fill(*zip(*points2), color='green', alpha=0.4)

	ps = [v1.lerp(v2, -10), v2.lerp(v1, -10)]
	axis.plot([p.x for p in ps], [p.y for p in ps], color='white', linestyle='dashed', linewidth=2, zorder=1)

	axis.scatter(v1.x, v1.y, color='blue', s=150, zorder=3)
	axis.scatter(v1.x, v1.y, color='white', s=60, zorder=7)
	axis.scatter(v2.x, v2.y, color='blue', s=150, zorder=3)
	axis.scatter(v2.x, v2.y, color='white', s=60, zorder=7)

	d1 = (v1 - polygon[index - 1]).normalize()# + (v1 - polygon[(index + 1) % len(polygon)]).normalize()
	d2 = (v2 - polygon[index]).normalize()# + (v2 - polygon[(index + 2) % len(polygon)]).normalize()

	axis.text(*(v1 + d1.scale_to_length(0.2)), f"$v^{index + 1}$", fontsize=20, zorder=4, backgroundcolor='white', ha='center', va='center')
	axis.text(*(v2 + d2.scale_to_length(0.2)), f"$v^{(index + 1) % len(polygon) + 1}$", fontsize=20, zorder=4, backgroundcolor='white', ha='center', va='center')

	axis.fill([p.x for p in polygon], [p.y for p in polygon], color='white', alpha=1.0, label='Polygon', zorder=2)
	axis.fill([p.x for p in polygon], [p.y for p in polygon], color='blue', alpha=0.3, label='Polygon', zorder=2)
	axis.plot([p.x for p in polygon + [polygon[0]]], [p.y for p in polygon + [polygon[0]]], color='blue', linewidth=2, zorder=3)

	axis.arrow(v1.x, v1.y, (v2 - v1).x, (v2 - v1).y, head_width=0.15, head_length=0.25, fc='white', ec='white', linewidth=0.0, length_includes_head=True, zorder=5)


	direction = (starts[i] - v1)
	direction.scale_to_length_ip(direction.length() * 0.95)

	if (v2 - v1).cross(direction) > 0:
		axis.plot(*zip(v1, v2), color='white', linewidth=4, zorder=4)
		axis.plot(*zip(v1, v2), color='red', linewidth=4, zorder=4)
	else:
		axis.plot(*zip(v1, v2), color='white', linewidth=4, zorder=4)
		axis.plot(*zip(v1, v2), color='green', linewidth=4, zorder=4)

	color = 'red' if (v2 - v1).cross(direction) > 0 else 'green'

	# Draw arrow:

	axis.arrow(v1.x, v1.y, direction.x, direction.y, head_width=0.1, head_length=0.2, fc='white', ec='black', linewidth=6, length_includes_head=True, zorder=5)
	axis.arrow(v1.x, v1.y, direction.x, direction.y, head_width=0.1, head_length=0.2, fc='white', ec='white', linewidth=2, length_includes_head=True, zorder=5)

	axis.scatter(starts[i].x, starts[i].y, color=color, s=150, label='Start')
	axis.scatter(starts[i].x, starts[i].y, color='white', s=60, zorder=5)

	axis.text(*(starts[i] + direction.scale_to_length(0.2)), "$s$", fontsize=20, zorder=6, backgroundcolor='white', ha='center', va='center')

	axis.scatter(v1.x, v1.y, color=color, s=150, label='End', zorder=5)
	axis.scatter(v1.x, v1.y, color='white', s=60, zorder=5)

	direction = (starts[i] - v2)
	direction.scale_to_length_ip(direction.length() * 0.95)

	# Draw arrow:
	axis.arrow(v2.x, v2.y, direction.x, direction.y, head_width=0.1, head_length=0.2, fc='white', ec='black', linewidth=6, length_includes_head=True, zorder=5)
	axis.arrow(v2.x, v2.y, direction.x, direction.y, head_width=0.1, head_length=0.2, fc='white', ec='white', linewidth=2, length_includes_head=True, zorder=5)

	axis.scatter(starts[i].x, starts[i].y, color=color, s=150, label='Start')
	axis.scatter(starts[i].x, starts[i].y, color='white', s=60, zorder=5)

	axis.scatter(v2.x, v2.y, color=color, s=150, label='End', zorder=5)
	axis.scatter(v2.x, v2.y, color='white', s=60, zorder=5)

fig.tight_layout()

plt.savefig('_test14_output.png', dpi=300, bbox_inches='tight')
