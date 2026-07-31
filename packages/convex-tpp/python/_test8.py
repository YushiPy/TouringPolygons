
from vector2 import Vector2

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

start1 = Vector2(0.8, 0.5)
start2 = Vector2(0.8, 0.5)

starts = [start1, start2]

index1 = 0
index2 = 2

indeces = [index1, index2]

polygon = [
	Vector2(0, -1.5),
	Vector2(0, 1.5),
	Vector2(-1.5, 0.5),
	Vector2(-1.5, -1),
]

minx, maxx, miny, maxy = get_bbox(starts + polygon, square=True, scale=1.2)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

for i, axis in enumerate(ax):

	axis.set_xlim(minx, maxx)
	axis.set_ylim(miny, maxy)
	axis.set_aspect('equal', adjustable='box')
	axis.set_xticks([])
	axis.set_yticks([])
	
	index = indeces[i]

	v1 = polygon[index]
	v2 = polygon[(index + 1) % len(polygon)]
	mid = v1.lerp(v2, 0.5)

	points1 = [v1.lerp(v2, -10), v2.lerp(v1, -10), v2.lerp(v2, -10) + (v2 - v1).perpendicular().scale_to_length(10), v1.lerp(v2, -10) + (v2 - v1).perpendicular().scale_to_length(10)]

	points2 = [v1.lerp(v2, -10), v2.lerp(v1, -10), v2.lerp(v2, -10) + (v2 - v1).perpendicular().scale_to_length(-10), v1.lerp(v2, -10) + (v2 - v1).perpendicular().scale_to_length(-10)]

	axis.fill([p.x for p in points1], [p.y for p in points1], color='red', alpha=0.4)
	axis.fill([p.x for p in points2], [p.y for p in points2], color='green', alpha=0.4)

	ps = [v1.lerp(v2, -10), v2.lerp(v1, -10)]

	axis.plot([p.x for p in ps], [p.y for p in ps], color='white', linestyle='dashed', linewidth=2, zorder=1)

	axis.scatter(v1.x, v1.y, color='blue', s=150, zorder=3)
	axis.scatter(v1.x, v1.y, color='white', s=60, zorder=7)
	axis.scatter(v2.x, v2.y, color='blue', s=150, zorder=3)
	axis.scatter(v2.x, v2.y, color='white', s=60, zorder=7)

	axis.fill([p.x for p in polygon], [p.y for p in polygon], color='white', alpha=1.0, label='Polygon')
	axis.fill([p.x for p in polygon], [p.y for p in polygon], color='blue', alpha=0.3, label='Polygon')
	axis.plot([p.x for p in polygon + [polygon[0]]], [p.y for p in polygon + [polygon[0]]], color='blue', linewidth=2, zorder=1)

	axis.arrow(v1.x, v1.y, (v2 - v1).x, (v2 - v1).y, head_width=0.15, head_length=0.25, fc='white', ec='blue', linewidth=1, length_includes_head=True, zorder=4)


	direction = (starts[i] - mid)
	direction.scale_to_length_ip(direction.length() * 0.95)

	# Draw arrow:

	axis.arrow(mid.x, mid.y, direction.x, direction.y, head_width=0.1, head_length=0.2, fc='white', ec='black', linewidth=6, length_includes_head=True, zorder=4)

	axis.arrow(mid.x, mid.y, direction.x, direction.y, head_width=0.1, head_length=0.2, fc='white', ec='white', linewidth=2, length_includes_head=True, zorder=4)

	axis.scatter(starts[i].x, starts[i].y, color='green', s=150, label='Start')
	axis.scatter(starts[i].x, starts[i].y, color='white', s=60, zorder=5)

	axis.scatter(mid.x, mid.y, color='red', s=150, label='End', zorder=5)
	axis.scatter(mid.x, mid.y, color='white', s=60, zorder=5)

fig.tight_layout()

plt.savefig('_test8_output.png', dpi=300)
