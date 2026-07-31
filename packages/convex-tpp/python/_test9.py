
from vector2 import Vector2
from u_tpp_filtered import Solution

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

start = Vector2(0.0, 2)
target = Vector2(0.4, 1.5)

polygon = [
	Vector2(1, 1), 
    Vector2(-1, 1),
    Vector2(-2, 0.5),
	Vector2(0, 0.0),
    Vector2(2, 0.5)
]

sol = Solution(start, target, [polygon])
sol.solve()

edge_index = 1

minx, maxx, miny, maxy = get_bbox(polygon + [start], square=False, scale=1.2)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

ax.set_xticks([])
ax.set_yticks([])

ax.set_aspect('equal', adjustable='box')

ax.fill([minx, maxx, maxx, minx], [miny, miny, maxy, maxy], color='lightgray', zorder=0)
#ax.scatter([start.x], [start.y], color='green', s=100, label='Start', zorder=3)

ax.fill(*zip(*polygon), color='white', alpha=1)
ax.fill([p.x for p in polygon], [p.y for p in polygon], color='blue', alpha=0.3)

ax.plot(*zip(*(polygon + [polygon[0]])), color='blue', linewidth=2, label='Path', zorder=4)

v1 = sol.filtered[0][edge_index]
v2 = sol.filtered[0][(edge_index + 1) % len(sol.filtered[0])]

r1 = sol.cones[0][edge_index][1]
r2 = sol.cones[0][(edge_index + 1) % len(sol.filtered[0])][0]

ax.plot([v1.x, v2.x], [v1.y, v2.y], color='green', linewidth=4, label='Edge', zorder=5)

ax.scatter([v1.x, v2.x], [v1.y, v2.y], color='green', s=150, zorder=20)
ax.scatter([v1.x, v2.x], [v1.y, v2.y], color='white', s=60, zorder=20)

ax.fill(*zip(v1, v1 + r1.scale_to_length(10), v2 + r2.scale_to_length(10), v2), color='white', alpha=1.0, zorder=2)
ax.fill(*zip(v1, v1 + r1.scale_to_length(10), v2 + r2.scale_to_length(10), v2), color='green', alpha=0.5, zorder=2)

ax.plot(*zip(*[v1, v1 + r1.scale_to_length(10)]), color='white', linewidth=3, linestyle='--', zorder=5)

ax.plot(*zip(*[v2, v2 + r2.scale_to_length(10)]), color='white', linewidth=3, linestyle='--', zorder=5)

ax.arrow(*v1, *r1.scale_to_length(0.6), head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=6, zorder=10)
ax.arrow(*v1, *r1.scale_to_length(0.6), head_width=0.1, head_length=0.1, fc='white', ec='white', linewidth=2, zorder=10)

ax.text(*(v1 + r1.scale_to_length(0.6) + Vector2(0.2, -0.03)), '$r^1$', color='black', fontsize=20, zorder=10)

ax.arrow(*v2, *r2.scale_to_length(0.6), head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=6, zorder=10)
ax.arrow(*v2, *r2.scale_to_length(0.6), head_width=0.1, head_length=0.1, fc='white', ec='white', linewidth=2, zorder=10)

ax.text(*(v2 + r2.scale_to_length(0.6) + Vector2(-0.3, -0.03)), '$r^2$', color='black', fontsize=20, zorder=10)

ax.arrow(*(v1.lerp(v2, 0.0)), *(v2 - v1) * 0.9, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=7, zorder=10)
ax.arrow(*(v1.lerp(v2, 0.0)), *(v2 - v1) * 0.9, head_width=0.1, head_length=0.1, fc='lightgreen', ec='lightgreen', linewidth=2, zorder=10)

ax.text(*(v1 + Vector2(-0.05, 0.15)), '$u$', color='black', fontsize=20, zorder=10)

ax.text(*(v2 + Vector2(0.00, 0.15)), '$v$', color='black', fontsize=20, zorder=10)


ax.arrow(*v1, *(target - v1) * 0.95, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=6, zorder=15, length_includes_head=True)
ax.arrow(*v1, *(target - v1) * 0.95, head_width=0.1, head_length=0.1, fc='pink', ec='pink', linewidth=2, zorder=15, length_includes_head=True)

ax.arrow(*v2, *(target - v2) * 0.95, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=6, zorder=15, length_includes_head=True)
ax.arrow(*v2, *(target - v2) * 0.95, head_width=0.1, head_length=0.1, fc='pink', ec='pink', linewidth=2, zorder=15, length_includes_head=True)

ax.scatter([target.x], [target.y], color='red', s=150, label='Target', zorder=20)
ax.scatter([target.x], [target.y], color='white', s=60, label='Target', zorder=20)

ax.text(*(target + Vector2(0.0, 0.1)), '$p$', color='black', fontsize=20, zorder=10)

fig.tight_layout()
plt.savefig('_edge_region_check.png', dpi=300)
