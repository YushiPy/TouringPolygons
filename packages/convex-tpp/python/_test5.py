
from math import tau
from vector2 import Vector2
from polygon2 import Polygon2
from UnconstrainedTPP.u_tpp_naive_hard import Solution

def regular_polygon(n: int, center: Vector2 = Vector2(0, 0), radius: float = 1.0, rotation: float = 0.0) -> Polygon2:
	points: list[Vector2] = []
	for i in range(n):
		angle = rotation + (i / n) * tau
		point = Vector2.from_polar(radius, angle) + center
		points.append(point)
	return Polygon2(points)


start = Vector2(0, 2)
target = Vector2(0, -2)
point = Vector2(0.5, 2)

# n_sides = 3
n_sides = 6
polygon = regular_polygon(n_sides, Vector2(0, 0), 1.0, tau / 4)

sol = Solution(start, target, [polygon])
sol.solve()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

xlim = -1.5, 1.5
ylim = -0, 3

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect('equal', adjustable='box')

ax.fill([xlim[0], xlim[1], xlim[1], xlim[0]], [ylim[0], ylim[0], ylim[1], ylim[1]], color='lightgray', zorder=-1)

v_index = 0
v = sol.polygons[0][v_index]
d1, d2 = sol.cones[0][v_index]

color1 = "green"
color2 = "purple"

ax.plot(*zip(v, v + d1 * 5), color=color1, linestyle='dashed', zorder=2)
ax.arrow(*v, *(0.7 * d1), color=color1, head_width=0.1, head_length=0.2, zorder=2)
ax.plot(*zip(v, v + d2 * 5), color=color2, linestyle='dashed', zorder=2)
ax.arrow(*v, *(0.7 * d2), color=color2, head_width=0.1, head_length=0.2, zorder=2)

scale = 3

p0 = v - d1 * scale
p1 = v + d1 * scale
p2 = v + (d1 + d1.perpendicular()) * scale
p3 = v + (-d1 + d1.perpendicular()) * scale

ax.fill(*zip(p0, p1, p2, p3), alpha=0.2, color=color1, zorder=1)
ax.plot(*zip(p0, v), color="white", zorder=1, linewidth=4)
ax.plot(*zip(p0, p1), color=color1, zorder=1, linestyle='dashed')

p0 = v + d2 * scale
p1 = v - d2 * scale
p2 = v - (d2 + d2.perpendicular()) * scale
p3 = v - (-d2 + d2.perpendicular()) * scale

ax.fill(*zip(p0, p1, p2, p3), alpha=0.2, color=color2, zorder=1)
ax.plot(*zip(v, p1), color="white", zorder=1, linewidth=4)
ax.plot(*zip(p0, p1), color=color2, zorder=1, linestyle='dashed')

ax.fill(*zip(*polygon), alpha=0.6, color="blue", zorder=3)
ax.plot(*zip(*polygon, polygon[0]), color='black', zorder=3, linewidth=2)

ax.scatter(*v, color='red', s=60, zorder=3)
ax.scatter(*v, color='white', s=20, zorder=3)
ax.text(*(v + Vector2(0.00, 0.1)), "v", fontsize=16, zorder=4)

# Text with background for d1
ax.text(*(v + d1.normalize() * 1.1), "$r^1$", fontsize=16, color=color1, zorder=4, bbox=dict(facecolor='white', edgecolor='none', pad=3, alpha=1.0), ha='center', va='center')

# Text with background for d2
ax.text(*(v + d2.normalize() * 1.1), "$r^2$", fontsize=16, color=color2, zorder=4, bbox=dict(facecolor='white', edgecolor='none', pad=3, alpha=1.0), ha='center', va='center')

# Text with background for d1 and d2
# ax.text(*(v + Vector2(-0.15, 1)), "$d^1, d^2$", fontsize=14, color='black', zorder=4, bbox=dict(facecolor='white', edgecolor='none', pad=3, alpha=0.7))

ax.scatter(*point, color='white', s=60, zorder=3)
ax.scatter(*point, color='orange', s=20, zorder=3)
ax.text(*(point + Vector2(0.06, 0.06)), "p", fontsize=14, zorder=4)

ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
plt.savefig("cone_diagram.png", dpi=300, bbox_inches='tight')

plt.show()