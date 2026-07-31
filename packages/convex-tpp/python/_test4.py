
import matplotlib.pyplot as plt

from vector2 import Vector2
from polygon2 import Polygon2

poligon = Polygon2([(5, 0.2), (2, 2.2), (0, 2), (-6, -2), (1, -10)])

def draw_point(point: Vector2, offset: Vector2, label: str, color: str) -> None:
    plt.scatter(*(point), color=color, zorder=3, s=60, edgecolor='white', linewidth=2)
    plt.text(*(point + offset), label, fontsize=14, zorder=4)

def draw_other(q: Vector2, offset: Vector2, label: str, color: str) -> None:

    proj = (q - u).project(v - u) + u

    draw_point(q, offset, label, color)
    plt.plot(*[(mid.x, q.x), (mid.y, q.y)], color='white', linewidth=3)
    plt.plot(*[(mid.x, q.x), (mid.y, q.y)], color=color, linestyle='dashed')

plt.xlim(-1.5, 3.5)
plt.ylim(0.45, 3.5)
plt.gca().set_aspect('equal', adjustable='box')

u = poligon[1]
v = poligon[2]

l1 = v.lerp(u, -10)
l2 = u.lerp(v, -10)

plt.fill(*zip(*[l1, l2, (10, 10), (-10, 10)]), alpha=0.3, color='green', zorder=1)
plt.fill(*zip(*[l1, l2, (10, -10), (-10, -10)]), alpha=0.3, color='red', zorder=1)

# plt.fill([-10, 10, 10, -10], [-10, -10, 10, 10], color='lightgray', zorder=0)

plt.fill(*zip(*poligon), alpha=1, color="white")
plt.fill(*zip(*poligon), alpha=0.7, edgecolor='black')


plt.plot(*zip(*poligon, poligon[0]), color='white', linewidth=5)
plt.plot(*zip(*poligon, poligon[0]), color='black')

plt.scatter(*zip(*poligon), color='green', zorder=2, s=40, edgecolor='white', linewidth=2)

# Add text to point 1

draw_point(u, Vector2(0.1, 0.1), "u", "blue")
draw_point(v, Vector2(-0.2, 0.1), "v", "blue")

plt.plot([v.x, l1.x], [v.y, l1.y], color='white', linewidth=3)
plt.plot([v.x, l1.x], [v.y, l1.y], color='black', linestyle='dashed')

plt.plot([u.x, l2.x], [u.y, l2.y], color='white', linewidth=3)
plt.plot([u.x, l2.x], [u.y, l2.y], color='black', linestyle='dashed')

mid = (u + v) / 2
draw_point(mid, Vector2(0, 0.2), "m", "orange")


draw_other(Vector2(3, 3), Vector2(0.1, 0.1), "$q_3$", "green")
draw_other(Vector2(-1, 2.7), Vector2(0.1, 0.1), "$q_4$", "green")
draw_other(Vector2(3.4, 1.7), Vector2(0.1, 0.1), "$q_1$", "red")
draw_other(Vector2(-1, 0.7), Vector2(-0.4, -0.1), "$q_2$", "red")

# draw arrow from u to v
plt.arrow(u.x, u.y, (v - u).x * 0.85, (v - u).y * 0.85, head_width=0.1, head_length=0.2, fc='blue', ec='blue', zorder=2)

xticks = plt.xticks()[0]
yticks = plt.yticks()[0]

plt.xticks(xticks, [""] * len(xticks))
plt.yticks(yticks, [""] * len(yticks))

plt.savefig("out.png", dpi=240, bbox_inches='tight')
