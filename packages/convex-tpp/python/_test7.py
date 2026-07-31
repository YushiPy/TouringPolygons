
from itertools import chain
from math import tau
from collections.abc import Callable
import matplotlib.pyplot as plt

from vector2 import Vector2
from polygon2 import Polygon2

def regular_polygon(center: Vector2, radius: float, sides: int, rotation: float = 0) -> Polygon2:
	return Polygon2(center + Vector2.from_polar(radius, rotation + i * (tau / sides)) for i in range(sides))

def get_filtered(polygon: list[Vector2], last: Vector2) -> list[Vector2]:

	start = -1
	end = -1

	for i in range(len(polygon)):

		if start != -1 and end != -1:
			break

		before = polygon[i - 1]
		v = polygon[i]
		after = polygon[(i + 1) % len(polygon)]
		
		last = last
		diff = v - last

		if start == -1 and diff.cross(after - v) < 1e-8 and diff.cross(v - before) > -1e-8:
			start = i
		
		if end == -1 and diff.cross(after - v) > -1e-8 and diff.cross(v - before) < 1e-8:
			end = i

	if start < end:
		return polygon[start:end + 1]
	else:
		return polygon[start:] + polygon[:end + 1]

def get_pass_through(polygon: list[Vector2], last: Vector2) -> list[Vector2]:

	def is_start(i: int) -> bool:

		before = polygon[i - 1]
		v = polygon[i]
		after = polygon[(i + 1) % len(polygon)]
		
		diff = v - last

		return diff.cross(after - v) < 1e-8 and diff.cross(v - before) > -1e-8

	def is_end(i: int) -> bool:

		before = polygon[i - 1]
		v = polygon[i]
		after = polygon[(i + 1) % len(polygon)]
		
		diff = v - last

		return diff.cross(after - v) > -1e-8 and diff.cross(v - before) < 1e-8

	def is_pass(i: int) -> bool:
		
		before = polygon[i - 1]
		v = polygon[i]
		after = polygon[(i + 1) % len(polygon)]
		
		diff = v - last

		return diff.cross(after - v) < -1e-8 or diff.cross(v - before) < -1e-8

	def is_sorted(i: int, j: int) -> bool:

		before1 = polygon[i - 1]
		v1 = polygon[i]
		after1 = polygon[(i + 1) % len(polygon)]

		diff1 = v1 - last

		before2 = polygon[j - 1]
		v2 = polygon[j]
		after2 = polygon[(j + 1) % len(polygon)]

		diff2 = v2 - last

		if is_pass(i):
			if is_pass(j):
				return diff1.normalize().dot((v1 - after1).normalize()) > diff2.normalize().dot((v2 - after2).normalize())
			else:
				return False
		else:
			if is_pass(j):
				return False
			else:
				return diff1.normalize().dot((v1 - before1).normalize()) < diff2.normalize().dot((v2 - before2).normalize())

	def find_start(left: int, right: int) -> int:

		if is_start(left):
			return left
		
		if is_start(right):
			return right

		mid = (left + right) // 2

		if is_pass(left):
			if is_pass(mid):
				if is_sorted(left, mid):
					return find_start(mid + 1, right)
				else:
					return find_start(left, mid)
			else:
				return find_start(mid + 1, right)
		else:
			if is_pass(mid):
				return find_start(left + 1, mid)
			else:
				if is_sorted(left, mid):
					return find_start(left + 1, mid - 1)
				else:
					return find_start(mid + 1, right)
		
	def find_end(left: int, right: int) -> int:

		if is_end(left):
			return left

		if is_end(right):
			return right

		mid = (left + right) // 2


		if is_pass(left):
			if is_pass(mid):
				if is_sorted(left, mid):
					return find_end(mid, right)
				else:
					return find_end(left, mid)
			else:
				return find_end(left, mid - 1)
		else:
			if is_pass(mid):
				return find_end(mid, right)
			else:
				if is_sorted(left, mid):
					return find_end(left + 1, mid - 1)
				else:
					return find_end(mid + 1, right)

	start = (find_start(0, len(polygon) - 1))
	end = (find_end(0, len(polygon) - 1))

	if start < end:
		return polygon[start:end + 1]
	else:
		return polygon[start:] + polygon[:end + 1]

polygon = Polygon2([
	Vector2(1, 1),
	Vector2(4, 1),
	Vector2(4, 3),
	Vector2(2, 4),
	Vector2(1, 3)
])

polygon = regular_polygon(Vector2(0, 0), 2, 19, rotation=tau * -0.1235)
# polygon = Polygon2(list(polygon)[:len(polygon) // 2] + list(polygon)[8 * len(polygon) // 9:])

start = Vector2(-3.0, 3.1)

filtered = get_filtered(list(polygon), start)
#filtered = get_pass_through(list(polygon), start)

if get_filtered(list(polygon), start) != get_pass_through(list(polygon), start):
	raise ValueError("Filtered and pass-through do not match.")

bbox = Polygon2.bbox(chain(polygon, [start]), extra=0.3)
minx, miny, maxx, maxy = bbox[0].x, bbox[0].y, bbox[1].x, bbox[1].y

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

ax.set_aspect('equal', 'box')

ax.fill([minx, maxx, maxx, minx], [miny, miny, maxy, maxy], color='black', alpha=0.1)

ax.fill(*zip(*polygon), edgecolor='black', alpha=0.5)

for i in range(1, len(filtered)):
	ax.plot([filtered[i - 1].x, filtered[i].x], [filtered[i - 1].y, filtered[i].y], color='blue', linewidth=3)

for i in range(len(polygon)):
	v = polygon[i]
	ax.scatter([v.x], [v.y], color='black', s=10, zorder=10)
	ax.text(v.x - 0.2, v.y + 0.2, str(i), color='black', fontsize=12, ha='center', va='center')

ax.plot(*zip(start, start.lerp(filtered[0], 2)), color='red', linestyle='dashed', linewidth=2)
ax.plot(*zip(start, start.lerp(filtered[-1], 2)), color='red', linestyle='dashed', linewidth=2)

ax.scatter([start.x], [start.y], color='white', s=60)
ax.scatter([start.x], [start.y], color='red', s=20)

