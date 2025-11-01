
import math
from typing import Any, Literal

from vector2 import Vector2


def locate_point(point: Vector2, directions: list[Vector2]) -> int:

	if len(directions) < 2:
		raise ValueError("Should have >= 2 directions")

	def locate_ray(ray: Vector2) -> Literal[0, 1, 2, 3]:
		return ray.rotate(math.pi / 4).quadrant()

	def extend_ray(ray: Vector2) -> Vector2:

		match locate_ray(ray):
			case 0: return ray * side_length / ray.x
			case 1: return ray * side_length / ray.y
			case 2: return ray * -side_length / ray.x
			case 3: return ray * -side_length / ray.y

	import matplotlib.pyplot as plt

	def point_in_cone(ray1: Vector2, ray2: Vector2, point: Vector2) -> bool:
		return (ray1.cross(point) >= 0 and ray2.cross(point) <= 0) ^ (ray1.cross(ray2) < 0)

	def get_points(ray1: Vector2, ray2: Vector2) -> list[Vector2]:

		ray1 = extend_ray(ray1)
		ray2 = extend_ray(ray2)

		side1 = locate_ray(ray1)
		side2 = locate_ray(ray2)

		if side1 == side2 and ray1.cross(ray2) > 0:
			return [Vector2(), ray1, ray2]

		points = [Vector2(), ray1]
		flag = True

		while flag or side1 != side2:
			flag = False
			points.append(Vector2.from_spherical(side_length * 2 ** .5, math.pi / 4 + side1 * math.pi / 2))
			side1 = (side1 + 1) % 4

		points.append(ray2)
	
		return points

	def draw_cone(ray1: Vector2, ray2: Vector2, *args: Any, **kwargs: Any) -> None:

		ray1 = extend_ray(ray1)
		ray2 = extend_ray(ray2)

		points = get_points(ray1, ray2)

		# plt.fill(*zip(*points), color="white", alpha=1)
		plt.fill(*zip(*points), *args, **kwargs)
		plt.plot([0.0, ray1.x], [0.0, ray1.y], color="black")

	side_length = point.magnitude() * 1.5

	fig, ax = plt.subplots()

	ax.set_xlim(-side_length, side_length)
	ax.set_ylim(-side_length, side_length)

	# directions.sort(key=Vector2.angle)

	for i in range(len(directions)):

		ray1 = directions[i]
		ray2 = directions[(i + 1) % len(directions)]

		color = ["blue", "red"][i % 2]

		draw_cone(ray1, ray2, alpha=0.5, color=color)

	# cone = next(i for i in range(len(directions)) if point_in_cone(directions[i], directions[(i + 1) % len(directions)], point))
	cone = find_point(point, directions)

	ray1 = directions[cone]
	ray2 = directions[(cone + 1) % len(directions)]

	draw_cone(ray1, ray2, alpha=0.8, color="green")

	ax.axhline(0, color='black', linewidth=1, alpha=0.3)  # x-axis
	ax.axvline(0, color='black', linewidth=1, alpha=0.3)  # y-axis

	ax.grid(True, which='both', linestyle='--', linewidth=0.5)

	ax.scatter(*point, alpha=1, color="white", edgecolor="black", zorder=5)

	plt.show()

def point_in_cone(ray1: Vector2, ray2: Vector2, point: Vector2) -> bool:

	if ray1.cross(ray2) < 0:
		return not point_in_cone(ray2, ray1, point)

	return ray1.cross(point) >= 0 and ray2.cross(point) <= 0

def find_point(point: Vector2, directions: list[Vector2]) -> int:

	def found(i: int, j: int) -> bool:
		return point_in_cone(directions[i], directions[j], point)

	if found(-1, 0):
		return len(directions) - 1

	left = 0
	right = len(directions) - 1

	while left + 1 != right:

		mid = (left + right) // 2

		if found(left, mid):
			right = mid
		else:
			left = mid
	
	return left


from random import random

directions = [Vector2.from_spherical(1, random() * math.tau) for _ in range(3)]
directions.sort(key=lambda v: v.angle() if v.angle() >= 0 else math.tau + v.angle())

# directions = [Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1), Vector2(1, 0)]

print(locate_point(Vector2(-1, -1), directions))
# print(locate_point(Vector2.from_spherical(1, random() * math.tau), directions))

# i = find_point(Vector2(1, 1), directions)
