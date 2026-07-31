
from collections.abc import Sequence


type Vector2 = tuple[float, float]

def vector_sub(v1: Vector2, v2: Vector2) -> Vector2:
	return (v1[0] - v2[0], v1[1] - v2[1])

def vector_cross(v1: Vector2, v2: Vector2) -> float:
	return v1[0] * v2[1] - v1[1] * v2[0]

def vector_dot(v1: Vector2, v2: Vector2) -> float:
	return v1[0] * v2[0] + v1[1] * v2[1]

def vector_is_same_direction(v1: Vector2, v2: Vector2) -> bool:
	return vector_cross(v1, v2) == 0 and vector_dot(v1, v2) >= 0


def remove_collinear_points(points: Sequence[Vector2]) -> list[Vector2]:
	"""
	Removes collinear points from a sequence of points.
	"""

	cleaned: list[Vector2] = [points[0], points[1]]

	for i in range(2, len(points)):

		a = cleaned[-2]
		b = cleaned[-1]
		candidate = points[i]

		v1 = vector_sub(b, a)
		v2 = vector_sub(candidate, b)

		if vector_is_same_direction(v1, v2):
			cleaned[-1] = candidate
		else:
			cleaned.append(candidate)

	return cleaned


string = """
{
		{{119.01, 110.12}, {123.63, 110.36}, {127.68, 110.56}, {128.63, 91.97}, {127.93, 91.92}, {128.38, 83.16}, {126.81, 84.19}, {125.52, 84.97}, {124.23, 85.71}, {122.9, 86.42}, {121.56, 87.09}, {120.83, 87.42}, {120.6, 92.76}, {119.9, 92.73}}, 
		{{104.59, 117.34}, {109.34, 117.49}, {110.1, 92.22}, {105.34, 92.09}}, 
		{{117.47, 125.13}, {123.05, 125.4}, {127.94, 125.66}, {127.97, 120.95}, {123.29, 120.71}, {123.47, 117.32}, {128.22, 117.57}, {128.53, 111.48}, {123.61, 111.23}, {117.85, 110.93}, {117.69, 116.73}}, 
		{{104.13, 132.71}, {112.44, 132.93}, {112.51, 130.6}, {114.9, 130.66}, {114.98, 127.33}, {117.41, 127.41}, {117.47, 125.13}, {117.69, 116.73}, {115.05, 116.66}, {114.97, 119.17}, {104.55, 118.86}}, 
		{{116.91, 146.89}, {121.57, 147.14}, {121.76, 143.67}, {127.71, 143.98}, {128.29, 132.48}, {127.91, 132.46}, {127.94, 125.66}, {123.05, 125.4}, {117.47, 125.13}, {117.41, 127.41}}
	}
"""

start = (0, 0)
target = (233.47, 164.63)

polygons = eval(string.replace("{", "[").replace("}", "]"))
polygons = [[(v[0], v[1]) for v in polygon] for polygon in polygons]
polygons = list(map(remove_collinear_points, polygons))

from tpp_bnb import tpp_solve

solution = tpp_solve(start, target, polygons)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(*start, color='green', label='Start')
ax.scatter(*target, color='red', label='Target')

for polygon in polygons:
	ax.fill(*zip(*polygon), alpha=0.5)

if solution is not None:
	solution.append(target)
	ax.plot(*zip(*solution), color='blue', label='Solution')

plt.show()