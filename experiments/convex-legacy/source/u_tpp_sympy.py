
from collections.abc import Sequence
import math
import numpy as np
from scipy.optimize import minimize

type Point = tuple[float, float]
type Polygon = list[Point]

def order_polygon(polygon: Sequence[Point]) -> Polygon:
	"""
	Order the vertices of a polygon in counterclockwise order.

	:param Polygon polygon: The input polygon as a list of points.

	:return: A new list of points representing the vertices of the polygon in counterclockwise order.
	"""
	center = (sum(x for x, y in polygon) / len(polygon), sum(y for x, y in polygon) / len(polygon))
	return sorted(polygon, key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]))

def tpp_solve(start: Point, target: Point, polygons: Sequence[Sequence[Point]]) -> list[Point]:

	polygons = [order_polygon(polygon) for polygon in polygons]

	# vars = [x1, y1, x2, y2, ..., xk, yk]
	def f(vars: list[float]) -> float:
		points = [start] + [(vars[i], vars[i + 1]) for i in range(0, len(vars), 2)] + [target]
		return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))

	# constraints: each must return >= 0
	constraints = []

	for i, polygon in enumerate(polygons):

		for j in range(len(polygon)):
			x1, y1 = polygon[j]
			x2, y2 = polygon[(j + 1) % len(polygon)]
			a = y2 - y1
			b = x1 - x2
			c = x2 * y1 - x1 * y2

			def constraint(vars: list[float], i=i, a=a, b=b, c=c) -> float:
				return -(a * vars[2 * i] + b * vars[2 * i + 1] + c)

			constraints.append({"type": "ineq", "fun": constraint})

	x0 = np.array([polygon[0] for polygon in polygons]).flatten()
	solution = minimize(f, x0, method="SLSQP", constraints=constraints)

	return [start] + [(float(solution.x[i]), float(solution.x[i + 1])) for i in range(0, len(solution.x), 2)] + [target]

