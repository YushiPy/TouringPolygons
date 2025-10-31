
from itertools import cycle, product

from pygame import Color
from problem1 import Solution
from vector2 import Vector2
from polygon2 import Polygon2

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import Delaunay
import numpy as np

def polygon_to_convex(polygon: Polygon):
	# Get polygon points
	points = np.array(polygon.exterior.coords[:-1])
	# Perform Delaunay triangulation
	tri = Delaunay(points)
	triangles = []
	for simplex in tri.simplices:
		tri_pts = points[simplex]
		tri_poly = Polygon(tri_pts)
		# Keep only triangles fully inside the polygon
		if polygon.contains(tri_poly.centroid):
			triangles.append(tri_poly)
	return triangles

def solve(start, end, polygons):

	start = Vector2(start)
	end = Vector2(end)
	
	parts = [[Polygon2(x.exterior.coords[:-1]) for x in polygon_to_convex(Polygon(poly))] for poly in polygons]
	possibles = list(product(*(range(len(part)) for part in parts)))

	colors = ["red", "blue", "green"]
	c = cycle(colors)

	WHITE = Color("white")
	BLACK = Color("black")

	for i, (base_poly, p) in enumerate(zip(polygons, parts), 1):

		base_color = Color(next(c))

		bc = "#" + "".join(hex(i)[2:].zfill(2) for i in tuple(base_color)[:3])
		plt.plot(*zip(*(list(base_poly) + [base_poly[0]])), color="white", alpha=1, linewidth=3)
		plt.plot(*zip(*(list(base_poly) + [base_poly[0]])), color=bc, linewidth=1.5)

		plt.fill([-1], [-1], label=f"Polygon {i}", color=bc)

		for j, poly in enumerate(p):

			rate = j / len(p) * 0.7
			color = base_color.lerp(WHITE if i % 2 else BLACK, rate)
			color = "#" + "".join(hex(i)[2:].zfill(2) for i in tuple(color)[:3])

			plt.fill(*zip(*poly), color="white", alpha=1)
			plt.fill(*zip(*poly), color=color, alpha=0.3)
			plt.plot(*zip(*(poly + (poly[0],))), alpha=0.3, color="black") # type: ignore

	paths = []

	for p in possibles:
		sol = Solution(start, end, [parts[i][a] for i, a in enumerate(p)])
		paths.append(sol.shortest_path())

	best = min(paths, key=lambda path: sum((path[i] - path[i-1]).magnitude() for i in range(1, len(path))))

	return best


def star(center, radius, points, rotation=0.0):

	"""
	Generate the points of a star-shaped polygon.

	:param center: The center of the star (x, y).
	:param radius: The radius of the outer points of the star.
	:param points: The number of points (or spikes) of the star.
	:param rotation: The rotation angle in radians.

	:return: A list of (x, y) tuples representing the vertices of the star.
	"""

	angle = np.linspace(0, 2 * np.pi, points, endpoint=False) + rotation
	outer_points = np.array([[np.cos(a), np.sin(a)] for a in angle]) * radius + center
	inner_angle = angle + np.pi / points
	inner_radius = radius / 2
	inner_points = np.array([[np.cos(a), np.sin(a)] for a in inner_angle]) * inner_radius + center
	star_points = np.empty((points * 2, 2))
	star_points[0::2] = outer_points
	star_points[1::2] = inner_points
	return star_points

x = star((0.5, 8.0), 5.0, 5, rotation=0.54)
y = star((-6.7, 2.0), 4.0, 6, 0.6)

start = (-6.0, 7.0)
end = (4.0, 2.0)

minx = min(min(p[0] for p in x), min(p[0] for p in y), start[0], end[0]) - 1
maxx = max(max(p[0] for p in x), max(p[0] for p in y), start[0], end[0]) + 1
miny = min(min(p[1] for p in x), min(p[1] for p in y), start[1], end[1]) - 1
maxy = max(max(p[1] for p in x), max(p[1] for p in y), start[1], end[1]) + 1

plt.figure(figsize=(6, 6))
plt.xlim(minx, maxx)
plt.ylim(miny, maxy)

plt.fill([minx, maxx, maxx, minx], [miny, miny, maxy, maxy], color="#ccd7d8", alpha=0.5) # type: ignore

plt.plot(*zip(*path), color="purple", linewidth=2, linestyle="dashed", marker="o", markersize=3) # type: ignore

plt.plot([start[0]], [start[1]], color="white", alpha=1, marker="o") # type: ignore
plt.scatter([start[0]], [start[1]], color="green", alpha=1, marker="o", label="Start") # type: ignore
plt.plot([start[0]], [start[1]], color="green", alpha=1, marker="o") # type: ignore

plt.plot([end[0]], [end[1]], color="white", alpha=1, marker="o") # type: ignore
plt.scatter([end[0]], [end[1]], color="red", alpha=1, marker="o", label="End") # type: ignore
plt.plot([end[0]], [end[1]], color="red", alpha=1, marker="o") # type: ignore

path = solve(start, end, [x, y])

plt.plot([minx - 1], [miny], color="purple", linewidth=2, linestyle="dashed", marker="o", markersize=3, label="Path") # type: ignore

plt.gca().set_aspect('equal', adjustable='box')

plt.grid() # type: ignore
plt.legend() # type: ignore
plt.tight_layout()

plt.savefig("output2.png", dpi=300) # type: ignore

plt.show() # type: ignore