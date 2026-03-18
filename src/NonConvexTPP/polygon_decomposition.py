
from collections.abc import Sequence

from vector2 import Vector2
from polygon2 import Polygon2

type Point = tuple[float, float]
type Polygon = Sequence[Point]


def point_in_triangle(p: Vector2, a: Vector2, b: Vector2, c: Vector2) -> bool:
	return ((b - a).cross(p - a) >= 0 and
	        (c - b).cross(p - b) >= 0 and
	        (a - c).cross(p - c) >= 0)

def is_ear(polygon: Sequence[Vector2], index: int) -> bool:
	"""
	Determines if the vertex at `index` in `polygon` is an ear.
	An ear is a triangle formed by three consecutive vertices that lies entirely within the polygon.
	"""

	v_prev, v, v_next = polygon[index - 1], polygon[index], polygon[(index + 1) % len(polygon)]

	if (v_next - v).cross(v_prev - v) <= 0:
		return False  # reflex vertex, cannot be an ear

	for i, p in enumerate(polygon):

		if i in (index - 1, index, (index + 1) % len(polygon)):
			continue

		if point_in_triangle(p, v_prev, v, v_next):
			return False

	return True

def triangulate(polygon: Polygon2) -> list[Polygon2]:
	"""
	Triangulates `polygon` using ear clipping.
	Returns a list of triangles, each represented as a list of 3 points.
	"""

	current = list(polygon)
	triangles: list[list[Vector2]] = []

	while len(current) > 3:

		for i in range(len(current)):
			if is_ear(current, i):
				v_prev, v, v_next = current[i - 1], current[i], current[(i + 1) % len(current)]
				triangles.append([v_prev, v, v_next])
				del current[i]
				break

	triangles.append(current)  # the last triangle

	return [Polygon2(tri) for tri in triangles]

def shared_edge(poly_a: Polygon2, poly_b: Polygon2) -> tuple[Vector2, Vector2] | None:
	"""
	Finds a shared edge between `poly_a` and `poly_b`, if it exists.
	Returns the two endpoints of the shared edge as a tuple, or None if no shared edge exists.
	"""
	s1 = set(poly_a)
	s2 = set(poly_b)
	shared = s1.intersection(s2)

	if len(shared) == 2:
		return tuple(shared) # type: ignore
	else:
		return None

def merge_polygons(poly_a: Polygon2, poly_b: Polygon2) -> Polygon2 | None:
	"""
	Attempts to merge `poly_a` and `poly_b` if they share an edge and the result is convex.
	Returns the merged polygon if successful, or None if they cannot be merged.
	"""

	edge = shared_edge(poly_a, poly_b)

	if edge is None:
		return None

	u, v = edge

	index1 = poly_a.index(u)

	if poly_a[(index1 - 1) % len(poly_a)] == v:
		# Edge is v->u in poly_a
		poly_a, poly_b = poly_b, poly_a  # swap to ensure consistent ordering
		index2 = index1
		index1 = poly_a.index(u)
	else:
		index2 = poly_b.index(u)

	# (0, 0), (2, 3), (0, 3)
	# (0, 0), (1.5, 2), (2, 3)
	# poly_a: ..., u, v, ...
	# poly_b: ..., v, u, ...
	# merged: 
	result = []

	index = (index1 + 1) % len(poly_a)

	while index != index1:
		result.append(poly_a[index])
		index = (index + 1) % len(poly_a)

	index = index2 % len(poly_b)

	while index != (index2 - 1) % len(poly_b):
		result.append(poly_b[index])
		index = (index + 1) % len(poly_b)

	result = Polygon2(result) # type: ignore

	if result.is_convex():
		return result
	else:
		return None

def hertel_mehlhorn(polygon: Sequence[tuple[float, float]]) -> list[list[tuple[float, float]]]:
	"""
	Decomposes a simple polygon into convex pieces using the Hertel-Mehlhorn algorithm.
	Returns a list of convex polygons that together cover the original polygon.
	"""

	input_polygon = Polygon2(polygon)

	if not input_polygon.is_simple():
		return [list(input_polygon)] # type: ignore

	# Step 1: Triangulate the polygon (using ear clipping)
	triangles = triangulate(input_polygon)
	
	# Step 2: Iteratively merge adjacent triangles into convex pieces
	changed = True
	current = triangles

	while changed:
		changed = False
		new_pieces = []
		merged_flags = [False] * len(current)

		for i in range(len(current)):

			if merged_flags[i]:
				continue

			for j in range(i + 1, len(current)):

				if merged_flags[j]:
					continue

				merged = merge_polygons(current[i], current[j])

				if merged is not None:
					new_pieces.append(merged)
					merged_flags[i] = True
					merged_flags[j] = True
					changed = True
					break
			else:
				new_pieces.append(current[i])

		current = new_pieces
	
	return [list(p) for p in current] # type: ignore
