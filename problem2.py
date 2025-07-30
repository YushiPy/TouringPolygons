
import heapq
from math import inf

from vector2 import Vector2
from polygon2 import Polygon2

def intersection_rates(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2) -> tuple[float, float] | None:

	cross = direction1.cross(direction2)

	if abs(cross) < 1e-8:
		return None
	
	sdiff = start2 - start1

	rate1 = sdiff.cross(direction2) / cross
	rate2 = sdiff.cross(direction1) / cross

	return rate1, rate2

def segment_segment_intersection(start1: Vector2, end1: Vector2, start2: Vector2, end2: Vector2) -> Vector2 | None:

	"""
	Returns the intersection point of two line segments if they intersect, otherwise returns None.

	:param Vector2 start1: The start point of the first segment as a Vector2.
	:param Vector2 end1: The end point of the first segment as a Vector2.
	:param Vector2 start2: The start point of the second segment as a Vector2.
	:param Vector2 end2: The end point of the second segment as a Vector2.

	:return: The intersection point as a Vector2 if the segments intersect, otherwise None.	
	"""

	diff1 = end1 - start1
	diff2 = end2 - start2

	rates = intersection_rates(start1, diff1, start2, diff2)

	if rates is not None and 0 <= rates[0] <= 1 and 0 <= rates[1] <= 1:
		return start1 + diff1 * rates[0]


def shortest_path_in_polygon(start: Vector2, end: Vector2, polygon: Polygon2) -> list[Vector2]:
	"""
	Calculate the shortest path from start to end, while staying inside the polygon.

	:param Vector2 start: The starting point.
	:param Vector2 end: The ending point.
	:param Polygon2 polygon: The polygon to stay within.

	:return: A list of Vector2 points representing the shortest path.
	"""

	vertices: list[Vector2] = [start, end] + list(polygon)
	edges: list[list[tuple[int, float]]] = [[] for _ in range(len(vertices))]

	for i in range(len(vertices)):
		for j in range(i + 1, len(vertices)):

			# If segments intersect, do not add an edge

			first = vertices[i]
			second = vertices[j]

			if not polygon.contains_segment(first, second, 1e-6):
				continue

			cost: float = (vertices[i] - vertices[j]).magnitude()

			edges[i].append((j, cost))
			edges[j].append((i, cost))
	
	# A star algorithm
	gscore = [inf] * len(vertices)
	gscore[0] = 0

	visited = [False] * len(vertices)
	previous = [-1] * len(vertices)

	queue: list[tuple[float, int]] = [(0, 0)] # (cost, index)

	while queue:

		_, vertex = heapq.heappop(queue)

		if vertex == 1: # Reached the end
			break

		if visited[vertex]:
			continue

		visited[vertex] = True
		gcost = gscore[vertex]

		for target, edge_cost in edges[vertex]:

			if visited[target]:
				continue

			new_cost = gcost + edge_cost

			if new_cost >= gscore[target]:
				continue

			gscore[target] = new_cost
			previous[target] = vertex
			heuristic = (vertices[target] - end).magnitude()
			heapq.heappush(queue, (new_cost + heuristic, target))

	if previous[1] == -1:
		raise ValueError("No path found from start to end.")
	
	path: list[Vector2] = []
	current = 1

	while current != 0:
		path.append(vertices[current])
		current = previous[current]

	path.append(start)
	path.reverse()

	return path


def point_in_segment(point: Vector2, start: Vector2, end: Vector2, eps: float = 1e-10) -> bool:
	"""
	Check if a point is within a line segment.

	:param Vector2 point: The point to check.
	:param Vector2 start: The start point of the segment.
	:param Vector2 end: The end point of the segment.
	:param float eps: A small epsilon value for numerical stability.

	:return: True if the point is within the segment, False otherwise.
	"""

	d1 = point - start
	d2 = end - start

	if abs(d1.cross(d2)) >= eps:
		return False

	rate = d1.dot(d2) / d2.magnitude_squared()

	return -eps <= rate <= 1 + eps

def get_outline(poly1: Polygon2, poly2: Polygon2) -> list[Vector2]:
	"""
	Get the outline of the intersection of two polygons.

	:param Polygon2 poly1: The first polygon.
	:param Polygon2 poly2: The second polygon.

	:return: A list of Vector2 points representing the outline of the intersection.
	"""

	def add(point: Vector2) -> None:
		if not any((point - v).magnitude() < 1e-8 for v in result):
			result.append(point)

	result: list[Vector2] = []

	for v1, v2 in poly1.edges():

		add(v1)

		ps = next(((a, b) for a, b in poly2.edges() if point_in_segment(a, v1, v2) and point_in_segment(b, v1, v2)), None)

		if ps is not None:

			if ps[1].distance_to(v1) < ps[0].distance_to(v1):
				ps = (ps[1], ps[0])

			add(ps[0])
			add(ps[1])

			continue

		for v3, v4 in poly2.edges():

			if point_in_segment(v3, v1, v2):
				add(v3)
				continue
			
			point = segment_segment_intersection(v1, v2, v3, v4)

			if point is not None:
				add(point)
	
	return result


def point_in_cone(point: Vector2, start: Vector2, ray1: Vector2, ray2: Vector2) -> bool:
	"""
	Check if a point is inside the cone defined by two rays starting from `start`.
	The rays are in clockwise order.

	:param Vector2 point: The point to check.
	:param Vector2 start: The starting point of the cone.
	:param Vector2 ray1: The first ray direction.
	:param Vector2 ray2: The second ray direction.

	:return: True if the point is inside the cone, False otherwise.
	"""

	vector = point - start

	cross1 = ray1.cross(vector)
	cross2 = ray2.cross(vector)

	# Is to the counter-clockwise side of the first ray and 
	# clockwise side of the second ray.
	return (cross1 >= 0 and cross2 <= 0)

def point_in_edge(point: Vector2, start1: Vector2, ray1: Vector2, start2: Vector2, ray2: Vector2) -> bool:
	"""
	Check if a point is inside the edge defined by two rays starting from `start1` and `start2`.

	:param Vector2 point: The point to check.
	:param Vector2 start1: The starting point of the first ray.
	:param Vector2 ray1: The direction vector of the first ray.
	:param Vector2 start2: The starting point of the second ray.
	:param Vector2 ray2: The direction vector of the second ray.

	:return: True if the point is inside the edge, False otherwise.
	"""

	vector1 = point - start1
	vector2 = point - start2

	ray3 = start2 - start1

	cross1 = ray1.cross(vector1)
	cross2 = ray2.cross(vector2)
	cross3 = ray3.cross(vector1)

	# Is to the counter-clockwise side of the first ray and 
	# clockwise side of the second ray.
	# Also checks if the point is clockwise to the segment from `start1` to `start2`.
	return (cross1 >= 0 and cross2 <= 0 and cross3 <= 0)


class Solution:

	start: Vector2
	end: Vector2

	polygons: list[Polygon2]
	fences: list[Polygon2]

	# Indicates whether edge `j` of polygon `i` is blocked
	blocked: list[list[bool]]

	cones: list[list[tuple[Vector2, Vector2]]]

	def __init__(self, start: Vector2, end: Vector2, polygons: list[Polygon2], fences: list[Polygon2]):
		
		self.start = start
		self.end = end
		self.polygons = polygons
		self.fences = fences

		self.blocked = []
		self.cones = []

	def query_a(self, point: Vector2, index: int) -> Vector2:

		if index == 0:
			return self.start

	def shortest_path(self) -> list[Vector2]:

		if len(self.polygons) == 0:
			return [self.start, self.end]

		polygon = self.polygons[0]

		blocked: list[bool] = []
		cones: list[tuple[Vector2, Vector2]] = []

		self.blocked.append(blocked)
		self.cones.append(cones)

		# Determining which edges are blocked.
		for v1, v2 in polygon.edges():

			middle = v1.lerp(v2, 0.5)

			last = self.query_a(middle, 0)
			last = shortest_path_in_polygon(last, middle, self.fences[0])[-2]

			# Check if the segment from `last` to `middle` intersects with 
			# any edge of the polygon that is not the segment from `v1` to `v2`.
			is_blocked = any((a, b) != (v1, v2) and segment_segment_intersection(last, middle, a, b) is not None for a, b in polygon.edges())
			blocked.append(is_blocked)

		# Determining the cones of visibility for each vertex.
		for j in range(len(polygon)):

			vertex = polygon[j]

			before = polygon[j - 1]
			after = polygon[j + 1]

			last = self.query_a(vertex, 0)
			last = shortest_path_in_polygon(last, vertex, self.fences[0])[-2]

			diff = vertex - last

			ray1 = diff.reflect((vertex - before).perpendicular()).normalize()
			ray2 = diff.reflect((vertex - after).perpendicular()).normalize()

			if blocked[j - 1]:
				ray1 = diff.normalize()
			if blocked[j]:
				ray2 = diff.normalize()

			cones.append((ray1, ray2))

		return []


def draw(start: Vector2, end: Vector2, polygons: list[Polygon2], fences: list[Polygon2]) -> None:
	"""
	Draw the polygons and fences along with the start and end points.
	"""
	import matplotlib.pyplot as plt

	plt.figure(figsize=(8, 8))

	for polygon in polygons:
		plt.fill(*zip(*polygon), alpha=0.5)

	for i, fence in enumerate(fences):
		plt.plot(*zip(*(list(fence) + [fence[0]])), label=f'Fence {i}')

	plt.plot(start.x, start.y, 'go', label='Start')
	plt.plot(end.x, end.y, 'ro', label='End')

	plt.legend()
	plt.grid()
	plt.axis('equal')
	plt.show()

start = Vector2(-3, 0)
end = Vector2(3, 0)

polygons = [
	Polygon2([Vector2(-1, 1), Vector2(1, 1), Vector2(0, 2)]),
]

fences = [
	Polygon2([Vector2(-4, 3), Vector2(-3, -1), Vector2(-2, 2), Vector2(-1, -1), Vector2(2, 0), Vector2(2, 2)]),
	Polygon2([Vector2(-2, 3), Vector2(2, 2), Vector2(3, 2), Vector2(4, 0), Vector2(3, -1), Vector2(2, 1), Vector2(-1, -1)]),
]

solution = Solution(start, end, polygons, fences)

# print(solution.shortest_path())

# draw(start, end, polygons, fences)

