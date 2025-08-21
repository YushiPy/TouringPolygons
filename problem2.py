
"""
Implementation of the second variation of the problem. 

We are given:
- A starting point `s`.
- A target point `t`.
- Convex polygons P_1, ..., P_k.
- Simple poligons called "fences" F_0, ..., F_k such that
for all 0 <= i <= k the polygons P_i and P_{i + 1} are inside the fence F_i.
We will conside that P_0 = s and P_{k + 1} = t.

We will also consider that:
- The polygons are non intersecting
- The problem is in 2D.
"""

import math
from typing import Iterable
import heapq


from vector2 import Vector2
from polygon2 import Polygon2


def shortest_path_in_polygon(start: Vector2, end: Vector2, polygon: Polygon2) -> list[Vector2]:
	"""
	Calculate the shortest path from start to end, while staying inside the polygon.

	:param Vector2 start: The starting point.
	:param Vector2 end: The ending point.
	:param Polygon2 polygon: The polygon to stay within.

	:return: A list of Vector2 points representing the shortest path.
	"""

	vertices: list[Vector2] = [start, end] + polygon.reflex_vertices
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
	gscore = [math.inf] * len(vertices)
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

def point_in_cone(point: Vector2, start: Vector2, ray1: Vector2, ray2: Vector2, eps: float = 1e-10) -> bool:
	"""
	Check if a point is inside the cone defined by two rays starting from `start`.
	The rays are in clockwise order.

	:param Vector2 point: The point to check.
	:param Vector2 start: The starting point of the cone.
	:param Vector2 ray1: The first ray direction.
	:param Vector2 ray2: The second ray direction.

	:return: True if the point is inside the cone, False otherwise.
	"""

	if ray1.cross(ray2) < 0:
		return not point_in_cone(point, start, ray2, ray1, eps)

	vector = point - start

	cross1 = ray1.cross(vector)
	cross2 = ray2.cross(vector)

	# Is to the counter-clockwise side of the first ray and 
	# clockwise side of the second ray.
	return (cross1 >= -eps and cross2 <= eps)

def intersection_rates(start1: Vector2, direction1: Vector2, start2: Vector2, direction2: Vector2) -> tuple[float, float] | None:

	cross = direction1.cross(direction2)

	if abs(cross) < 1e-8:
		return None
	
	sdiff = start2 - start1

	rate1 = sdiff.cross(direction2) / cross
	rate2 = sdiff.cross(direction1) / cross

	return rate1, rate2

def segment_segment_intersection(start1: Vector2, end1: Vector2, start2: Vector2, end2: Vector2, eps: float = 1e-10) -> Vector2 | None:
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

	if rates is not None and -eps <= rates[0] <= 1 + eps and -eps <= rates[1] <= 1 + eps:
		return start1 + diff1 * rates[0]



class Solution:

	start: Vector2
	end: Vector2

	polygons: list[Polygon2]
	fences: list[Polygon2]

	# For each vertex j in the polygon P_i, we store 
	# two vectors that represent the the start and end of the cone
	# of the vertex. The vectors are in counter-clockwise order.
	# If the vectors are the same, it means that the vertex
	# is not part of T_i.
	cones: list[list[tuple[Vector2, Vector2]]]

	# Indicates whether the vertex j of P_i is blocked by P_i.
	blocked: list[list[bool]]

	# For each polygon P_i, we store the direction of the 
	# last segment that reaches the vertex.
	# These vertices are the reflex vertices of the fences.
	last_fence_vertex: list[list[Vector2]]

	def __init__(self, start: Vector2, end: Vector2, polygons: Iterable[Polygon2], fences: Iterable[Polygon2]) -> None:
		"""
		Initialize the solution with the starting point, end point, polygons, and fences.

		:param Vector2 start: The starting point of the path.
		:param Vector2 end: The end point of the path.
		:param Iterable[Polygon2] polygons: An iterable of convex polygons representing the obstacles.
		:param Iterable[Polygon2] fences: An iterable of simple polygons representing the fences.

		:raises ValueError: If any of the polygons are not convex.
		"""

		self.start = start
		self.end = end

		self.polygons = [Polygon2(polygon) for polygon in polygons]
		self.fences = [Polygon2(fence) for fence in fences]

		self.cones = [[] for _ in range(len(self.polygons))]
		self.blocked = [[] for _ in range(len(self.polygons))]

		self.last_fence_vertex = [[] for _ in range(len(self.polygons))]

		if not all(polygon.is_convex() for polygon in self.polygons):
			raise ValueError("All polygons must be convex.")
	
	def shortest_fenced_path(self, start: Vector2, end: Vector2, index: int) -> Vector2:
		return shortest_path_in_polygon(start, end, self.fences[index])[-2]

	def query(self, index: int, point: Vector2) -> Vector2:
		"""
		Returns the start of the last segment of the shortest 
		`index`-path that reaches `point`.
		"""

		if index == 0:
			return self.shortest_fenced_path(self.start, point, 0)

		polygon = self.polygons[index - 1]
		cones = self.cones[index - 1]
		blocked = self.blocked[index - 1]

		# Check if point is inside a cone region
		for j in range(len(polygon)):

			vertex = polygon[j]

			if blocked[j]:
				continue

			first, second = cones[j]

			if point_in_cone(point, vertex, first, second):
				return vertex

		# TODO: Check if point is inside edge region

		# Point is in pass-through region

		# Check if a straight segment from any point in T_{index - 1} to point is optimal.
		for i, j in enumerate(self.fences[index - 1].reflex_vertices_indices + [-1]):

			if j == -1:
				vertex = self.start
			else:
				vertex = self.fences[index - 1][j]
				before = self.fences[index - 1][j - 1]
				last = self.last_fence_vertex[index - 1][i]

				rates = intersection_rates(vertex, before - vertex, last, point - last)

				# If the intersection is not valid, then the path is not optimal.
				if rates is None or rates[0] < 0:
					continue

			# Path is optimal, need to calculate intersection point with polygon P_{index - 1}
			poligon_point = min(
				(s for a, b in self.polygons[index - 1].edges() if (s := segment_segment_intersection(vertex, point, a, b)) is not None), 
				key=vertex.distance_squared_to
			)

			# Path is optimal, but need to check if it stays inside the fence F_{index - 1}
			# while is doesn't reach the polygon P_{index - 1}.
			if any(
				(p := segment_segment_intersection(vertex, point, a, b)) is not None and 
				p.distance_to(vertex) < poligon_point.distance_to(vertex)
				for a, b in self.fences[index - 1].far_edges(vertex)
			):
				continue

			# We only need to check if the path leaves the fence F_{index} now
			# Negative eps avoids matching the vertex itself, if point if a vertex of the new fence.
			if any(
				segment_segment_intersection(poligon_point, point, a, b, -1e-10) is not None 
				for a, b in self.fences[index].edges()
			):
				continue

			return vertex

		# Path can't be directly reached, so we need to find the last reflex vertex of the fence F_{index}
		# TODO: Implement this.

		return Vector2(0, 0) # TODO: implement

	def shortest_path(self) -> list[Vector2]:

		index = 0

		polygon = self.polygons[index]
		cones = self.cones[index]
		blocked = self.blocked[index]
		fence = self.fences[index]

		last_vertex: list[Vector2] = []

		last_fence_vertex: list[Vector2] = []
		self.last_fence_vertex[index] = last_fence_vertex

		for vertex in fence.reflex_vertices:
			last = self.query(index, vertex)
			last_fence_vertex.append(last)
		
		for j in range(len(polygon)):
			
			vertex = polygon[j]
			last = self.query(index, vertex)

			last_vertex.append(last)

			blocked.append(polygon.intersects_segment(last, vertex))

		for j in range(len(polygon)):

			if blocked[j]:
				cones.append((Vector2(), Vector2()))
				continue

			vertex = polygon[j]
			last = last_vertex[j]

			diff = vertex - last

			v_before = polygon[j - 1]
			v_after = polygon[j + 1]

			if blocked[j - 1]:
				dir1 = diff.normalize()
			else:
				dir1 = diff.reflect((vertex - v_before).perpendicular()).normalize()

			if blocked[(j + 1) % len(polygon)]:
				dir2 = diff.normalize()
			else:
				dir2 = diff.reflect((vertex - v_after).perpendicular()).normalize()

			cones.append((dir1, dir2))

		return []

start = Vector2(0, 1)
end = Vector2(0, -3.5)

polygons = [
	Polygon2([
		Vector2(-2, -1), Vector2(2, -1), Vector2(0, -2)
	])
]

fences = [
	Polygon2([
		Vector2(-4, 2), Vector2(-4, 1), Vector2(-0.5, 0), Vector2(-4, 0), Vector2(-4, -3),
		Vector2(4, -3), Vector2(4, 0), Vector2(0.5, 0), Vector2(4, 1), Vector2(4, 2),
	]),
	Polygon2([
		Vector2(-3, -1.5), Vector2(-3, -4), Vector2(3, -4), Vector2(3, -1.5) 
	])
]

#solution = Solution(start, end, polygons, fences)
#path = solution.shortest_path()
