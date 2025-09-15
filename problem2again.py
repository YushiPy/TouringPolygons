
from typing import Callable, Iterable

import heapq

from vector2 import Vector2
from polygon2 import Polygon2


type _Vector2 = Iterable[float]
type _Polygon2 = Iterable[_Vector2]


def astar(start: int, target: int, edges: list[list[tuple[int, float]]], heuristic: Callable[[int, int], float]) -> list[int]:
	"""
	Perform A* search to find the shortest path from `start` to `target`.

	:param start: The starting node index.
	:param target: The target node index.
	:param edges: A list where each element is a list of tuples representing the neighboring nodes and the cost to reach them.
	:param heuristic: A function that takes two node indices and returns an estimated cost from the first node to the second.

	:return: A list of node indices representing the shortest path from `start` to `target`. If no path is found, returns an empty list.
	"""

	n = len(edges)
	previous = [-1] * n
	costs = [float("inf")] * n
	costs[start] = 0

	# min-heap of (estimated total cost, node index)
	queue = [(heuristic(start, target), start)]

	while queue:

		_, current = heapq.heappop(queue)

		if current == target:

			path: list[int] = []

			while current != -1:
				path.append(current)
				current = previous[current]

			return path[::-1]

		for neighbor, edge_cost in edges[current]:

			new_cost = costs[current] + edge_cost

			if new_cost < costs[neighbor]:
				costs[neighbor] = new_cost
				previous[neighbor] = current
				priority = new_cost + heuristic(neighbor, target)
				heapq.heappush(queue, (priority, neighbor))

	return []


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
	target: Vector2

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

	# A mapping from (fence_index, reflex_vertex_index) to (fence_index, reflex_vertex_index)
	# This mapping tells us which vertices can be reached from which other vertices.
	# Note that the mapping is only for reflex vertices.
	mapping: dict[tuple[int, int], list[tuple[int, int]]]

	def __init__(self, start: _Vector2, target: _Vector2, polygons: Iterable[_Polygon2], fences: Iterable[_Polygon2]) -> None:
		
		self.start = Vector2(start)
		self.target = Vector2(target)
		self.polygons = list(map(Polygon2, polygons))
		self.fences = list(map(Polygon2, fences))

		if len(self.fences) != len(self.polygons) + 1:
			raise ValueError("Number of fences must be exactly one more than number of polygons")
		
		self.cones = [[] for _ in range(len(self.polygons))]
		self.blocked = [[] for _ in range(len(self.polygons))]

		self.make_mapping()

	def respects_fences(self, start: Vector2, target: Vector2, start_index: int, end_index: int) -> bool:
		"""
		Returns whether the segment from `start` to `target` respects the fences from `start_index` to `end_index`.
		"""

		# Check whether the segment respects all fences from start_index to end_index - 1.
		# That is, it must intersect all polygons from start_index to end_index - 1.
		# Finally, it must not intersect the fence at end_index.
		for i in range(start_index, end_index):

			# We must first determine whether the segment intersects the current polygon.
			# If it does not, then it does not respect the next fence, as it never reaches the current polygon.
			polygon = self.polygons[i]
			intersection = polygon.segment_intersection(start, target)

			# Q: is this correct?
			# A: I think so...
			if intersection is None:
				return False

			# We need to check if the segment from start to intersection intersects the fence.
			# If it does, then the segment does not respect the fences.
			if self.fences[i].intersects_segment(start, intersection):
				return False

			start = intersection

		# Finally, we need to check if the segment intersects the fence at end_index.
		# If it does, then the segment does not respect the fences.
		return not self.fences[end_index].intersects_segment(start, target)

	def make_mapping(self) -> None:

		self.mapping = {}

		# For each reflex vertex in each polygon, check which reflex vertices in other polygons
		# can be reached from it.

		for i in range(len(self.fences)):
			for j in range(i, len(self.fences)):
				print(i, j)
				fence1 = self.fences[i]
				fence2 = self.fences[j]

				for vi, v1 in fence1.reflex_vertices_pairs:
					for vj, v2 in fence2.reflex_vertices_pairs:

						# Skip if they are the same vertex in the same polygon
						# this is a valid path, but we don't want to store it in the mapping.
						if i == j and vi == vj:
							continue

						if not self.respects_fences(v1, v2, i, j):
							continue

						print(f"Mapping ({i}, {vi}) to ({j}, {vj}); {v1} -> {v2}")
						
						if (i, vi) not in self.mapping:
							self.mapping[(i, vi)] = []
						
						self.mapping[(i, vi)].append((j, vj))


	def _path_in_fence0(self, start: Vector2, target: Vector2) -> list[Vector2]:
		"""
		Returns the smallest path from `start` to `target` that is inside the first fence (index 0).
		"""

		fence = self.fences[0]
		vertices = [start] + [target] + fence.reflex_vertices
		edges: list[list[tuple[int, float]]] = [[] for _ in range(len(vertices))]

		for i in range(len(vertices)):
			for j in range(i + 1, len(vertices)):

				edge = (vertices[i], vertices[j])

				if not fence.intersects_segment(edge[0], edge[1]):
					edges[i].append((j, edge[0].distance_to(edge[1])))
					edges[j].append((i, edge[0].distance_to(edge[1])))
		
		path_indices = astar(0, 1, edges, lambda a, b: vertices[a].distance_to(vertices[b]))
		path = [vertices[i] for i in path_indices]

		return path

	def path_in_fence(self, start: Vector2, target: Vector2, index: int) -> list[Vector2]:
		"""
		Returns the smallest path from `start` to `target` that is inside the fence `index` and `index + 1`.
		"""

		if index == 0:
			return self._path_in_fence0(start, target)

		return []

	def fenced_path(self, start: Vector2, target: Vector2, start_index: int, end_index: int) -> list[Vector2]:
		"""
		Returns the smallest path from `start` to `target` that is inside all fences from `start_index` to `end_index`.
		"""

		if start_index == end_index:
			return self.path_in_fence(start, target, start_index)

		vertices = [start, target]
		edges: list[list[tuple[int, float]]] = []



		return []

	def point_in_cone(self, point: Vector2, index: int, vertex_index: int) -> bool:
		"""
		Returns whether the `point` is inside the cone of the vertex `vertex_index` of polygon `index`.
		"""

		ray1, ray2 = self.cones[index][vertex_index]
		vertex = self.polygons[index][vertex_index]

		return point_in_cone(point, vertex, ray1, ray2)

	def point_in_edge(self, point: Vector2, index: int, edge_index: int) -> bool:
		"""
		Returns whether the `point` is inside the edge defined by the edge `edge_index` of polygon `index`.
		"""

		polygon = self.polygons[index]
		ray1 = self.cones[index][edge_index][1]
		ray2 = self.cones[index][(edge_index + 1) % len(polygon)][0]

		return point_in_edge(point, polygon[edge_index], ray1, polygon[(edge_index + 1) % len(polygon)], ray2)

	def point_in_pass_through(self, point: Vector2, index: int) -> bool:
		"""
		Returns whether the `point` is inside the pass through region of polygon `index`.
		"""

		polygon = self.polygons[index]
		blocked = self.blocked[index]
		cones = self.cones[index]

		first = next(i for i in range(len(blocked)) if blocked[i] and not blocked[i - 1])
		last = next(i for i in range(len(blocked)) if blocked[i] and not blocked[(i + 1) % len(blocked)])

		ray1 = cones[first][1]
		ray2 = cones[(last + 1) % len(blocked)][0]

		vertex1 = polygon[first]
		vertex2 = polygon[(last + 1) % len(blocked)]

		# Check if the point is inside the big edge
		if point_in_edge(point, vertex1, ray1, vertex2, ray2):
			return True

		# Check if the point is inside the polygon
		return polygon.contains_point(point) >= 0

	def query(self, point: Vector2, index: int) -> Vector2:
		"""
		Returns the last point on a `index`-path to `point`.
		"""

		if index == 0:
			return self.path_in_fence(self.start, point, 0)[-2]

		polygon = self.polygons[index - 1]

		# Check for cone region
		for i in range(len(polygon)):
			if self.point_in_cone(point, index - 1, i):
				return polygon[i]
		
		# Check for edge region
		for i in range(len(polygon)):

			if self.blocked[index - 1][i] or not self.point_in_edge(point, index - 1, i):
				continue

			raise NotImplementedError("Reflecting on edges is not implemented yet")

		if not self.point_in_pass_through(point, index - 1):
			raise ValueError("WTF! The point is not in any region?!?")

		return point.inf()

	def solve0(self) -> None:

		polygon = self.polygons[0]

		# Compute the blocked edges for polygon 0 by checking if the
		# last segment from start to the midpoint of each edge intersects the polygon.		
		for a, b in polygon.edges():

			mid = (a + b) / 2
			last = self.query(a, 0)
			
			self.blocked[0].append(polygon.intersects_segment(last, mid))

		# Compute the cones for polygon 0
		for i in range(len(polygon)):

			v = polygon[i]
			last = self.query(v, 0)
			diff = (v - last).normalize()

			before = polygon[i - 1]
			after = polygon[(i + 1) % len(polygon)]

			dir1 = diff.normalize()
			dir2 = diff.normalize()

			if not self.blocked[0][i - 1]:
				dir1 = dir1.reflect((before - v).perpendicular())


			if not self.blocked[0][i]:
				dir2 = dir2.reflect((after - v).perpendicular())

			self.cones[0].append((dir1, dir2))

	def solve(self) -> list[Vector2]:

		self.solve0()

		print(self.query(self.target, 1))

		return []

test1 = (
	(0.0, 4.009), 
	(-2.499, -6.247), 
	[
		[(-2.499, -0.0), (1.249, -1.249), (4.998, -0.0)], 
		[(-7.497, -4.998), (-6.247, -3.748), (-4.998, -3.748), (-6.247, -4.998)]
	], 
	[
		[(-6.247, 6.247), (-6.247, 1.249), (-1.249, 1.249), (-6.247, -0.0), (-6.247, -4.998), (8.746, -4.998), (8.746, -0.0), (2.499, 1.249), (8.746, 1.249), (8.746, 4.998), (2.499, 3.748), (6.247, 6.247)], 
		[(-8.746, 3.748), (-7.497, 2.499), (-4.998, 7.497), (-4.998, 2.499), (-3.748, 9.996), (0.0, 8.746), (-1.249, 4.998), (2.499, 7.497), (3.748, 9.996), (3.748, 4.998), (6.247, 8.746), (8.746, 7.497), (6.247, 4.998), (9.996, 2.499), (2.499, 2.499), (9.996, 1.249), (6.247, -1.249), (0.0, -2.499), (9.996, -1.249), (9.996, -3.748), (1.249, -3.748), (8.746, -6.247), (-8.746, -7.497), (-9.996, -4.998), (-3.748, -2.499), (-6.247, -6.247), (-1.249, -2.499), (-9.996, 1.249)], 
		[(-9.079, -10.088), (-7.062, -7.062), (-5.044, -11.097), (1.249, -3.748), (-5.044, -9.079), (-4.035, -2.018), (-7.062, -3.026), (-9.079, -7.062)]
	]
)

s = Solution(*test1)
s.solve()
