
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


class Solution:

	start: Vector2
	end: Vector2

	polygons: list[Polygon2]
	fences: list[Polygon2]

	# For each vertex j in the polygon P_i, we store the last vertex of the 
	# i-path that reaches j.
	vertex_last: list[list[Vector2]]

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

		if not all(polygon.is_convex() for polygon in self.polygons):
			raise ValueError("All polygons must be convex.")
	
	def shortest_fenced_path(self, start: Vector2, end: Vector2, index: int) -> Vector2:
		return shortest_path_in_polygon(start, end, self.fences[index])[-1]

	def query(self, index: int, end: Vector2) -> Vector2:
		"""
		Returns the start of the last segment of the shortest 
		`index`-path that reaches `end`.
		"""

		if index == 0:
			return self.shortest_fenced_path(self.start, end, 0)
	
	def shortest_path(self) -> list[Vector2]:



		return []

start = Vector2(0, 3)
end = Vector2(0, -3)

polygons = [
	Polygon2([
		Vector2(-2, 0), Vector2(2, 0), Vector2(0, 1)
	])
]

fences = [
	Polygon2([
		Vector2(-4, 4), Vector2(-4, 1), Vector2(-1, 1), Vector2(-4, 0), Vector2(-4, -4),
		Vector2(4, -4), Vector2(4, 0), Vector2(1, 1), Vector2(4, 1), Vector2(4, 4),
	]),
	Polygon2([
		Vector2(-3, 2), Vector2(-3, -4), Vector2(3, -4), Vector2(3, 2) 
	])
]

solution = Solution(start, end, polygons, fences)
path = solution.shortest_path()
