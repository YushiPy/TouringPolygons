
import heapq
from math import inf

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
