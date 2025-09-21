
import heapq
from itertools import accumulate, chain
from typing import Callable, Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


from vector2 import Vector2
from polygon2 import Polygon2


type _Vector2 = Iterable[float]
type _Polygon2 = Iterable[_Vector2]


def debug_func(skip_count: int = 0) -> None:
	"""
	Prints the name and arguments of the caller function.
	"""
	import inspect

	# Get the caller's frame
	frame = inspect.currentframe().f_back # type: ignore
	func_name = frame.f_code.co_name # type: ignore
	
	# Get local variables in the caller (arguments are part of locals)
	args, _, _, values = inspect.getargvalues(frame) # type: ignore
	args = args[skip_count:]

	# Format the arguments nicely
	arg_str = ", ".join(f"{arg}={values[arg]!r}" for arg in args)
	
	print(f"{func_name}({arg_str})")


class InputError(ValueError):
	pass

class RuntimeError(Exception):
	pass

class Graph:

	vertices: list[Vector2]
	edges: list[list[int]]

	def __init__(self, vertices: Iterable[Vector2] = [], edges: Iterable[Iterable[int]] = []) -> None:
		self.vertices = list(vertices)
		self.edges = list(map(list, edges))

	def astar(self, start: int = 0, end: int = 1) -> list[Vector2]:

		n = len(self.vertices)
		v_end = self.vertices[end]

		previous = [-1] * n
		costs = [float('inf')] * n
		costs[start] = 0

		# The priority queue stores tuples of (estimated total cost, vertex index).
		queue: list[tuple[float, int]] = [(-1, start)] # The -1 doesn't matter, will become _.

		while queue:
			
			_, current = queue.pop(0)

			if current == end:
				break


			for neighbor in self.edges[current]:

				v1 = self.vertices[current]
				v2 = self.vertices[neighbor]
				new_cost = costs[current] + v1.distance_to(v2)

				if new_cost < costs[neighbor]:
					costs[neighbor] = new_cost
					previous[neighbor] = current
					heapq.heappush(queue, (new_cost + v2.distance_to(v_end), neighbor))

		if previous[end] == -1:
			return []
		
		path: list[Vector2] = []
		current = end

		while current != -1:
			path.append(self.vertices[current])
			current = previous[current]
		
		path.reverse()

		return path

# This is the same as functools.cache, but it does not remove the type hints.
def cache[T, **P](func: Callable[P, T]) -> Callable[P, T]:

	from functools import lru_cache

	return lru_cache(maxsize=None)(func) # type: ignore

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

def bend_is_optimal(start: Vector2, end: Vector2, vertex: Vector2, before: Vector2, after: Vector2) -> bool:
	"""
	Returns whether a bend at `vertex` that 
	comes from `before` and goes to `after` is 
	optimal on a path from `start` to `end`.
	"""

	if point_in_cone(end, vertex, before - vertex, after - vertex):
		return False

	rates1 = intersection_rates(start, end - start, vertex, before - vertex)
	rates2 = intersection_rates(start, end - start, vertex, after - vertex)

	if rates1 is not None and 0 <= rates1[0] <= 1 and rates1[1] >= 0:
		return True
	
	if rates2 is not None and 0 <= rates2[0] <= 1 and rates2[1] >= 0:
		return True

	return False


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
	reflex_mapping: dict[tuple[int, int], list[tuple[int, int]]]

	fig: Figure
	ax: Axes

	def __init__(self, start: _Vector2, target: _Vector2, polygons: Iterable[_Polygon2], fences: Iterable[_Polygon2]) -> None:

		self.start = Vector2(*start)
		self.target = Vector2(*target)

		self.polygons = list(map(Polygon2, polygons))
		self.fences = list(map(Polygon2, fences))

		self.cones = [[] for _ in self.polygons]
		self.blocked = [[] for _ in self.polygons]

		if len(self.fences) != len(self.polygons) + 1:
			raise InputError("Number of fences must be one more than number of polygons.")
		
		if not self.fences[0].contains_point(self.start) >= 0:
			raise InputError("Start point must be inside the first fence.")
	
		if not self.fences[-1].contains_point(self.target) >= 0:
			raise InputError("Target point must be inside the last fence.")
	
		if any(not p.is_convex() for p in self.polygons):
			raise InputError("All polygons must be convex.")

		polys = [[self.start]] + self.polygons + [[self.target]]

		for i in range(len(self.polygons)):

			if not self.fences[i].contains_polygon(polys[i]) >= 0:
				raise InputError(f"Polygon {i} must be inside fence {i}.")
			
			if not self.fences[i].contains_polygon(polys[i + 1]) >= 0:
				raise InputError(f"Polygon {i + 1} must be inside fence {i}.")

		self.fig, self.ax = plt.subplots() # type: ignore

	def basic_draw(self, show: bool = True) -> None:
		
		self.ax.set_aspect('equal', 'box')

		self.ax.plot(self.start.x, self.start.y, 'go') # type: ignore
		self.ax.plot(self.target.x, self.target.y, 'ro') # type: ignore

		for p in self.polygons:
			xs, ys = zip(*[(v.x, v.y) for v in p] + [(p[0].x, p[0].y)])
			self.ax.fill(xs, ys) # type: ignore

		for f in self.fences:
			xs, ys = zip(*[(v.x, v.y) for v in f] + [(f[0].x, f[0].y)])
			self.ax.plot(xs, ys, alpha=0.7) # type: ignore

		self.ax.grid() # type: ignore

		if show:
			plt.show() # type: ignore

	def basic_cones(self, index: int) -> None:

		options = ["red", "purple", "blue", "orange", "green", "purple", "brown", "pink", "gray", "olive", "cyan"]
		color = options[index % len(options)]

		for v, (ray1, ray2) in zip(self.polygons[index], self.cones[index]):

			if ray1 == ray2:
				continue

			ray1 = ray1.scale_to_length(7)
			ray2 = ray2.scale_to_length(7)

			self.ax.plot([v.x, v.x + ray1.x], [v.y, v.y + ray1.y], '-', color=color, alpha=0.7) # type: ignore
			self.ax.plot([v.x, v.x + ray2.x], [v.y, v.y + ray2.y], '-', color=color, alpha=0.7) # type: ignore

	def make_mapping(self) -> None:

		self.reflex_mapping = {}

		for i, fence in enumerate(self.fences):
			for j, v in fence.reflex_vertices_pairs:

				reachable: list[tuple[int, int]] = []

				for k in range(i, len(self.fences)):

					other_fence = self.fences[k]

					for l, u in other_fence.reflex_vertices_pairs:
						if self.respects_fences(v, u, i, k):
							reachable.append((k, l))

				self.reflex_mapping[(i, j)] = reachable


	def valid_last_step(self, start: Vector2, end: Vector2, start_index: int, end_index: int) -> bool:
		"""
		Checks if a path from `start` to `end` is valid.
		This means a standard valid path or a path that 
		goes outside the last fence only once and intersects
		the polygon from the previous fence.

		For example, a path from a start point inside fence 0
		to a target point that respects all fences from 0 to 1.
		The path will be valid if it respects fence 0 and
		intersects polygon 1 and then goes outside fence 1 to reach end.
		"""

		if self.respects_fences(start, end, start_index, end_index):
			return True

		# If there are previous fences, it must respect them.
		if 0 < start_index < end_index and not self.respects_fences(start, end, start_index, end_index - 1):
			return False

		fence = self.fences[end_index]
		
		# Standard case
		if fence.contains_segment(start, end):
			return True

		if fence.contains_point(end) >= 0:
			return False

		# This query can't be from an edge query, because
		# edge querys reduce the problem to the previous polygon.
		if end_index == len(self.polygons):
			return False

		# Check if the path goes outside the last fence only once
		if sum(segment_segment_intersection(a, b, start, end) is not None for a, b in fence.far_edges(start, end)) > 1:
			return False

		polygon = self.polygons[end_index]
		intersection = polygon.segment_intersection(start, end)
		
		return intersection is not None

	def make_graph(self, start: Vector2, end: Vector2, start_index: int, end_index: int) -> Graph:

		@cache
		def vertex_index(fence_index: int, vertex_index: int) -> int:
			return accum_reflex_counts[fence_index] + self.fences[fence_index].reflex_vertices_indices.index(vertex_index)

		vertices = [start, end]

		for fence in self.fences:
			vertices.extend(fence.reflex_vertices)

		reflex_counts = [len(fence.reflex_vertices) for fence in self.fences]
		accum_reflex_counts = list(accumulate(reflex_counts, initial=2)) # +2 for start and target

		total_vertices = accum_reflex_counts[-1]
		edges: list[list[int]] = [[] for _ in range(total_vertices)]

		for fence_index in range(start_index, end_index + 1):

			fence = self.fences[fence_index]

			for j, v in fence.reflex_vertices_pairs:

				from_index = vertex_index(fence_index, j)

				for k, l in self.reflex_mapping[(fence_index, j)]:

					if not (start_index <= k <= end_index):
						continue

					to_index = vertex_index(k, l)

					if from_index == to_index:
						continue

					edges[from_index].append(to_index)

		for fence_index in range(start_index, end_index + 1):

			fence = self.fences[fence_index]

			for j, v in fence.reflex_vertices_pairs:

				index = vertex_index(fence_index, j)

				if self.respects_fences(start, v, start_index, fence_index):
					edges[0].append(index)

				if self.valid_last_step(v, end, fence_index, end_index):
					edges[index].append(1)

		if self.valid_last_step(start, end, start_index, end_index):
			edges[0].append(1)

		return Graph(vertices, edges)

	def respects_fences(self, start: Vector2, end: Vector2, start_index: int, end_index: int, eps: float = 1e-12) -> bool:
		"""
		Check if the line segment from `start` to `end` respects all fences from `start_index` to `end_index`.
		"""

		if start_index > end_index:
			raise RuntimeError(f"{start_index=} must be less than or equal to {end_index=}.")

		for i in range(start_index, end_index):
	
			fence = self.fences[i]
			polygon = self.polygons[i]

			intersection = polygon.segment_intersection(start, end, eps)

			# Path does not reach the polygon.
			if intersection is None:
				return False
			
			# Path goes outside the fence.
			if not fence.contains_segment(start, intersection, eps):
				return False

			# Update start with least distant point that touches polygon
			start = intersection

		return self.fences[end_index].contains_segment(start, end, eps)

	def fenced_path(self, start: Vector2, end: Vector2, start_index: int, end_index: int) -> list[Vector2]:
		"""
		Computes the shortest path from `start` to `end` that respects all fences from `start_index` to `end_index`.
		"""

		if start_index > end_index:
			raise RuntimeError(f"{start_index=} must be less than or equal to {end_index=}.")

		if self.respects_fences(start, end, start_index, end_index):
			return [start, end]
		
		graph = self.make_graph(start, end, start_index, end_index)
		path = graph.astar(0, 1)

		return path

	def point_in_cone(self, point: Vector2, index: int, vertex_index: int) -> bool:
		"""
		Returns whether the `point` is inside the cone of the vertex `vertex_index` of polygon `index`.
		"""

		ray1, ray2 = self.cones[index][vertex_index]
		vertex = self.polygons[index][vertex_index]

		if ray1 == ray2:
			return False

		return point_in_cone(point, vertex, ray1, ray2)

	def point_in_edge(self, point: Vector2, index: int, edge_index: int) -> bool:
		"""
		Returns whether the `point` is inside the edge defined by the edge `edge_index` of polygon `index`.
		"""

		if self.blocked[index][edge_index]:
			return False

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

	def query_cone(self, point: Vector2, index: int, end_index: int) -> list[Vector2]:
		"""
		Checks if the point is inside any cone of polygon `index`.
		Returns the path to the point if it is, otherwise returns an empty list.
		"""
		for i, v in enumerate(self.polygons[index - 1]):

			if not self.point_in_cone(point, index - 1, i):
				continue

			if not self.valid_last_step(v, point, index - 1, end_index):
				continue
			#if not self.fences[index].contains_segment(v, point):
			#	continue

			path = self.query(v, index - 1, index - 1)

			# TODO: Check if we can just return path + [point] here,
			# instead of calling fenced_path again.
			return path[:-1] + self.fenced_path(path[-1], point, index - 1, end_index)

		return []
	
	def query_edge(self, point: Vector2, index: int, end_index: int) -> list[Vector2]:
		"""
		Checks if the point is inside any edge of polygon `index`.
		Returns the path to the point if it is, otherwise returns an empty list.
		"""

		polygon = self.polygons[index - 1]
		blocked = self.blocked[index - 1]

		for i in range(len(polygon)):

			if blocked[i]:
				continue

			if not self.point_in_edge(point, index - 1, i):
				continue

			v1 = polygon[i]
			v2 = polygon[(i + 1) % len(polygon)]

			reflected = point.reflect_segment(v1, v2)
			
			path = self.query(reflected, index - 1, index - 1)

			if len(path) < 2:
				raise RuntimeError(f"query_edge({point}, {index}, {end_index}) -> path={path} must have at least two points.")

			last = path[-2]
			intersection = polygon.segment_intersection(last, reflected)

			if intersection is None:
				raise RuntimeError(f"query_edge({point}, {index}, {end_index}) -> last={tuple(round(last, 3))} to reflected={tuple(round(reflected, 3))} does not intersect polygon {index - 1}.")
			
			# TODO: Check if we can just return path + [point] here,
			# instead of calling fenced_path again.
			return path[:-1] + self.fenced_path(intersection, point, index - 1, end_index)

		return []

	def query_pass_through(self, point: Vector2, index: int, end_index: int) -> list[Vector2]:
		"""
		Checks if the point is inside the pass through region of polygon `index`.
		Returns the path to the point if it is, otherwise returns an empty list.
		"""

		if not self.point_in_pass_through(point, index - 1):
			return []

		# This may not work if the path would be blocked by a fence.
		# This will cause the query to fail and return an empty list.
		# However, this will just pass the problem to the
		# reflex vertex queries, which will handle it correctly.
		return self.query(point, index - 1, end_index)

	def query_reflex_vertex(self, point: Vector2, index: int, end_index: int) -> list[Vector2]:
		"""
		Checks if the point is reachable from a reflex vertex of polygon `index`.
		Returns the path to the point if it is, otherwise returns an empty list.
		"""

		def rotate[T](values: Sequence[T], start: int) -> Iterator[T]:
			return map(values.__getitem__, chain(range(start, len(values)), range(0, start)))

		fence = self.fences[index]

		if not fence.reflex_vertices:
			return []

		start = next((i for i, v in enumerate(fence.reflex_vertices) if point.is_close(v)), - 1)
		start = (start + 1) % len(fence.reflex_vertices)

		for v_index in rotate(fence.reflex_vertices_indices, start):
			
			vertex = fence[v_index]

			# The path from the vertex to the point must be direct.
			if not self.valid_last_step(vertex, point, index, end_index):
				continue

			path = self.query(vertex, index, index)

			if len(path) < 2:
				raise RuntimeError(f"query_reflex_vertex({point}, {index}, {end_index}) -> path={path} must have at least two points.")

			before = fence[v_index - 1]
			after = fence[(v_index + 1) % len(fence)]
			last = path[-2]

			# The bend at the vertex must be optimal.
			if not bend_is_optimal(last, point, vertex, before, after):
				continue
			
			# TODO: Check if we can just return path + [point] here,
			# instead of calling fenced_path again.
			return path[:-1] + self.fenced_path(vertex, point, index, end_index)

		return []

	def query(self, point: Vector2, index: int, end_index: int) -> list[Vector2]:
		"""
		Computes the shortest `index`-path to `point` that respects all fences up to `end_index`.
		"""

		# Base case, just go straight to the point, 
		# while respecting all fences.
		if index == 0:
			return self.fenced_path(self.start, point, 0, end_index)
		
		if (path := self.query_cone(point, index, end_index)):
			return path

		if (path := self.query_edge(point, index, end_index)):
			return path

		if (path := self.query_pass_through(point, index, end_index)):
			return path

		# All cases failed, there is no direct path to the point.
		# We must go through a reflex vertex.

		if (path := self.query_reflex_vertex(point, index, end_index)):
			return path

		raise RuntimeError(f"query({point}, {index}, {end_index}) -> No path found.")

	def query2(self, point: Vector2, index: int, end_index: int) -> Vector2:
		"""
		Same as `query`, but only returns the last point before `point`.
		"""

		path = self.query(point, index, end_index)

		if len(path) < 2:
			raise RuntimeError(f"{path=} must have at least two points.")

		return path[-2]

	def compute_blocked(self, index: int) -> None:

		polygon = self.polygons[index]
		block: list[bool] = []

		for a, b in polygon.edges():

			mid = (a + b) / 2
			last = self.query2(mid, index, index)

			blocked = polygon.intersects_segment(last, mid)
			block.append(blocked)

		self.blocked[index] = block

	def compute_cones(self, index: int) -> None:

		polygon = self.polygons[index]
		blocked = self.blocked[index]
		cones: list[tuple[Vector2, Vector2]] = []

		for i, v in enumerate(polygon):

			last = self.query2(v, index, index)
			diff = (v - last).normalize()

			before = polygon[i - 1]
			after = polygon[(i + 1) % len(polygon)]

			dir1 = diff.reflect((before - v).perpendicular())
			dir2 = diff.reflect((after - v).perpendicular())

			if blocked[i - 1]:
				dir1 = diff

			if blocked[i]:
				dir2 = diff

			if dir1 == dir2:
				cones.append((Vector2(), Vector2()))
			else:
				cones.append((dir1, dir2))

		self.cones[index] = cones

	def solve(self) -> list[Vector2]:

		self.make_mapping()

		n = len(self.polygons)

		for i in range(n):
			self.compute_blocked(i)
			self.compute_cones(i)
		
		path = self.query(self.target, n, n)

		if len(path) < 2:
			raise RuntimeError(f"{path=} must have at least two points.")
	
		return path
