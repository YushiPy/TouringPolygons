
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from vector2 import Vector2
from polygon2 import Polygon2


type _Vector2 = Iterable[float]
type _Polygon2 = Iterable[_Vector2]

class InputError(ValueError):
	pass

class RuntimeError(Exception):
	pass

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

	fig: Figure
	ax: Axes

	def __init__(self, start: _Vector2, target: _Vector2, polygons: Iterable[_Polygon2], fences: Iterable[_Polygon2]) -> None:

		self.start = Vector2(*start)
		self.target = Vector2(*target)

		self.polygons = list(map(Polygon2, polygons))
		self.fences = list(map(Polygon2, fences))

		self.cones = [[] for _ in self.polygons]
		self.blocked = [[] for _ in self.polygons]

		self.mapping = {}

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

	def respects_fences(self, start: Vector2, end: Vector2, start_index: int, end_index: int, eps: float = 1e-12) -> bool:
		"""
		Check if the line segment from `start` to `end` respects all fences from `start_index` to `end_index`.
		"""

		if start_index > end_index:
			raise RuntimeError(f"{start_index=} must be less than or equal to {end_index=}.")

		for i in range(start_index, end_index):
	
			fence = self.fences[i]
			polygon = self.polygons[i]

		return True

	def fenced_path(self, start: Vector2, end: Vector2, start_index: int, end_index: int) -> list[Vector2]:
		"""
		Computes the shortest path from `start` to `end` that respects all fences from `start_index` to `end_index`.
		"""

		if start_index > end_index:
			raise RuntimeError(f"{start_index=} must be less than or equal to {end_index=}.")

		pass

	def query(self, point: Vector2, index: int, end_index: int) -> list[Vector2]:
		"""
		Computes the shortest `index`-path to `point` that respects all fences up to `end_index`.
		"""

		return []

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

	def solve(self) -> None:
		pass
		# self.compute_blocked(0)
		# print(self.blocked)

test1 = (
	(-6.247, 6.247), 
	(-1.599, -6.447), 
	[
		[(-2.499, -0.0), (1.249, -1.249), (4.998, -0.0)], 
		[(-7.497, -4.998), (-6.247, -3.748), (-4.998, -3.748), (-6.247, -4.998)]
	], 
	[
		[(-6.247, 6.247), (-6.247, 1.249), (-1.249, 1.249), (-6.247, -0.0), (-6.247, -3.0), (8.746, -3.0), (8.746, -0.0), (2.499, 1.249), (8.746, 1.249), (8.746, 4.998), (2.499, 3.748), (6.247, 6.247)], 
		[(-8.746, 3.748), (-7.497, 2.499), (-4.998, 7.497), (-4.998, 2.499), (-3.748, 9.996), (0.0, 8.746), (-1.249, 4.998), (2.499, 7.497), (3.748, 9.996), (3.748, 4.998), (6.247, 8.746), (8.746, 7.497), (6.247, 4.998), (9.996, 2.499), (2.499, 2.499), (9.996, 1.249), (6.247, -1.249), (0.0, -2.499), (9.996, -1.249), (9.996, -3.748), (1.249, -3.748), (8.746, -6.247), (-8.746, -7.497), (-9.996, -4.998), (-3.748, -2.499), (-6.247, -6.247), (-1.249, -2.499), (-9.996, 1.249)], 
		[(-9.079, -10.088), (-7.062, -7.062), (-5.044, -11.097), (1.249, -3.748), (-5.044, -9.079), (-4.035, -2.018), (-7.062, -3.026), (-9.079, -7.062)]
	]
)

sol = Solution(*test1)
sol.basic_draw(False)
sol.solve()


