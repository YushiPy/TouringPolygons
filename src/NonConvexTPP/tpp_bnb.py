
from collections import deque
import itertools
import math

from polygon2 import Polygon2
from polygon_decomposition import hertel_mehlhorn
import u_tpp

from collections.abc import Sequence

def path_length(path: Sequence[tuple[float, float]]) -> float:
	return sum(math.dist(path[i + 1], path[i]) for i in range(len(path) - 1))

def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[tuple[float, float]]:

	def bound(instance: Sequence[int]) -> float:

		selected_polygons = [convex_pieces[i][instance[i]] for i in range(len(instance))]
		input_polygons = selected_polygons + [convex_hulls[i] for i in range(len(instance), len(polygons))]

		path = u_tpp.tpp_solve(start, target, input_polygons, simplify=False)

		return path_length(path)

	if simplify:
		polygons = [u_tpp.clean_polygon(poly) for poly in polygons]

	convex_hulls = [Polygon2(poly).convex_hull().to_list() for poly in polygons]
	convex_pieces = [hertel_mehlhorn(poly) for poly in polygons]

	minimal_path = u_tpp.tpp_solve(start, target, convex_hulls, simplify=False)
	minimal_length = path_length(minimal_path)

	# Each element of the queue is a list containing the index of the convex piece chosen for each polygon, in order. 
	# We start the queue as a list containing just the empty list, which represents 
	# a combination where no convex pieces have been chosen yet. 
	queue = deque([[]])

	while queue:

		current = queue.popleft()

		if len(current) == len(polygons):

			# We have a complete combination of convex pieces, so we can solve the TPP for this combination.

			instance = [convex_pieces[i][current[i]] for i in range(len(polygons))]

			try:
				path = u_tpp.tpp_solve(start, target, instance, simplify=False)
			except ValueError:
				print("Failed to solve TPP for input:\n", (start, target, instance), flush=True)
				continue

			length = path_length(path)

			if length < minimal_length:
				minimal_length = length
				minimal_path = path
		else:
			
			# We have a partial combination, so we need to add the next convex piece to the queue.

			next_polygon_index = len(current)

			for i in range(len(convex_pieces[next_polygon_index])):
				
				selected = current + [i]

				if bound(selected) > minimal_length:
					continue

				queue.append(selected)

	if minimal_path is None:
		raise ValueError("No path found")

	return minimal_path
