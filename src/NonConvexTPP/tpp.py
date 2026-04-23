
import itertools
from time import perf_counter

from polygon_decomposition import hertel_mehlhorn
import u_tpp

from collections.abc import Sequence

def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> list[tuple[float, float]]:

	if simplify:
		polygons = [u_tpp.clean_polygon(poly) for poly in polygons]

	convex_pieces = [hertel_mehlhorn(poly) for poly in polygons]

	minimal_path = None
	minimal_length = float('inf')

	for comb in itertools.product(*convex_pieces):
		# comb is a tuple of convex pieces, one from each polygon

		try:
			path = u_tpp.tpp_solve(start, target, comb, simplify=False)
		except ValueError:
			print("Failed to solve TPP for input:\n", (start, target, comb), flush=True)
			continue

		length = sum(((path[i][0] - path[i-1][0]) ** 2 + (path[i][1] - path[i-1][1]) ** 2) ** 0.5 for i in range(1, len(path)))

		if length < minimal_length:
			minimal_length = length
			minimal_path = path
	
	if minimal_path is None:
		raise ValueError("No path found")

	return minimal_path


def tpp_solve_tracked(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]], *, simplify: bool = False) -> tuple[float, int]:

	start_time = perf_counter()

	if simplify:
		polygons = [u_tpp.clean_polygon(poly) for poly in polygons]

	convex_pieces = [hertel_mehlhorn(poly) for poly in polygons]

	minimal_path = None
	minimal_length = float('inf')

	count = 0

	for comb in itertools.product(*convex_pieces):
		# comb is a tuple of convex pieces, one from each polygon

		try:
			count += 1
			path = u_tpp.tpp_solve(start, target, comb, simplify=False)
		except ValueError:
			print("Failed to solve TPP for input:\n", (start, target, comb), flush=True)
			continue

		length = sum(((path[i][0] - path[i-1][0]) ** 2 + (path[i][1] - path[i-1][1]) ** 2) ** 0.5 for i in range(1, len(path)))

		if length < minimal_length:
			minimal_length = length
			minimal_path = path
	
	if minimal_path is None:
		raise ValueError("No path found")

	end_time = perf_counter()

	return end_time - start_time, count

