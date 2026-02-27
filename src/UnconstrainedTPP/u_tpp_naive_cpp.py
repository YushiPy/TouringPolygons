
from export_tests import export_test_cases, read_test_results

from collections.abc import Sequence
import subprocess

PATH = "./C++/tests"


def tpp_solve(start: tuple[float, float], target: tuple[float, float], polygons: Sequence[Sequence[tuple[float, float]]]) -> list[tuple[float, float]]:
	"""
	Returns the shortest path from `start` to `target` that visits all polygons in order.
	"""
	
	export_test_cases([(start, target, polygons)], "/tmp/test_cases.bin")

	subprocess.run([PATH, "/tmp/test_cases.bin", "/tmp/solution.bin"], check=True)

	result = read_test_results("/tmp/solution.bin")

	return [(x, y) for x, y in result[0][1]]
