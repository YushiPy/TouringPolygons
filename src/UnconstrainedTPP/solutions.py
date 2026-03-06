
from collections.abc import Callable, Sequence
import time

# tpp_solve_linear, tpp_solve_binary, tpp_solve_binary2, tpp_solve_dynamic, tpp_solve_linear_jit, tpp_solve_binary_jit, tpp_solve_binary2_jit, tpp_solve_dynamic_jit
import common
import u_tpp
import u_tpp_naive_cpp

type Point = tuple[float, float]
type Points = Sequence[Point]

type TestCase = tuple[Point, Point, Sequence[Points], int]
type TestCases = Sequence[TestCase]
type TestResults = Sequence[tuple[float, Sequence[Point]]]

type Solution = Callable[[TestCases], TestResults]

def wrap_python_solution(solve: Callable[[Point, Point, Sequence[Points]], Sequence[Point]]) -> Callable[[Sequence[tuple[tuple[float, float], tuple[float, float], Sequence[Sequence[tuple[float, float]]], int]]], Sequence[tuple[float, Sequence[tuple[float, float]]]]]:

	def run_test_cases(test_cases: TestCases) -> TestResults:

		results = []

		for start, target, polygons, n in test_cases:
			
			total_time = 0.0
			path = []

			for _ in range(n):

				start_time = time.perf_counter()
				path = solve(start, target, polygons)
				end_time = time.perf_counter()

				total_time += end_time - start_time

			results.append((total_time, path))

		return results

	return run_test_cases

# Solutions used for benchmarking. They return both the path and the time taken to compute it.

timed_tpp_solve_linear = wrap_python_solution(common.tpp_solve_linear)
timed_tpp_solve_binary = wrap_python_solution(common.tpp_solve_binary)
timed_tpp_solve_binary2 = wrap_python_solution(common.tpp_solve_binary2)
timed_tpp_solve_dynamic = wrap_python_solution(common.tpp_solve_dynamic)
timed_tpp_solve_linear_jit = wrap_python_solution(common.tpp_solve_linear_jit)
timed_tpp_solve_binary_jit = wrap_python_solution(common.tpp_solve_binary_jit)
timed_tpp_solve_binary2_jit = wrap_python_solution(common.tpp_solve_binary2_jit)
timed_tpp_solve_dynamic_jit = wrap_python_solution(common.tpp_solve_dynamic_jit)

timed_tpp_solve = wrap_python_solution(u_tpp.tpp_solve)
timed_tpp_solve_naive_cpp = u_tpp_naive_cpp.solve_test_cases

# Solutions used for testing correctness. They return only the path, without the time taken.

tpp_solve_linear = common.tpp_solve_linear
tpp_solve_binary = common.tpp_solve_binary
tpp_solve_binary2 = common.tpp_solve_binary2
tpp_solve_dynamic = common.tpp_solve_dynamic
tpp_solve_linear_jit = common.tpp_solve_linear_jit
tpp_solve_binary_jit = common.tpp_solve_binary_jit
tpp_solve_binary2_jit = common.tpp_solve_binary2_jit
tpp_solve_dynamic_jit = common.tpp_solve_dynamic_jit

tpp_solve = u_tpp.tpp_solve
tpp_solve_naive_cpp = u_tpp_naive_cpp.tpp_solve

