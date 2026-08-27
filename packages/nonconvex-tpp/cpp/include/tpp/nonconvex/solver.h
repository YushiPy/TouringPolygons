#pragma once

#include "vector2.h"

#include <cstddef>
#include <vector>

namespace tpp {

	struct NonconvexTppSolveOptions {
		size_t max_calls = 200000;
		double max_seconds = 3.0;
	};

	struct NonconvexTppSolveResult {
		std::vector<Vector2> path;
		bool exact = true;
		size_t calls = 0;
		double seconds = 0.0;
	};

	NonconvexTppSolveResult tpp_nonconvex_solve(
		const Vector2 &start,
		const Vector2 &target,
		const std::vector<std::vector<Vector2>> &polygons,
		const NonconvexTppSolveOptions &options = {}
	);

	NonconvexTppSolveResult tpp_nonconvex_solve_decomposed(
		const Vector2 &start,
		const Vector2 &target,
		const std::vector<std::vector<std::vector<Vector2>>> &pieces,
		const NonconvexTppSolveOptions &options = {}
	);

}
