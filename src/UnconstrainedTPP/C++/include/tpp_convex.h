
#pragma once

#include "vector2.h"

#include <vector>
#include <functional>

namespace tpp {

	using Solver = std::function<std::vector<Vector2>(const Vector2&, const Vector2&, const std::vector<std::vector<Vector2>>&)>;

	// Solver tpp_convex_solve_binary_search;
	// Solver tpp_convex_solve_linear_search;
	// Solver tpp_convex_solve_tamc;
	// Solver tpp_convex_solve_gurobi;

	std::vector<Vector2> tpp_convex_solve_binary_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_linear_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_tamc(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);

	/*
	Note: This function requires a Gurobi license to use. Also it is not very precise, so don't use it.
	*/
	std::vector<Vector2> tpp_convex_solve_gurobi(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
}
