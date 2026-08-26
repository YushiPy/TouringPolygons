
#pragma once

#include "tpp/convex/options.h"
#include "tpp/convex/workspace.h"
#include "vector2.h"

#include <vector>
#include <functional>

namespace tpp {

	using Solver = std::function<std::vector<Vector2>(const Vector2&, const Vector2&, const std::vector<std::vector<Vector2>>&)>;

	std::vector<Vector2> tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_tan_jiang(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);

	double tpp_convex_solve_length_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	double tpp_convex_solve_length_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	double tpp_convex_solve_length_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	double tpp_convex_solve_length_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	double tpp_convex_solve_length_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	double tpp_convex_solve_length_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	double tpp_convex_solve_length_tan_jiang(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	double tpp_convex_solve_length_gurobi(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);

	void tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_tan_jiang(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output);

	void tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output);
	void tpp_convex_solve_tan_jiang(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output);

	// Compatibility wrappers for older callers.
	std::vector<Vector2> tpp_convex_solve(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_binary_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_binary_search_dp(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_linear_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_linear_search_dp(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
	std::vector<Vector2> tpp_convex_solve_tamc(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);

	/*
	Note: This function requires a Gurobi license to use. Also it is not very precise, so don't use it.
	*/
	std::vector<Vector2> tpp_convex_solve_gurobi(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
}
