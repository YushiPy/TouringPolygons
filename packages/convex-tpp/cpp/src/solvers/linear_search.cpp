
#include "vector2.h"
#include "common.h"
#include "tpp_convex_common.h"
#include "tpp_convex.h"

using std::vector;
using std::pair;

class SolutionLinearSearch : public tpp::Solution {

	using tpp::Solution::Solution;

	int64_t locate_point(const Vector2& point, size_t i) override {

		const auto &polygon = polygons[i - 1];

		for (size_t j = 0; j < polygon.size(); j++) {

			const auto &v = polygon[j];
			const auto &[ray1, ray2] = get_cone(i - 1, j);

			size_t prev = (j + polygon.size() - 1) % polygon.size();

			if (!is_first_contact(i - 1, j) && !is_first_contact(i - 1, prev)) {
				continue;
			}

			if (tpp::point_in_cone(point, v, ray1, ray2)) {
				return 2 * j;
			}
		}

		for (size_t j = 0; j < polygon.size(); j++) {

			if (!is_first_contact(i - 1, j)) {
				continue;
			}

			const auto &v1 = polygon[j];
			const auto &v2 = polygon[(j + 1) % polygon.size()];

			const auto &ray1 = get_cone(i - 1, j).second;
			const auto &ray2 = get_cone(i - 1, (j + 1) % polygon.size()).first;

			if (tpp::point_in_edge(point, v1, v2, ray1, ray2)) {
				return 2 * j + 1;
			}
		}

		return -1;
	}
};

namespace tpp {

	void tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		SolutionLinearSearch(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
	}

	void tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		SolutionLinearSearch(start, target, polygons, workspace).solve(PreloadPolicy::Eager, output);
	}

	void tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_linear_search_lazy(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	void tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_linear_search_eager(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionLinearSearch(start, target, polygons).solve(PreloadPolicy::Lazy);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionLinearSearch(start, target, polygons).solve(PreloadPolicy::Eager);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_linear_search_lazy(start, target, polygons);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_dp(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_linear_search_eager(start, target, polygons);
	}
}
