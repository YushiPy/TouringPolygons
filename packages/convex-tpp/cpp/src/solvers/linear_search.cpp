
#include "vector2.h"
#include "common.h"
#include "tpp_convex_common.h"
#include "tpp_convex.h"
#include "tpp/convex/detail/intersecting_maps.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

using std::vector;
using std::pair;

namespace {

	constexpr double LOCAL_EPSILON = 1e-8;
	constexpr double LOCAL_EPSILON_SQUARED = LOCAL_EPSILON * LOCAL_EPSILON;

	bool point_in_convex_polygon_closed(const Vector2 &point, const vector<Vector2> &polygon) {
		bool has_positive = false;
		bool has_negative = false;

		for (size_t j = 0; j < polygon.size(); j++) {
			const auto &v1 = polygon[j];
			const auto &v2 = polygon[(j + 1) % polygon.size()];
			const double cross = (v2 - v1).cross(point - v1);

			if (cross > LOCAL_EPSILON_SQUARED) {
				has_positive = true;
			} else if (cross < -LOCAL_EPSILON_SQUARED) {
				has_negative = true;
			}

			if (has_positive && has_negative) {
				return false;
			}
		}

		return true;
	}

	bool polygons_intersect_or_touch(const vector<Vector2> &a, const vector<Vector2> &b) {
		auto bounds = [](const vector<Vector2> &polygon) {
			double min_x = polygon.front().x;
			double max_x = polygon.front().x;
			double min_y = polygon.front().y;
			double max_y = polygon.front().y;

			for (const auto &point : polygon) {
				min_x = std::min(min_x, point.x);
				max_x = std::max(max_x, point.x);
				min_y = std::min(min_y, point.y);
				max_y = std::max(max_y, point.y);
			}

			return std::tuple(min_x, max_x, min_y, max_y);
		};

		const auto [a_min_x, a_max_x, a_min_y, a_max_y] = bounds(a);
		const auto [b_min_x, b_max_x, b_min_y, b_max_y] = bounds(b);

		if (
			a_max_x < b_min_x - LOCAL_EPSILON ||
			b_max_x < a_min_x - LOCAL_EPSILON ||
			a_max_y < b_min_y - LOCAL_EPSILON ||
			b_max_y < a_min_y - LOCAL_EPSILON
		) {
			return false;
		}

		for (size_t i = 0; i < a.size(); i++) {
			for (size_t j = 0; j < b.size(); j++) {
				if (tpp::segment_segment_intersection_safe(a[i], a[(i + 1) % a.size()], b[j], b[(j + 1) % b.size()]).is_finite()) {
					return true;
				}
			}
		}

		return point_in_convex_polygon_closed(a.front(), b) || point_in_convex_polygon_closed(b.front(), a);
	}

	bool polygons_are_pairwise_disjoint(const vector<vector<Vector2>> &polygons) {
		for (size_t i = 0; i < polygons.size(); i++) {
			for (size_t j = i + 1; j < polygons.size(); j++) {
				if (polygons_intersect_or_touch(polygons[i], polygons[j])) {
					return false;
				}
			}
		}

		return true;
	}


}

class SolutionLinearSearchDisjoint : public tpp::Solution {

	using tpp::Solution::Solution;

	protected:

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
		if (polygons_are_pairwise_disjoint(polygons)) {
			SolutionLinearSearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
		} else {
			output = detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Lazy);
		}
	}

	void tpp_convex_solve_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		SolutionLinearSearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
	}

	void tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			SolutionLinearSearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Eager, output);
		} else {
			output = detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Eager);
		}
	}

	void tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_linear_search_lazy(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	void tpp_convex_solve_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_linear_search_disjoint(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	void tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_linear_search_eager(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionLinearSearchDisjoint(start, target, polygons).solve(PreloadPolicy::Lazy);
		} else {
			return detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Lazy);
		}
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionLinearSearchDisjoint(start, target, polygons).solve(PreloadPolicy::Lazy);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionLinearSearchDisjoint(start, target, polygons).solve(PreloadPolicy::Eager);
		} else {
			return detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Eager);
		}
	}

	double tpp_convex_solve_length_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionLinearSearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Lazy);
		} else {
			return detail::length_intersecting_maps(start, target, polygons, PreloadPolicy::Lazy);
		}
	}

	double tpp_convex_solve_length_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionLinearSearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Lazy);
	}

	double tpp_convex_solve_length_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionLinearSearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Eager);
		} else {
			return detail::length_intersecting_maps(start, target, polygons, PreloadPolicy::Eager);
		}
	}

	std::vector<Vector2> tpp_convex_solve_linear_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_linear_search_lazy(start, target, polygons);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_dp(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_linear_search_eager(start, target, polygons);
	}
}
