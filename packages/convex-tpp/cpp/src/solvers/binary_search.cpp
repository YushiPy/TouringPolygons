
#include "common.h"
#include "tpp_convex_common.h"
#include "tpp_convex.h"

#include <algorithm>
#include <cmath>
#include <tuple>

using std::vector;
using std::pair;

namespace {

	constexpr double LOCAL_EPSILON = 1e-8;
	constexpr double LOCAL_EPSILON_SQUARED = LOCAL_EPSILON * LOCAL_EPSILON;

	bool point_on_segment(const Vector2 &point, const Vector2 &a, const Vector2 &b);

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

	bool point_in_convex_polygon_open(const Vector2 &point, const vector<Vector2> &polygon) {

		if (!point_in_convex_polygon_closed(point, polygon)) {
			return false;
		}

		for (size_t j = 0; j < polygon.size(); j++) {
			const auto &v1 = polygon[j];
			const auto &v2 = polygon[(j + 1) % polygon.size()];

			if (point_on_segment(point, v1, v2)) {
				return false;
			}
		}

		return true;
	}

	bool point_on_segment(const Vector2 &point, const Vector2 &a, const Vector2 &b) {
		return std::fabs((b - a).cross(point - a)) <= LOCAL_EPSILON_SQUARED
			&& (point - a).dot(point - b) <= LOCAL_EPSILON_SQUARED;
	}

	bool segment_enters_polygon_before_endpoint(const Vector2 &from, const Vector2 &to, const vector<Vector2> &polygon) {

		if (point_in_convex_polygon_closed(from, polygon)) {
			return !from.is_equal_approx(to, LOCAL_EPSILON);
		}

		for (size_t j = 0; j < polygon.size(); j++) {
			const auto &v1 = polygon[j];
			const auto &v2 = polygon[(j + 1) % polygon.size()];
			const auto intersection = tpp::segment_segment_intersection_safe(from, to, v1, v2);

			if (!intersection.is_finite()) {
				continue;
			}

			if (!intersection.is_equal_approx(to, LOCAL_EPSILON)) {
				return true;
			}
		}

		return false;
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
			const auto &a1 = a[i];
			const auto &a2 = a[(i + 1) % a.size()];

			for (size_t j = 0; j < b.size(); j++) {
				const auto &b1 = b[j];
				const auto &b2 = b[(j + 1) % b.size()];

				if (tpp::segment_segment_intersection_safe(a1, a2, b1, b2).is_finite()) {
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

	double path_length(const vector<Vector2> &path) {
		double length = 0.0;

		for (size_t i = 1; i < path.size(); i++) {
			length += path[i - 1].distance_to(path[i]);
		}

		return length;
	}
}

class SolutionBinarySearchDisjoint : public tpp::Solution {

	using tpp::Solution::Solution;

	public:

	/*
	Uses binary search to locate `point` in the visibility map of `polygon[i]`.
	Returns index as follows:
	- `2n` -> cone in vertex `n`
	- `2n + 1` -> edge between vertex `n` and `n + 1`

	The returned vertex or edge may not be in the first contact region, 
	so the caller should check for that and return -1 if it's not in the first contact region.
	*/
	size_t _locate_point(const Vector2& point, size_t i) {

		const auto polygon_index = i - 1;
		const auto &polygon = polygons[polygon_index];
		const auto vertex_count = polygon.size();

		const auto &first_vertex = polygon[0];
		const auto &[first_ray1, first_ray2] = get_cone(polygon_index, 0);

		if (tpp::point_in_cone_plus(point, first_vertex, first_ray1, first_ray2)) {
			return 0;
		}

		size_t left = 0;
		size_t right = vertex_count - 1;

		while (left != right) {

			const auto mid = left + (right - left) / 2;
			const auto mid_vertex_index = mid + 1;
			const auto &mid_vertex = polygon[mid_vertex_index];
			const auto &[mid_ray1, mid_ray2] = get_cone(polygon_index, mid_vertex_index);

			if (tpp::point_in_cone_plus(point, mid_vertex, mid_ray1, mid_ray2)) {
				return 2 * mid_vertex_index;
			}

			const auto &left_vertex = polygon[left];
			const auto &left_ray2 = get_cone(polygon_index, left).second;

			if (tpp::point_in_edge_plus(point, left_vertex, mid_vertex, left_ray2, mid_ray1)) {
				right = mid;
			} else {
				left = mid + 1;
			}
		}

		return 2 * left + 1;
	}

	int64_t locate_point_linear(const Vector2& point, size_t i) {
		const auto &polygon = polygons[i - 1];

		for (size_t j = 0; j < polygon.size(); j++) {
			const auto &vertex = polygon[j];
			const auto &[ray1, ray2] = get_cone(i - 1, j);
			const size_t previous = (j + polygon.size() - 1) % polygon.size();

			if (!is_first_contact(i - 1, j) && !is_first_contact(i - 1, previous)) {
				continue;
			}

			if (tpp::point_in_cone(point, vertex, ray1, ray2)) {
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

	int64_t locate_point(const Vector2& point, size_t i) override {

		size_t location = _locate_point(point, i);
		const auto polygon_index = i - 1;
		const auto vertex_count = polygons[polygon_index].size();
		
		size_t previous_index = location == 0 ? vertex_count - 1 : (location - 1) / 2;

		if (is_first_contact(polygon_index, location / 2) || is_first_contact(polygon_index, previous_index)) {
			return location;
		} else {
			return locate_point_linear(point, i);
		}
	}
};

class SolutionBinarySearchIntersecting : public SolutionBinarySearchDisjoint {

	using SolutionBinarySearchDisjoint::SolutionBinarySearchDisjoint;
	using SolutionBinarySearchDisjoint::_locate_point;

	vector<uint8_t> intersects_previous_cache;

	bool intersects_previous_polygon(size_t polygon_index) {

		if (intersects_previous_cache.size() != polygons.size()) {
			intersects_previous_cache.assign(polygons.size(), 2);
		}

		auto &cached = intersects_previous_cache[polygon_index];

		if (cached != 2) {
			return cached != 0;
		}

		for (size_t h = 0; h < polygon_index; h++) {
			if (polygons_intersect_or_touch(polygons[h], polygons[polygon_index])) {
				cached = 1;
				return true;
			}
		}

		cached = 0;
		return false;
	}

	int64_t locate_point(const Vector2& point, size_t i) override {

		const auto polygon_index = i - 1;
		const auto &polygon = polygons[polygon_index];
		
		if (point_in_convex_polygon_open(point, polygon)) {
			return -1;
		}

		if (intersects_previous_polygon(polygon_index)) {
			const auto previous = query(point, i - 1);

			if (segment_enters_polygon_before_endpoint(previous, point, polygon)) {
				return -1;
			}
		}

		size_t location = _locate_point(point, i);
		const auto vertex_count = polygons[polygon_index].size();
		
		size_t previous_index = location == 0 ? vertex_count - 1 : (location - 1) / 2;

		if (is_first_contact(polygon_index, location / 2) || is_first_contact(polygon_index, previous_index)) {
			return location;
		} else {
			return locate_point_linear(point, i);
		}
	}

	};

namespace tpp {

	void tpp_convex_solve_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			SolutionBinarySearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
		} else {
			tpp_convex_solve_linear_search_lazy(start, target, polygons, workspace, output);
		}
	}

	void tpp_convex_solve_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		SolutionBinarySearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
	}

	void tpp_convex_solve_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			SolutionBinarySearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Eager, output);
		} else {
			tpp_convex_solve_linear_search_eager(start, target, polygons, workspace, output);
		}
	}

	void tpp_convex_solve_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_binary_search_lazy(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	void tpp_convex_solve_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_binary_search_disjoint(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	void tpp_convex_solve_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_binary_search_eager(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	std::vector<Vector2> tpp_convex_solve_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionBinarySearchDisjoint(start, target, polygons).solve(PreloadPolicy::Lazy);
		} else {
			return tpp_convex_solve_linear_search_lazy(start, target, polygons);
		}
	}

	std::vector<Vector2> tpp_convex_solve_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionBinarySearchDisjoint(start, target, polygons).solve(PreloadPolicy::Lazy);
	}

	std::vector<Vector2> tpp_convex_solve_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionBinarySearchDisjoint(start, target, polygons).solve(PreloadPolicy::Eager);
		} else {
			return tpp_convex_solve_linear_search_eager(start, target, polygons);
		}
	}

	double tpp_convex_solve_length_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionBinarySearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Lazy);
		} else {
			return path_length(tpp_convex_solve_linear_search_lazy(start, target, polygons));
		}
	}

	double tpp_convex_solve_length_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionBinarySearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Lazy);
	}

	double tpp_convex_solve_length_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionBinarySearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Eager);
		} else {
			return path_length(tpp_convex_solve_linear_search_eager(start, target, polygons));
		}
	}

	std::vector<Vector2> tpp_convex_solve_binary_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_binary_search_lazy(start, target, polygons);
	}

	std::vector<Vector2> tpp_convex_solve(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_binary_search_lazy(start, target, polygons);
	}

	std::vector<Vector2> tpp_convex_solve_binary_search_dp(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_binary_search_eager(start, target, polygons);
	}
}
