
#include "common.h"
#include "tpp_convex_common.h"
#include "tpp_convex.h"
#include "tpp/convex/detail/intersecting_maps.h"

#include <algorithm>
#include <cmath>
#include <optional>
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

	// The disjoint core assumes CCW boundaries. Normalize only when necessary;
	// already-CCW inputs retain the existing solver and storage representation.
	std::optional<vector<vector<Vector2>>> normalized_winding(const vector<vector<Vector2>> &polygons) {
		auto clockwise = [](const vector<Vector2> &p) {
			for (size_t j = 0; j < p.size(); ++j) {
				const auto &a = p[j], &b = p[(j + 1) % p.size()], &c = p[(j + 2) % p.size()];
				const long double turn = ((long double)b.x - a.x) * ((long double)c.y - b.y)
					- ((long double)b.y - a.y) * ((long double)c.x - b.x);
				if (turn != 0) return turn < 0;
			}
			return false;
		};
		std::optional<vector<vector<Vector2>>> result;
		for (size_t i = 0; i < polygons.size(); ++i) {
			if (!clockwise(polygons[i])) continue;
			if (!result) result = polygons;
			std::reverse((*result)[i].begin(), (*result)[i].end());
		}
		return result;
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

class SolutionBinarySearchDisjoint : public tpp::Solution {

	using tpp::Solution::Solution;

	public:

	void query_trace(const Vector2 &point,size_t i,vector<tpp::detail::DirectionalTraceStep> &trace) {
		if(i==0)return;
		const auto location=locate_point(point,i);
		tpp::detail::DirectionalTraceStep step;step.level=i;
		if(location<0) {
			step.region=tpp::detail::DirectionalTraceRegion::Crossing;
			trace.push_back(step);query_trace(point,i-1,trace);return;
		}
		const auto &polygon=polygons[i-1];const size_t vertex_index=size_t(location)/2;
		step.original_edge=vertex_index;
		if(location%2==0) {
			step.region=tpp::detail::DirectionalTraceRegion::Vertex;
			step.defining_point=polygon[vertex_index];
			trace.push_back(step);query_trace(polygon[vertex_index],i-1,trace);return;
		}
		step.region=tpp::detail::DirectionalTraceRegion::Edge;trace.push_back(step);
		query_trace(point.reflect_line(polygon[vertex_index],polygon[(vertex_index+1)%polygon.size()]),i-1,trace);
	}

	vector<tpp::detail::DirectionalTraceStep> trace(tpp::PreloadPolicy preload) {
		initialize_storage();
		if(preload==tpp::PreloadPolicy::Eager)
			for(size_t i=0;i<polygons.size();++i)for(size_t j=0;j<polygons[i].size();++j)build_cone(i,j);
		else preload_cones();
		vector<tpp::detail::DirectionalTraceStep> result;result.reserve(polygons.size());
		query_trace(target,polygons.size(),result);
		return result;
	}

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

	int64_t locate_point(const Vector2& point, size_t i) override {

		size_t location = _locate_point(point, i);
		const auto polygon_index = i - 1;
		const auto vertex_count = polygons[polygon_index].size();
		
		size_t previous_index = location == 0 ? vertex_count - 1 : (location - 1) / 2;

		// Binary search locates the circular fan cell. A cell with no incident
		// first-contact edge inherits the preceding path (crossing).
		if (is_first_contact(polygon_index, location / 2) || is_first_contact(polygon_index, previous_index))
			return location;
		return -1;
	}
};

namespace tpp {
	namespace detail {
		bool pairwise_disjoint_unchecked_double(const std::vector<std::vector<Vector2>> &polygons) {
			return polygons_are_pairwise_disjoint(polygons);
		}
		std::vector<DirectionalTraceStep> solve_binary_search_disjoint_trace_unchecked(
			const Vector2 &start,const Vector2 &target,const std::vector<std::vector<Vector2>> &polygons,
			PreloadPolicy preload) {
			if(auto normalized=normalized_winding(polygons))
				return solve_binary_search_disjoint_trace_unchecked(start,target,*normalized,preload);
			return SolutionBinarySearchDisjoint(start,target,polygons).trace(preload);
		}
	}

	void tpp_convex_solve_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		if (auto normalized = normalized_winding(polygons)) {
			tpp_convex_solve_binary_search_lazy(start, target, *normalized, workspace, output);
			return;
		}
		if (polygons_are_pairwise_disjoint(polygons)) {
			SolutionBinarySearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
		} else {
			output = detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Lazy);
		}
	}

	void tpp_convex_solve_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		SolutionBinarySearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
	}

	void tpp_convex_solve_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		if (auto normalized = normalized_winding(polygons)) {
			tpp_convex_solve_binary_search_eager(start, target, *normalized, workspace, output);
			return;
		}
		if (polygons_are_pairwise_disjoint(polygons)) {
			SolutionBinarySearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Eager, output);
		} else {
			output = detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Eager);
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
		if (auto normalized = normalized_winding(polygons))
			return tpp_convex_solve_binary_search_lazy(start, target, *normalized);
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionBinarySearchDisjoint(start, target, polygons).solve(PreloadPolicy::Lazy);
		} else {
			return detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Lazy);
		}
	}

	std::vector<Vector2> tpp_convex_solve_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if(auto normalized=normalized_winding(polygons))
			return tpp_convex_solve_binary_search_disjoint(start,target,*normalized);
		return SolutionBinarySearchDisjoint(start, target, polygons).solve(PreloadPolicy::Lazy);
	}

	std::vector<Vector2> tpp_convex_solve_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (auto normalized = normalized_winding(polygons))
			return tpp_convex_solve_binary_search_eager(start, target, *normalized);
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionBinarySearchDisjoint(start, target, polygons).solve(PreloadPolicy::Eager);
		} else {
			return detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Eager);
		}
	}

	double tpp_convex_solve_length_binary_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (auto normalized = normalized_winding(polygons))
			return tpp_convex_solve_length_binary_search_lazy(start, target, *normalized);
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionBinarySearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Lazy);
		} else {
			return detail::length_intersecting_maps(start, target, polygons, PreloadPolicy::Lazy);
		}
	}

	double tpp_convex_solve_length_binary_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if(auto normalized=normalized_winding(polygons))
			return tpp_convex_solve_length_binary_search_disjoint(start,target,*normalized);
		return SolutionBinarySearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Lazy);
	}

	double tpp_convex_solve_length_binary_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (auto normalized = normalized_winding(polygons))
			return tpp_convex_solve_length_binary_search_eager(start, target, *normalized);
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionBinarySearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Eager);
		} else {
			return detail::length_intersecting_maps(start, target, polygons, PreloadPolicy::Eager);
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
