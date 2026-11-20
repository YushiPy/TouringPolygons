
#include "tpp/geometry/common.h"
#include "tpp_convex_common.h"
#include "tpp_convex.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "tpp/convex/detail/convex_contacts.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>

using std::vector;
using std::pair;

namespace {

	using tpp::detail::normalized_winding;

	// O(kn) in all, by the linear-time contacts of convex_contacts.h.
	// Points and segments are not "disjoint" here: the disjoint solver needs
	// two distinct edges at every vertex, so they take the directional maps.
	bool polygons_are_pairwise_disjoint(const vector<vector<Vector2>> &polygons) {
		return !tpp::detail::has_lower_dimensional_region(polygons) && tpp::detail::pairwise_disjoint_double(polygons);
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

		++tpp::detail::point_locations();
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
			if(has_lower_dimensional_region(polygons))
				throw std::invalid_argument("The disjoint trace needs polygons with at least three vertices");
			return SolutionBinarySearchDisjoint(start,target,polygons).trace(preload);
		}
		double length_binary_search_disjoint_unchecked(const Vector2 &start,const Vector2 &target,
			const std::vector<std::vector<Vector2>> &polygons,PreloadPolicy preload) {
			if(auto normalized=normalized_winding(polygons))
				return length_binary_search_disjoint_unchecked(start,target,*normalized,preload);
			if(has_lower_dimensional_region(polygons))
				return length_disjoint_maps(start,target,polygons,preload);
			return SolutionBinarySearchDisjoint(start,target,polygons).solve_length(preload);
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
		if (tpp::detail::has_lower_dimensional_region(polygons)) {
			output = detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Lazy);
			return;
		}
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
		if (tpp::detail::has_lower_dimensional_region(polygons))
			return detail::solve_intersecting_maps(start, target, polygons, PreloadPolicy::Lazy);
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
		if (tpp::detail::has_lower_dimensional_region(polygons))
			return detail::length_intersecting_maps(start, target, polygons, PreloadPolicy::Lazy);
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

	namespace {
		std::vector<double> disjoint_lengths(const Vector2& start, const std::vector<Vector2>& targets,
				const std::vector<std::vector<Vector2>>& polygons, PreloadPolicy policy) {
			if (auto normalized = normalized_winding(polygons))
				return disjoint_lengths(start, targets, *normalized, policy);
			if (!polygons_are_pairwise_disjoint(polygons))
				throw std::invalid_argument("Many-target lengths need pairwise disjoint polygons");
			const Vector2 &first = targets.empty() ? start : targets.front();
			return SolutionBinarySearchDisjoint(start, first, polygons).query_lengths(targets, policy);
		}
	}

	std::vector<double> tpp_convex_solve_lengths_binary_search_lazy(const Vector2& start, const std::vector<Vector2>& targets, const std::vector<std::vector<Vector2>>& polygons) {
		return disjoint_lengths(start, targets, polygons, PreloadPolicy::Lazy);
	}

	std::vector<double> tpp_convex_solve_lengths_binary_search_eager(const Vector2& start, const std::vector<Vector2>& targets, const std::vector<std::vector<Vector2>>& polygons) {
		return disjoint_lengths(start, targets, polygons, PreloadPolicy::Eager);
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
