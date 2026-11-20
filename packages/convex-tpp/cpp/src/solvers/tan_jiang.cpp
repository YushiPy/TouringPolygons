/*
This solution is based on this paper:

Tan, X., Jiang, B. (2017). 
Efficient Algorithms for Touring a Sequence of Convex Polygons and Related Problems.
In: Gopal, T., Jäger , G., Steila, S. (eds)
Theory and Applications of Models of Computation. TAMC 2017.
Lecture Notes in Computer Science(), vol 10185. Springer, Cham.
https://doi.org/10.1007/978-3-319-55911-7_44
*/

#include "tpp/geometry/vec2.h"
#include "tpp/geometry/common.h"
#include "tpp_convex_common.h"
#include "tpp_convex.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "tpp/convex/detail/convex_contacts.h"

#include <algorithm>
#include <tuple>

// Maximum number of "up and down" scans before falling back to independent point location.
#define MAX_SCAN_COUNT 3

using std::vector;
using std::pair;

namespace {

	bool is_tan_jiang_supported_disjoint_case(const vector<vector<Vector2>> &polygons) {
		// Its statuses read three corners of each polygon: points and segments
		// take the directional maps.
		return !tpp::detail::has_lower_dimensional_region(polygons) && tpp::detail::pairwise_disjoint_double(polygons);
	}
}

class SolutionTAMC : public tpp::Solution {

	using tpp::Solution::Solution;
	using tpp::Solution::query;

	public:

	// The cells chosen by the query to the target, as SolutionBinarySearchDisjoint
	// records them for the certified flow; every cone is built first.
	void query_trace(const Vector2 &point, size_t i, vector<tpp::detail::DirectionalTraceStep> &trace) {
		if (i == 0) return;
		const auto location = locate_point(point, i);
		tpp::detail::DirectionalTraceStep step; step.level = i;
		if (location < 0) {
			step.region = tpp::detail::DirectionalTraceRegion::Crossing;
			trace.push_back(step); query_trace(point, i - 1, trace); return;
		}
		const auto &polygon = polygons[i - 1]; const size_t vertex_index = size_t(location) / 2;
		step.original_edge = vertex_index;
		if (location % 2 == 0) {
			step.region = tpp::detail::DirectionalTraceRegion::Vertex;
			step.defining_point = polygon[vertex_index];
			trace.push_back(step); query_trace(polygon[vertex_index], i - 1, trace); return;
		}
		step.region = tpp::detail::DirectionalTraceRegion::Edge; trace.push_back(step);
		query_trace(point.reflect_line(polygon[vertex_index], polygon[(vertex_index + 1) % polygon.size()]), i - 1, trace);
	}

	vector<tpp::detail::DirectionalTraceStep> trace() {
		initialize_storage();
		preload_cones();
		vector<tpp::detail::DirectionalTraceStep> result; result.reserve(polygons.size());
		query_trace(target, polygons.size(), result);
		return result;
	}

	private:

	bool is_active_location(size_t polygon_index, size_t location) const {

		const auto vertex_count = polygons[polygon_index].size();

		if (location % 2 == 0) {
			const auto vertex_index = location / 2;
			const auto previous_index = vertex_index == 0 ? vertex_count - 1 : vertex_index - 1;
			return is_first_contact(polygon_index, vertex_index) || is_first_contact(polygon_index, previous_index);
		}

		return is_first_contact(polygon_index, location / 2);
	}

	/*
	Uses binary search to locate `point` in the visibility map of `polygon[i - 1]`.
	Returns index as follows:
	- `2n` -> cone in vertex `n`
	- `2n + 1` -> edge between vertex `n` and `n + 1`

	The returned vertex or edge may not be in the first contact region, 
	so the caller should check for that and return -1 if it's not in the first contact region.
	*/
	size_t _locate_point(const Vector2& point, size_t i) {

		auto check_vertex = [&](size_t j) -> bool {

			const auto &v = polygons[i - 1][j];
			const auto &[ray1, ray2] = get_cone(i - 1, j);

			return tpp::point_in_cone_plus(point, v, ray1, ray2);
		};

		auto check_edge = [&](size_t l, size_t r) -> bool {

			r = (r + 1) % polygons[i - 1].size();

			const auto &v1 = polygons[i - 1][l];
			const auto &v2 = polygons[i - 1][r];

			const auto &ray1 = get_cone(i - 1, l).second;
			const auto &ray2 = get_cone(i - 1, r).first;

			return tpp::point_in_edge_plus(point, v1, v2, ray1, ray2);
		};

		if (check_vertex(0)) {
			return 0;
		}

		size_t left = 0;
		size_t right = polygons[i - 1].size() - 1;

		while (left != right) {

			auto mid = left + (right - left) / 2;

			if (check_vertex(mid + 1)) {
				return 2 * (mid + 1);
			}

			if (check_edge(left, mid)) {
				right = mid;
			} else {
				left = mid + 1;
			}
		}

		return 2 * left + 1;
	}

	int64_t locate_point(const Vector2& point, size_t i) override {

		size_t location = _locate_point(point, i);

		if (is_active_location(i - 1, location)) {
			return location;
		} else {
			return -1;
		}
	}

	vector<Vector2> query_points(const vector<Vector2> &points, size_t i) {

		if (points.empty()) {
			return {};
		}

		if (i == 0) {
			return vector<Vector2>(points.size(), start);
		}

		const auto &polygon = polygons[i - 1];

		vector<size_t> locations;
		locations.reserve(points.size());

		auto point_in_location = [&](const Vector2 &point, size_t location) -> bool {

			const auto index1 = location / 2;
			const auto index2 = (location + 1) / 2 % polygon.size();

			if (location % 2 == 0) {

				const auto &vertex = polygon[index1];
				const auto &[ray1, ray2] = get_cone(i - 1, index1);

				return tpp::point_in_cone_plus(point, vertex, ray1, ray2);

			} else {

				const auto &v1 = polygon[index1];
				const auto &v2 = polygon[index2];
				const auto &ray1 = get_cone(i - 1, index1).second;
				const auto &ray2 = get_cone(i - 1, index2).first;

				return tpp::point_in_edge_plus(point, v1, v2, ray1, ray2);
			}
		};

		// Perform 4 scans to ensure that we find the location of all points, two in each direction.
		size_t point_index = 0;
		size_t last_location = 0;
		size_t scan_count = 0;

		while (point_index < points.size()) {

			const size_t first_unlocated_point = point_index;
			size_t original_location = last_location;
			size_t location = last_location;

			while (point_index < points.size()) {

				if (point_in_location(points[point_index], location)) {
					locations.push_back(location);
					last_location = location;
					point_index++;
				} else {
					
					location = (location + 1) % (2 * polygon.size());

					if (location == original_location) {
						break;
					}
				}
			}

			original_location = last_location;
			location = last_location;

			while (point_index < points.size()) {

				if (point_in_location(points[point_index], location)) {
					locations.push_back(location);
					last_location = location;
					point_index++;
				} else {
					
					location = (location + 2 * polygon.size() - 1) % (2 * polygon.size());

					if (location == original_location) {
						break;
					}
				}
			}

			scan_count++;

			if (point_index == first_unlocated_point) {
				location = _locate_point(points[point_index], i);
				locations.push_back(location);
				last_location = location;
				point_index++;
				scan_count = 0;
				continue;
			}

			if (scan_count > MAX_SCAN_COUNT) {
				while (point_index < points.size()) {
					location = _locate_point(points[point_index], i);
					locations.push_back(location);
					last_location = location;
					point_index++;
				}
			}
		}

		vector<Vector2> input_points;

		for (size_t j = 0; j < points.size(); j++) {

			const auto &point = points[j];
			const auto &location = locations[j];

			if (!is_active_location(i - 1, location)) {
				input_points.push_back(point);
			} else if (location % 2 == 0) {
				// Do nothing, we can tell the point come from the vertex directly without needing to query the previous polygon.
			} else {
				const auto &v1 = polygon[location / 2];
				const auto &v2 = polygon[(location / 2 + 1) % polygon.size()];
				const auto reflected = point.reflect_line(v1, v2);
				input_points.push_back(reflected);
			}
		}

		vector<Vector2> returned_points = query_points(input_points, i - 1);
		size_t index = 0;

		vector<Vector2> results;

		for (size_t j = 0; j < points.size(); j++) {

			const auto &location = locations[j];

			if (!is_active_location(i - 1, location)) {
				results.push_back(returned_points[index++]);
			} else if (location % 2 == 0) {
				results.push_back(polygon[location / 2]);
			} else {
				
				const auto &v1 = polygon[location / 2];
				const auto &v2 = polygon[(location / 2 + 1) % polygon.size()];
				const auto reflected = input_points[index];
				const auto last = returned_points[index++];

				const auto intersection = tpp::segment_segment_intersection(last, reflected, v1, v2);

				results.push_back(intersection);
			}
		}

		return results;
	}

	void preload_cones() override {
		
		// If there are no polygons, there is nothing to preload.
		// If there is a sinlge polygon, queries to the vertices take O(1) time, 
		// so there is no need to preload the cones.
		if (polygons.size() <= 1) {
			return;
		}

		for (size_t j = 0; j < polygons[0].size(); j++) {
			build_cone(0, j, start);
		}
		
		for (size_t i = 1; i < polygons.size(); i++) {

			const auto &polygon = polygons[i];
			const auto last_points = query_points(polygon, i);
			
			for (size_t j = 0; j < polygon.size(); j++) {
				build_cone(i, j, last_points[j]);
			}
		}
	}
};

namespace tpp {

	void tpp_convex_solve_tan_jiang(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		if (auto normalized = detail::normalized_winding(polygons)) {
			tpp_convex_solve_tan_jiang(start, target, *normalized, workspace, output);
			return;
		}

		if (!is_tan_jiang_supported_disjoint_case(polygons)) {
			output = detail::solve_intersecting_maps_tan_jiang(start, target, polygons);
			return;
		}

		SolutionTAMC(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
	}

	void tpp_convex_solve_tan_jiang(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_tan_jiang(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	std::vector<Vector2> tpp_convex_solve_tan_jiang(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (auto normalized = detail::normalized_winding(polygons))
			return tpp_convex_solve_tan_jiang(start, target, *normalized);

		if (!is_tan_jiang_supported_disjoint_case(polygons)) {
			return detail::solve_intersecting_maps_tan_jiang(start, target, polygons);
		}

		return SolutionTAMC(start, target, polygons).solve();
	}

	double tpp_convex_solve_length_tan_jiang(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (auto normalized = detail::normalized_winding(polygons))
			return tpp_convex_solve_length_tan_jiang(start, target, *normalized);

		if (!is_tan_jiang_supported_disjoint_case(polygons)) {
			return detail::length_intersecting_maps_tan_jiang(start, target, polygons);
		}

		return SolutionTAMC(start, target, polygons).solve_length();
	}

	std::vector<double> tpp_convex_solve_lengths_tan_jiang(const Vector2& start, const std::vector<Vector2>& targets, const std::vector<std::vector<Vector2>>& polygons) {
		if (auto normalized = detail::normalized_winding(polygons))
			return tpp_convex_solve_lengths_tan_jiang(start, targets, *normalized);

		if (!is_tan_jiang_supported_disjoint_case(polygons))
			throw std::invalid_argument("Tan and Jiang's construction needs pairwise disjoint polygons");
		const Vector2 &first = targets.empty() ? start : targets.front();
		return SolutionTAMC(start, first, polygons).query_lengths(targets, PreloadPolicy::Lazy);
	}

	namespace detail {
		std::vector<DirectionalTraceStep> solve_tan_jiang_trace_unchecked(const Vector2 &start, const Vector2 &target,
				const std::vector<std::vector<Vector2>> &polygons) {
			if (auto normalized = normalized_winding(polygons))
				return solve_tan_jiang_trace_unchecked(start, target, *normalized);
			if (!is_tan_jiang_supported_disjoint_case(polygons))
				throw std::invalid_argument("Tan and Jiang's construction needs pairwise disjoint polygons");
			return SolutionTAMC(start, target, polygons).trace();
		}
		double length_tan_jiang_unchecked(const Vector2 &start, const Vector2 &target,
				const std::vector<std::vector<Vector2>> &polygons) {
			if (auto normalized = normalized_winding(polygons))
				return length_tan_jiang_unchecked(start, target, *normalized);
			return SolutionTAMC(start, target, polygons).solve_length();
		}
	}

	std::vector<Vector2> tpp_convex_solve_tamc(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_tan_jiang(start, target, polygons);
	}
}
