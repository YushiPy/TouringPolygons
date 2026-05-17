/*
This solution is based on this paper:

Tan, X., Jiang, B. (2017). 
Efficient Algorithms for Touring a Sequence of Convex Polygons and Related Problems.
In: Gopal, T., Jäger , G., Steila, S. (eds)
Theory and Applications of Models of Computation. TAMC 2017.
Lecture Notes in Computer Science(), vol 10185. Springer, Cham.
https://doi.org/10.1007/978-3-319-55911-7_44
*/

#include "vector2.h"
#include "common.h"
#include "tpp_convex_common.h"


#include "tests.h"
#include <iostream>
#include <print>

using std::vector;
using std::pair;

class SolutionTAMC : public tpp::Solution {

	using tpp::Solution::Solution;
	using tpp::Solution::query;

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
		const auto &fc = first_contact[i - 1];
		
		size_t previous_index = location == 0 ? fc.size() - 1 : (location - 1) / 2;

		if (fc[location / 2] || fc[previous_index]) {
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

		const auto &point = points[0];
		auto location = _locate_point(point, i);

		vector<size_t> locations;
		locations.reserve(points.size());
		locations.push_back(location);

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

		auto scan = [&](size_t &start_location, size_t &point_index, int shift) -> void {

			size_t location = start_location;
			size_t original_location = location;

			while (point_index < points.size()) {

				const Vector2 &point = points[point_index];

				if (point_in_location(point, location)) {
					locations.push_back(location);
					point_index++;
					start_location = location;
				} else {
					
					location = (location + shift + 2 * polygon.size()) % (2 * polygon.size());

					if (location == original_location) {
						break;
					}
				}
			}
		};

		// Perform 4 scans to ensure that we find the location of all points, two in each direction.
		size_t last_location = location;
		size_t point_index = 1;

		scan(last_location, point_index, 1);
		scan(last_location, point_index, -1);
		scan(last_location, point_index, 1);
		scan(last_location, point_index, -1);

		vector<Vector2> reflected_points, pass_through_points;

		for (size_t j = 0; j < points.size(); j++) {

			const auto &point = points[j];
			const auto &location = locations[j];

			if (location % 2 == 1) {
				if (first_contact[i - 1][location / 2]) {

					const auto &v1 = polygon[location / 2];
					const auto &v2 = polygon[(location / 2 + 1) % polygon.size()];
					const auto reflected = point.reflect_line(v1, v2);

					reflected_points.push_back(reflected);

				} else {
					pass_through_points.push_back(point);
				}
			}
		}

		auto reflected_results = query_points(reflected_points, i - 1);
		auto pass_through_results = query_points(pass_through_points, i - 1);

		vector<Vector2> results;
		results.reserve(points.size());

		size_t reflected_index = 0;
		size_t pass_through_index = 0;

		for (size_t j = 0; j < points.size(); j++) {

			const auto &point = points[j];
			const auto &location = locations[j];

			if (location % 2 == 1) {
				if (first_contact[i - 1][location / 2]) {

					const auto &v1 = polygon[location / 2];
					const auto &v2 = polygon[(location / 2 + 1) % polygon.size()];

					const auto &reflected_point = reflected_points[reflected_index];
					const auto &reflected_result = reflected_results[reflected_index++];
					
					const auto intersection = tpp::segment_segment_intersection(reflected_point, reflected_result, v1, v2);

					results.push_back(intersection);

				} else {
					results.push_back(pass_through_results[pass_through_index]);
					pass_through_index++;
				}
			} else {
				results.push_back(polygon[location / 2]);
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

	std::vector<Vector2> tpp_convex_solve_tamc(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionTAMC(start, target, polygons).solve();
	}
}
