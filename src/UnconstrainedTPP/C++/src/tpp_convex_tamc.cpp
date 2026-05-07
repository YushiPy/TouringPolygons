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

	/*
	We allow the caller to pass the location of the point in the last step map.
	This is useful if the caller ha already determined the location of the point in the last step map, 
	as it avoids doing the same binary search again in the `query` function.

	However, it is important to note that the location passed by the caller only informs
	us about the location of the point in the current last step map. Thus, we might have to perform
	additional binary searches if the location is not a vertex region.
	*/
	Vector2 query(const Vector2& point, size_t i, int64_t location) {

		const auto &fc = first_contact[i - 1];
		size_t previous_index = location == 0 ? fc.size() - 1 : (location - 1) / 2;

		if (!fc[location / 2] && !fc[previous_index]) {
			location = -1;
		}

		if (location == -1) {
			return query(point, i - 1);
		}

		const auto &polygon = polygons[i - 1];
		auto vertex_index = location / 2;
		
		if (location % 2 == 0) {
			return polygon[vertex_index];
		}

		const auto &v1 = polygon[vertex_index];
		const auto &v2 = polygon[(vertex_index + 1) % polygon.size()];

		const auto reflected = point.reflect_line(v1, v2);
		const auto last = query(reflected, i - 1);
		const auto intersection = tpp::segment_segment_intersection(last, reflected, v1, v2);

		return intersection;
	};

	void build_cone(size_t i, size_t j, const Vector2 &last) {
		
		auto j_prev = (j - 1 + polygons[i].size()) % polygons[i].size();
		auto j_next = (j + 1) % polygons[i].size();

		const auto &polygon = polygons[i];
		const auto &_first_contact = first_contact[i];

		const auto before = polygon[j_prev];
		const auto vertex = polygon[j];
		const auto after = polygon[j_next];

		// const auto last = query(vertex, i);
		const auto diff = (vertex - last).normalized(); // Normalizing is not necessary, but makes debugging easier and does not affect the correctness of the algorithm.

		auto ray1 = diff.reflect(vertex - before);
		auto ray2 = diff.reflect(vertex - after);

		first_contact[i][j_prev] = diff.cross(vertex - before) < 0;
		first_contact[i][j] = diff.cross(vertex - after) > 0;

		if (!_first_contact[j_prev]) {
			ray1 = diff;
		}

		if (!_first_contact[j]) {
			ray2 = diff;
		}

		cones[i][j] = {ray1, ray2};
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
		
		for (size_t i = 0; i < polygons.size() - 1; i++) {
			
			const auto &polygon1 = polygons[i]; // Solved polygon
			const auto &polygon2 = polygons[i + 1]; // Next polygon to solve

			const auto &v = polygon2[0];

			auto location = _locate_point(v, i + 1);
			const auto last = query(v, i + 1, location);

			build_cone(i + 1, 0, last);

			auto point_in_location = [&](size_t vertex_index, size_t location) -> bool {

				const auto &target = polygon2[vertex_index];
				const auto index1 = location / 2;
				const auto index2 = (location + 1) / 2 % polygon1.size();

				if (location % 2 == 0) {
					const auto &vertex = polygon1[index1];
					const auto &[ray1, ray2] = get_cone(i, index1);
					return tpp::point_in_cone_plus(target, vertex, ray1, ray2);
				} else {

					const auto &v1 = polygon1[index1];
					const auto &v2 = polygon1[index2];
					const auto &ray1 = get_cone(i, index1).second;
					const auto &ray2 = get_cone(i, index2).first;

					return tpp::point_in_edge_plus(target, v1, v2, ray1, ray2);
				}
			};

			auto scan = [&](size_t start_location, size_t vertex_index, int shift) -> pair<size_t, size_t> {

				size_t location = start_location;
				size_t last_location = location;

				while (vertex_index < polygon2.size()) {

					if (point_in_location(vertex_index, location)) {
						build_cone(i + 1, vertex_index, query(polygon2[vertex_index], i + 1, location));
						vertex_index++;
						last_location = location;
					} else {
						
						location = (location + shift + 2 * polygon1.size()) % (2 * polygon1.size());

						if (location == start_location) {
							break;
						}
					}
				}

				return {last_location, vertex_index};
			};

			size_t last_location = location;
			size_t vertex_index = 1;

			std::tie(last_location, vertex_index) = scan(last_location, vertex_index, 1);
			std::tie(last_location, vertex_index) = scan(last_location, vertex_index, -1);
			std::tie(last_location, vertex_index) = scan(last_location, vertex_index, 1);
			std::tie(last_location, vertex_index) = scan(last_location, vertex_index, -1);
		}
	}
};

namespace tpp {

	std::vector<Vector2> tpp_convex_solve_tamc(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionTAMC(start, target, polygons).solve();
	}
}
