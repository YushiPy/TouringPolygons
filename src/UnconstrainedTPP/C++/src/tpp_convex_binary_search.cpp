
#include "vector2.h"
#include "common.h"
#include "tpp_convex_common.h"
#include "tpp_convex_binary_search.h"

using std::vector;
using std::pair;

class SolutionBinarySearch : public tpp::Solution {

	using tpp::Solution::Solution;

	/*
	Uses binary search to locate `point` in the visibility map of `polygon[i]`.
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
};

namespace tpp {

	std::vector<Vector2> tpp_convex_solve_binary_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionBinarySearch(start, target, polygons).solve();
	}
}
