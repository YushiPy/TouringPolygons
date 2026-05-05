
#include "vector2.h"
#include "common.h"
#include "tpp_convex_common.h"
#include "tpp_convex_linear_search.h"

using std::vector;
using std::pair;

class SolutionLinearSearch : public tpp::Solution {

	using tpp::Solution::Solution;

	int64_t locate_point(const Vector2& point, size_t i) override {

		const auto &polygon = polygons[i - 1];
		const auto &_first_contact = first_contact[i - 1];

		for (size_t j = 0; j < polygon.size(); j++) {

			const auto &v = polygon[j];
			const auto &[ray1, ray2] = get_cone(i - 1, j);

			size_t prev = (j + polygon.size() - 1) % polygon.size();

			if (!_first_contact[j] && !_first_contact[prev]) {
				continue;
			}

			if (tpp::point_in_cone(point, v, ray1, ray2)) {
				return 2 * j;
			}
		}

		for (size_t j = 0; j < polygon.size(); j++) {

			if (!_first_contact[j]) {
				continue;
			}

			const auto &v1 = polygon[j];
			const auto &v2 = polygon[(j + 1) % polygon.size()];

			const auto &ray1 = get_cone(i - 1, j).second;
			const auto &ray2 = get_cone(i - 1, (j + 1) % cones[i - 1].size()).first;

			if (tpp::point_in_edge(point, v1, ray1, v2, ray2)) {
				return 2 * j + 1;
			}
		}

		return -1;
	}
};

namespace tpp {

	std::vector<Vector2> tpp_convex_solve_linear_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionLinearSearch(start, target, polygons).solve();
	}
}
