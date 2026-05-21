
#include "vector2.h"
#include "common.h"
#include "tpp_convex.h"

using std::vector;
using std::pair;


class Solution {

	public:

	const Vector2 &start;
	const Vector2 &target;
	const vector<vector<Vector2>> &polygons;

	vector<vector<bool>> first_contact;
	vector<vector<pair<Vector2, Vector2>>> cones;

	Solution(const Vector2& start, const Vector2& target, const vector<vector<Vector2>>& polygons) : start(start), target(target), polygons(polygons) {}

	void build_cone(size_t i, size_t j) {
		
		auto j_prev = (j - 1 + polygons[i].size()) % polygons[i].size();
		auto j_next = (j + 1) % polygons[i].size();

		const auto &polygon = polygons[i];
		const auto &_first_contact = first_contact[i];

		const auto before = polygon[j_prev];
		const auto vertex = polygon[j];
		const auto after = polygon[j_next];

		const auto last = query(polygons[i][j], i);
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

	pair<Vector2, Vector2> &get_cone(size_t i, size_t j) {

		if (cones[i][j].first.is_nan()) {
			build_cone(i, j);
		}

		return cones[i][j];
	}

	void query_full(const Vector2 &point, size_t i, vector<Vector2> &accumulator) {

		if (i == 0) return;

		const auto &polygon = polygons[i - 1];
		auto location = locate_point(point, i);
		auto vertex_index = location / 2;

		if (location == -1) {
			query_full(point, i - 1, accumulator);
		} else if (location % 2 == 0) {

			const auto &vertex = polygon[vertex_index];
			query_full(vertex, i - 1, accumulator);
			accumulator.push_back(vertex);

		} else {

			const auto &v1 = polygon[vertex_index];
			const auto &v2 = polygon[(vertex_index + 1) % polygon.size()];

			const auto &reflected = point.reflect_line(v1, v2);

			query_full(reflected, i - 1, accumulator);

			const auto &last = accumulator.empty() ? start : accumulator.back();
			const auto &intersection = tpp::segment_segment_intersection(last, reflected, v1, v2);

			if (!intersection.is_finite()) {
				throw std::runtime_error(
					std::format("Intersection not found for point {} in polygon {} at edge {}", point, i, vertex_index)
				);
			}

			accumulator.push_back(intersection);
		}
	}

	vector<Vector2> query_full(const Vector2& point, size_t i) {

		std::vector<Vector2> path;
		query_full(point, i, path);

		return path;
	}

	Vector2 query(const Vector2& point, size_t i) {

		if (i == 0) {
			return start;
		}
		
		const auto &polygon = polygons[i - 1];
		auto location = locate_point(point, i);
		auto vertex_index = location / 2;

		if (location == -1) {
			return query(point, i - 1);
		} else if (location % 2 == 0) {
			return polygon[vertex_index];
		} else {
			const auto &v1 = polygon[vertex_index];
			const auto &v2 = polygon[(vertex_index + 1) % polygon.size()];

			const auto reflected = point.reflect_line(v1, v2);
			const auto last = query(reflected, i - 1);
			const auto intersection = tpp::segment_segment_intersection(last, reflected, v1, v2);

			return intersection;
		}
	}

	std::vector<Vector2> solve() {

		first_contact.resize(polygons.size());
		cones.resize(polygons.size());
		
		for (size_t i = 0; i < polygons.size(); i++) {
			first_contact[i].resize(polygons[i].size(), false);
			cones[i].resize(polygons[i].size(), {Vector2::NaN, Vector2::NaN});
		}

		return tpp::remove_collinear_points(query_full(target, polygons.size()));
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

	int64_t locate_point(const Vector2& point, size_t i) {

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

	vector<Vector2> tpp_convex_solve(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons) {
		Solution solution(start, target, polygons);
		return solution.solve();
	}
}
