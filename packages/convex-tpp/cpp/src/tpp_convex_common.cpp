
#include "vector2.h"
#include "common.h"
#include "tpp_convex_common.h"

using std::vector;
using std::pair;

namespace tpp {

	Solution::Solution(const Vector2& start, const Vector2& target, const vector<vector<Vector2>>& polygons) : start(start), target(target), polygons(polygons), first_contact(), cones() {}

	void Solution::build_cone(size_t i, size_t j, const Vector2 &last) {
		
		auto j_prev = (j - 1 + polygons[i].size()) % polygons[i].size();
		auto j_next = (j + 1) % polygons[i].size();

		const auto &polygon = polygons[i];
		const auto &_first_contact = first_contact[i];

		const auto before = polygon[j_prev];
		const auto vertex = polygon[j];
		const auto after = polygon[j_next];

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

	void Solution::build_cone(size_t i, size_t j) {
		const auto last = query(polygons[i][j], i);
		build_cone(i, j, last);
	}

	pair<Vector2, Vector2>& Solution::get_cone(size_t i, size_t j) {

		if (cones[i][j].first.is_nan()) {
			build_cone(i, j);
		}

		return cones[i][j];
	}

	void Solution::query_full(const Vector2 &point, size_t i, vector<Vector2> &accumulator) {
		
		if (i == 0) {
			accumulator.push_back(start);
			return;
		}

		auto location = locate_point(point, i);

		if (location == -1) {
			query_full(point, i - 1, accumulator);
			return;
		}

		const auto &polygon = polygons[i - 1];
		auto vertex_index = location / 2;

		if (location % 2 == 0) {
			const auto &vertex = polygon[vertex_index];
			query_full(vertex, i - 1, accumulator);
			accumulator.push_back(vertex);
			return;
		}

		const auto &v1 = polygon[vertex_index];
		const auto &v2 = polygon[(vertex_index + 1) % polygon.size()];

		const auto &reflected = point.reflect_line(v1, v2);

		query_full(reflected, i - 1, accumulator);

		const auto &last = accumulator.back();
		const auto &intersection = tpp::segment_segment_intersection(last, reflected, v1, v2);

		if (!intersection.is_finite()) {
			throw std::runtime_error(
				std::format("Intersection not found for point {} in polygon {} at edge {}", point, i, vertex_index)
			);
		}

		accumulator.push_back(intersection);

		return;
	}

	vector<Vector2> Solution::query_full(const Vector2& point, size_t i) {

		vector<Vector2> path;
		query_full(point, i, path);
		path.push_back(point);

		return path;
	}

	Vector2 Solution::query(const Vector2& point, size_t i) {

		if (i == 0) {
			return start;
		}

		auto location = locate_point(point, i);

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
	}
	
	vector<Vector2> Solution::solve() {
		
		first_contact.resize(polygons.size());
		cones.resize(polygons.size());
		
		for (size_t i = 0; i < polygons.size(); i++) {
			first_contact[i].resize(polygons[i].size(), false);
			cones[i].resize(polygons[i].size(), {Vector2::NaN, Vector2::NaN});
		}

		preload_cones();

		return tpp::remove_collinear_points(query_full(target, polygons.size()));
	}
}
