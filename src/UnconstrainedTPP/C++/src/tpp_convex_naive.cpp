
#include "vector2.h"
#include "common.h"

using std::vector;
using std::pair;

class Solution {

	public:

	const Vector2 &start;
	const Vector2 &target;
	const vector<vector<Vector2>> &polygons;

	vector<vector<bool>> first_contact;
	vector<vector<pair<Vector2, Vector2>>> cones;

	Solution(const Vector2& start, const Vector2& target, const vector<vector<Vector2>>& polygons) : start(start), target(target), polygons(polygons), first_contact(), cones() {}

	/*
	Returns `cones[i][j] = (ray1, ray2)`. If the rays haven't been computed yet, computes the cone of vertex `j` in polygon `i` 
	and stores it in `cones[i][j]` before returning it. It also uses this value to update the `first_contact` array.
	
	`cones` should be initialized with `(Vector2.NAN, Vector2.NAN)` for all vertices before calling
	any of the `locate_point` or `query` functions, and should only
	be accessed through this function to ensure that the cones are properly computed and cached.
	*/
	pair<Vector2, Vector2> &get_cone(size_t i, size_t j) {

		if (cones[i][j].first.is_nan()) {
			
			auto j_prev = (j - 1 + polygons[i].size()) % polygons[i].size();
			auto j_next = (j + 1) % polygons[i].size();

			const auto &polygon = polygons[i];
			const auto &_first_contact = first_contact[i];

			const auto before = polygon[j_prev];
			const auto vertex = polygon[j];
			const auto after = polygon[j_next];

			const auto last = query(vertex, i);
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

		return cones[i][j];
	}

	/*
	Locates `point` in the shortest last step map of `i (0 to k)` 
	and returns the index of the region as follows:
	- `2n` if the point is in the region of vertex `n`
	- `2n + 1` if the point is between vertices `n` and `n + 1`.
	- `-1` if the point is in the pass through region.
	*/
	int64_t locate_point(const Vector2& point, size_t i) {

		const auto &polygon = polygons[i - 1];
		const auto &_first_contact = first_contact[i - 1];
		const auto &_cones = cones[i - 1];

		for (size_t j = 0; j < polygon.size(); j++) {

			size_t prev = (j + polygon.size() - 1) % polygon.size();

			if (!_first_contact[j] && !_first_contact[prev]) {
				continue;
			}

			const auto &v = polygon[j];
			const auto &[ray1, ray2] = _cones[j];

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

			const auto &ray1 = _cones[j].second;
			const auto &ray2 = _cones[(j + 1) % _cones.size()].first;

			if (tpp::point_in_edge(point, v1, ray1, v2, ray2)) {
				return 2 * j + 1;
			}
		}

		return -1;
	}

	/*
	Returns the `i`-path to `point`, i.e. the shortest path from
	`start` that visits all polygons from `1` to `i` (inclusive) in order and ends at `point`.

	The last point, i.e. `point` is not included in the returned path, 
	is it the caller's responsibility to add it to the path if needed.
	
	The function should be called with an empty `accumulator` vector, 
	and the path will be constructed in the `accumulator` vector.
	*/
	void query_full(const Vector2 &point, size_t i, vector<Vector2> &accumulator) {
		
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

	/*
	Returns the `i`-path to `point`, i.e. the shortest path from
	`start` that visits all polygons from `1` to `i` (inclusive) in order and ends at `point`.
	*/
	vector<Vector2> query_full(const Vector2& point, size_t i) {

		vector<Vector2> path;
		query_full(point, i, path);
		path.push_back(point);

		return path;
	}

	/*
	Returns the last step of the `i`-path to `point`, i.e. the point that precedes
	`point` in the shortest path from `start` that visits all polygons from `1` to `i` (inclusive) 
	in order and ends at `point`.
	*/
	Vector2 query(const Vector2& point, size_t i) {

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

		if (!intersection.is_finite()) {
			throw std::runtime_error(
				std::format("Intersection not found for point {} in polygon {} at edge {}", point, i, vertex_index)
			);
		}

		return intersection;
	}

	/*
	Returns the shortest path from `start` to `target` that visits all polygons in `polygons` in order.
	*/
	vector<Vector2> solve() {

		first_contact.resize(polygons.size());
		cones.resize(polygons.size());

		for (size_t i = 0; i < polygons.size(); i++) {
			first_contact[i].resize(polygons[i].size(), false);
			cones[i].resize(polygons[i].size(), {Vector2::NaN, Vector2::NaN});
		}

		for (size_t i = 0; i < polygons.size(); i++) {
			for (size_t j = 0; j < polygons[i].size(); j++) {
				get_cone(i, j); // Computes the cones[i][j]
			}
		}

		return query_full(target, polygons.size());
	}
};

namespace tpp {

	std::vector<Vector2> tpp_convex_solve(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		
		Solution solution(start, target, polygons);
		auto path = solution.solve();
		path = tpp::remove_collinear_points(path);

		return path;
	}
}
