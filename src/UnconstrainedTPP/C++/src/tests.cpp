
#include "vector2.h"
#include "common.h"
#include "tests.h"

#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <random>

using std::vector;
using std::tuple;

vector<Vector2> regular_polygon(size_t n, const Vector2 &center, double radius) {

	vector<Vector2> vertices;

	for (size_t i = 0; i < n; i++) {
		double angle = 2.0 * M_PI * i / n;
		auto vertex = Vector2::from_angle(angle) * radius + center;
		vertices.push_back(vertex);
	}

	return vertices;
}

bool segment_polygon_intersection(const Vector2 &p1, const Vector2 &p2, const vector<Vector2> &polygon, Vector2 *intersection) {

	for (size_t i = 0; i < polygon.size(); i++) {
		const auto &v1 = polygon[i];
		const auto &v2 = polygon[(i + 1) % polygon.size()];

		auto current_intersection = tpp::segment_segment_intersection(p1, p2, v1, v2);

		if (current_intersection.is_finite()) {
			
			if (intersection != nullptr) {
				*intersection = current_intersection;
			}

			return true;
		}
	}

	return false;
}


namespace tpp {

	auto rng = std::default_random_engine {};

	tuple<Vector2, Vector2, vector<vector<Vector2>>> generate_random_test(const vector<size_t> &polygon_sizes) {

		const size_t height = std::ceil(std::sqrt(polygon_sizes.size()));
		const size_t width = std::ceil((double) polygon_sizes.size() / height);

		vector<size_t> indeces(height * width);
		
		for (size_t i = 0; i < indeces.size(); i++) {
			indeces[i] = i;
		}

		// Shuffle the indeces to randomize the polygon positions
		
		std::ranges::shuffle(indeces, rng);

		vector<vector<Vector2>> polygons;

		for (size_t i = 0; i < polygon_sizes.size(); i++) {
			
			size_t index = indeces[i];
			size_t row = index / width;
			size_t col = index % width;

			Vector2 center(col, row);
			double radius = (double) rng() / rng.max() * 0.4 + 0.1; // Random radius between 0.1 and 0.5

			polygons.push_back(regular_polygon(polygon_sizes[i], center, radius));
		}

		Vector2 start(-0.5, -0.5);
		Vector2 target(width - 0.5, height - 0.5);

		return {start, target, polygons};
	}

	bool is_valid_solution(const Vector2 &start, const Vector2 & target, const vector<vector<Vector2>> &polygons, const vector<Vector2> &solution) {

		// A correct solution must start at `start`, end at `target`, have no consecutive collinear points, and visit the polygons in order without skipping any.
		if (solution.size() < 2) {
			return false;
		}

		if (!solution.front().is_equal_approx(start) || !solution.back().is_equal_approx(target)) {
			return false;
		}

		if (solution.size() > 2) {

			for (size_t i = 1; i < solution.size() - 1; i++) {

				const auto &p1 = solution[i - 1];
				const auto &p2 = solution[i];
				const auto &p3 = solution[i + 1];

				const auto &d1 = p2 - p1;
				const auto &d2 = p3 - p2;

				if (d1.is_same_direction(d2)) {
					return false;
				}
			}
		}

		size_t polygon_index = 0;
		size_t path_index = 1;

		// Tracks whether the current segment has visited at least one polygon. 
		// Every segment must visit at least one polygon, but it can visit more than one.
		bool segment_visits_a_polygon = false;

		while (polygon_index < polygons.size() && path_index < solution.size()) {

			const auto &polygon = polygons[polygon_index];

			const auto &point = solution[path_index];
			const auto &previous_point = solution[path_index - 1];

			bool is_on_vertex = false;

			// Verify if the point is on a vertex and if it is a valid bend.
			for (size_t i = 0; i < polygon.size(); i++) {
				
				const auto &vertex = polygon[i];

				if (!point.is_equal_approx(vertex)) {
					continue;
				}

				is_on_vertex = true;

				// If the target point is on a vertex, it is valid.
				if (path_index == solution.size() - 1) {
					return true;
				}

				const auto &before = polygon[(i + polygon.size() - 1) % polygon.size()];
				const auto &after = polygon[(i + 1) % polygon.size()];

				const auto last = previous_point;
				const auto diff = vertex - last;

				auto ray1 = diff.reflect(vertex - before);
				auto ray2 = diff.reflect(vertex - after);

				if (diff.cross(vertex - before) >= 0) {
					ray1 = diff;
				}

				if (diff.cross(vertex - after) <= 0) {
					ray2 = diff;
				}

				const auto &next_point = solution[path_index + 1];

				if (!tpp::point_in_cone_plus(next_point, vertex, ray1, ray2)) {
					return false;
				}

				break;
			}

			if (is_on_vertex) {
				segment_visits_a_polygon = true;
				polygon_index++;
				continue;
			}

			bool is_on_edge = false;

			// Verify if the point is on an edge and if it follows the reflection rule.
			for (size_t i = 0; i < polygon.size(); i++) {
				
				const auto &v1 = polygon[i];
				const auto &v2 = polygon[(i + 1) % polygon.size()];
				
				// Verify if the point is on the edge
				if (!(point - v1).is_same_direction(v2 - v1) || (point - v2).dot(v1 - v2) <= 0) {
					continue;
				}

				is_on_edge = true;

				// If the target point is on an edge, it is valid if it follows the reflection rule.
				if (path_index == solution.size() - 1) {
					return true;
				}

				const auto &next_point = solution[path_index + 1];
				const auto reflected = next_point.reflect_line(v1, v2);

				const auto d1 = reflected - point;
				const auto d2 = point - previous_point;

				// We verify the reflection rule
				if (!d1.is_same_direction(d2)) {
					return false;
				}

				break;
			}

			if (is_on_edge) {
				segment_visits_a_polygon = true;
				polygon_index++;
				continue;
			}

			bool crosses_polygon = false;

			for (size_t i = 0; i < polygon.size(); i++) {
				
				const auto &v1 = polygon[i];
				const auto &v2 = polygon[(i + 1) % polygon.size()];

				if (tpp::segment_segment_intersection(previous_point, point, v1, v2).is_finite()) {
					crosses_polygon = true;
					break;
				}
			}

			if (crosses_polygon) {
				segment_visits_a_polygon = true;
				polygon_index++;
				continue;
			}

			if (!segment_visits_a_polygon) {
				return false;
			}

			path_index++;

		}

		return true;
	}

}
