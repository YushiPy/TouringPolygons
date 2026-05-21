
#include "vector2.h"
#include "tests.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>
#include <CGAL/point_generators_2.h>

#include <vector>
#include <utility>
#include <queue>

using K      = CGAL::Exact_predicates_inexact_constructions_kernel;
using Traits = CGAL::Partition_traits_2<K>;
using Poly   = Traits::Polygon_2;
using Point  = Traits::Point_2;
using PolyList = std::list<Poly>;

using std::vector;

vector<vector<Vector2>> decompose_polygon(const vector<Vector2> &polygon) {

	vector<Point> points;

	for (const auto &v : polygon) {
		points.emplace_back(v.x, v.y);
	}

	Poly poly(points.begin(), points.end());

	if (poly.orientation() == CGAL::CLOCKWISE) {
		poly.reverse_orientation();
	}

	// Ensure the polygon is simple (no self-intersections)
	assert(poly.is_simple());

	PolyList pieces;
	CGAL::optimal_convex_partition_2(poly.vertices_begin(), poly.vertices_end(), std::back_inserter(pieces));

	// Validate the partitioning
	assert(CGAL::convex_partition_is_valid_2(
		poly.vertices_begin(), poly.vertices_end(),
		pieces.begin(), pieces.end()
	));

	vector<vector<Vector2>> result;

	for (const auto& p : pieces) {
	
		vector<Vector2> piece;
	
		for (auto v = p.vertices_begin(); v != p.vertices_end(); ++v) {
			piece.emplace_back(v->x(), v->y());
		}
	}

	return result;
}

vector<Vector2> tpp_astar3(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons) {

	if (polygons.empty()) {
		return {};
	}

	// Precompute accumulated sizes for indexing into a single array
	vector<size_t> accumulated_sizes(polygons.size() + 1, 0);

	for (size_t i = 0; i < polygons.size(); i++) {
		accumulated_sizes[i + 1] = accumulated_sizes[i] + polygons[i].size();
	}

	const size_t vertex_count = accumulated_sizes.back();

	vector<double> dp(vertex_count, std::numeric_limits<double>::infinity());
	vector<size_t> predecessors(vertex_count, SIZE_MAX);

	// Explore start point to first polygon
	for (size_t j = 0; j < polygons[0].size(); j++) {
		dp[j] = start.distance_to(polygons[0][j]);
	};

	// For each polygon[i], visit each vertex of polygon[i + 1]
	for (size_t i = 0; i < polygons.size() - 1; i++) {
		for (size_t j = 0; j < polygons[i].size(); j++) {

			const Vector2 &current_vertex = polygons[i][j];
			const size_t current_index = accumulated_sizes[i] + j;
			const double current_cost = dp[current_index];

			for (size_t k = 0; k < polygons[i + 1].size(); k++) {

				const double new_cost = current_cost + current_vertex.distance_to(polygons[i + 1][k]);
				const size_t next_index = accumulated_sizes[i + 1] + k;

				if (new_cost < dp[next_index]) {
					dp[next_index] = new_cost;
					predecessors[next_index] = current_index;
				}
			}
		}
	}

	double best_target_distance = std::numeric_limits<double>::infinity();
	size_t best_target_predecessor = SIZE_MAX;

	// For each vertex in the last polygon, try reaching target
	for (size_t j = 0; j < polygons.back().size(); j++) {

		const Vector2 &current_vertex = polygons.back()[j];
		const size_t current_index = accumulated_sizes[polygons.size() - 1] + j;
		const double current_cost = dp[current_index];

		const double target_distance = current_cost + current_vertex.distance_to(target);

		if (target_distance < best_target_distance) {
			best_target_distance = target_distance;
			best_target_predecessor = current_index;
		}
	}

	// This should never happen since the problem guarantees a solution.
	// However, it may be possible if some points in the input are infinite or NaN, so we check just in case.
	if (best_target_predecessor == SIZE_MAX) {
		throw std::runtime_error("No path found");
	}

	// Reconstruct path (without start/target)
	vector<Vector2> path;
	size_t current = best_target_predecessor;
	size_t polygon_index = polygons.size() - 1;

	while (current != SIZE_MAX) {

		while (polygon_index > 0 && current < accumulated_sizes[polygon_index]) {
			polygon_index--;
		}

		const size_t vertex_index = current - accumulated_sizes[polygon_index];
		const Vector2 &vertex = polygons[polygon_index][vertex_index];

		path.push_back(vertex);
		current = predecessors[current];
	}

	std::reverse(path.begin(), path.end());

	return path;
}



int main() {
	
	const auto &[_start, _target, polygons] = tpp::generate_test_bad({3, 3, 3, 6, 5, 10});
	const auto start = _start + Vector2(0.0, 0.5); // Perturb start to avoid numerical issues
	const auto target = _target + Vector2(0.0, -0.5); // Perturb target to avoid numerical issues

	auto solution = tpp_astar3(start, target, polygons);

	solution.insert(solution.begin(), start);
	solution.push_back(target);

	tpp::plot_solution(start, target, polygons, solution);
}