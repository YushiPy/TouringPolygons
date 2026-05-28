
#include "vector2.h"
#include "tests.h"
#include "common.h"
#include "tpp_convex.h"

#include <vector>
#include <utility>
#include <queue>
#include <algorithm>

#include <chrono>
#include <print>

using std::vector;

vector<Vector2> tpp_approximation(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons) {

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


Vector2 segment_segment_intersection(const Vector2 &p1, const Vector2 &p2, const Vector2 &q1, const Vector2 &q2) {

	const Vector2 s1 = p2 - p1;
	const Vector2 s2 = q2 - q1;

	const double denominator = s1.cross(s2);

	if (std::abs(denominator) < 1e-10) {
		return Vector2::INF; // Lines are parallel or collinear, treat as no intersection for simplicity
	}

	const Vector2 delta = q1 - p1;

	const double s = delta.cross(s2) / denominator;
	const double t = delta.cross(s1) / denominator;

	if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
		return p1 + s1 * s; // Intersection point
	} else {
		return Vector2::INF; // No intersection
	}
}

bool point_in_polygon(const Vector2 &point, const vector<Vector2> &polygon) {

	bool inside = false;

	for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {

		const Vector2 &pi = polygon[i];
		const Vector2 &pj = polygon[j];

		if ((pi.y > point.y) != (pj.y > point.y) &&
			point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x) {
			inside = !inside;
		}
	}

	return inside;
}

Vector2 segment_intersects_polygon(const Vector2 &p1, const Vector2 &p2, const vector<Vector2> &polygon) {

	if (point_in_polygon(p1, polygon)) {
		return p1;
	}

	Vector2 solution = Vector2::INF;

	for (size_t i = 0; i < polygon.size(); i++) {
		
		const Vector2 &q1 = polygon[i];
		const Vector2 &q2 = polygon[(i + 1) % polygon.size()];

		const Vector2 intersection = segment_segment_intersection(p1, p2, q1, q2);

		if (intersection.is_finite() && (solution == Vector2::INF || p1.distance_to(intersection) < p1.distance_to(solution))) {
			solution = intersection;
		}
	}

	return solution;
}


vector<Vector2> tpp_approximation2(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons) {

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

	auto is_valid_edge = [&](const Vector2 &from, const Vector2 &to, size_t from_polygon_index, size_t to_polygon_index) {
		
		Vector2 segment_start = from;

		for (size_t i = from_polygon_index; i < to_polygon_index; i++) {

			Vector2 intersection = segment_intersects_polygon(segment_start, to, polygons[i]);

			if (intersection.is_finite()) {
				segment_start = intersection;
			} else {
				return false;
			}
		}

		return true;
	};

	if (is_valid_edge(start, target, 0, polygons.size())) {
		return {};
	}

	for (size_t i = 1; i < polygons.size(); i++) {

		const auto &new_polygon = polygons[i];

		for (size_t j = 0; j < new_polygon.size(); j++) {

			const Vector2 &current_vertex = new_polygon[j];
			const size_t current_index = accumulated_sizes[i] + j;
			
			if (is_valid_edge(start, current_vertex, 0, i)) {
				dp[current_index] = start.distance_to(current_vertex);
				predecessors[current_index] = SIZE_MAX; // Start is the predecessor
			}
		}

		for (size_t j = 0; j < i; j++) {

			const auto &old_polygon = polygons[j];

			for (size_t k = 0; k < old_polygon.size(); k++) {

				const Vector2 &current_vertex = old_polygon[k];
				const size_t current_index = accumulated_sizes[j] + k;
				const double current_cost = dp[current_index];

				for (size_t l = 0; l < new_polygon.size(); l++) {

					const Vector2 &next_vertex = new_polygon[l];
					const size_t next_index = accumulated_sizes[i] + l;

					if (is_valid_edge(current_vertex, next_vertex, j + 1, i)) {

						const double new_cost = current_cost + current_vertex.distance_to(next_vertex);

						if (new_cost < dp[next_index]) {
							dp[next_index] = new_cost;
							predecessors[next_index] = current_index;
						}
					}
				}
			}
		}
	}

	double best_target_distance = std::numeric_limits<double>::infinity();
	size_t best_target_predecessor = SIZE_MAX;

	for (size_t i = 0; i < polygons.size(); i++) {
		for (size_t j = 0; j < polygons[i].size(); j++) {

			const Vector2 &current_vertex = polygons[i][j];
			const size_t current_index = accumulated_sizes[i] + j;
			const double current_cost = dp[current_index];

			if (is_valid_edge(current_vertex, target, i + 1, polygons.size())) {

				const double target_distance = current_cost + current_vertex.distance_to(target);

				if (target_distance < best_target_distance) {
					best_target_distance = target_distance;
					best_target_predecessor = current_index;
				}
			}
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


bool is_ccw_turn(const Vector2 &p0, const Vector2 &p1, const Vector2 &p2) {
	return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y) > 0;
}

vector<Vector2> half_hull(const vector<Vector2> &sorted_points) {

	vector<Vector2> hull;

	for (const auto &p : sorted_points) {
		
		while (hull.size() > 1 && !is_ccw_turn(hull[hull.size() - 2], hull[hull.size() - 1], p)) {
			hull.pop_back();
		}
		
		hull.push_back(p);
	}

	hull.pop_back(); // Remove last point to avoid duplication in full hull

	return hull;
}

vector<Vector2> convex_hull(const vector<Vector2> &points) {

	vector<Vector2> sorted_points = points;
	
	std::sort(sorted_points.begin(), sorted_points.end(), [](const Vector2 &a, const Vector2 &b) {
		return std::tie(a.x, a.y) < std::tie(b.x, b.y);
	});

	vector<Vector2> lower = half_hull(sorted_points);
	vector<Vector2> upper = half_hull(vector<Vector2>(sorted_points.rbegin(), sorted_points.rend()));

	lower.insert(lower.end(), upper.begin(), upper.end());

	return lower;
}


Vector2 project_point_on_polygon(const Vector2 &point, const vector<Vector2> &polygon) {

	Vector2 closest_point;
	double closest_distance = std::numeric_limits<double>::infinity();

	for (size_t i = 0; i < polygon.size(); i++) {

		const Vector2 &a = polygon[i];
		const Vector2 &b = polygon[(i + 1) % polygon.size()];

		const Vector2 ab = b - a;
		const Vector2 ap = point - a;

		double t = ap.dot(ab) / ab.dot(ab);
		t = std::clamp(t, 0.0, 1.0);

		Vector2 projection = a + t * ab;
		double distance = projection.distance_to(point);

		if (distance < closest_distance) {
			closest_distance = distance;
			closest_point = projection;
		}
	}

	return closest_point;
}

vector<Vector2> tpp_solve(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons) {

	auto path_length = [&](const vector<Vector2> &path) {
		
		if (path.empty()) {
			return start.distance_to(target);
		}

		double length = start.distance_to(path[0]);

		for (size_t i = 1; i < path.size(); i++) {
			length += path[i - 1].distance_to(path[i]);
		}

		length += path.back().distance_to(target);

		return length;
	};

	if (polygons.empty()) {
		return {};
	}

	vector<vector<Vector2>> convex_hulls;
	vector<vector<vector<Vector2>>> convex_pieces;

	convex_hulls.reserve(polygons.size());
	convex_pieces.reserve(polygons.size());

	for (const auto &polygon : polygons) {
		convex_hulls.push_back(convex_hull(polygon));
		convex_pieces.push_back(tpp::decompose_polygon(polygon));
	}

	vector<Vector2> best_path = tpp_approximation2(start, target, convex_hulls);

	vector<Vector2> full_path;
	full_path.reserve(best_path.size() + 2);

	full_path.push_back(start);

	for (const auto &p : best_path) {
		full_path.push_back(p);
	}

	full_path.push_back(target);

	auto point_on_boundary = [](const Vector2 &p, const vector<Vector2> &polygon) {

		constexpr double EPS = 1e-9;

		for (size_t i = 0; i < polygon.size(); i++) {

			const auto &a = polygon[i];
			const auto &b = polygon[(i + 1) % polygon.size()];

			Vector2 ab = b - a;
			Vector2 ap = p - a;

			if (std::abs(ab.cross(ap)) > EPS) {
				continue;
			}

			double dot = ap.dot(ab);

			if (dot < -EPS) {
				continue;
			}

			if (dot > ab.dot(ab) + EPS) {
				continue;
			}

			return true;
		}

		return false;
	};

	auto segment_intersects_polygon = [](const Vector2 &a, const Vector2 &b, const vector<Vector2> &polygon) {

		for (size_t i = 0; i < polygon.size(); i++) {

			const auto &p1 = polygon[i];
			const auto &p2 = polygon[(i + 1) % polygon.size()];

			if (tpp::segment_segment_intersection(a, b, p1, p2).is_finite()) {
				return true;
			}
		}

		return false;
	};

	size_t current_polygon = 0;
	size_t current_path = 1;

	while (current_polygon < polygons.size() && current_path + 1 < full_path.size()) {

		const Vector2 &curr = full_path[current_path];

		while (point_on_boundary(curr, polygons[current_polygon])) {
			
			auto &pieces = convex_pieces[current_polygon];

			std::sort(
				pieces.begin(),
				pieces.end(),
				[&](const vector<Vector2> &a, const vector<Vector2> &b) {

					double da =
						project_point_on_polygon(curr, a)
						.distance_squared_to(curr);

					double db =
						project_point_on_polygon(curr, b)
						.distance_squared_to(curr);

					return da < db;
				}
			);

			current_polygon++;

			if (current_polygon >= polygons.size()) {
				break;
			}
		}

		auto &pieces = convex_pieces[current_polygon];

		const Vector2 &prev = full_path[current_path - 1];
		const Vector2 &next = full_path[current_path + 1];

		vector<vector<Vector2>> intersecting;
		vector<vector<Vector2>> non_intersecting;

		for (const auto &piece : pieces) {

			if (segment_intersects_polygon(prev, next, piece)) {
				intersecting.push_back(piece);
			} else {
				non_intersecting.push_back(piece);
			}
		}

		pieces.clear();

		for (auto &piece : intersecting) {
			pieces.push_back(std::move(piece));
		}

		for (auto &piece : non_intersecting) {
			pieces.push_back(std::move(piece));
		}

		current_polygon++;
	}
	
	// Initial solution from the heuristic ordering.
	vector<vector<Vector2>> selected_pieces;

	for (const auto &pieces : convex_pieces) {
		selected_pieces.push_back(pieces.front());
	}

	best_path = tpp::tpp_convex_solve(start, target, selected_pieces);

	double minimal_length = path_length(best_path);
	vector<Vector2> minimal_path = best_path;

	size_t count = 0;
	size_t changes = 0;

	auto bound = [&](const vector<size_t> &instance) {

		vector<vector<Vector2>> input_polygons;
		input_polygons.reserve(polygons.size());

		for (size_t i = 0; i < instance.size(); i++) {
			input_polygons.push_back(convex_pieces[i][instance[i]]);
		}

		for (size_t i = instance.size(); i < polygons.size(); i++) {
			input_polygons.push_back(convex_hulls[i]);
		}

		vector<Vector2> path;

		try {
			path = tpp::tpp_convex_solve(start, target, input_polygons);
			count++;
		} catch (...) {
			return std::numeric_limits<double>::infinity();
		}

		return path_length(path);
	};

	std::vector<std::vector<size_t>> queue;
	queue.push_back({});

	while (!queue.empty()) {

		vector<size_t> current = std::move(queue.back());
		queue.pop_back();
		
		// std::println("{}", current);

		if (current.size() == polygons.size()) {

			vector<vector<Vector2>> instance;
			instance.reserve(polygons.size());

			for (size_t i = 0; i < polygons.size(); i++) {
				instance.push_back(convex_pieces[i][current[i]]);
			}

			vector<Vector2> path;

			try {
				path = tpp::tpp_convex_solve(start, target, instance);
				count++;
			} catch (...) {
				continue;
			}

			double length = path_length(path);

			if (length < minimal_length) {
				minimal_length = length;
				minimal_path = std::move(path);
				changes++;
				std::println("New best length: {} ({} changes)", minimal_length, changes);
			}

			continue;
		}

		size_t next_polygon_index = current.size();

		for (size_t i = 0; i < convex_pieces[next_polygon_index].size(); i++) {

			vector<size_t> selected = current;
			selected.push_back(i);

			if (bound(selected) > minimal_length) {
				continue;
			}

			queue.push_back(std::move(selected));
		}
	}

	best_path = std::move(minimal_path);

	best_path.insert(best_path.begin(), start);
	best_path.push_back(target);

	
	size_t prod = 1;
	
	for (const auto &pieces : convex_pieces) {
		prod *= pieces.size();
	}
	
	std::println("Total convex calls: {}", count);
	std::println("Total combinations: {}", prod);
	std::println("Ratio: {}", static_cast<double>(prod) / count);

	return best_path;
}


int main() {
	
	const auto test_cases = tpp::load_test_cases("tests/test_cases_simplified2.bin");
	auto [start, target, polygons, _] = test_cases[0];
	polygons = vector<vector<Vector2>>(polygons.begin(), polygons.begin() + 25); // Limit to first 10 polygons for testing

	std::println("{}", polygons.size());

	//std::println("Vector2 start = {};", start);
	//std::println("Vector2 target = {};", target);
	//std::println("std::vector<std::vector<Vector2>> polygons = {};", polygons);

	const auto start_time = std::chrono::high_resolution_clock::now();
	auto solution = tpp_solve(start, target, polygons);
	// auto solution = tpp_solve(start, target, limited_polygons);
	const auto end_time = std::chrono::high_resolution_clock::now();
	const double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
	std::println("Solution found in {} seconds", elapsed_seconds);

	tpp::plot_solution(start, target, polygons, solution);
}