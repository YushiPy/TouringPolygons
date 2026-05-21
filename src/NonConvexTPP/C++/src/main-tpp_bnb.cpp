
#include "vector2.h"
#include "tests.h"
#include "common.h"
#include "tpp_convex.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>
#include <CGAL/point_generators_2.h>

#include <vector>
#include <utility>
#include <queue>

#include <chrono>
#include <print>

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

		result.push_back(std::move(piece));
	}

	return result;
}

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


bool segment_segment_intersection(const Vector2 &p1, const Vector2 &p2, const Vector2 &q1, const Vector2 &q2) {

	const Vector2 s1 = p2 - p1;
	const Vector2 s2 = q2 - q1;

	const double denominator = s1.cross(s2);

	if (std::abs(denominator) < 1e-10) {
		return false; // Lines are parallel or collinear, treat as no intersection for simplicity
	}

	const Vector2 delta = q1 - p1;

	const double s = delta.cross(s2) / denominator;
	const double t = delta.cross(s1) / denominator;

	return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
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

bool segment_intersects_polygon(const Vector2 &p1, const Vector2 &p2, const vector<Vector2> &polygon) {

	if (point_in_polygon(p1, polygon) || point_in_polygon(p2, polygon)) {
		return true;
	}

	for (size_t i = 0; i < polygon.size(); i++) {
		
		const Vector2 &q1 = polygon[i];
		const Vector2 &q2 = polygon[(i + 1) % polygon.size()];

		if (segment_segment_intersection(p1, p2, q1, q2)) {
			return true;
		}
	}

	return false;
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
		
		for (size_t i = from_polygon_index; i < to_polygon_index; i++) {
			if (!segment_intersects_polygon(from, to, polygons[i])) {
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
		convex_pieces.push_back(decompose_polygon(polygon));
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

	vector<vector<Vector2>> selected_pieces;

	while (current_polygon < polygons.size() && current_path + 1 < full_path.size()) {

		const auto &polygon = polygons[current_polygon];

		const Vector2 &prev = full_path[current_path - 1];
		const Vector2 &curr = full_path[current_path];
		const Vector2 &next = full_path[current_path + 1];

		auto &pieces = convex_pieces[current_polygon];

		if (point_on_boundary(curr, polygon)) {

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

			selected_pieces.push_back(pieces[0]);

			current_polygon++;
			current_path++;

			continue;
		}

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
	best_path = tpp::tpp_convex_solve(start, target, selected_pieces);

	double minimal_length = path_length(best_path);
	vector<Vector2> minimal_path = best_path;

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

		if (current.size() == polygons.size()) {

			vector<vector<Vector2>> instance;
			instance.reserve(polygons.size());

			for (size_t i = 0; i < polygons.size(); i++) {
				instance.push_back(convex_pieces[i][current[i]]);
			}

			vector<Vector2> path;

			try {
				path = tpp::tpp_convex_solve(start, target, instance);
			} catch (...) {
				continue;
			}

			double length = path_length(path);

			if (length < minimal_length) {
				minimal_length = length;
				minimal_path = std::move(path);
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

	return best_path;
}


int main() {
	
	Vector2 start(-2.0, 0.177198062845644);
	Vector2 target(10.0, -10.0);
	
	vector<vector<Vector2>> polygons = {

		{{0.4497354497354502, 4.761904761904762}, {-1.851851851851852, 2.5793650793650795}, {-0.5423280423280419, 3.32010582010582}, {0.3306878306878307, 3.373015873015873}, {0.846560846560847, 2.552910052910053}, {-1.0, 1.0}, {2.0, 2.0}}, 
		{{4.969135802469138, -3.1481481481481484}, {3.9021164021164005, -2.2751322751322762}, {3.505291005291004, -3.240740740740742}, {1.970899470899469, -4.232804232804234}, {0.6084656084656066, -4.034391534391536}, {1.7217813051146398, -5.02300914134991}}, 
		{{-1.7900939934870728, -5.1503424152681445}, {-4.797306307390922, -6.105278916062125}, {-6.405620413991311, -6.281188271471543}, {-5.928152163594321, -4.522094717377367}, {-3.2895118324530586, -3.7681974799084355}, {-7.9594863867745005, -1.588442056267219}, {-7.959486386774497, -8.712242774269072}}, 
		{{-4.353344600881444, 4.926750658899912}, {-9.521728995529566, 3.1425271968901045}, {-9.722768258854616, 4.273373053093503}, {-8.46627286307306, 5.705777804284474}, {-6.80769894064141, 5.152919830140591}, {-10.522736994168872, 8.488651017900835}, {-10.522736994168868, 1.3648502998989844}}, 
		{{17.65625045019001, 7.99160221179721}, {16.13829923570195, 10.530476367002898}, {19.64436354527171, 10.973771854419763}, {16.702493492414327, 13.149949701738926}, {14.16361933720864, 10.40957759770739}, {15.493505799459237, 7.508007134615176}, {10.053061181161336, 5.573626825887033}, {8.844073488206247, 7.185610416493819}, {8.118680872433194, 9.522986622873656}, {9.126170616562435, 12.102160367844515}, {7.762701162840861, 13.703645556089068}, {7.762701162840866, 2.2795588675053517}}, 
		{{23.016095888957572, -6.717748052489705}, {19.201068057854844, -7.32224189896725}, {15.735303338050254, -9.65961810534709}, {14.16361933720864, -7.765537386384116}, {13.96212138838279, -5.6699587185952955}, {15.856202107345764, -3.372882101980626}, {13.122546601608423, -1.0057047081978485}, {13.122546601608427, -12.429791396781566}}
	};

	const auto start_time = std::chrono::high_resolution_clock::now();
	auto solution = tpp_solve(start, target, polygons);
	const auto end_time = std::chrono::high_resolution_clock::now();
	const double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
	std::println("Solution found in {} seconds", elapsed_seconds);

	tpp::plot_solution(start, target, polygons, solution);
}