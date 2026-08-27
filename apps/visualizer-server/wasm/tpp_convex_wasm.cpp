#include "common.h"
#include "tpp_convex.h"
#include "vector2.h"

#include <emscripten/emscripten.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

std::vector<double> last_path_points;
bool last_exact = true;
size_t last_calls = 0;
double last_seconds = 0.0;

std::vector<std::vector<Vector2>> read_polygons(const double *points, const int *polygon_sizes, int polygon_count) {
	std::vector<std::vector<Vector2>> polygons;
	polygons.reserve(static_cast<size_t>(polygon_count));

	size_t point_index = 0;
	for (int i = 0; i < polygon_count; i++) {
		std::vector<Vector2> polygon;
		polygon.reserve(static_cast<size_t>(polygon_sizes[i]));

		for (int j = 0; j < polygon_sizes[i]; j++) {
			polygon.push_back({points[2 * point_index], points[2 * point_index + 1]});
			point_index++;
		}

		polygons.push_back(std::move(polygon));
	}

	return polygons;
}

bool is_ccw_turn(const Vector2 &p0, const Vector2 &p1, const Vector2 &p2) {
	return (p1 - p0).cross(p2 - p0) > 0;
}

std::vector<Vector2> half_hull(const std::vector<Vector2> &sorted_points) {
	std::vector<Vector2> hull;

	for (const auto &p : sorted_points) {
		while (hull.size() > 1 && !is_ccw_turn(hull[hull.size() - 2], hull[hull.size() - 1], p)) {
			hull.pop_back();
		}

		hull.push_back(p);
	}

	hull.pop_back();
	return hull;
}

std::vector<Vector2> convex_hull(const std::vector<Vector2> &points) {
	std::vector<Vector2> sorted_points = points;

	std::sort(sorted_points.begin(), sorted_points.end(), [](const Vector2 &a, const Vector2 &b) {
		return std::tie(a.x, a.y) < std::tie(b.x, b.y);
	});

	std::vector<Vector2> lower = half_hull(sorted_points);
	std::vector<Vector2> upper = half_hull(std::vector<Vector2>(sorted_points.rbegin(), sorted_points.rend()));
	lower.insert(lower.end(), upper.begin(), upper.end());

	return lower;
}

bool is_convex(const std::vector<Vector2> &polygon) {
	bool has_positive = false;
	bool has_negative = false;

	for (size_t i = 0; i < polygon.size(); i++) {
		const auto &p0 = polygon[i];
		const auto &p1 = polygon[(i + 1) % polygon.size()];
		const auto &p2 = polygon[(i + 2) % polygon.size()];
		const double cross = (p1 - p0).cross(p2 - p1);

		if (cross > 1e-10) {
			has_positive = true;
		} else if (cross < -1e-10) {
			has_negative = true;
		}

		if (has_positive && has_negative) {
			return false;
		}
	}

	return true;
}

double path_length(const std::vector<Vector2> &path) {
	double length = 0.0;

	for (size_t i = 1; i < path.size(); i++) {
		length += path[i - 1].distance_to(path[i]);
	}

	return length;
}

std::vector<Vector2> sampled_vertex_path(
	const Vector2 &start,
	const Vector2 &target,
	const std::vector<std::vector<Vector2>> &polygons
) {
	if (polygons.empty()) {
		return {start, target};
	}

	std::vector<size_t> offsets(polygons.size() + 1, 0);
	for (size_t i = 0; i < polygons.size(); i++) {
		offsets[i + 1] = offsets[i] + polygons[i].size();
	}

	std::vector<double> dp(offsets.back(), std::numeric_limits<double>::infinity());
	std::vector<size_t> pred(offsets.back(), SIZE_MAX);

	for (size_t j = 0; j < polygons.front().size(); j++) {
		dp[j] = start.distance_to(polygons.front()[j]);
	}

	for (size_t i = 0; i + 1 < polygons.size(); i++) {
		for (size_t j = 0; j < polygons[i].size(); j++) {
			const size_t from_index = offsets[i] + j;
			for (size_t k = 0; k < polygons[i + 1].size(); k++) {
				const size_t to_index = offsets[i + 1] + k;
				const double candidate = dp[from_index] + polygons[i][j].distance_to(polygons[i + 1][k]);

				if (candidate < dp[to_index]) {
					dp[to_index] = candidate;
					pred[to_index] = from_index;
				}
			}
		}
	}

	double best = std::numeric_limits<double>::infinity();
	size_t current = SIZE_MAX;
	const size_t last_polygon = polygons.size() - 1;

	for (size_t j = 0; j < polygons.back().size(); j++) {
		const size_t index = offsets[last_polygon] + j;
		const double candidate = dp[index] + polygons.back()[j].distance_to(target);

		if (candidate < best) {
			best = candidate;
			current = index;
		}
	}

	if (current == SIZE_MAX) {
		throw std::runtime_error("No initial path found.");
	}

	std::vector<Vector2> path;
	path.push_back(target);

	size_t polygon_index = last_polygon;
	while (current != SIZE_MAX) {
		while (polygon_index > 0 && current < offsets[polygon_index]) {
			polygon_index--;
		}

		const size_t vertex_index = current - offsets[polygon_index];
		path.push_back(polygons[polygon_index][vertex_index]);
		current = pred[current];
	}

	path.push_back(start);
	std::reverse(path.begin(), path.end());
	return path;
}

struct SolveResult {
	std::vector<Vector2> path;
	bool exact = true;
	size_t calls = 0;
	double seconds = 0.0;
};

SolveResult solve_tpp(
	const Vector2 &start,
	const Vector2 &target,
	const std::vector<std::vector<Vector2>> &polygons,
	size_t max_calls,
	double max_seconds
) {
	const auto start_time = std::chrono::steady_clock::now();

	if (polygons.empty()) {
		return {{start, target}, true, 0, 0.0};
	}

	if (std::ranges::all_of(polygons, is_convex)) {
		return {tpp::tpp_convex_solve_binary_search_lazy(start, target, polygons), true, 1, 0.0};
	}

	std::vector<std::vector<Vector2>> hulls;
	std::vector<std::vector<std::vector<Vector2>>> pieces;
	hulls.reserve(polygons.size());
	pieces.reserve(polygons.size());

	for (const auto &polygon : polygons) {
		hulls.push_back(convex_hull(polygon));
		pieces.push_back(is_convex(polygon) ? std::vector<std::vector<Vector2>>{polygon} : tpp::decompose_polygon(polygon));
	}

	std::vector<Vector2> best_path;
	std::vector<std::vector<Vector2>> selected;
	selected.reserve(polygons.size());
	for (const auto &polygon_pieces : pieces) {
		selected.push_back(polygon_pieces.front());
	}

	try {
		best_path = tpp::tpp_convex_solve_binary_search_lazy(start, target, selected);
	} catch (...) {
		best_path = sampled_vertex_path(start, target, polygons);
	}

	double best_length = path_length(best_path);
	SolveResult result{best_path, true, 0, 0.0};

	auto elapsed = [&] {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
	};

	auto bound = [&](const std::vector<size_t> &instance) {
		std::vector<std::vector<Vector2>> input;
		input.reserve(polygons.size());

		for (size_t i = 0; i < instance.size(); i++) {
			input.push_back(pieces[i][instance[i]]);
		}

		for (size_t i = instance.size(); i < polygons.size(); i++) {
			input.push_back(hulls[i]);
		}

		result.calls++;
		return tpp::tpp_convex_solve_length_binary_search_lazy(start, target, input);
	};

	std::queue<std::vector<size_t>> queue;
	queue.push({});

	while (!queue.empty()) {
		if (result.calls >= max_calls || elapsed() >= max_seconds) {
			result.exact = false;
			break;
		}

		auto current = std::move(queue.front());
		queue.pop();

		if (current.size() == polygons.size()) {
			std::vector<std::vector<Vector2>> input;
			input.reserve(polygons.size());

			for (size_t i = 0; i < current.size(); i++) {
				input.push_back(pieces[i][current[i]]);
			}

			result.calls++;
			const auto path = tpp::tpp_convex_solve_binary_search_lazy(start, target, input);
			const double length = path_length(path);

			if (length < best_length) {
				best_length = length;
				result.path = path;
			}

			continue;
		}

		const size_t next_polygon = current.size();
		for (size_t i = 0; i < pieces[next_polygon].size(); i++) {
			auto next = current;
			next.push_back(i);

			if (bound(next) <= best_length) {
				queue.push(std::move(next));
			}
		}
	}

	result.seconds = elapsed();
	return result;
}

int store_result(const SolveResult &result) {
	last_path_points.clear();
	last_path_points.reserve(result.path.size() * 2);

	for (const auto &point : result.path) {
		last_path_points.push_back(point.x);
		last_path_points.push_back(point.y);
	}

	last_exact = result.exact;
	last_calls = result.calls;
	last_seconds = result.seconds;

	return static_cast<int>(result.path.size());
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int tpp_solve(
	double start_x,
	double start_y,
	double target_x,
	double target_y,
	const double *points,
	const int *polygon_sizes,
	int polygon_count,
	int max_calls,
	double max_seconds
) {
	try {
		return store_result(solve_tpp(
			{start_x, start_y},
			{target_x, target_y},
			read_polygons(points, polygon_sizes, polygon_count),
			static_cast<size_t>(max_calls),
			max_seconds
		));
	} catch (const std::exception &) {
		last_path_points.clear();
		return -1;
	} catch (...) {
		last_path_points.clear();
		return -1;
	}
}

EMSCRIPTEN_KEEPALIVE
int tpp_solve_convex(
	double start_x,
	double start_y,
	double target_x,
	double target_y,
	const double *points,
	const int *polygon_sizes,
	int polygon_count
) {
	return tpp_solve(start_x, start_y, target_x, target_y, points, polygon_sizes, polygon_count, 200000, 3.0);
}

EMSCRIPTEN_KEEPALIVE
const double *tpp_get_path_points() {
	return last_path_points.empty() ? nullptr : last_path_points.data();
}

EMSCRIPTEN_KEEPALIVE
int tpp_solution_exact() {
	return last_exact ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
int tpp_solution_calls() {
	return static_cast<int>(last_calls);
}

EMSCRIPTEN_KEEPALIVE
double tpp_solution_seconds() {
	return last_seconds;
}

}
