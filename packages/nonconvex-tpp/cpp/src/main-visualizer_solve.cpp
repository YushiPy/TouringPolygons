#include "common.h"
#include "tpp_convex.h"
#include "vector2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <ranges>
#include <tuple>
#include <vector>

using std::vector;

namespace {

bool is_ccw_turn(const Vector2 &p0, const Vector2 &p1, const Vector2 &p2) {
	return (p1 - p0).cross(p2 - p0) > 0;
}

vector<Vector2> half_hull(const vector<Vector2> &sorted_points) {
	vector<Vector2> hull;

	for (const auto &p : sorted_points) {
		while (hull.size() > 1 && !is_ccw_turn(hull[hull.size() - 2], hull[hull.size() - 1], p)) {
			hull.pop_back();
		}

		hull.push_back(p);
	}

	hull.pop_back();
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

bool is_convex(const vector<Vector2> &polygon) {
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

double path_length(const vector<Vector2> &path) {
	double length = 0.0;

	for (size_t i = 1; i < path.size(); i++) {
		length += path[i - 1].distance_to(path[i]);
	}

	return length;
}

vector<Vector2> sampled_vertex_path(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons) {
	if (polygons.empty()) {
		return {start, target};
	}

	vector<size_t> offsets(polygons.size() + 1, 0);
	for (size_t i = 0; i < polygons.size(); i++) {
		offsets[i + 1] = offsets[i] + polygons[i].size();
	}

	vector<double> dp(offsets.back(), std::numeric_limits<double>::infinity());
	vector<size_t> pred(offsets.back(), SIZE_MAX);

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

	vector<Vector2> path;
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
	vector<Vector2> path;
	bool exact = true;
	size_t calls = 0;
	double seconds = 0.0;
};

SolveResult solve_tpp(
	const Vector2 &start,
	const Vector2 &target,
	const vector<vector<Vector2>> &polygons,
	size_t max_calls,
	double max_seconds
) {
	const auto start_time = std::chrono::steady_clock::now();

	if (polygons.empty()) {
		return {{start, target}, true, 0, 0.0};
	}

	if (std::ranges::all_of(polygons, is_convex)) {
		return {tpp::tpp_convex_solve(start, target, polygons), true, 1, 0.0};
	}

	vector<vector<Vector2>> hulls;
	vector<vector<vector<Vector2>>> pieces;
	hulls.reserve(polygons.size());
	pieces.reserve(polygons.size());

	for (const auto &polygon : polygons) {
		hulls.push_back(convex_hull(polygon));
		pieces.push_back(is_convex(polygon) ? vector<vector<Vector2>>{polygon} : tpp::decompose_polygon(polygon));
	}

	vector<vector<Vector2>> selected;
	selected.reserve(polygons.size());
	for (const auto &polygon_pieces : pieces) {
		selected.push_back(polygon_pieces.front());
	}

	vector<Vector2> best_path;
	try {
		best_path = tpp::tpp_convex_solve(start, target, selected);
	} catch (...) {
		best_path = sampled_vertex_path(start, target, polygons);
	}

	double best_length = path_length(best_path);
	SolveResult result{best_path, true, 0, 0.0};

	auto elapsed = [&] {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
	};

	auto bound = [&](const vector<size_t> &instance) {
		vector<vector<Vector2>> input;
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

	std::queue<vector<size_t>> queue;
	queue.push({});

	while (!queue.empty()) {
		if (result.calls >= max_calls || elapsed() >= max_seconds) {
			result.exact = false;
			break;
		}

		auto current = std::move(queue.front());
		queue.pop();

		if (current.size() == polygons.size()) {
			vector<vector<Vector2>> input;
			input.reserve(polygons.size());

			for (size_t i = 0; i < current.size(); i++) {
				input.push_back(pieces[i][current[i]]);
			}

			result.calls++;
			const auto path = tpp::tpp_convex_solve(start, target, input);
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

Vector2 read_point() {
	double x = 0.0;
	double y = 0.0;
	std::cin >> x >> y;
	return {x, y};
}

void write_response(const SolveResult &result) {
	std::cout << std::setprecision(17);
	std::cout << "OK " << (result.exact ? 1 : 0) << ' ' << result.calls << ' ' << result.seconds << ' ' << result.path.size() << '\n';

	for (const auto &point : result.path) {
		std::cout << point.x << ' ' << point.y << '\n';
	}
}

} // namespace

int main() {
	try {
		const Vector2 start = read_point();
		const Vector2 target = read_point();

		size_t polygon_count = 0;
		size_t max_calls = 200000;
		double max_seconds = 3.0;
		std::cin >> polygon_count >> max_calls >> max_seconds;

		vector<vector<Vector2>> polygons(polygon_count);
		for (auto &polygon : polygons) {
			size_t vertex_count = 0;
			std::cin >> vertex_count;
			polygon.reserve(vertex_count);

			for (size_t i = 0; i < vertex_count; i++) {
				polygon.push_back(read_point());
			}
		}

		if (!std::cin) {
			throw std::runtime_error("Invalid input.");
		}

		write_response(solve_tpp(start, target, polygons, max_calls, max_seconds));
		return 0;
	} catch (const std::exception &error) {
		std::cout << "ERR " << error.what() << '\n';
		return 1;
	}
}
