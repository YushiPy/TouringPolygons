#include "common.h"
#include "tpp/convex/solver.h"
#include "tpp/nonconvex/solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <tuple>

namespace {

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

	double signed_area(const std::vector<Vector2> &polygon) {
		double area = 0.0;

		for (size_t i = 0; i < polygon.size(); i++) {
			const auto &current = polygon[i];
			const auto &next = polygon[(i + 1) % polygon.size()];
			area += current.cross(next);
		}

		return area / 2.0;
	}

	std::vector<Vector2> counter_clockwise_polygon(const std::vector<Vector2> &polygon) {
		std::vector<Vector2> normalized = polygon;

		if (signed_area(normalized) < 0.0) {
			std::reverse(normalized.begin(), normalized.end());
		}

		return normalized;
	}

	std::vector<std::vector<Vector2>> counter_clockwise_polygons(
		const std::vector<std::vector<Vector2>> &polygons
	) {
		std::vector<std::vector<Vector2>> normalized;
		normalized.reserve(polygons.size());

		for (const auto &polygon : polygons) {
			normalized.push_back(counter_clockwise_polygon(polygon));
		}

		return normalized;
	}

	std::vector<std::vector<std::vector<Vector2>>> counter_clockwise_piece_groups(
		const std::vector<std::vector<std::vector<Vector2>>> &pieces
	) {
		std::vector<std::vector<std::vector<Vector2>>> normalized;
		normalized.reserve(pieces.size());

		for (const auto &polygon_pieces : pieces) {
			normalized.push_back(counter_clockwise_polygons(polygon_pieces));
		}

		return normalized;
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

	std::vector<Vector2> convex_hull_of_pieces(const std::vector<std::vector<Vector2>> &pieces) {
		std::vector<Vector2> vertices;

		for (const auto &piece : pieces) {
			vertices.insert(vertices.end(), piece.begin(), piece.end());
		}

		return convex_hull(vertices);
	}

	std::vector<std::vector<Vector2>> first_piece_instance(
		const std::vector<std::vector<std::vector<Vector2>>> &pieces
	) {
		std::vector<std::vector<Vector2>> selected;
		selected.reserve(pieces.size());

		for (const auto &polygon_pieces : pieces) {
			if (polygon_pieces.empty()) {
				throw std::runtime_error("Every polygon must have at least one convex piece.");
			}

			selected.push_back(polygon_pieces.front());
		}

		return selected;
	}

}

namespace tpp {

	NonconvexTppSolveResult tpp_nonconvex_solve(
		const Vector2 &start,
		const Vector2 &target,
		const std::vector<std::vector<Vector2>> &polygons,
		const NonconvexTppSolveOptions &options
	) {
		if (polygons.empty()) {
			return {{start, target}, true, 0, 0.0};
		}

		if (std::ranges::all_of(polygons, is_convex)) {
			const auto normalized_polygons = counter_clockwise_polygons(polygons);
			return {tpp_convex_solve_binary_search_lazy(start, target, normalized_polygons), true, 1, 0.0};
		}

		std::vector<std::vector<std::vector<Vector2>>> pieces;
		pieces.reserve(polygons.size());

		for (const auto &polygon : polygons) {
			pieces.push_back(is_convex(polygon) ? std::vector<std::vector<Vector2>>{polygon} : decompose_polygon(polygon));
		}

		return tpp_nonconvex_solve_decomposed(start, target, pieces, options);
	}

	NonconvexTppSolveResult tpp_nonconvex_solve_decomposed(
		const Vector2 &start,
		const Vector2 &target,
		const std::vector<std::vector<std::vector<Vector2>>> &pieces,
		const NonconvexTppSolveOptions &options
	) {
		const auto start_time = std::chrono::steady_clock::now();

		if (pieces.empty()) {
			return {{start, target}, true, 0, 0.0};
		}

		const auto normalized_pieces = counter_clockwise_piece_groups(pieces);

		std::vector<std::vector<Vector2>> hulls;
		hulls.reserve(normalized_pieces.size());

		for (const auto &polygon_pieces : normalized_pieces) {
			if (polygon_pieces.empty()) {
				throw std::runtime_error("Every polygon must have at least one convex piece.");
			}

			hulls.push_back(polygon_pieces.size() == 1 ? polygon_pieces.front() : convex_hull_of_pieces(polygon_pieces));
		}

		std::vector<Vector2> best_path;
		const auto selected = first_piece_instance(normalized_pieces);

		try {
			best_path = tpp_convex_solve_binary_search_lazy(start, target, selected);
		} catch (...) {
			best_path = sampled_vertex_path(start, target, hulls);
		}

		double best_length = path_length(best_path);
		NonconvexTppSolveResult result{best_path, true, 0, 0.0};

		auto elapsed = [&] {
			return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
		};

		auto bound = [&](const std::vector<size_t> &instance) {
			std::vector<std::vector<Vector2>> input;
			input.reserve(normalized_pieces.size());

			for (size_t i = 0; i < instance.size(); i++) {
				input.push_back(normalized_pieces[i][instance[i]]);
			}

			for (size_t i = instance.size(); i < normalized_pieces.size(); i++) {
				input.push_back(hulls[i]);
			}

			result.calls++;
			return tpp_convex_solve_length_binary_search_lazy(start, target, input);
		};

		std::queue<std::vector<size_t>> queue;
		queue.push({});

		while (!queue.empty()) {
			if (result.calls >= options.max_calls || elapsed() >= options.max_seconds) {
				result.exact = false;
				break;
			}

			auto current = std::move(queue.front());
			queue.pop();

			if (current.size() == normalized_pieces.size()) {
				std::vector<std::vector<Vector2>> input;
				input.reserve(normalized_pieces.size());

				for (size_t i = 0; i < current.size(); i++) {
					input.push_back(normalized_pieces[i][current[i]]);
				}

				result.calls++;
				const auto path = tpp_convex_solve_binary_search_lazy(start, target, input);
				const double length = path_length(path);

				if (length < best_length) {
					best_length = length;
					result.path = path;
				}

				continue;
			}

			const size_t next_polygon = current.size();
			for (size_t i = 0; i < normalized_pieces[next_polygon].size(); i++) {
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

}
