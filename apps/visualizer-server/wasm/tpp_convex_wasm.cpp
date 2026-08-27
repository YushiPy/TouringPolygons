#include "tpp/nonconvex/solver.h"
#include "vector2.h"

#include <emscripten/emscripten.h>

#include <cstddef>
#include <exception>
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

	std::vector<std::vector<std::vector<Vector2>>> read_piece_groups(
		const double *points,
		const int *piece_sizes,
		const int *group_sizes,
		int group_count
	) {
		std::vector<std::vector<std::vector<Vector2>>> groups;
		groups.reserve(static_cast<size_t>(group_count));

		size_t point_index = 0;
		size_t piece_index = 0;
		for (int i = 0; i < group_count; i++) {
			std::vector<std::vector<Vector2>> group;
			group.reserve(static_cast<size_t>(group_sizes[i]));

			for (int j = 0; j < group_sizes[i]; j++) {
				std::vector<Vector2> piece;
				piece.reserve(static_cast<size_t>(piece_sizes[piece_index]));

				for (int k = 0; k < piece_sizes[piece_index]; k++) {
					piece.push_back({points[2 * point_index], points[2 * point_index + 1]});
					point_index++;
				}

				group.push_back(std::move(piece));
				piece_index++;
			}

			groups.push_back(std::move(group));
		}

		return groups;
	}

	int store_result(const tpp::NonconvexTppSolveResult &result) {
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

}

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
		return store_result(tpp::tpp_nonconvex_solve(
			{start_x, start_y},
			{target_x, target_y},
			read_polygons(points, polygon_sizes, polygon_count),
			{static_cast<size_t>(max_calls), max_seconds}
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
int tpp_solve_piece_groups(
	double start_x,
	double start_y,
	double target_x,
	double target_y,
	const double *points,
	const int *piece_sizes,
	const int *group_sizes,
	int group_count,
	int max_calls,
	double max_seconds
) {
	try {
		return store_result(tpp::tpp_nonconvex_solve_decomposed(
			{start_x, start_y},
			{target_x, target_y},
			read_piece_groups(points, piece_sizes, group_sizes, group_count),
			{static_cast<size_t>(max_calls), max_seconds}
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
