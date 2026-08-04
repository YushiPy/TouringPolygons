#include "common.h"
#include "tests.h"
#include "tpp_convex.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <numeric>
#include <print>
#include <string>
#include <tuple>
#include <vector>

using std::vector;

namespace {

struct BenchmarkOptions {
	std::string input_path = "tests/test_cases_simplified2.bin";
	size_t max_polygons = 40;
	size_t max_instances = 6;
	size_t max_calls_per_instance = 512;
	size_t max_branching = 6;
};

struct BoundCall {
	vector<vector<Vector2>> polygons;
	size_t selected_count = 0;
	size_t total_vertices = 0;
};

bool is_ccw_turn(const Vector2 &p0, const Vector2 &p1, const Vector2 &p2) {
	return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y) > 0;
}

vector<Vector2> half_hull(const vector<Vector2> &sorted_points) {

	vector<Vector2> hull;
	hull.reserve(sorted_points.size());

	for (const auto &p : sorted_points) {
		while (hull.size() > 1 && !is_ccw_turn(hull[hull.size() - 2], hull[hull.size() - 1], p)) {
			hull.pop_back();
		}

		hull.push_back(p);
	}

	if (!hull.empty()) {
		hull.pop_back();
	}

	return hull;
}

vector<Vector2> convex_hull(const vector<Vector2> &points) {

	vector<Vector2> sorted_points = points;

	std::sort(sorted_points.begin(), sorted_points.end(), [](const Vector2 &a, const Vector2 &b) {
		return std::tie(a.x, a.y) < std::tie(b.x, b.y);
	});

	vector<Vector2> lower = half_hull(sorted_points);
	vector<Vector2> upper = half_hull(vector<Vector2>(sorted_points.rbegin(), sorted_points.rend()));

	lower.reserve(lower.size() + upper.size());
	lower.insert(lower.end(), upper.begin(), upper.end());

	return lower;
}

double path_length(const Vector2 &start, const Vector2 &target, const vector<Vector2> &path) {

	if (path.empty()) {
		return start.distance_to(target);
	}

	double length = start.distance_to(path.front());

	for (size_t i = 1; i < path.size(); i++) {
		length += path[i - 1].distance_to(path[i]);
	}

	length += path.back().distance_to(target);

	return length;
}

size_t vertex_count(const vector<vector<Vector2>> &polygons) {
	return std::accumulate(polygons.begin(), polygons.end(), static_cast<size_t>(0), [](size_t sum, const auto &polygon) {
		return sum + polygon.size();
	});
}

double combination_count(const vector<vector<vector<Vector2>>> &convex_pieces) {

	double combinations = 1.0;

	for (const auto &pieces : convex_pieces) {
		combinations *= static_cast<double>(pieces.size());
	}

	return combinations;
}

void append_bound_call(
	vector<BoundCall> &calls,
	const vector<size_t> &selected,
	const vector<vector<vector<Vector2>>> &convex_pieces,
	const vector<vector<Vector2>> &convex_hulls,
	size_t max_calls
) {

	if (calls.size() >= max_calls) {
		return;
	}

	BoundCall call;
	call.selected_count = selected.size();
	call.polygons.reserve(convex_hulls.size());

	for (size_t i = 0; i < selected.size(); i++) {
		call.polygons.push_back(convex_pieces[i][selected[i]]);
	}

	for (size_t i = selected.size(); i < convex_hulls.size(); i++) {
		call.polygons.push_back(convex_hulls[i]);
	}

	call.total_vertices = vertex_count(call.polygons);
	calls.push_back(std::move(call));
}

vector<BoundCall> make_bound_calls(
	const vector<vector<vector<Vector2>>> &convex_pieces,
	const vector<vector<Vector2>> &convex_hulls,
	size_t max_calls,
	size_t max_branching
) {

	vector<BoundCall> calls;
	calls.reserve(max_calls);

	vector<vector<size_t>> stack;
	stack.push_back({});

	while (!stack.empty() && calls.size() < max_calls) {
		vector<size_t> selected = std::move(stack.back());
		stack.pop_back();

		append_bound_call(calls, selected, convex_pieces, convex_hulls, max_calls);

		if (selected.size() == convex_pieces.size()) {
			continue;
		}

		const size_t next_polygon = selected.size();
		const size_t branch_count = std::min(max_branching, convex_pieces[next_polygon].size());

		for (size_t i = branch_count; i > 0; i--) {
			vector<size_t> child = selected;
			child.push_back(i - 1);
			stack.push_back(std::move(child));
		}
	}

	return calls;
}

BenchmarkOptions parse_options(int argc, char **argv) {

	BenchmarkOptions options;

	if (argc > 1) {
		options.input_path = argv[1];
	}

	if (argc > 2) {
		options.max_polygons = std::stoull(argv[2]);
	}

	if (argc > 3) {
		options.max_instances = std::stoull(argv[3]);
	}

	if (argc > 4) {
		options.max_calls_per_instance = std::stoull(argv[4]);
	}

	if (argc > 5) {
		options.max_branching = std::stoull(argv[5]);
	}

	return options;
}

} // namespace

int main(int argc, char **argv) {

	const BenchmarkOptions options = parse_options(argc, argv);
	const auto test_cases = tpp::load_test_cases(options.input_path);

	std::println(
		"source,case_index,polygons,decomposed_pieces,total_combinations,calls,mean_selected,total_vertices_min,total_vertices_max,seconds_per_call,checksum"
	);

	size_t benchmarked_instances = 0;

	for (size_t case_index = 0; case_index < test_cases.size() && benchmarked_instances < options.max_instances; case_index++) {

		const auto &[start, target, raw_polygons, _] = test_cases[case_index];

		if (raw_polygons.empty() || raw_polygons.size() > options.max_polygons) {
			continue;
		}

		vector<vector<Vector2>> polygons;
		polygons.reserve(raw_polygons.size());

		for (const auto &polygon : raw_polygons) {
			polygons.push_back(tpp::remove_collinear_points(polygon));
		}

		vector<vector<Vector2>> convex_hulls;
		vector<vector<vector<Vector2>>> convex_pieces;
		convex_hulls.reserve(polygons.size());
		convex_pieces.reserve(polygons.size());

		try {
			for (const auto &polygon : polygons) {
				convex_hulls.push_back(convex_hull(polygon));
				convex_pieces.push_back(tpp::decompose_polygon(polygon));
			}
		} catch (const std::exception &error) {
			std::println(stderr, "Skipping case {}: decomposition failed: {}", case_index, error.what());
			continue;
		}

		const vector<BoundCall> calls = make_bound_calls(
			convex_pieces,
			convex_hulls,
			options.max_calls_per_instance,
			options.max_branching
		);

		if (calls.empty()) {
			continue;
		}

		size_t total_pieces = 0;
		size_t selected_sum = 0;
		size_t min_vertices = std::numeric_limits<size_t>::max();
		size_t max_vertices = 0;
		double checksum = 0.0;

		for (const auto &pieces : convex_pieces) {
			total_pieces += pieces.size();
		}

		const auto start_time = std::chrono::steady_clock::now();

		for (const auto &call : calls) {
			try {
				const vector<Vector2> path = tpp::tpp_convex_solve(start, target, call.polygons);
				checksum += path_length(start, target, path);
			} catch (...) {
				checksum += std::numeric_limits<double>::infinity();
			}

			selected_sum += call.selected_count;
			min_vertices = std::min(min_vertices, call.total_vertices);
			max_vertices = std::max(max_vertices, call.total_vertices);
		}

		const auto end_time = std::chrono::steady_clock::now();
		const double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
		const double seconds_per_call = elapsed_seconds / static_cast<double>(calls.size());
		const double mean_selected = static_cast<double>(selected_sum) / static_cast<double>(calls.size());

		std::println(
			"nonconvex_bnb_cgal,{},{},{},{:.0f},{},{:.3f},{},{},{:.12f},{:.12f}",
			case_index,
			polygons.size(),
			total_pieces,
			combination_count(convex_pieces),
			calls.size(),
			mean_selected,
			min_vertices,
			max_vertices,
			seconds_per_call,
			checksum
		);

		benchmarked_instances++;
	}
}
