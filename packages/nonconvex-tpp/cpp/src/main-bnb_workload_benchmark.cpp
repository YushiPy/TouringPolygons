#include "common.h"
#include "tests.h"
#include "tpp_convex.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

using std::vector;

namespace {

constexpr char NUMBER_SEPARATOR = '_';
constexpr size_t PROGRESS_BAR_WIDTH = 30;
constexpr size_t BRANCH_BUCKET_COUNT = 5;

struct BenchmarkOptions {
	std::string input_path = "tests/test_cases_simplified2.bin";
	size_t max_polygons = 40;
	size_t max_instances = 6;
	size_t max_calls_per_instance = 512;
	size_t max_branching = 6;
	size_t repeat_count = 1;
	size_t thread_count = 0;
	std::optional<std::string> output_path;
	std::optional<std::string> summary_output_path;
};

struct BoundCall {
	vector<vector<Vector2>> polygons;
	size_t selected_count = 0;
	size_t total_vertices = 0;
};

struct BoundCallSet {
	vector<BoundCall> calls;
	bool exhausted = false;
	bool branch_limited = false;
	size_t max_observed_branching = 0;
};

struct BenchmarkSummary {
	size_t total_instances = 0;
	size_t skipped_empty = 0;
	size_t skipped_max_polygons = 0;
	size_t skipped_decomposition = 0;
	size_t skipped_no_calls = 0;
	size_t benchmarked_instances = 0;
	size_t fully_covered_instances = 0;
	size_t capped_by_calls_instances = 0;
	size_t branch_limited_instances = 0;
	size_t total_calls = 0;
	size_t total_incumbent_solves = 0;
	size_t total_bound_solves = 0;
	size_t total_leaf_solves = 0;
	size_t total_visited_nodes = 0;
	size_t total_pruned_nodes = 0;
	size_t total_best_updates = 0;
	size_t max_observed_branching = 0;
	double decomposition_seconds = 0.0;
	double approximation_seconds = 0.0;
	double bnb_seconds = 0.0;
	double solver_seconds = 0.0;
	double checksum = 0.0;
};

std::string format_count(size_t value) {

	std::string digits = std::to_string(value);
	std::string formatted;
	formatted.reserve(digits.size() + digits.size() / 3);

	const size_t first_group_size = digits.size() % 3 == 0 ? 3 : digits.size() % 3;

	for (size_t i = 0; i < digits.size(); i++) {
		if (i != 0 && i >= first_group_size && (i - first_group_size) % 3 == 0) {
			formatted.push_back(NUMBER_SEPARATOR);
		}

		formatted.push_back(digits[i]);
	}

	return formatted;
}

std::string format_percent(double part, double total) {
	if (total <= 0.0 || !std::isfinite(total)) {
		return "0.00%";
	}

	return std::format("{:.2f}%", part / total * 100.0);
}

std::string format_seconds_with_percent(double seconds, double total_seconds) {
	return std::format("{:.6f}s ({})", seconds, format_percent(seconds, total_seconds));
}

std::string progress_bar(size_t current, size_t total) {

	if (total == std::numeric_limits<size_t>::max()) {
		return std::format("{} calls", format_count(current));
	}

	const double ratio = total == 0 ? 1.0 : std::min(1.0, static_cast<double>(current) / static_cast<double>(total));
	const size_t filled = static_cast<size_t>(ratio * static_cast<double>(PROGRESS_BAR_WIDTH));
	std::string bar;
	bar.reserve(PROGRESS_BAR_WIDTH);

	for (size_t i = 0; i < PROGRESS_BAR_WIDTH; i++) {
		bar.push_back(i < filled ? '#' : '-');
	}

	return std::format("[{}] {} / {} ({})", bar, format_count(current), format_count(total), format_percent(current, total));
}

void increment_histogram(vector<size_t> &histogram, size_t index) {
	if (histogram.size() <= index) {
		histogram.resize(index + 1, 0);
	}

	histogram[index]++;
}

size_t branching_bucket(size_t branching) {
	if (branching <= 1) {
		return 0;
	}

	if (branching == 2) {
		return 1;
	}

	if (branching <= 5) {
		return 2;
	}

	if (branching <= 10) {
		return 3;
	}

	return 4;
}

struct BranchAndBoundResult {
	size_t convex_calls = 0;
	size_t incumbent_solves = 0;
	size_t bound_solves = 0;
	size_t leaf_solves = 0;
	size_t visited_nodes = 0;
	size_t pruned_nodes = 0;
	size_t best_updates = 0;
	size_t selected_sum = 0;
	size_t min_vertices = std::numeric_limits<size_t>::max();
	size_t max_vertices = 0;
	size_t max_observed_branching = 0;
	bool exhausted = true;
	bool branch_limited = false;
	double initial_length = std::numeric_limits<double>::infinity();
	double incumbent_length = std::numeric_limits<double>::infinity();
	double final_length = std::numeric_limits<double>::infinity();
	double solver_seconds = 0.0;
	double incumbent_solver_seconds = 0.0;
	double bound_solver_seconds = 0.0;
	double leaf_solver_seconds = 0.0;
	size_t failed_prune_count = 0;
	double failed_prune_ratio_sum = 0.0;
	double failed_prune_gap_sum = 0.0;
	double failed_prune_depth_sum = 0.0;
	vector<size_t> visited_depth_histogram;
	vector<size_t> bound_depth_histogram;
	vector<size_t> leaf_depth_histogram;
	std::array<size_t, BRANCH_BUCKET_COUNT> branching_histogram = {};
	double checksum = 0.0;
};

struct InstanceRecord {
	size_t case_index = 0;
	size_t repeat_index = 0;
	size_t polygons = 0;
	size_t decomposed_pieces = 0;
	double total_combinations = 0.0;
	size_t calls = 0;
	size_t incumbent_solves = 0;
	size_t bound_solves = 0;
	size_t leaf_solves = 0;
	size_t visited_nodes = 0;
	size_t pruned_nodes = 0;
	size_t best_updates = 0;
	double mean_selected = 0.0;
	size_t total_vertices_min = 0;
	size_t total_vertices_max = 0;
	double initial_length = 0.0;
	double incumbent_length = 0.0;
	double final_length = 0.0;
	double decomposition_seconds = 0.0;
	double approximation_seconds = 0.0;
	double bnb_seconds = 0.0;
	double solver_seconds = 0.0;
	double incumbent_solver_seconds = 0.0;
	double bound_solver_seconds = 0.0;
	double leaf_solver_seconds = 0.0;
	double seconds_per_call = 0.0;
	bool exhausted = false;
	bool branch_limited = false;
	size_t max_observed_branching = 0;
	size_t failed_prune_count = 0;
	double failed_prune_ratio_mean = 0.0;
	double failed_prune_gap_mean = 0.0;
	double failed_prune_depth_mean = 0.0;
	double instance_seconds = 0.0;
	double checksum = 0.0;
};

struct CaseBenchmarkResult {
	size_t case_index = 0;
	bool decomposed = false;
	size_t skipped_decomposition = 0;
	size_t skipped_no_calls = 0;
	std::string decomposition_error;
	vector<InstanceRecord> records;
	vector<size_t> visited_depth_histogram;
	vector<size_t> bound_depth_histogram;
	vector<size_t> leaf_depth_histogram;
	std::array<size_t, BRANCH_BUCKET_COUNT> branching_histogram = {};
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

vector<Vector2> make_interpolated_polygon(const vector<Vector2> &polygon, size_t points_per_edge) {

	vector<Vector2> interpolated;
	interpolated.reserve(polygon.size() * points_per_edge);

	for (size_t i = 0; i < polygon.size(); i++) {
		const auto &a = polygon[i];
		const auto &b = polygon[(i + 1) % polygon.size()];

		for (size_t j = 0; j < points_per_edge; j++) {
			const double weight = static_cast<double>(j) / static_cast<double>(points_per_edge);
			interpolated.push_back(a.lerp(b, weight));
		}
	}

	return interpolated;
}

vector<Vector2> tpp_approximation(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons) {

	if (polygons.empty()) {
		return {};
	}

	vector<size_t> accumulated_sizes(polygons.size() + 1, 0);

	for (size_t i = 0; i < polygons.size(); i++) {
		accumulated_sizes[i + 1] = accumulated_sizes[i] + polygons[i].size();
	}

	const size_t point_count = accumulated_sizes.back();
	vector<double> dp(point_count, std::numeric_limits<double>::infinity());
	vector<size_t> predecessors(point_count, SIZE_MAX);

	for (size_t j = 0; j < polygons.front().size(); j++) {
		dp[j] = start.distance_to(polygons.front()[j]);
	}

	for (size_t i = 0; i + 1 < polygons.size(); i++) {
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

	for (size_t j = 0; j < polygons.back().size(); j++) {
		const Vector2 &current_vertex = polygons.back()[j];
		const size_t current_index = accumulated_sizes[polygons.size() - 1] + j;
		const double target_distance = dp[current_index] + current_vertex.distance_to(target);

		if (target_distance < best_target_distance) {
			best_target_distance = target_distance;
			best_target_predecessor = current_index;
		}
	}

	if (best_target_predecessor == SIZE_MAX) {
		return {};
	}

	vector<Vector2> path;
	size_t current = best_target_predecessor;
	size_t polygon_index = polygons.size() - 1;

	while (current != SIZE_MAX) {
		while (polygon_index > 0 && current < accumulated_sizes[polygon_index]) {
			polygon_index--;
		}

		const size_t vertex_index = current - accumulated_sizes[polygon_index];
		path.push_back(polygons[polygon_index][vertex_index]);
		current = predecessors[current];
	}

	std::reverse(path.begin(), path.end());

	return path;
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

double log2_combination_count(const InstanceRecord &record) {
	if (record.total_combinations <= 0.0) {
		return 0.0;
	}

	return std::log2(record.total_combinations);
}

double pieces_per_polygon(const InstanceRecord &record) {
	return record.polygons == 0 ? 0.0 : static_cast<double>(record.decomposed_pieces) / static_cast<double>(record.polygons);
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

		const Vector2 projection = a + ab * t;
		const double distance = projection.distance_squared_to(point);

		if (distance < closest_distance) {
			closest_distance = distance;
			closest_point = projection;
		}
	}

	return closest_point;
}

bool point_on_boundary(const Vector2 &point, const vector<Vector2> &polygon) {

	constexpr double EPS = 1e-9;

	for (size_t i = 0; i < polygon.size(); i++) {
		const auto &a = polygon[i];
		const auto &b = polygon[(i + 1) % polygon.size()];
		const Vector2 ab = b - a;
		const Vector2 ap = point - a;

		if (std::abs(ab.cross(ap)) > EPS) {
			continue;
		}

		const double dot = ab.dot(ap);

		if (dot >= -EPS && dot <= ab.dot(ab) + EPS) {
			return true;
		}
	}

	return false;
}

bool segment_intersects_polygon(const Vector2 &a, const Vector2 &b, const vector<Vector2> &polygon) {

	for (size_t i = 0; i < polygon.size(); i++) {
		const auto &p1 = polygon[i];
		const auto &p2 = polygon[(i + 1) % polygon.size()];

		if (tpp::segment_segment_intersection_safe(a, b, p1, p2).is_finite()) {
			return true;
		}
	}

	return false;
}

void order_pieces_from_approximation(
	const vector<vector<Vector2>> &polygons,
	const vector<Vector2> &approximate_path,
	vector<vector<vector<Vector2>>> &convex_pieces,
	const Vector2 &start,
	const Vector2 &target
) {

	vector<Vector2> full_path;
	full_path.reserve(approximate_path.size() + 2);
	full_path.push_back(start);
	full_path.insert(full_path.end(), approximate_path.begin(), approximate_path.end());
	full_path.push_back(target);

	size_t current_polygon = 0;
	size_t current_path = 1;

	while (current_polygon < polygons.size() && current_path + 1 < full_path.size()) {
		const Vector2 &current_point = full_path[current_path];
		const size_t original_polygon_index = current_polygon;

		while (current_polygon < polygons.size() && point_on_boundary(current_point, polygons[current_polygon])) {
			auto &pieces = convex_pieces[current_polygon];

			std::sort(pieces.begin(), pieces.end(), [&](const vector<Vector2> &a, const vector<Vector2> &b) {
				return project_point_on_polygon(current_point, a).distance_squared_to(current_point)
					< project_point_on_polygon(current_point, b).distance_squared_to(current_point);
			});

			current_polygon++;
		}

		if (current_polygon != original_polygon_index) {
			current_path++;
			continue;
		}

		auto &pieces = convex_pieces[current_polygon];
		const Vector2 &prev = full_path[current_path - 1];
		const Vector2 &next = full_path[current_path + 1];

		std::stable_sort(pieces.begin(), pieces.end(), [&](const vector<Vector2> &a, const vector<Vector2> &b) {
			return segment_intersects_polygon(prev, next, a) > segment_intersects_polygon(prev, next, b);
		});

		current_polygon++;
	}
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

BoundCallSet make_bound_calls(
	const vector<vector<vector<Vector2>>> &convex_pieces,
	const vector<vector<Vector2>> &convex_hulls,
	size_t max_calls,
	size_t max_branching
) {

	BoundCallSet result;
	result.calls.reserve(max_calls);

	vector<vector<size_t>> stack;
	stack.push_back({});

	while (!stack.empty() && result.calls.size() < max_calls) {
		vector<size_t> selected = std::move(stack.back());
		stack.pop_back();

		append_bound_call(result.calls, selected, convex_pieces, convex_hulls, max_calls);

		if (selected.size() == convex_pieces.size()) {
			continue;
		}

		const size_t next_polygon = selected.size();
		const size_t observed_branching = convex_pieces[next_polygon].size();
		const size_t branch_count = std::min(max_branching, observed_branching);

		result.max_observed_branching = std::max(result.max_observed_branching, observed_branching);
		result.branch_limited = result.branch_limited || branch_count < observed_branching;

		for (size_t i = branch_count; i > 0; i--) {
			vector<size_t> child = selected;
			child.push_back(i - 1);
			stack.push_back(std::move(child));
		}
	}

	result.exhausted = stack.empty();

	return result;
}

BranchAndBoundResult run_branch_and_bound(
	const Vector2 &start,
	const Vector2 &target,
	const vector<vector<vector<Vector2>>> &convex_pieces,
	const vector<vector<Vector2>> &convex_hulls,
	const vector<Vector2> &approximate_path,
	size_t max_calls,
	size_t max_branching,
	size_t case_index,
	bool show_progress
) {

	BranchAndBoundResult result;
	result.initial_length = path_length(start, target, approximate_path);
	auto last_progress_time = std::chrono::steady_clock::now();

	auto update_progress = [&](bool force) {
		if (!show_progress) {
			return;
		}

		const auto now = std::chrono::steady_clock::now();

		if (!force && std::chrono::duration<double>(now - last_progress_time).count() < 0.25) {
			return;
		}

		last_progress_time = now;
		std::cerr << '\r'
			<< "case " << format_count(case_index)
			<< " | " << progress_bar(result.convex_calls, max_calls)
			<< " | best_updates " << format_count(result.best_updates)
			<< " | pruned " << format_count(result.pruned_nodes)
			<< std::flush;
	};

	update_progress(true);

	auto solve_convex = [&](const vector<vector<Vector2>> &input_polygons, vector<Vector2> *output_path, double &kind_seconds) -> double {
		if (result.convex_calls >= max_calls) {
			result.exhausted = false;
			return std::numeric_limits<double>::infinity();
		}

		const size_t vertices = vertex_count(input_polygons);
		result.min_vertices = std::min(result.min_vertices, vertices);
		result.max_vertices = std::max(result.max_vertices, vertices);

		const auto solver_start_time = std::chrono::steady_clock::now();

		try {
			vector<Vector2> path = tpp::tpp_convex_solve(start, target, input_polygons);
			const auto solver_end_time = std::chrono::steady_clock::now();
			const double elapsed_seconds = std::chrono::duration<double>(solver_end_time - solver_start_time).count();
			result.solver_seconds += elapsed_seconds;
			kind_seconds += elapsed_seconds;
			result.convex_calls++;
			result.checksum += path_length(start, target, path);
			update_progress(false);

			const double length = path_length(start, target, path);

			if (output_path != nullptr) {
				*output_path = std::move(path);
			}

			return length;
		} catch (...) {
			const auto solver_end_time = std::chrono::steady_clock::now();
			const double elapsed_seconds = std::chrono::duration<double>(solver_end_time - solver_start_time).count();
			result.solver_seconds += elapsed_seconds;
			kind_seconds += elapsed_seconds;
			result.convex_calls++;
			result.checksum += std::numeric_limits<double>::infinity();
			update_progress(false);
			return std::numeric_limits<double>::infinity();
		}
	};

	vector<vector<Vector2>> selected_pieces;
	selected_pieces.reserve(convex_pieces.size());

	for (const auto &pieces : convex_pieces) {
		selected_pieces.push_back(pieces.front());
	}

	vector<Vector2> best_path;
	best_path = approximate_path;
	result.incumbent_length = result.initial_length;
	result.final_length = result.initial_length;

	vector<Vector2> selected_path;
	const size_t calls_before_incumbent = result.convex_calls;
	const double selected_length = solve_convex(selected_pieces, &selected_path, result.incumbent_solver_seconds);

	if (result.convex_calls > calls_before_incumbent) {
		result.incumbent_solves++;
	}

	if (selected_length < result.final_length) {
		result.final_length = selected_length;
		result.incumbent_length = selected_length;
		best_path = std::move(selected_path);
		result.best_updates++;
	}

	vector<vector<size_t>> stack;
	stack.push_back({});

	while (!stack.empty() && result.exhausted) {
		vector<size_t> current = std::move(stack.back());
		stack.pop_back();
		result.visited_nodes++;
		result.selected_sum += current.size();
		increment_histogram(result.visited_depth_histogram, current.size());

		if (current.size() == convex_pieces.size()) {
			vector<vector<Vector2>> instance;
			instance.reserve(convex_pieces.size());

			for (size_t i = 0; i < convex_pieces.size(); i++) {
				instance.push_back(convex_pieces[i][current[i]]);
			}

			vector<Vector2> path;
			const size_t calls_before = result.convex_calls;
			increment_histogram(result.leaf_depth_histogram, current.size());
			const double length = solve_convex(instance, &path, result.leaf_solver_seconds);

			if (result.convex_calls > calls_before) {
				result.leaf_solves++;
			}

			if (!result.exhausted) {
				break;
			}

			if (length < result.final_length) {
				result.final_length = length;
				best_path = std::move(path);
				result.best_updates++;
			}

			continue;
		}

		const size_t next_polygon = current.size();
		const size_t observed_branching = convex_pieces[next_polygon].size();
		const size_t branch_count = std::min(max_branching, observed_branching);

		result.max_observed_branching = std::max(result.max_observed_branching, observed_branching);
		result.branch_limited = result.branch_limited || branch_count < observed_branching;
		result.branching_histogram[branching_bucket(observed_branching)]++;

		for (size_t i = branch_count; i > 0; i--) {
			vector<size_t> selected = current;
			selected.push_back(i - 1);

			vector<vector<Vector2>> bound_instance;
			bound_instance.reserve(convex_pieces.size());

			for (size_t j = 0; j < selected.size(); j++) {
				bound_instance.push_back(convex_pieces[j][selected[j]]);
			}

			for (size_t j = selected.size(); j < convex_hulls.size(); j++) {
				bound_instance.push_back(convex_hulls[j]);
			}

			const size_t calls_before = result.convex_calls;
			increment_histogram(result.bound_depth_histogram, selected.size());
			const double incumbent_before_bound = result.final_length;
			const double bound = solve_convex(bound_instance, nullptr, result.bound_solver_seconds);

			if (result.convex_calls > calls_before) {
				result.bound_solves++;
			}

			if (!result.exhausted) {
				break;
			}

			if (bound > result.final_length) {
				result.pruned_nodes++;
				continue;
			}

			if (std::isfinite(bound) && std::isfinite(incumbent_before_bound) && incumbent_before_bound > 0.0) {
				result.failed_prune_count++;
				result.failed_prune_ratio_sum += bound / incumbent_before_bound;
				result.failed_prune_gap_sum += incumbent_before_bound - bound;
				result.failed_prune_depth_sum += static_cast<double>(selected.size());
			}

			stack.push_back(std::move(selected));
		}
	}

	if (result.convex_calls == 0) {
		result.min_vertices = 0;
	}

	update_progress(true);

	if (show_progress) {
		std::cerr << '\n';
	}

	return result;
}

struct Distribution {
	double min = 0.0;
	double median = 0.0;
	double p90 = 0.0;
	double p99 = 0.0;
	double max = 0.0;
	double mean = 0.0;
};

Distribution distribution(vector<double> values) {

	if (values.empty()) {
		return {};
	}

	std::sort(values.begin(), values.end());

	auto percentile = [&](double p) {
		const size_t index = std::min(values.size() - 1, static_cast<size_t>(std::ceil(p * static_cast<double>(values.size())) - 1.0));
		return values[index];
	};

	const double sum = std::accumulate(values.begin(), values.end(), 0.0);

	return {
		.min = values.front(),
		.median = percentile(0.50),
		.p90 = percentile(0.90),
		.p99 = percentile(0.99),
		.max = values.back(),
		.mean = sum / static_cast<double>(values.size()),
	};
}

std::string format_double(double value) {
	if (!std::isfinite(value)) {
		return "inf";
	}

	return std::format("{:.6f}", value);
}

std::string format_count_double(double value) {
	if (!std::isfinite(value)) {
		return "inf";
	}

	return format_count(static_cast<size_t>(std::llround(value)));
}

double initial_gap_percent(const InstanceRecord &record) {
	if (record.final_length <= 0.0 || !std::isfinite(record.final_length)) {
		return 0.0;
	}

	return (record.initial_length - record.final_length) / record.final_length * 100.0;
}

double incumbent_gap_percent(const InstanceRecord &record) {
	if (record.final_length <= 0.0 || !std::isfinite(record.final_length)) {
		return 0.0;
	}

	return (record.incumbent_length - record.final_length) / record.final_length * 100.0;
}

double prune_rate_percent(const InstanceRecord &record) {
	const size_t decisions = record.pruned_nodes + record.visited_nodes;
	return decisions == 0 ? 0.0 : static_cast<double>(record.pruned_nodes) / static_cast<double>(decisions) * 100.0;
}

double calls_per_visited_node(const InstanceRecord &record) {
	return record.visited_nodes == 0 ? 0.0 : static_cast<double>(record.calls) / static_cast<double>(record.visited_nodes);
}

double bound_calls_per_leaf(const InstanceRecord &record) {
	return record.leaf_solves == 0 ? 0.0 : static_cast<double>(record.bound_solves) / static_cast<double>(record.leaf_solves);
}

void add_histogram(vector<size_t> &target, const vector<size_t> &source) {
	if (target.size() < source.size()) {
		target.resize(source.size(), 0);
	}

	for (size_t i = 0; i < source.size(); i++) {
		target[i] += source[i];
	}
}

std::string histogram_to_string(const vector<size_t> &histogram, size_t max_entries = 12) {

	std::string output;
	const size_t count = std::min(max_entries, histogram.size());

	for (size_t i = 0; i < count; i++) {
		if (!output.empty()) {
			output += ", ";
		}

		output += std::format("{}: {}", i, format_count(histogram[i]));
	}

	if (histogram.size() > max_entries) {
		size_t remaining = 0;

		for (size_t i = max_entries; i < histogram.size(); i++) {
			remaining += histogram[i];
		}

		output += std::format(", {}+: {}", max_entries, format_count(remaining));
	}

	return output.empty() ? "none" : output;
}

std::string branching_histogram_to_string(const std::array<size_t, BRANCH_BUCKET_COUNT> &histogram) {
	return std::format(
		"1: {}, 2: {}, 3-5: {}, 6-10: {}, 11+: {}",
		format_count(histogram[0]),
		format_count(histogram[1]),
		format_count(histogram[2]),
		format_count(histogram[3]),
		format_count(histogram[4])
	);
}

void write_csv_record(std::ostream &output, const InstanceRecord &record) {
	output
		<< "nonconvex_bnb_cgal;"
		<< record.case_index << ';'
		<< record.repeat_index << ';'
		<< record.polygons << ';'
		<< record.decomposed_pieces << ';'
		<< std::format("{:.0f}", record.total_combinations) << ';'
		<< record.calls << ';'
		<< record.incumbent_solves << ';'
		<< record.bound_solves << ';'
		<< record.leaf_solves << ';'
		<< record.visited_nodes << ';'
		<< record.pruned_nodes << ';'
		<< record.best_updates << ';'
		<< std::format("{:.3f}", record.mean_selected) << ';'
		<< record.total_vertices_min << ';'
		<< record.total_vertices_max << ';'
		<< std::format("{:.12f}", record.initial_length) << ';'
		<< std::format("{:.12f}", record.incumbent_length) << ';'
		<< std::format("{:.12f}", record.final_length) << ';'
		<< std::format("{:.6f}", initial_gap_percent(record)) << ';'
		<< std::format("{:.6f}", incumbent_gap_percent(record)) << ';'
		<< std::format("{:.6f}", prune_rate_percent(record)) << ';'
		<< std::format("{:.6f}", calls_per_visited_node(record)) << ';'
		<< std::format("{:.6f}", bound_calls_per_leaf(record)) << ';'
		<< std::format("{:.6f}", record.decomposition_seconds) << ';'
		<< format_percent(record.decomposition_seconds, record.instance_seconds) << ';'
		<< std::format("{:.6f}", record.approximation_seconds) << ';'
		<< format_percent(record.approximation_seconds, record.instance_seconds) << ';'
		<< std::format("{:.6f}", record.bnb_seconds) << ';'
		<< format_percent(record.bnb_seconds, record.instance_seconds) << ';'
		<< std::format("{:.6f}", record.solver_seconds) << ';'
		<< format_percent(record.solver_seconds, record.instance_seconds) << ';'
		<< std::format("{:.6f}", record.incumbent_solver_seconds) << ';'
		<< std::format("{:.6f}", record.bound_solver_seconds) << ';'
		<< std::format("{:.6f}", record.leaf_solver_seconds) << ';'
		<< std::format("{:.12f}", record.seconds_per_call) << ';'
		<< (record.exhausted ? "true" : "false") << ';'
		<< (record.branch_limited ? "true" : "false") << ';'
		<< record.max_observed_branching << ';'
		<< record.failed_prune_count << ';'
		<< std::format("{:.6f}", record.failed_prune_ratio_mean) << ';'
		<< std::format("{:.6f}", record.failed_prune_gap_mean) << ';'
		<< std::format("{:.3f}", record.failed_prune_depth_mean) << ';'
		<< std::format("{:.12f}", record.checksum)
		<< '\n';
}

CaseBenchmarkResult run_case_benchmark(size_t case_index, const tpp::TestCase &test_case, const BenchmarkOptions &options) {
	CaseBenchmarkResult result;
	result.case_index = case_index;

	const auto &[start, target, raw_polygons, _] = test_case;

	vector<vector<Vector2>> polygons;
	polygons.reserve(raw_polygons.size());

	for (const auto &polygon : raw_polygons) {
		polygons.push_back(tpp::remove_collinear_points(polygon));
	}

	vector<vector<Vector2>> convex_hulls;
	vector<vector<vector<Vector2>>> convex_pieces;
	convex_hulls.reserve(polygons.size());
	convex_pieces.reserve(polygons.size());

	const auto decomposition_start_time = std::chrono::steady_clock::now();

	try {
		for (const auto &polygon : polygons) {
			convex_hulls.push_back(convex_hull(polygon));
			convex_pieces.push_back(tpp::decompose_polygon(polygon));
		}
	} catch (const std::exception &error) {
		result.skipped_decomposition = 1;
		result.decomposition_error = error.what();
		return result;
	}

	const auto decomposition_end_time = std::chrono::steady_clock::now();
	const double decomposition_seconds = std::chrono::duration<double>(decomposition_end_time - decomposition_start_time).count();
	result.decomposed = true;

	const auto approximation_start_time = std::chrono::steady_clock::now();
	vector<vector<Vector2>> interpolated_polygons;
	interpolated_polygons.reserve(polygons.size());

	for (const auto &polygon : polygons) {
		interpolated_polygons.push_back(make_interpolated_polygon(polygon, 10));
	}

	const vector<Vector2> approximate_path = tpp_approximation(start, target, interpolated_polygons);
	const auto approximation_end_time = std::chrono::steady_clock::now();
	const double approximation_seconds = std::chrono::duration<double>(approximation_end_time - approximation_start_time).count();

	order_pieces_from_approximation(polygons, approximate_path, convex_pieces, start, target);

	size_t total_pieces = 0;

	for (const auto &pieces : convex_pieces) {
		total_pieces += pieces.size();
	}

	result.records.reserve(options.repeat_count);

	for (size_t repeat_index = 0; repeat_index < options.repeat_count; repeat_index++) {
		const auto bnb_start_time = std::chrono::steady_clock::now();
		const BranchAndBoundResult bnb = run_branch_and_bound(
			start,
			target,
			convex_pieces,
			convex_hulls,
			approximate_path,
			options.max_calls_per_instance,
			options.max_branching,
			case_index,
			false
		);
		const auto bnb_end_time = std::chrono::steady_clock::now();
		const double bnb_seconds = std::chrono::duration<double>(bnb_end_time - bnb_start_time).count();

		if (bnb.convex_calls == 0) {
			result.skipped_no_calls++;
			continue;
		}

		InstanceRecord record;
		record.case_index = case_index;
		record.repeat_index = repeat_index;
		record.polygons = polygons.size();
		record.decomposed_pieces = total_pieces;
		record.total_combinations = combination_count(convex_pieces);
		record.calls = bnb.convex_calls;
		record.incumbent_solves = bnb.incumbent_solves;
		record.bound_solves = bnb.bound_solves;
		record.leaf_solves = bnb.leaf_solves;
		record.visited_nodes = bnb.visited_nodes;
		record.pruned_nodes = bnb.pruned_nodes;
		record.best_updates = bnb.best_updates;
		record.mean_selected = bnb.visited_nodes == 0 ? 0.0 : static_cast<double>(bnb.selected_sum) / static_cast<double>(bnb.visited_nodes);
		record.total_vertices_min = bnb.min_vertices;
		record.total_vertices_max = bnb.max_vertices;
		record.initial_length = bnb.initial_length;
		record.incumbent_length = bnb.incumbent_length;
		record.final_length = bnb.final_length;
		record.decomposition_seconds = repeat_index == 0 ? decomposition_seconds : 0.0;
		record.approximation_seconds = repeat_index == 0 ? approximation_seconds : 0.0;
		record.bnb_seconds = bnb_seconds;
		record.solver_seconds = bnb.solver_seconds;
		record.incumbent_solver_seconds = bnb.incumbent_solver_seconds;
		record.bound_solver_seconds = bnb.bound_solver_seconds;
		record.leaf_solver_seconds = bnb.leaf_solver_seconds;
		record.seconds_per_call = bnb.solver_seconds / static_cast<double>(bnb.convex_calls);
		record.exhausted = bnb.exhausted;
		record.branch_limited = bnb.branch_limited;
		record.max_observed_branching = bnb.max_observed_branching;
		record.failed_prune_count = bnb.failed_prune_count;
		record.failed_prune_ratio_mean = bnb.failed_prune_count == 0 ? 0.0 : bnb.failed_prune_ratio_sum / static_cast<double>(bnb.failed_prune_count);
		record.failed_prune_gap_mean = bnb.failed_prune_count == 0 ? 0.0 : bnb.failed_prune_gap_sum / static_cast<double>(bnb.failed_prune_count);
		record.failed_prune_depth_mean = bnb.failed_prune_count == 0 ? 0.0 : bnb.failed_prune_depth_sum / static_cast<double>(bnb.failed_prune_count);
		record.instance_seconds = record.decomposition_seconds + record.approximation_seconds + bnb_seconds;
		record.checksum = bnb.checksum;

		add_histogram(result.visited_depth_histogram, bnb.visited_depth_histogram);
		add_histogram(result.bound_depth_histogram, bnb.bound_depth_histogram);
		add_histogram(result.leaf_depth_histogram, bnb.leaf_depth_histogram);

		for (size_t i = 0; i < BRANCH_BUCKET_COUNT; i++) {
			result.branching_histogram[i] += bnb.branching_histogram[i];
		}

		result.records.push_back(record);
	}

	return result;
}

size_t default_thread_count() {
	const unsigned int hardware_threads = std::thread::hardware_concurrency();
	return hardware_threads == 0 ? 1 : static_cast<size_t>(hardware_threads);
}

void print_usage(const char *program) {
	std::println(stderr, "Usage:");
	std::println(stderr, "  {} <input_file> <max_polygons> <max_instances> <max_calls_per_instance> <max_branching> [repeat_count] [csv_output_file] [summary_md_file]", program);
	std::println(stderr, "");
	std::println(stderr, "Example:");
	std::println(stderr, "  {} packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin 40 6 512 6", program);
	std::println(stderr, "  {} packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin -1 -1 1000000 -1 results.csv", program);
	std::println(stderr, "  {} packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin 40 6 512 6 5 results.csv", program);
	std::println(stderr, "  {} packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin 40 6 512 6 5 results.csv summary.md", program);
	std::println(stderr, "");
	std::println(stderr, "Arguments:");
	std::println(stderr, "  input_file              Binary test case file.");
	std::println(stderr, "  max_polygons            Skip instances with more polygons than this.");
	std::println(stderr, "  max_instances           Stop after benchmarking this many accepted instances.");
	std::println(stderr, "  max_calls_per_instance  Cap actual convex solver calls per instance.");
	std::println(stderr, "  max_branching           Cap explored children per polygon during B&B.");
	std::println(stderr, "  repeat_count            Optional repeated runs per accepted instance.");
	std::println(stderr, "  csv_output_file         Optional file path for per-instance CSV rows.");
	std::println(stderr, "  summary_md_file         Optional file path for the markdown summary.");
	std::println(stderr, "");
	std::println(stderr, "All numeric arguments must be non-negative integers or -1 for unlimited.");
	std::println(stderr, "Set TPP_BENCH_THREADS to override the default hardware thread count.");
}

std::optional<size_t> parse_size_arg(const char *text) {

	const std::string value = text;

	if (value == "-1") {
		return std::numeric_limits<size_t>::max();
	}

	if (value.empty() || value.front() == '-') {
		return std::nullopt;
	}

	size_t consumed = 0;

	try {
		const size_t parsed = std::stoull(value, &consumed);
		if (consumed != value.size()) {
			return std::nullopt;
		}

		return parsed;
	} catch (...) {
		return std::nullopt;
	}
}

std::optional<BenchmarkOptions> parse_options(int argc, char **argv) {

	BenchmarkOptions options;

	if (argc != 1 && argc != 6 && argc != 7 && argc != 8 && argc != 9) {
		print_usage(argv[0]);
		return std::nullopt;
	}

	if (argc == 1) {
		return options;
	}

	options.input_path = argv[1];

	const auto max_polygons = parse_size_arg(argv[2]);
	const auto max_instances = parse_size_arg(argv[3]);
	const auto max_calls_per_instance = parse_size_arg(argv[4]);
	const auto max_branching = parse_size_arg(argv[5]);

	if (!max_polygons || !max_instances || !max_calls_per_instance || !max_branching) {
		print_usage(argv[0]);
		return std::nullopt;
	}

	options.max_polygons = *max_polygons;
	options.max_instances = *max_instances;
	options.max_calls_per_instance = *max_calls_per_instance;
	options.max_branching = *max_branching;

	if (argc == 7) {
		const auto repeat_count = parse_size_arg(argv[6]);

		if (repeat_count) {
			if (*repeat_count == 0 || *repeat_count == std::numeric_limits<size_t>::max()) {
				print_usage(argv[0]);
				return std::nullopt;
			}

			options.repeat_count = *repeat_count;
		} else {
			options.output_path = argv[6];
		}
	}

	if (argc == 8) {
		const auto repeat_count = parse_size_arg(argv[6]);

		if (repeat_count) {
			if (*repeat_count == 0 || *repeat_count == std::numeric_limits<size_t>::max()) {
				print_usage(argv[0]);
				return std::nullopt;
			}

			options.repeat_count = *repeat_count;
			options.output_path = argv[7];
		} else {
			options.output_path = argv[6];
			options.summary_output_path = argv[7];
		}
	}

	if (argc == 9) {
		const auto repeat_count = parse_size_arg(argv[6]);

		if (!repeat_count || *repeat_count == 0 || *repeat_count == std::numeric_limits<size_t>::max()) {
			print_usage(argv[0]);
			return std::nullopt;
		}

		options.repeat_count = *repeat_count;
		options.output_path = argv[7];
		options.summary_output_path = argv[8];
	}

	return options;
}

} // namespace

int main(int argc, char **argv) {

	const auto parsed_options = parse_options(argc, argv);

	if (!parsed_options) {
		return 2;
	}

	BenchmarkOptions options = *parsed_options;
	if (const char *thread_count_text = std::getenv("TPP_BENCH_THREADS")) {
		const auto thread_count = parse_size_arg(thread_count_text);

		if (!thread_count || *thread_count == 0 || *thread_count == std::numeric_limits<size_t>::max()) {
			std::println(stderr, "Invalid TPP_BENCH_THREADS value: {}", thread_count_text);
			print_usage(argv[0]);
			return 2;
		}

		options.thread_count = *thread_count;
	}

	if (options.thread_count == 0) {
		options.thread_count = default_thread_count();
	}

	const auto program_start_time = std::chrono::steady_clock::now();
	const auto test_cases = tpp::load_test_cases(options.input_path);
	BenchmarkSummary summary;
	summary.total_instances = test_cases.size();
	vector<InstanceRecord> records;
	vector<size_t> total_visited_depth_histogram;
	vector<size_t> total_bound_depth_histogram;
	vector<size_t> total_leaf_depth_histogram;
	std::array<size_t, BRANCH_BUCKET_COUNT> total_branching_histogram = {};

	constexpr const char *csv_header =
		"source;case_index;repeat_index;polygons;decomposed_pieces;total_combinations;calls;incumbent_solves;bound_solves;leaf_solves;visited_nodes;pruned_nodes;best_updates;mean_selected;total_vertices_min;total_vertices_max;initial_length;incumbent_length;final_length;initial_gap_percent;incumbent_gap_percent;prune_rate_percent;calls_per_visited_node;bound_calls_per_leaf;decomposition_seconds;decomposition_percent;approximation_seconds;approximation_percent;bnb_seconds;bnb_percent;solver_seconds;solver_percent;incumbent_solver_seconds;bound_solver_seconds;leaf_solver_seconds;seconds_per_call;exhausted;branch_limited;max_observed_branching;failed_prune_count;failed_prune_ratio_mean;failed_prune_gap_mean;failed_prune_depth_mean;checksum";

	std::ofstream csv_file;
	std::ostream *csv_output = &std::cout;

	if (options.output_path) {
		csv_file.open(*options.output_path);

		if (!csv_file) {
			std::println(stderr, "Could not open CSV output file: {}", *options.output_path);
			return 1;
		}

		csv_output = &csv_file;
		std::println(stderr, "Writing per-instance CSV rows to {}", *options.output_path);
	}

	vector<size_t> case_jobs;
	case_jobs.reserve(std::min(test_cases.size(), options.max_instances));

	for (size_t case_index = 0; case_index < test_cases.size() && case_jobs.size() < options.max_instances; case_index++) {
		const auto &[start, target, raw_polygons, _] = test_cases[case_index];
		if (raw_polygons.empty()) {
			summary.skipped_empty++;
			continue;
		}

		if (raw_polygons.size() > options.max_polygons) {
			summary.skipped_max_polygons++;
			continue;
		}

		case_jobs.push_back(case_index);
	}

	const size_t worker_count = case_jobs.empty() ? 0 : std::min(options.thread_count, case_jobs.size());
	vector<CaseBenchmarkResult> case_results(case_jobs.size());
	std::atomic<size_t> next_job = 0;
	std::atomic<size_t> completed_jobs = 0;
	std::mutex progress_mutex;
	const bool show_case_progress = options.output_path.has_value() && case_jobs.size() > 1;

	if (show_case_progress) {
		std::println(stderr, "Benchmarking {} accepted cases with {} worker threads", format_count(case_jobs.size()), format_count(worker_count));
	}

	vector<std::thread> workers;
	workers.reserve(worker_count);

	for (size_t worker_index = 0; worker_index < worker_count; worker_index++) {
		workers.emplace_back([&]() {
			while (true) {
				const size_t job_index = next_job.fetch_add(1);

				if (job_index >= case_jobs.size()) {
					break;
				}

				const size_t case_index = case_jobs[job_index];
				case_results[job_index] = run_case_benchmark(case_index, test_cases[case_index], options);
				const size_t done = completed_jobs.fetch_add(1) + 1;

				if (show_case_progress) {
					const std::scoped_lock lock(progress_mutex);
					std::cerr << '\r'
						<< "cases | " << progress_bar(done, case_jobs.size())
						<< std::flush;
				}
			}
		});
	}

	for (auto &worker : workers) {
		worker.join();
	}

	if (show_case_progress) {
		std::cerr << '\n';
	}

	for (const auto &case_result : case_results) {
		if (case_result.skipped_decomposition != 0) {
			std::println(stderr, "Skipping case {}: decomposition failed: {}", case_result.case_index, case_result.decomposition_error);
		}

		summary.skipped_decomposition += case_result.skipped_decomposition;
		summary.skipped_no_calls += case_result.skipped_no_calls;

		if (case_result.decomposed) {
			summary.benchmarked_instances++;
		}

		add_histogram(total_visited_depth_histogram, case_result.visited_depth_histogram);
		add_histogram(total_bound_depth_histogram, case_result.bound_depth_histogram);
		add_histogram(total_leaf_depth_histogram, case_result.leaf_depth_histogram);

		for (size_t i = 0; i < BRANCH_BUCKET_COUNT; i++) {
			total_branching_histogram[i] += case_result.branching_histogram[i];
		}

		for (const auto &record : case_result.records) {
			summary.fully_covered_instances += record.exhausted && !record.branch_limited ? 1 : 0;
			summary.capped_by_calls_instances += record.exhausted ? 0 : 1;
			summary.branch_limited_instances += record.branch_limited ? 1 : 0;
			summary.total_calls += record.calls;
			summary.total_incumbent_solves += record.incumbent_solves;
			summary.total_bound_solves += record.bound_solves;
			summary.total_leaf_solves += record.leaf_solves;
			summary.total_visited_nodes += record.visited_nodes;
			summary.total_pruned_nodes += record.pruned_nodes;
			summary.total_best_updates += record.best_updates;
			summary.max_observed_branching = std::max(summary.max_observed_branching, record.max_observed_branching);
			summary.decomposition_seconds += record.decomposition_seconds;
			summary.approximation_seconds += record.approximation_seconds;
			summary.bnb_seconds += record.bnb_seconds;
			summary.solver_seconds += record.solver_seconds;
			summary.checksum += record.checksum + record.final_length;
			records.push_back(record);
		}
	}

	std::sort(records.begin(), records.end(), [](const InstanceRecord &a, const InstanceRecord &b) {
		return std::tie(a.case_index, a.repeat_index) < std::tie(b.case_index, b.repeat_index);
	});

	*csv_output << csv_header << '\n';

	for (const auto &record : records) {
		write_csv_record(*csv_output, record);
	}

	const auto program_end_time = std::chrono::steady_clock::now();
	const double total_seconds = std::chrono::duration<double>(program_end_time - program_start_time).count();
	const double measured_work_seconds = summary.decomposition_seconds + summary.approximation_seconds + summary.bnb_seconds;
	const double parallel_speedup_estimate = total_seconds == 0.0 ? 0.0 : measured_work_seconds / total_seconds;
	const double mean_seconds_per_call =
		summary.total_calls == 0 ? 0.0 : summary.solver_seconds / static_cast<double>(summary.total_calls);
	const double mean_failed_prune_ratio =
		records.empty() ? 0.0 : std::accumulate(records.begin(), records.end(), 0.0, [](double sum, const InstanceRecord &record) {
			return sum + record.failed_prune_ratio_mean;
		}) / static_cast<double>(records.size());
	const double mean_failed_prune_gap =
		records.empty() ? 0.0 : std::accumulate(records.begin(), records.end(), 0.0, [](double sum, const InstanceRecord &record) {
			return sum + record.failed_prune_gap_mean;
		}) / static_cast<double>(records.size());
	const double mean_failed_prune_depth =
		records.empty() ? 0.0 : std::accumulate(records.begin(), records.end(), 0.0, [](double sum, const InstanceRecord &record) {
			return sum + record.failed_prune_depth_mean;
		}) / static_cast<double>(records.size());

	auto values = [&](auto getter) {
		vector<double> result;
		result.reserve(records.size());

		for (const auto &record : records) {
			result.push_back(static_cast<double>(getter(record)));
		}

		return result;
	};

	std::ostringstream summary_markdown;

	auto emit = [&](std::string_view line) {
		summary_markdown << line << '\n';
	};

	auto emitf = [&]<typename... Args>(std::format_string<Args...> fmt, Args&&... args) {
		emit(std::format(fmt, std::forward<Args>(args)...));
	};

	auto print_distribution = [&](const std::string &name, const Distribution &dist, bool count_like) {
		if (count_like) {
			emitf(
				"| {} | {} | {} | {} | {} | {} | {} |",
				name,
				format_count_double(dist.min),
				format_count_double(dist.median),
				format_count_double(dist.p90),
				format_count_double(dist.p99),
				format_count_double(dist.max),
				format_count_double(dist.mean)
			);
		} else {
			emitf(
				"| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |",
				name,
				dist.min,
				dist.median,
				dist.p90,
				dist.p99,
				dist.max,
				dist.mean
			);
		}
	};

	auto print_top = [&](const std::string &title, auto getter, const std::string &value_label, bool count_like) {
		vector<InstanceRecord> sorted = records;
		std::sort(sorted.begin(), sorted.end(), [&](const InstanceRecord &a, const InstanceRecord &b) {
			return getter(a) > getter(b);
		});

		emitf("### {}", title);
		emit("");
		emitf("| Case | Repeat | Value ({}) | Calls | Pieces | Max Branch | Initial Gap | Exhausted |", value_label);
		emit("|---:|---:|---:|---:|---:|---:|---:|---:|");

		const size_t rows = std::min(static_cast<size_t>(10), sorted.size());

		for (size_t i = 0; i < rows; i++) {
			const auto &record = sorted[i];
			const double value = getter(record);
			const std::string formatted_value = count_like ? format_count_double(value) : format_double(value);

			emitf(
				"| {} | {} | {} | {} | {} | {} | {:.6f}% | {} |",
				format_count(record.case_index),
				format_count(record.repeat_index),
				formatted_value,
				format_count(record.calls),
				format_count(record.decomposed_pieces),
				format_count(record.max_observed_branching),
				initial_gap_percent(record),
				record.exhausted
			);
		}

		emit("");
	};

	emit("");
	emit("## Benchmark Summary");
	emit("");

	if (options.output_path) {
		emitf("CSV output: `{}`", *options.output_path);
		emit("");
	}

	if (options.summary_output_path) {
		emitf("Summary output: `{}`", *options.summary_output_path);
		emit("");
	}

	emit("| Metric | Value |");
	emit("|---|---:|");
	emitf("| Total instances | {} |", format_count(summary.total_instances));
	emitf("| Benchmarked instances | {} |", format_count(summary.benchmarked_instances));
	emitf("| Benchmark runs | {} |", format_count(records.size()));
	emitf("| Repeat count | {} |", format_count(options.repeat_count));
	emitf("| Worker threads | {} |", format_count(worker_count));
	emitf("| Fully solved runs | {} |", format_count(summary.fully_covered_instances));
	emitf("| Capped by calls runs | {} |", format_count(summary.capped_by_calls_instances));
	emitf("| Branch limited runs | {} |", format_count(summary.branch_limited_instances));
	emitf("| Skipped by max polygons | {} |", format_count(summary.skipped_max_polygons));
	emitf("| Skipped empty | {} |", format_count(summary.skipped_empty));
	emitf("| Skipped decomposition | {} |", format_count(summary.skipped_decomposition));
	emitf("| Skipped no calls | {} |", format_count(summary.skipped_no_calls));
	emitf("| Max observed branching | {} |", format_count(summary.max_observed_branching));
	emit("");
	emit("| B&B Counter | Value |");
	emit("|---|---:|");
	emitf("| Total convex calls | {} |", format_count(summary.total_calls));
	emitf("| Incumbent solves | {} |", format_count(summary.total_incumbent_solves));
	emitf("| Bound solves | {} |", format_count(summary.total_bound_solves));
	emitf("| Leaf solves | {} |", format_count(summary.total_leaf_solves));
	emitf("| Visited nodes | {} |", format_count(summary.total_visited_nodes));
	emitf("| Pruned nodes | {} |", format_count(summary.total_pruned_nodes));
	emitf("| Best updates | {} |", format_count(summary.total_best_updates));
	emit("");
	emit("| Timing | Value |");
	emit("|---|---:|");
	emitf("| Decomposition | {} |", format_seconds_with_percent(summary.decomposition_seconds, measured_work_seconds));
	emitf("| Approximation | {} |", format_seconds_with_percent(summary.approximation_seconds, measured_work_seconds));
	emitf("| B&B | {} |", format_seconds_with_percent(summary.bnb_seconds, measured_work_seconds));
	emitf("| Convex solver | {} of measured work |", format_seconds_with_percent(summary.solver_seconds, measured_work_seconds));
	emitf("| Measured work | {:.6f}s (100.00%) |", measured_work_seconds);
	emitf("| Wall-clock total | {:.6f}s |", total_seconds);
	emitf("| Parallel speedup estimate | {:.2f}x |", parallel_speedup_estimate);
	emitf("| Mean seconds per call | {:.12f}s |", mean_seconds_per_call);
	emitf("| Checksum | {:.12f} |", summary.checksum);
	emit("");
	emit("## Distributions");
	emit("");
	emit("| Metric | Min | Median | P90 | P99 | Max | Mean |");
	emit("|---|---:|---:|---:|---:|---:|---:|");
	print_distribution("Seconds per call", distribution(values([](const InstanceRecord &r) { return r.seconds_per_call; })), false);
	print_distribution("Polygons", distribution(values([](const InstanceRecord &r) { return r.polygons; })), true);
	print_distribution("Calls", distribution(values([](const InstanceRecord &r) { return r.calls; })), true);
	print_distribution("Best updates", distribution(values([](const InstanceRecord &r) { return r.best_updates; })), true);
	print_distribution("Initial gap %", distribution(values([](const InstanceRecord &r) { return initial_gap_percent(r); })), false);
	print_distribution("Incumbent gap %", distribution(values([](const InstanceRecord &r) { return incumbent_gap_percent(r); })), false);
	print_distribution("Max branching", distribution(values([](const InstanceRecord &r) { return r.max_observed_branching; })), true);
	print_distribution("Decomposed pieces", distribution(values([](const InstanceRecord &r) { return r.decomposed_pieces; })), true);
	print_distribution("Pieces per polygon", distribution(values([](const InstanceRecord &r) { return pieces_per_polygon(r); })), false);
	print_distribution("log2(total combinations)", distribution(values([](const InstanceRecord &r) { return log2_combination_count(r); })), false);
	emit("");
	emit("## Derived Metrics");
	emit("");
	emit("| Metric | Value |");
	emit("|---|---:|");
	emitf("| Overall prune rate | {} |", format_percent(static_cast<double>(summary.total_pruned_nodes), static_cast<double>(summary.total_pruned_nodes + summary.total_visited_nodes)));
	emitf("| Calls per visited node | {:.6f} |", summary.total_visited_nodes == 0 ? 0.0 : static_cast<double>(summary.total_calls) / static_cast<double>(summary.total_visited_nodes));
	emitf("| Bound calls per leaf | {:.6f} |", summary.total_leaf_solves == 0 ? 0.0 : static_cast<double>(summary.total_bound_solves) / static_cast<double>(summary.total_leaf_solves));
	emitf("| Solver time share | {} |", format_percent(summary.solver_seconds, measured_work_seconds));
	emitf("| Mean failed-prune bound/incumbent | {:.6f} |", mean_failed_prune_ratio);
	emitf("| Mean failed-prune incumbent-bound gap | {:.6f} |", mean_failed_prune_gap);
	emitf("| Mean failed-prune depth | {:.3f} |", mean_failed_prune_depth);
	emit("");
	emit("## Histograms");
	emit("");
	emit("| Histogram | Buckets |");
	emit("|---|---|");
	emitf("| Visited node depth | {} |", histogram_to_string(total_visited_depth_histogram));
	emitf("| Bound solve depth | {} |", histogram_to_string(total_bound_depth_histogram));
	emitf("| Leaf solve depth | {} |", histogram_to_string(total_leaf_depth_histogram));
	emitf("| Branching | {} |", branching_histogram_to_string(total_branching_histogram));
	emit("");
	emit("## Worst Runs");
	emit("");
	print_top("By Runtime", [](const InstanceRecord &r) { return r.instance_seconds; }, "Seconds", false);
	print_top("By Convex Calls", [](const InstanceRecord &r) { return r.calls; }, "Calls", true);
	print_top("By Decomposed Pieces", [](const InstanceRecord &r) { return r.decomposed_pieces; }, "Pieces", true);
	print_top("By log2(Total Combinations)", [](const InstanceRecord &r) { return log2_combination_count(r); }, "log2(combinations)", false);
	print_top("By Max Branching", [](const InstanceRecord &r) { return r.max_observed_branching; }, "Max Branch", true);
	print_top("By Initial Gap", [](const InstanceRecord &r) { return initial_gap_percent(r); }, "Gap %", false);
	emit("Tip: with summary output enabled, render it with `glow summary.md`.");

	const std::string markdown = summary_markdown.str();

	if (options.summary_output_path) {
		std::ofstream summary_file(*options.summary_output_path);

		if (!summary_file) {
			std::println(stderr, "Could not open summary output file: {}", *options.summary_output_path);
			return 1;
		}

		summary_file << markdown;
	}

	std::cout << markdown;
}
