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
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <print>
#include <set>
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
constexpr double APPROXIMATION_WORK_BUDGET = 1'000'000.0;
constexpr double APPROXIMATION_ADAPTIVE_MAX_FACTOR = 8.0;

using ConvexSolverFunction = std::vector<Vector2> (*)(
	const Vector2&,
	const Vector2&,
	const std::vector<std::vector<Vector2>>&
);

using ConvexLengthSolverFunction = double (*)(
	const Vector2&,
	const Vector2&,
	const std::vector<std::vector<Vector2>>&
);

struct BenchmarkOptions {
	std::string input_path = "benchmarks/suites/canonical-v1.bin";
	size_t max_polygons = 40;
	size_t max_instances = 6;
	size_t max_calls_per_instance = 512;
	size_t max_branching = 6;
	double max_seconds_per_instance = std::numeric_limits<double>::infinity();
	size_t repeat_count = 1;
	size_t thread_count = 0;
	std::string solver_name = "binary_search_lazy";
	ConvexSolverFunction solver = tpp::tpp_convex_solve_binary_search_lazy;
	ConvexLengthSolverFunction length_solver = tpp::tpp_convex_solve_length_binary_search_lazy;
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
	size_t skipped_max_polygons = 0;
	size_t skipped_decomposition = 0;
	size_t skipped_intersecting_hulls = 0;
	size_t skipped_no_calls = 0;
	size_t benchmarked_instances = 0;
	size_t fully_covered_instances = 0;
	size_t capped_by_calls_instances = 0;
	size_t capped_by_time_instances = 0;
	size_t branch_limited_instances = 0;
	size_t grouped_pieces = 0;
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
	double piece_graph_precompute_seconds = 0.0;
	double piece_graph_bound_seconds = 0.0;
	size_t piece_graph_bound_calls = 0;
	double port_bound_precompute_seconds = 0.0;
	double port_bound_seconds = 0.0;
	size_t port_bound_calls = 0;
	size_t hull_bound_prunes = 0;
	size_t piece_graph_extra_prunes = 0;
	size_t piece_graph_dominates = 0;
	size_t port_extra_prunes = 0;
	size_t port_dominates = 0;
	double refinement_bound_seconds = 0.0;
	size_t refinement_bound_calls = 0;
	size_t refinement_extra_prunes = 0;
	size_t refinement_dominates = 0;
	double contact_bound_seconds = 0.0;
	size_t contact_path_calls = 0;
	size_t contact_bound_calls = 0;
	size_t contact_extra_prunes = 0;
	size_t contact_dominates = 0;
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
	bool time_limited = false;
	bool branch_limited = false;
	double initial_length = std::numeric_limits<double>::infinity();
	double incumbent_length = std::numeric_limits<double>::infinity();
	double final_length = std::numeric_limits<double>::infinity();
	double solver_seconds = 0.0;
	double incumbent_solver_seconds = 0.0;
	double bound_solver_seconds = 0.0;
	double leaf_solver_seconds = 0.0;
	double piece_graph_precompute_seconds = 0.0;
	double piece_graph_bound_seconds = 0.0;
	size_t piece_graph_bound_calls = 0;
	double port_bound_precompute_seconds = 0.0;
	double port_bound_seconds = 0.0;
	size_t port_bound_calls = 0;
	size_t hull_bound_prunes = 0;
	size_t piece_graph_extra_prunes = 0;
	size_t piece_graph_dominates = 0;
	size_t port_extra_prunes = 0;
	size_t port_dominates = 0;
	double refinement_bound_seconds = 0.0;
	size_t refinement_bound_calls = 0;
	size_t refinement_extra_prunes = 0;
	size_t refinement_dominates = 0;
	double contact_bound_seconds = 0.0;
	size_t contact_path_calls = 0;
	size_t contact_bound_calls = 0;
	size_t contact_extra_prunes = 0;
	size_t contact_dominates = 0;
	size_t failed_prune_count = 0;
	double failed_prune_ratio_sum = 0.0;
	double failed_prune_gap_sum = 0.0;
	double failed_prune_depth_sum = 0.0;
	vector<Vector2> best_path;
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
	size_t grouped_pieces = 0;
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
	double piece_graph_precompute_seconds = 0.0;
	double piece_graph_bound_seconds = 0.0;
	size_t piece_graph_bound_calls = 0;
	double port_bound_precompute_seconds = 0.0;
	double port_bound_seconds = 0.0;
	size_t port_bound_calls = 0;
	size_t hull_bound_prunes = 0;
	size_t piece_graph_extra_prunes = 0;
	size_t piece_graph_dominates = 0;
	size_t port_extra_prunes = 0;
	size_t port_dominates = 0;
	double refinement_bound_seconds = 0.0;
	size_t refinement_bound_calls = 0;
	size_t refinement_extra_prunes = 0;
	size_t refinement_dominates = 0;
	double contact_bound_seconds = 0.0;
	size_t contact_path_calls = 0;
	size_t contact_bound_calls = 0;
	size_t contact_extra_prunes = 0;
	size_t contact_dominates = 0;
	bool exhausted = false;
	bool time_limited = false;
	bool branch_limited = false;
	size_t max_observed_branching = 0;
	size_t failed_prune_count = 0;
	double failed_prune_ratio_mean = 0.0;
	double failed_prune_gap_mean = 0.0;
	double failed_prune_depth_mean = 0.0;
	double instance_seconds = 0.0;
	double checksum = 0.0;
	std::string solution_preview_path;
};

struct CaseBenchmarkResult {
	size_t case_index = 0;
	bool decomposed = false;
	size_t skipped_decomposition = 0;
	size_t skipped_intersecting_hulls = 0;
	size_t skipped_no_calls = 0;
	std::string decomposition_error;
	vector<InstanceRecord> records;
	vector<size_t> visited_depth_histogram;
	vector<size_t> bound_depth_histogram;
	vector<size_t> leaf_depth_histogram;
	std::array<size_t, BRANCH_BUCKET_COUNT> branching_histogram = {};
};

struct SyntheticCoverSuite {
	vector<tpp::TestCase> cases;
	vector<vector<vector<vector<Vector2>>>> covers;
};

bool is_ccw_turn(const Vector2 &p0, const Vector2 &p1, const Vector2 &p2) {
	return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y) > 0;
}

double cross(const Vector2 &a, const Vector2 &b, const Vector2 &c) {
	return (b - a).cross(c - a);
}

bool point_on_segment(const Vector2 &point, const Vector2 &a, const Vector2 &b) {

	constexpr double EPS = 1e-9;

	if (std::abs(cross(a, b, point)) > EPS) {
		return false;
	}

	return point.x >= std::min(a.x, b.x) - EPS
		&& point.x <= std::max(a.x, b.x) + EPS
		&& point.y >= std::min(a.y, b.y) - EPS
		&& point.y <= std::max(a.y, b.y) + EPS;
}

bool segments_intersect_or_touch(const Vector2 &a, const Vector2 &b, const Vector2 &c, const Vector2 &d) {

	constexpr double EPS = 1e-9;

	const double c1 = cross(a, b, c);
	const double c2 = cross(a, b, d);
	const double c3 = cross(c, d, a);
	const double c4 = cross(c, d, b);

	if (((c1 > EPS && c2 < -EPS) || (c1 < -EPS && c2 > EPS))
		&& ((c3 > EPS && c4 < -EPS) || (c3 < -EPS && c4 > EPS))) {
		return true;
	}

	return point_on_segment(c, a, b)
		|| point_on_segment(d, a, b)
		|| point_on_segment(a, c, d)
		|| point_on_segment(b, c, d);
}

bool point_in_polygon_or_on_boundary(const Vector2 &point, const vector<Vector2> &polygon) {

	bool inside = false;

	for (size_t i = 0; i < polygon.size(); i++) {
		const Vector2 &a = polygon[i];
		const Vector2 &b = polygon[(i + 1) % polygon.size()];

		if (point_on_segment(point, a, b)) {
			return true;
		}

		if ((a.y > point.y) != (b.y > point.y)) {
			const double intersection_x = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
			if (intersection_x >= point.x) {
				inside = !inside;
			}
		}
	}

	return inside;
}

bool polygons_intersect_or_touch(const vector<Vector2> &a, const vector<Vector2> &b) {

	for (size_t i = 0; i < a.size(); i++) {
		const Vector2 &a1 = a[i];
		const Vector2 &a2 = a[(i + 1) % a.size()];

		for (size_t j = 0; j < b.size(); j++) {
			const Vector2 &b1 = b[j];
			const Vector2 &b2 = b[(j + 1) % b.size()];

			if (segments_intersect_or_touch(a1, a2, b1, b2)) {
				return true;
			}
		}
	}

	return (!a.empty() && point_in_polygon_or_on_boundary(a.front(), b))
		|| (!b.empty() && point_in_polygon_or_on_boundary(b.front(), a));
}

double point_segment_distance(const Vector2 &point, const Vector2 &a, const Vector2 &b) {

	const Vector2 ab = b - a;
	const double denominator = ab.dot(ab);

	if (denominator == 0.0) {
		return point.distance_to(a);
	}

	const double t = std::clamp((point - a).dot(ab) / denominator, 0.0, 1.0);
	return point.distance_to(a + ab * t);
}

double segment_segment_distance(const Vector2 &a, const Vector2 &b, const Vector2 &c, const Vector2 &d) {

	if (segments_intersect_or_touch(a, b, c, d)) {
		return 0.0;
	}

	return std::min({
		point_segment_distance(a, c, d),
		point_segment_distance(b, c, d),
		point_segment_distance(c, a, b),
		point_segment_distance(d, a, b),
	});
}

double point_polygon_distance(const Vector2 &point, const vector<Vector2> &polygon) {

	if (polygon.empty()) {
		return std::numeric_limits<double>::infinity();
	}

	if (point_in_polygon_or_on_boundary(point, polygon)) {
		return 0.0;
	}

	double distance = std::numeric_limits<double>::infinity();

	for (size_t i = 0; i < polygon.size(); i++) {
		distance = std::min(distance, point_segment_distance(point, polygon[i], polygon[(i + 1) % polygon.size()]));
	}

	return distance;
}

double polygon_polygon_distance(const vector<Vector2> &a, const vector<Vector2> &b) {

	if (a.empty() || b.empty()) {
		return std::numeric_limits<double>::infinity();
	}

	if (polygons_intersect_or_touch(a, b)) {
		return 0.0;
	}

	double distance = std::numeric_limits<double>::infinity();

	for (size_t i = 0; i < a.size(); i++) {
		for (size_t j = 0; j < b.size(); j++) {
			distance = std::min(
				distance,
				segment_segment_distance(a[i], a[(i + 1) % a.size()], b[j], b[(j + 1) % b.size()])
			);
		}
	}

	return distance;
}

bool use_piece_graph_bound() {

	const char *enabled = std::getenv("TPP_PIECE_GRAPH_BOUND");
	return enabled != nullptr && std::string_view(enabled) != "0";
}

bool use_piece_grouping() {

	const char *enabled = std::getenv("TPP_GROUP_PIECES");
	return enabled != nullptr && std::string_view(enabled) != "0";
}

bool use_adaptive_piece_grouping() {

	const char *enabled = std::getenv("TPP_GROUP_ADAPTIVE");
	return enabled != nullptr && std::string_view(enabled) != "0";
}

double piece_group_max_excess_ratio() {

	const char *value = std::getenv("TPP_GROUP_MAX_EXCESS_RATIO");

	if (value == nullptr) {
		return use_adaptive_piece_grouping() ? 0.25 : 0.05;
	}

	return std::strtod(value, nullptr);
}

size_t piece_group_max_size() {

	const char *value = std::getenv("TPP_GROUP_MAX_SIZE");

	if (value == nullptr) {
		return 3;
	}

	return std::strtoull(value, nullptr, 10);
}

double adaptive_group_max_local_slack_ratio() {

	const char *value = std::getenv("TPP_GROUP_ADAPTIVE_MAX_LOCAL_SLACK_RATIO");

	if (value == nullptr) {
		return 0.025;
	}

	return std::strtod(value, nullptr);
}

double adaptive_group_excess_weight() {

	const char *value = std::getenv("TPP_GROUP_ADAPTIVE_EXCESS_WEIGHT");

	if (value == nullptr) {
		return 1.0;
	}

	return std::strtod(value, nullptr);
}

double adaptive_group_local_slack_weight() {

	const char *value = std::getenv("TPP_GROUP_ADAPTIVE_LOCAL_SLACK_WEIGHT");

	if (value == nullptr) {
		return 3.0;
	}

	return std::strtod(value, nullptr);
}

double adaptive_group_size_weight() {

	const char *value = std::getenv("TPP_GROUP_ADAPTIVE_SIZE_WEIGHT");

	if (value == nullptr) {
		return 0.05;
	}

	return std::strtod(value, nullptr);
}

bool piece_group_require_touch() {

	const char *enabled = std::getenv("TPP_GROUP_REQUIRE_TOUCH");
	return enabled != nullptr && std::string_view(enabled) != "0";
}

double piece_group_order_penalty() {

	const char *value = std::getenv("TPP_GROUP_ORDER_PENALTY");

	if (value == nullptr) {
		return 0.0;
	}

	return std::strtod(value, nullptr);
}

bool use_synthetic_cover() {

	const char *enabled = std::getenv("TPP_USE_SYNTHETIC_COVER");
	return enabled != nullptr && std::string_view(enabled) != "0";
}

std::string synthetic_cover_pattern() {

	const char *value = std::getenv("TPP_SYNTHETIC_COVER_PATTERN");

	if (value == nullptr) {
		return "stair";
	}

	return value;
}

bool use_port_bound() {

	const char *enabled = std::getenv("TPP_PORT_BOUND");
	return enabled != nullptr && std::string_view(enabled) != "0";
}

bool use_refinement_bound() {

	const char *enabled = std::getenv("TPP_REFINEMENT_BOUND");
	return enabled != nullptr && std::string_view(enabled) != "0";
}

double refinement_gap_ratio() {

	const char *value = std::getenv("TPP_REFINEMENT_GAP_RATIO");

	if (value == nullptr) {
		return std::numeric_limits<double>::infinity();
	}

	return std::strtod(value, nullptr);
}

size_t refinement_min_depth() {

	const char *value = std::getenv("TPP_REFINEMENT_MIN_DEPTH");

	if (value == nullptr) {
		return 0;
	}

	return std::strtoull(value, nullptr, 10);
}

size_t refinement_window_size() {

	const char *value = std::getenv("TPP_REFINEMENT_WINDOW_SIZE");

	if (value == nullptr) {
		return 1;
	}

	return std::strtoull(value, nullptr, 10);
}

size_t refinement_max_combinations() {

	const char *value = std::getenv("TPP_REFINEMENT_MAX_COMBINATIONS");

	if (value == nullptr) {
		return 64;
	}

	return std::strtoull(value, nullptr, 10);
}

bool use_contact_bound() {

	const char *enabled = std::getenv("TPP_CONTACT_BOUND");
	return enabled != nullptr && std::string_view(enabled) != "0";
}

double contact_gap_ratio() {

	const char *value = std::getenv("TPP_CONTACT_GAP_RATIO");

	if (value == nullptr) {
		return 1e-5;
	}

	return std::strtod(value, nullptr);
}

size_t contact_min_depth() {

	const char *value = std::getenv("TPP_CONTACT_MIN_DEPTH");

	if (value == nullptr) {
		return 8;
	}

	return std::strtoull(value, nullptr, 10);
}

size_t contact_max_polygons() {

	const char *value = std::getenv("TPP_CONTACT_MAX_POLYGONS");

	if (value == nullptr) {
		return 2;
	}

	return std::strtoull(value, nullptr, 10);
}

size_t contact_max_combinations() {

	const char *value = std::getenv("TPP_CONTACT_MAX_COMBINATIONS");

	if (value == nullptr) {
		return 64;
	}

	return std::strtoull(value, nullptr, 10);
}

struct PieceGraphBoundCache {
	vector<vector<double>> start_distances;
	vector<vector<double>> target_distances;
	vector<vector<vector<double>>> transitions;

	explicit PieceGraphBoundCache(
		const Vector2 &start,
		const Vector2 &target,
		const vector<vector<vector<Vector2>>> &convex_pieces
	) {
		start_distances.resize(convex_pieces.size());
		target_distances.resize(convex_pieces.size());

		for (size_t i = 0; i < convex_pieces.size(); i++) {
			start_distances[i].reserve(convex_pieces[i].size());
			target_distances[i].reserve(convex_pieces[i].size());

			for (const auto &piece : convex_pieces[i]) {
				start_distances[i].push_back(point_polygon_distance(start, piece));
				target_distances[i].push_back(point_polygon_distance(target, piece));
			}
		}

		if (convex_pieces.size() >= 2) {
			transitions.resize(convex_pieces.size() - 1);
		}

		for (size_t i = 0; i + 1 < convex_pieces.size(); i++) {
			transitions[i].resize(convex_pieces[i].size());

			for (size_t a = 0; a < convex_pieces[i].size(); a++) {
				transitions[i][a].reserve(convex_pieces[i + 1].size());

				for (size_t b = 0; b < convex_pieces[i + 1].size(); b++) {
					transitions[i][a].push_back(polygon_polygon_distance(convex_pieces[i][a], convex_pieces[i + 1][b]));
				}
			}
		}
	}

	double bound(const vector<size_t> &selected) const {

		if (start_distances.empty()) {
			return 0.0;
		}

		vector<double> dp = start_distances.front();

		if (!selected.empty()) {
			for (size_t piece = 0; piece < dp.size(); piece++) {
				if (piece != selected.front()) {
					dp[piece] = std::numeric_limits<double>::infinity();
				}
			}
		}

		for (size_t i = 0; i + 1 < start_distances.size(); i++) {
			vector<double> next(start_distances[i + 1].size(), std::numeric_limits<double>::infinity());

			for (size_t a = 0; a < dp.size(); a++) {
				if (!std::isfinite(dp[a])) {
					continue;
				}

				for (size_t b = 0; b < next.size(); b++) {
					if (i + 1 < selected.size() && b != selected[i + 1]) {
						continue;
					}

					next[b] = std::min(next[b], dp[a] + transitions[i][a][b]);
				}
			}

			dp = std::move(next);
		}

		double result = std::numeric_limits<double>::infinity();

		for (size_t piece = 0; piece < dp.size(); piece++) {
			result = std::min(result, dp[piece] + target_distances.back()[piece]);
		}

		return result;
	}
};

double triangle_cover_radius(const Vector2 &a, const Vector2 &b, const Vector2 &c) {

	const double ab = a.distance_to(b);
	const double bc = b.distance_to(c);
	const double ca = c.distance_to(a);
	const double longest = std::max({ab, bc, ca});
	const double longest_squared = longest * longest;
	const double side_sum_squared = ab * ab + bc * bc + ca * ca;

	if (longest_squared * 2.0 >= side_sum_squared - 1e-12) {
		return longest * 0.5;
	}

	const double area2 = std::abs((b - a).cross(c - a));

	if (area2 <= 1e-12) {
		return longest * 0.5;
	}

	return ab * bc * ca / (2.0 * area2);
}

struct PortState {
	size_t piece = 0;
	Vector2 point;
	double radius = 0.0;
};

struct PortBoundCache {
	vector<vector<PortState>> layers;

	explicit PortBoundCache(const vector<vector<vector<Vector2>>> &convex_pieces) {
		layers.reserve(convex_pieces.size());

		for (const auto &polygon_pieces : convex_pieces) {
			vector<PortState> layer;

			for (size_t piece_index = 0; piece_index < polygon_pieces.size(); piece_index++) {
				const auto &piece = polygon_pieces[piece_index];

				if (piece.empty()) {
					continue;
				}

				Vector2 centroid;

				for (const auto &point : piece) {
					centroid = centroid + point;
				}

				centroid = centroid / static_cast<double>(piece.size());

				vector<Vector2> boundary_ports;
				boundary_ports.reserve(piece.size() * 2);

				for (size_t i = 0; i < piece.size(); i++) {
					const Vector2 &current = piece[i];
					const Vector2 &next = piece[(i + 1) % piece.size()];
					boundary_ports.push_back(current);
					boundary_ports.push_back((current + next) * 0.5);
				}

				double radius = 0.0;

				for (size_t i = 0; i < boundary_ports.size(); i++) {
					radius = std::max(
						radius,
						triangle_cover_radius(centroid, boundary_ports[i], boundary_ports[(i + 1) % boundary_ports.size()])
					);
				}

				for (const auto &port : boundary_ports) {
					layer.push_back({piece_index, port, radius});
				}

				layer.push_back({piece_index, centroid, radius});
			}

			layers.push_back(std::move(layer));
		}
	}

	double bound(const Vector2 &start, const Vector2 &target, const vector<size_t> &selected) const {

		if (layers.empty()) {
			return start.distance_to(target);
		}

		vector<double> dp(layers.front().size(), std::numeric_limits<double>::infinity());

		for (size_t state = 0; state < layers.front().size(); state++) {
			const auto &port = layers.front()[state];

			if (!selected.empty() && port.piece != selected.front()) {
				continue;
			}

			dp[state] = std::max(0.0, start.distance_to(port.point) - port.radius);
		}

		for (size_t layer_index = 0; layer_index + 1 < layers.size(); layer_index++) {
			vector<double> next_dp(layers[layer_index + 1].size(), std::numeric_limits<double>::infinity());

			for (size_t previous = 0; previous < layers[layer_index].size(); previous++) {
				if (!std::isfinite(dp[previous])) {
					continue;
				}

				const auto &previous_port = layers[layer_index][previous];

				for (size_t next = 0; next < layers[layer_index + 1].size(); next++) {
					const auto &next_port = layers[layer_index + 1][next];

					if (layer_index + 1 < selected.size() && next_port.piece != selected[layer_index + 1]) {
						continue;
					}

					const double segment_bound = std::max(
						0.0,
						previous_port.point.distance_to(next_port.point) - previous_port.radius - next_port.radius
					);
					next_dp[next] = std::min(next_dp[next], dp[previous] + segment_bound);
				}
			}

			dp = std::move(next_dp);
		}

		double result = std::numeric_limits<double>::infinity();

		for (size_t state = 0; state < layers.back().size(); state++) {
			const auto &port = layers.back()[state];
			result = std::min(result, dp[state] + std::max(0.0, port.point.distance_to(target) - port.radius));
		}

		return result;
	}
};

bool any_polygons_intersect_or_touch(const vector<vector<Vector2>> &polygons) {

	for (size_t i = 0; i < polygons.size(); i++) {
		for (size_t j = i + 1; j < polygons.size(); j++) {
			if (polygons_intersect_or_touch(polygons[i], polygons[j])) {
				return true;
			}
		}
	}

	return false;
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

double polygon_area(const vector<Vector2> &polygon) {

	double twice_area = 0.0;

	for (size_t i = 0; i < polygon.size(); i++) {
		twice_area += polygon[i].cross(polygon[(i + 1) % polygon.size()]);
	}

	return std::abs(twice_area) * 0.5;
}

struct PieceGroup {
	vector<size_t> piece_indices;
	vector<Vector2> hull;
	double piece_area_sum = 0.0;
};

vector<Vector2> group_hull_from_indices(const vector<vector<Vector2>> &pieces, const vector<size_t> &indices) {

	vector<Vector2> points;

	for (const size_t index : indices) {
		points.insert(points.end(), pieces[index].begin(), pieces[index].end());
	}

	return convex_hull(points);
}

PieceGroup make_piece_group(const vector<vector<Vector2>> &pieces, vector<size_t> indices) {

	PieceGroup group;
	group.piece_indices = std::move(indices);
	group.hull = group_hull_from_indices(pieces, group.piece_indices);

	for (const size_t index : group.piece_indices) {
		group.piece_area_sum += polygon_area(pieces[index]);
	}

	return group;
}

double group_excess_ratio(const PieceGroup &group) {

	if (group.piece_area_sum <= 0.0) {
		return std::numeric_limits<double>::infinity();
	}

	return std::max(0.0, polygon_area(group.hull) - group.piece_area_sum) / group.piece_area_sum;
}

size_t group_min_piece_index(const PieceGroup &group) {
	return *std::min_element(group.piece_indices.begin(), group.piece_indices.end());
}

size_t group_max_piece_index(const PieceGroup &group) {
	return *std::max_element(group.piece_indices.begin(), group.piece_indices.end());
}

size_t group_order_gap(const PieceGroup &a, const PieceGroup &b) {

	const size_t a_min = group_min_piece_index(a);
	const size_t a_max = group_max_piece_index(a);
	const size_t b_min = group_min_piece_index(b);
	const size_t b_max = group_max_piece_index(b);

	if (a_max < b_min) {
		return b_min - a_max - 1;
	}

	if (b_max < a_min) {
		return a_min - b_max - 1;
	}

	return 0;
}

vector<PieceGroup> make_piece_groups_for_polygon(
	const vector<vector<Vector2>> &pieces,
	double max_excess_ratio,
	size_t max_group_size
) {

	vector<PieceGroup> groups;
	groups.reserve(pieces.size());

	for (size_t i = 0; i < pieces.size(); i++) {
		groups.push_back(make_piece_group(pieces, {i}));
	}

	const bool require_touch = piece_group_require_touch();
	const double order_penalty = piece_group_order_penalty();

	while (true) {
		std::optional<std::tuple<size_t, size_t, double, PieceGroup>> best_merge;

		for (size_t i = 0; i < groups.size(); i++) {
			for (size_t j = i + 1; j < groups.size(); j++) {
				if (groups[i].piece_indices.size() + groups[j].piece_indices.size() > max_group_size) {
					continue;
				}

				if (require_touch && !polygons_intersect_or_touch(groups[i].hull, groups[j].hull)) {
					continue;
				}

				vector<size_t> merged_indices = groups[i].piece_indices;
				merged_indices.insert(merged_indices.end(), groups[j].piece_indices.begin(), groups[j].piece_indices.end());
				std::sort(merged_indices.begin(), merged_indices.end());

				PieceGroup merged = make_piece_group(pieces, std::move(merged_indices));
				const double excess_ratio = group_excess_ratio(merged);
				const double merge_score = excess_ratio + order_penalty * static_cast<double>(group_order_gap(groups[i], groups[j]));

				if (excess_ratio > max_excess_ratio) {
					continue;
				}

				if (!best_merge || merge_score < std::get<2>(*best_merge)) {
					best_merge = std::make_tuple(i, j, merge_score, std::move(merged));
				}
			}
		}

		if (!best_merge) {
			break;
		}

		const size_t first = std::get<0>(*best_merge);
		const size_t second = std::get<1>(*best_merge);
		PieceGroup merged = std::move(std::get<3>(*best_merge));
		groups[first] = std::move(merged);
		groups.erase(groups.begin() + static_cast<std::ptrdiff_t>(second));
	}

	std::sort(groups.begin(), groups.end(), [](const PieceGroup &a, const PieceGroup &b) {
		return group_min_piece_index(a) < group_min_piece_index(b);
	});

	return groups;
}

double local_distance_through_piece(
	const std::optional<Vector2> &previous_point,
	const vector<Vector2> *previous_polygon,
	const vector<Vector2> &piece,
	const std::optional<Vector2> &next_point,
	const vector<Vector2> *next_polygon
) {

	double distance = 0.0;

	if (previous_point) {
		distance += point_polygon_distance(*previous_point, piece);
	} else if (previous_polygon != nullptr) {
		distance += polygon_polygon_distance(*previous_polygon, piece);
	}

	if (next_point) {
		distance += point_polygon_distance(*next_point, piece);
	} else if (next_polygon != nullptr) {
		distance += polygon_polygon_distance(piece, *next_polygon);
	}

	return distance;
}

double group_local_slack_ratio(
	const vector<vector<Vector2>> &pieces,
	const PieceGroup &group,
	const std::optional<Vector2> &previous_point,
	const vector<Vector2> *previous_polygon,
	const std::optional<Vector2> &next_point,
	const vector<Vector2> *next_polygon
) {

	double best_piece_distance = std::numeric_limits<double>::infinity();

	for (const size_t piece_index : group.piece_indices) {
		best_piece_distance = std::min(
			best_piece_distance,
			local_distance_through_piece(previous_point, previous_polygon, pieces[piece_index], next_point, next_polygon)
		);
	}

	const double group_distance = local_distance_through_piece(
		previous_point,
		previous_polygon,
		group.hull,
		next_point,
		next_polygon
	);
	const double slack = std::max(0.0, best_piece_distance - group_distance);
	return slack / std::max(1.0, best_piece_distance);
}

vector<PieceGroup> make_adaptive_piece_groups_for_polygon(
	const vector<vector<Vector2>> &pieces,
	const std::optional<Vector2> &previous_point,
	const vector<Vector2> *previous_polygon,
	const std::optional<Vector2> &next_point,
	const vector<Vector2> *next_polygon,
	double max_excess_ratio,
	size_t max_group_size
) {

	vector<PieceGroup> groups;
	groups.reserve(pieces.size());

	for (size_t i = 0; i < pieces.size(); i++) {
		groups.push_back(make_piece_group(pieces, {i}));
	}

	const bool require_touch = piece_group_require_touch();
	const double order_penalty = piece_group_order_penalty();
	const double max_local_slack_ratio = adaptive_group_max_local_slack_ratio();
	const double excess_weight = adaptive_group_excess_weight();
	const double local_slack_weight = adaptive_group_local_slack_weight();
	const double size_weight = adaptive_group_size_weight();

	while (true) {
		std::optional<std::tuple<size_t, size_t, double, PieceGroup>> best_merge;

		for (size_t i = 0; i < groups.size(); i++) {
			for (size_t j = i + 1; j < groups.size(); j++) {
				const size_t merged_size = groups[i].piece_indices.size() + groups[j].piece_indices.size();

				if (merged_size > max_group_size) {
					continue;
				}

				if (require_touch && !polygons_intersect_or_touch(groups[i].hull, groups[j].hull)) {
					continue;
				}

				vector<size_t> merged_indices = groups[i].piece_indices;
				merged_indices.insert(merged_indices.end(), groups[j].piece_indices.begin(), groups[j].piece_indices.end());
				std::sort(merged_indices.begin(), merged_indices.end());

				PieceGroup merged = make_piece_group(pieces, std::move(merged_indices));
				const double excess_ratio = group_excess_ratio(merged);

				if (excess_ratio > max_excess_ratio) {
					continue;
				}

				const double local_slack_ratio = group_local_slack_ratio(
					pieces,
					merged,
					previous_point,
					previous_polygon,
					next_point,
					next_polygon
				);

				if (local_slack_ratio > max_local_slack_ratio) {
					continue;
				}

				const double size_ratio = static_cast<double>(merged_size - 1) / static_cast<double>(std::max<size_t>(1, max_group_size - 1));
				const double merge_score =
					excess_weight * excess_ratio
					+ local_slack_weight * local_slack_ratio
					+ size_weight * size_ratio
					+ order_penalty * static_cast<double>(group_order_gap(groups[i], groups[j]));

				if (!best_merge || merge_score < std::get<2>(*best_merge)) {
					best_merge = std::make_tuple(i, j, merge_score, std::move(merged));
				}
			}
		}

		if (!best_merge) {
			break;
		}

		const size_t first = std::get<0>(*best_merge);
		const size_t second = std::get<1>(*best_merge);
		PieceGroup merged = std::move(std::get<3>(*best_merge));
		groups[first] = std::move(merged);
		groups.erase(groups.begin() + static_cast<std::ptrdiff_t>(second));
	}

	std::sort(groups.begin(), groups.end(), [](const PieceGroup &a, const PieceGroup &b) {
		return group_min_piece_index(a) < group_min_piece_index(b);
	});

	return groups;
}

struct GridPoint {
	int x = 0;
	int y = 0;

	auto operator<=>(const GridPoint&) const = default;
};

struct GridRect {
	int x0 = 0;
	int y0 = 0;
	int x1 = 0;
	int y1 = 0;
};

vector<Vector2> rectangle_polygon(double x0, double y0, double x1, double y1) {

	return {
		{x0, y0},
		{x1, y0},
		{x1, y1},
		{x0, y1},
	};
}

vector<Vector2> union_boundary_from_grid_rectangles(const vector<GridRect> &rectangles, double scale, const Vector2 &offset) {

	std::set<GridPoint> cells;

	for (const auto &rectangle : rectangles) {
		for (int x = rectangle.x0; x < rectangle.x1; x++) {
			for (int y = rectangle.y0; y < rectangle.y1; y++) {
				cells.insert({x, y});
			}
		}
	}

	auto occupied = [&](int x, int y) {
		return cells.contains({x, y});
	};

	std::map<GridPoint, GridPoint> next_vertex;

	for (const auto &cell : cells) {
		const int x = cell.x;
		const int y = cell.y;

		if (!occupied(x, y - 1)) {
			next_vertex[{x, y}] = {x + 1, y};
		}

		if (!occupied(x + 1, y)) {
			next_vertex[{x + 1, y}] = {x + 1, y + 1};
		}

		if (!occupied(x, y + 1)) {
			next_vertex[{x + 1, y + 1}] = {x, y + 1};
		}

		if (!occupied(x - 1, y)) {
			next_vertex[{x, y + 1}] = {x, y};
		}
	}

	if (next_vertex.empty()) {
		return {};
	}

	GridPoint start = next_vertex.begin()->first;

	for (const auto &[point, _] : next_vertex) {
		if (std::tie(point.y, point.x) < std::tie(start.y, start.x)) {
			start = point;
		}
	}

	vector<Vector2> polygon;
	GridPoint current = start;

	do {
		polygon.push_back({offset.x + static_cast<double>(current.x) * scale, offset.y + static_cast<double>(current.y) * scale});
		current = next_vertex.at(current);
	} while (current != start && polygon.size() <= next_vertex.size() + 1);

	return tpp::remove_collinear_points(polygon);
}

vector<GridRect> synthetic_cover_rectangles(size_t polygon_index, size_t case_index, std::string_view pattern) {

	const int variant = static_cast<int>((polygon_index + case_index) % 4);

	if (pattern == "cross") {
		if (variant == 0) {
			return {{0, 2, 12, 5}, {4, 0, 8, 8}};
		}

		if (variant == 1) {
			return {{0, 1, 10, 4}, {3, 0, 6, 7}, {6, 3, 12, 6}};
		}

		if (variant == 2) {
			return {{0, 3, 14, 6}, {2, 1, 5, 8}, {8, 0, 11, 9}};
		}

		return {{0, 2, 13, 5}, {3, 0, 6, 8}, {7, 1, 10, 7}};
	}

	if (pattern == "overlap") {
		if (variant == 0) {
			return {{0, 0, 8, 3}, {2, 1, 10, 5}, {5, 3, 13, 7}};
		}

		if (variant == 1) {
			return {{0, 2, 12, 5}, {1, 0, 5, 7}, {4, 1, 9, 8}, {8, 3, 14, 6}};
		}

		if (variant == 2) {
			return {{0, 0, 6, 4}, {3, 1, 10, 6}, {7, 0, 13, 4}, {9, 3, 15, 7}};
		}

		return {{0, 1, 9, 4}, {2, 0, 6, 8}, {5, 3, 13, 6}, {9, 2, 15, 9}};
	}

	if (pattern == "comb") {
		if (variant == 0) {
			return {{0, 0, 14, 2}, {1, 1, 3, 7}, {5, 1, 7, 6}, {9, 1, 11, 8}};
		}

		if (variant == 1) {
			return {{0, 4, 14, 6}, {2, 0, 4, 5}, {6, 2, 8, 8}, {10, 1, 12, 5}};
		}

		if (variant == 2) {
			return {{0, 0, 16, 2}, {2, 1, 4, 8}, {6, 1, 8, 6}, {10, 1, 12, 9}, {13, 1, 15, 5}};
		}

		return {{0, 5, 16, 7}, {1, 0, 3, 6}, {5, 2, 7, 6}, {9, 1, 11, 8}, {13, 3, 15, 6}};
	}

	if (pattern == "longstair") {
		if (variant == 0) {
			return {{0, 0, 8, 2}, {3, 1, 11, 3}, {6, 2, 14, 4}, {9, 3, 17, 5}};
		}

		if (variant == 1) {
			return {{0, 3, 8, 5}, {3, 2, 11, 4}, {6, 1, 14, 3}, {9, 0, 17, 2}};
		}

		if (variant == 2) {
			return {{0, 0, 7, 3}, {2, 2, 10, 5}, {5, 4, 13, 7}, {8, 6, 16, 9}};
		}

		return {{0, 6, 7, 9}, {2, 4, 10, 7}, {5, 2, 13, 5}, {8, 0, 16, 3}};
	}

	if (variant == 0) {
		return {{0, 0, 4, 2}, {2, 1, 6, 3}, {4, 2, 8, 4}};
	}

	if (variant == 1) {
		return {{0, 1, 5, 3}, {1, 0, 3, 5}, {3, 2, 7, 4}};
	}

	if (variant == 2) {
		return {{0, 0, 3, 3}, {2, 1, 6, 4}, {5, 0, 8, 2}, {6, 1, 9, 5}};
	}

	return {{0, 0, 5, 2}, {1, 1, 4, 5}, {3, 3, 8, 5}, {6, 2, 9, 4}};
}

SyntheticCoverSuite make_synthetic_cover_suite(size_t case_count) {

	SyntheticCoverSuite suite;
	suite.cases.reserve(case_count);
	suite.covers.reserve(case_count);
	const std::string pattern = synthetic_cover_pattern();

	for (size_t case_index = 0; case_index < case_count; case_index++) {
		const size_t polygon_count = 8 + case_index % 9;
		const double scale = 0.75;
		const double spacing = pattern == "stair" ? 9.0 : 14.0;
		tpp::TestCase test_case;
		test_case.start = {-2.0, 1.5};
		test_case.target = {static_cast<double>(polygon_count) * spacing + 1.0, 3.0};
		vector<vector<vector<Vector2>>> case_cover;

		for (size_t polygon_index = 0; polygon_index < polygon_count; polygon_index++) {
			const Vector2 offset{
				static_cast<double>(polygon_index) * spacing,
				static_cast<double>((polygon_index + case_index) % 3) * 1.75,
			};
			const vector<GridRect> rectangles = synthetic_cover_rectangles(polygon_index, case_index, pattern);
			vector<vector<Vector2>> polygon_cover;
			polygon_cover.reserve(rectangles.size());

			for (const auto &rectangle : rectangles) {
				polygon_cover.push_back(rectangle_polygon(
					offset.x + static_cast<double>(rectangle.x0) * scale,
					offset.y + static_cast<double>(rectangle.y0) * scale,
					offset.x + static_cast<double>(rectangle.x1) * scale,
					offset.y + static_cast<double>(rectangle.y1) * scale
				));
			}

			test_case.polygons.push_back(union_boundary_from_grid_rectangles(rectangles, scale, offset));
			case_cover.push_back(std::move(polygon_cover));
		}

		suite.cases.push_back(std::move(test_case));
		suite.covers.push_back(std::move(case_cover));
	}

	return suite;
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

double polygon_perimeter(const vector<Vector2> &polygon) {

	double perimeter = 0.0;

	for (size_t i = 0; i < polygon.size(); i++) {
		perimeter += polygon[i].distance_to(polygon[(i + 1) % polygon.size()]);
	}

	return perimeter;
}

vector<Vector2> make_fixed_edge_sampled_polygon(const vector<Vector2> &polygon, size_t points_per_edge) {

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

vector<Vector2> make_evenly_spaced_polygon(const vector<Vector2> &polygon, size_t point_count) {

	if (polygon.empty() || point_count == 0) {
		return {};
	}

	const double perimeter = polygon_perimeter(polygon);

	if (perimeter == 0.0) {
		return vector<Vector2>(point_count, polygon.front());
	}

	vector<Vector2> interpolated;
	interpolated.reserve(point_count);

	for (const auto &vertex : polygon) {
		interpolated.push_back(vertex);
	}

	if (point_count <= polygon.size()) {
		return interpolated;
	}

	const size_t extra_point_count = point_count - polygon.size();
	const double spacing = perimeter / static_cast<double>(extra_point_count);
	size_t edge_index = 0;
	double edge_start_distance = 0.0;
	double edge_length = polygon[0].distance_to(polygon[1 % polygon.size()]);

	for (size_t sample_index = 0; sample_index < extra_point_count; sample_index++) {
		const double target_distance = spacing * (static_cast<double>(sample_index) + 0.5);

		while (edge_index + 1 < polygon.size() && edge_start_distance + edge_length < target_distance) {
			edge_start_distance += edge_length;
			edge_index++;
			edge_length = polygon[edge_index].distance_to(polygon[(edge_index + 1) % polygon.size()]);
		}

		const auto &a = polygon[edge_index];
		const auto &b = polygon[(edge_index + 1) % polygon.size()];
		const double weight = edge_length == 0.0 ? 0.0 : (target_distance - edge_start_distance) / edge_length;
		interpolated.push_back(a.lerp(b, weight));
	}

	return interpolated;
}

double approximation_work_budget(double total_combinations) {

	double budget = APPROXIMATION_WORK_BUDGET;

	if (const char *raw_budget = std::getenv("TPP_APPROX_WORK_BUDGET")) {
		const double parsed_budget = std::atof(raw_budget);

		if (parsed_budget > 0.0) {
			budget = parsed_budget;
		}
	}

	const char *mode = std::getenv("TPP_APPROX_BUDGET_MODE");

	if (mode != nullptr && std::string_view(mode) == "adaptive") {
		const double complexity = total_combinations > 0.0 ? std::log2(total_combinations) : 0.0;
		const double factor = std::min(APPROXIMATION_ADAPTIVE_MAX_FACTOR, std::exp2(complexity / 32.0));
		budget *= factor;
	}

	return budget;
}

bool use_legacy_approximation_sampler() {

	const char *sampler = std::getenv("TPP_APPROX_SAMPLER");
	return sampler != nullptr && std::string_view(sampler) == "legacy";
}

vector<size_t> choose_approximation_point_counts(const vector<vector<Vector2>> &polygons, double work_budget) {

	vector<double> perimeters;
	perimeters.reserve(polygons.size());

	for (const auto &polygon : polygons) {
		perimeters.push_back(polygon_perimeter(polygon));
	}

	double weighted_work = 0.0;

	for (size_t i = 0; i + 1 < perimeters.size(); i++) {
		weighted_work += perimeters[i] * perimeters[i + 1];
	}

	double scale = 1.0;

	if (weighted_work > 0.0) {
		scale = std::sqrt(work_budget / weighted_work);
	} else if (!perimeters.empty() && perimeters.front() > 0.0) {
		scale = std::sqrt(work_budget) / perimeters.front();
	}

	vector<size_t> point_counts;
	point_counts.reserve(polygons.size());

	for (size_t i = 0; i < polygons.size(); i++) {
		const auto requested_count = static_cast<size_t>(std::ceil(perimeters[i] * scale));
		point_counts.push_back(std::max(polygons[i].size(), requested_count));
	}

	return point_counts;
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
	const vector<vector<PieceGroup>> *piece_groups,
	const vector<Vector2> &approximate_path,
	size_t max_calls,
	size_t max_branching,
	double max_seconds,
	ConvexSolverFunction solver,
	ConvexLengthSolverFunction length_solver,
	size_t case_index,
	bool show_progress
) {

	BranchAndBoundResult result;
	result.initial_length = path_length(start, target, approximate_path);
	const auto bnb_start_time = std::chrono::steady_clock::now();
	auto last_progress_time = std::chrono::steady_clock::now();
	vector<vector<vector<Vector2>>> grouped_branch_pieces;
	const vector<vector<vector<Vector2>>> *branch_pieces = &convex_pieces;

	if (piece_groups != nullptr) {
		grouped_branch_pieces.reserve(piece_groups->size());

		for (const auto &polygon_groups : *piece_groups) {
			vector<vector<Vector2>> polygon_branch_pieces;
			polygon_branch_pieces.reserve(polygon_groups.size());

			for (const auto &group : polygon_groups) {
				polygon_branch_pieces.push_back(group.hull);
			}

			grouped_branch_pieces.push_back(std::move(polygon_branch_pieces));
		}

		branch_pieces = &grouped_branch_pieces;
	}

	std::optional<PieceGraphBoundCache> piece_graph_bound_cache;
	std::optional<PortBoundCache> port_bound_cache;
	const bool piece_graph_bound_enabled = use_piece_graph_bound();
	const bool port_bound_enabled = use_port_bound();
	const bool refinement_bound_enabled = use_refinement_bound();
	const double refinement_gap_ratio_value = refinement_gap_ratio();
	const size_t refinement_min_depth_value = refinement_min_depth();
	const size_t refinement_window_size_value = refinement_window_size();
	const size_t refinement_max_combinations_value = refinement_max_combinations();
	const bool contact_bound_enabled = use_contact_bound();
	const double contact_gap_ratio_value = contact_gap_ratio();
	const size_t contact_min_depth_value = contact_min_depth();
	const size_t contact_max_polygons_value = contact_max_polygons();
	const size_t contact_max_combinations_value = contact_max_combinations();

	if (piece_graph_bound_enabled) {
		const auto graph_precompute_start_time = std::chrono::steady_clock::now();
		piece_graph_bound_cache.emplace(start, target, *branch_pieces);
		const auto graph_precompute_end_time = std::chrono::steady_clock::now();
		result.piece_graph_precompute_seconds =
			std::chrono::duration<double>(graph_precompute_end_time - graph_precompute_start_time).count();
	}

	if (port_bound_enabled) {
		const auto port_precompute_start_time = std::chrono::steady_clock::now();
		port_bound_cache.emplace(*branch_pieces);
		const auto port_precompute_end_time = std::chrono::steady_clock::now();
		result.port_bound_precompute_seconds =
			std::chrono::duration<double>(port_precompute_end_time - port_precompute_start_time).count();
	}

	auto elapsed_seconds = [&]() {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - bnb_start_time).count();
	};

	auto stop_by_time = [&]() {
		if (elapsed_seconds() < max_seconds) {
			return false;
		}

		result.exhausted = false;
		result.time_limited = true;
		return true;
	};

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

	auto before_convex_call = [&](const vector<vector<Vector2>> &input_polygons) -> bool {
		if (stop_by_time()) {
			return false;
		}

		if (result.convex_calls >= max_calls) {
			result.exhausted = false;
			return false;
		}

		const size_t vertices = vertex_count(input_polygons);
		result.min_vertices = std::min(result.min_vertices, vertices);
		result.max_vertices = std::max(result.max_vertices, vertices);

		return true;
	};

	auto solve_convex_length = [&](const vector<vector<Vector2>> &input_polygons, double &kind_seconds) -> double {
		if (!before_convex_call(input_polygons)) {
			return std::numeric_limits<double>::infinity();
		}

		const auto solver_start_time = std::chrono::steady_clock::now();

		try {
			const double length = length_solver(start, target, input_polygons);
			const auto solver_end_time = std::chrono::steady_clock::now();
			const double elapsed_seconds = std::chrono::duration<double>(solver_end_time - solver_start_time).count();
			result.solver_seconds += elapsed_seconds;
			kind_seconds += elapsed_seconds;
			result.convex_calls++;
			result.checksum += length;
			update_progress(false);
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

	auto solve_convex_path = [&](const vector<vector<Vector2>> &input_polygons, vector<Vector2> &output_path, double &kind_seconds) -> double {
		if (!before_convex_call(input_polygons)) {
			return std::numeric_limits<double>::infinity();
		}

		const auto solver_start_time = std::chrono::steady_clock::now();

		try {
			vector<Vector2> path = solver(start, target, input_polygons);
			const auto solver_end_time = std::chrono::steady_clock::now();
			const double elapsed_seconds = std::chrono::duration<double>(solver_end_time - solver_start_time).count();
			result.solver_seconds += elapsed_seconds;
			kind_seconds += elapsed_seconds;
			result.convex_calls++;
			const double length = path_length(start, target, path);
			result.checksum += length;
			update_progress(false);
			output_path = std::move(path);
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

	auto solve_piece_graph_bound = [&](const vector<size_t> &selected) {
		if (!piece_graph_bound_cache) {
			return 0.0;
		}

		const auto graph_bound_start_time = std::chrono::steady_clock::now();
		const double bound = piece_graph_bound_cache->bound(selected);
		const auto graph_bound_end_time = std::chrono::steady_clock::now();
		result.piece_graph_bound_seconds += std::chrono::duration<double>(graph_bound_end_time - graph_bound_start_time).count();
		result.piece_graph_bound_calls++;
		return bound;
	};

	auto solve_port_bound = [&](const vector<size_t> &selected) {
		if (!port_bound_cache) {
			return 0.0;
		}

		const auto port_bound_start_time = std::chrono::steady_clock::now();
		const double bound = port_bound_cache->bound(start, target, selected);
		const auto port_bound_end_time = std::chrono::steady_clock::now();
		result.port_bound_seconds += std::chrono::duration<double>(port_bound_end_time - port_bound_start_time).count();
		result.port_bound_calls++;
		return bound;
	};

	auto solve_refinement_bound = [&](const vector<size_t> &selected, double incumbent) {
		if (!refinement_bound_enabled || selected.size() >= branch_pieces->size()) {
			return 0.0;
		}

		const auto refinement_start_time = std::chrono::steady_clock::now();
		const size_t first_refined_polygon = selected.size();
		size_t refined_polygon_count = 0;
		size_t combination_count = 1;

		while (
			refined_polygon_count < refinement_window_size_value
			&& first_refined_polygon + refined_polygon_count < branch_pieces->size()
		) {
			const size_t polygon = first_refined_polygon + refined_polygon_count;
			const size_t next_count = combination_count * (*branch_pieces)[polygon].size();

			if (next_count > refinement_max_combinations_value) {
				break;
			}

			combination_count = next_count;
			refined_polygon_count++;
		}

		if (refined_polygon_count == 0) {
			return 0.0;
		}

		double refined_bound = std::numeric_limits<double>::infinity();
		double refinement_solver_seconds = 0.0;
		vector<size_t> refined_piece_indices(refined_polygon_count, 0);
		bool done = false;

		while (!done) {
			vector<vector<Vector2>> refined_instance;
			refined_instance.reserve(convex_pieces.size());

			for (size_t j = 0; j < selected.size(); j++) {
				refined_instance.push_back((*branch_pieces)[j][selected[j]]);
			}

			for (size_t j = 0; j < refined_polygon_count; j++) {
				const size_t polygon = first_refined_polygon + j;
				refined_instance.push_back((*branch_pieces)[polygon][refined_piece_indices[j]]);
			}

			for (size_t j = first_refined_polygon + refined_polygon_count; j < convex_hulls.size(); j++) {
				refined_instance.push_back(convex_hulls[j]);
			}

			const size_t calls_before_candidate = result.convex_calls;
			const double candidate = solve_convex_length(refined_instance, refinement_solver_seconds);
			if (result.convex_calls > calls_before_candidate) {
				result.refinement_bound_calls++;
			}
			refined_bound = std::min(refined_bound, candidate);

			if (!result.exhausted || refined_bound <= incumbent) {
				// The minimum can only decrease as we inspect more pieces, so
				// no pruning proof is possible after this point.
				break;
			}

			for (size_t index = refined_piece_indices.size(); index > 0; index--) {
				const size_t piece_slot = index - 1;
				refined_piece_indices[piece_slot]++;

				if (refined_piece_indices[piece_slot] < (*branch_pieces)[first_refined_polygon + piece_slot].size()) {
					break;
				}

				refined_piece_indices[piece_slot] = 0;

				if (piece_slot == 0) {
					done = true;
				}
			}
		}

		const auto refinement_end_time = std::chrono::steady_clock::now();
		result.refinement_bound_seconds += std::chrono::duration<double>(refinement_end_time - refinement_start_time).count();
		return refined_bound;
	};

	auto solve_contact_bound = [&](const vector<size_t> &selected, const vector<vector<Vector2>> &bound_instance, double incumbent) {
		if (!contact_bound_enabled || selected.size() >= branch_pieces->size()) {
			return 0.0;
		}

		const auto contact_start_time = std::chrono::steady_clock::now();
		vector<Vector2> hull_path;
		double contact_solver_seconds = 0.0;
		const size_t calls_before_path = result.convex_calls;
		const double hull_path_length = solve_convex_path(bound_instance, hull_path, contact_solver_seconds);

		if (result.convex_calls > calls_before_path) {
			result.contact_path_calls++;
		}

		if (!result.exhausted || !std::isfinite(hull_path_length) || hull_path.size() != branch_pieces->size()) {
			const auto contact_end_time = std::chrono::steady_clock::now();
			result.contact_bound_seconds += std::chrono::duration<double>(contact_end_time - contact_start_time).count();
			return 0.0;
		}

		struct FakeContact {
			size_t polygon = 0;
			double distance = 0.0;
		};

		vector<FakeContact> fake_contacts;

		for (size_t polygon = selected.size(); polygon < hull_path.size(); polygon++) {
			double distance = std::numeric_limits<double>::infinity();

			for (const auto &piece : (*branch_pieces)[polygon]) {
				if (point_in_polygon_or_on_boundary(hull_path[polygon], piece)) {
					distance = 0.0;
					break;
				}

				distance = std::min(distance, point_polygon_distance(hull_path[polygon], piece));
			}

			if (distance > 1e-9 && std::isfinite(distance)) {
				fake_contacts.push_back({polygon, distance});
			}
		}

		std::sort(fake_contacts.begin(), fake_contacts.end(), [](const FakeContact &a, const FakeContact &b) {
			return a.distance > b.distance;
		});

		vector<size_t> refined_polygons;
		size_t combination_count = 1;

		for (const auto &contact : fake_contacts) {
			if (refined_polygons.size() >= contact_max_polygons_value) {
				break;
			}

			const size_t next_count = combination_count * (*branch_pieces)[contact.polygon].size();

			if (next_count > contact_max_combinations_value) {
				continue;
			}

			refined_polygons.push_back(contact.polygon);
			combination_count = next_count;
		}

		if (refined_polygons.empty()) {
			const auto contact_end_time = std::chrono::steady_clock::now();
			result.contact_bound_seconds += std::chrono::duration<double>(contact_end_time - contact_start_time).count();
			return 0.0;
		}

		double contact_bound = std::numeric_limits<double>::infinity();
		vector<size_t> refined_piece_indices(refined_polygons.size(), 0);
		bool done = false;

		while (!done) {
			vector<vector<Vector2>> refined_instance;
			refined_instance.reserve(branch_pieces->size());

			for (size_t polygon = 0; polygon < branch_pieces->size(); polygon++) {
				if (polygon < selected.size()) {
					refined_instance.push_back((*branch_pieces)[polygon][selected[polygon]]);
					continue;
				}

				auto refined_it = std::find(refined_polygons.begin(), refined_polygons.end(), polygon);

				if (refined_it == refined_polygons.end()) {
					refined_instance.push_back(convex_hulls[polygon]);
					continue;
				}

				const size_t refined_index = static_cast<size_t>(std::distance(refined_polygons.begin(), refined_it));
				refined_instance.push_back((*branch_pieces)[polygon][refined_piece_indices[refined_index]]);
			}

			const size_t calls_before_candidate = result.convex_calls;
			const double candidate = solve_convex_length(refined_instance, contact_solver_seconds);

			if (result.convex_calls > calls_before_candidate) {
				result.contact_bound_calls++;
			}

			contact_bound = std::min(contact_bound, candidate);

			if (!result.exhausted || contact_bound <= incumbent) {
				break;
			}

			for (size_t index = refined_piece_indices.size(); index > 0; index--) {
				const size_t piece_slot = index - 1;
				refined_piece_indices[piece_slot]++;

				if (refined_piece_indices[piece_slot] < (*branch_pieces)[refined_polygons[piece_slot]].size()) {
					break;
				}

				refined_piece_indices[piece_slot] = 0;

				if (piece_slot == 0) {
					done = true;
				}
			}
		}

		const auto contact_end_time = std::chrono::steady_clock::now();
		result.contact_bound_seconds += std::chrono::duration<double>(contact_end_time - contact_start_time).count();
		return contact_bound;
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
	const double selected_length = solve_convex_path(selected_pieces, selected_path, result.incumbent_solver_seconds);

	if (result.convex_calls > calls_before_incumbent) {
		result.incumbent_solves++;
	}

	if (selected_length < result.final_length) {
		result.final_length = selected_length;
		result.incumbent_length = selected_length;
		best_path = std::move(selected_path);
	}

	auto refine_selected_groups = [&](const vector<size_t> &selected_groups) {
		if (piece_groups == nullptr) {
			return;
		}

		vector<vector<Vector2>> selected_group_hulls;
		selected_group_hulls.reserve(selected_groups.size());

		for (size_t i = 0; i < selected_groups.size(); i++) {
			selected_group_hulls.push_back((*piece_groups)[i][selected_groups[i]].hull);
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
					const size_t piece_index = (*piece_groups)[i][selected_groups[i]].piece_indices[current[i]];
					instance.push_back(convex_pieces[i][piece_index]);
				}

				vector<Vector2> path;
				const size_t calls_before = result.convex_calls;
				increment_histogram(result.leaf_depth_histogram, current.size());
				const double length = solve_convex_length(instance, result.leaf_solver_seconds);

				if (result.convex_calls > calls_before) {
					result.leaf_solves++;
				}

				if (!result.exhausted) {
					break;
				}

				if (length < result.final_length) {
					const double path_length = solve_convex_path(instance, path, result.leaf_solver_seconds);

					if (!result.exhausted) {
						break;
					}

					if (path_length < result.final_length) {
						result.final_length = path_length;
						best_path = std::move(path);
						result.best_updates++;
					}
				}

				continue;
			}

			const size_t next_polygon = current.size();
			const auto &group = (*piece_groups)[next_polygon][selected_groups[next_polygon]];
			const size_t observed_branching = group.piece_indices.size();
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
					const size_t piece_index = (*piece_groups)[j][selected_groups[j]].piece_indices[selected[j]];
					bound_instance.push_back(convex_pieces[j][piece_index]);
				}

				for (size_t j = selected.size(); j < selected_group_hulls.size(); j++) {
					bound_instance.push_back(selected_group_hulls[j]);
				}

				const size_t calls_before = result.convex_calls;
				increment_histogram(result.bound_depth_histogram, selected_groups.size() + selected.size());
				const double incumbent_before_bound = result.final_length;
				const double bound = solve_convex_length(bound_instance, result.bound_solver_seconds);

				if (result.convex_calls > calls_before) {
					result.bound_solves++;
				}

				if (!result.exhausted) {
					break;
				}

				if (bound > result.final_length) {
					result.hull_bound_prunes++;
					result.pruned_nodes++;
					continue;
				}

				if (std::isfinite(bound) && std::isfinite(incumbent_before_bound) && incumbent_before_bound > 0.0) {
					result.failed_prune_count++;
					result.failed_prune_ratio_sum += bound / incumbent_before_bound;
					result.failed_prune_gap_sum += incumbent_before_bound - bound;
					result.failed_prune_depth_sum += static_cast<double>(selected_groups.size() + selected.size());
				}

				stack.push_back(std::move(selected));
			}
		}
	};

	vector<vector<size_t>> stack;
	stack.push_back({});

	while (!stack.empty() && result.exhausted) {
		vector<size_t> current = std::move(stack.back());
		stack.pop_back();
		result.visited_nodes++;
		result.selected_sum += current.size();
		increment_histogram(result.visited_depth_histogram, current.size());

		if (current.size() == branch_pieces->size()) {
			if (piece_groups != nullptr) {
				refine_selected_groups(current);
				continue;
			}

			vector<vector<Vector2>> instance;
			instance.reserve(branch_pieces->size());

			for (size_t i = 0; i < branch_pieces->size(); i++) {
				instance.push_back((*branch_pieces)[i][current[i]]);
			}

			vector<Vector2> path;
			const size_t calls_before = result.convex_calls;
			increment_histogram(result.leaf_depth_histogram, current.size());
			const double length = solve_convex_length(instance, result.leaf_solver_seconds);

			if (result.convex_calls > calls_before) {
				result.leaf_solves++;
			}

			if (!result.exhausted) {
				break;
			}

			if (length < result.final_length) {
				const double path_length = solve_convex_path(instance, path, result.leaf_solver_seconds);

				if (!result.exhausted) {
					break;
				}

				if (path_length < result.final_length) {
					result.final_length = path_length;
					best_path = std::move(path);
					result.best_updates++;
				}
			}

			continue;
		}

		const size_t next_polygon = current.size();
		const size_t observed_branching = (*branch_pieces)[next_polygon].size();
		const size_t branch_count = std::min(max_branching, observed_branching);

		result.max_observed_branching = std::max(result.max_observed_branching, observed_branching);
		result.branch_limited = result.branch_limited || branch_count < observed_branching;
		result.branching_histogram[branching_bucket(observed_branching)]++;

		for (size_t i = branch_count; i > 0; i--) {
			vector<size_t> selected = current;
			selected.push_back(i - 1);

			vector<vector<Vector2>> bound_instance;
			bound_instance.reserve(branch_pieces->size());

			for (size_t j = 0; j < selected.size(); j++) {
				bound_instance.push_back((*branch_pieces)[j][selected[j]]);
			}

			for (size_t j = selected.size(); j < convex_hulls.size(); j++) {
				bound_instance.push_back(convex_hulls[j]);
			}

			const size_t calls_before = result.convex_calls;
			increment_histogram(result.bound_depth_histogram, selected.size());
			const double incumbent_before_bound = result.final_length;
			const double hull_bound = solve_convex_length(bound_instance, result.bound_solver_seconds);
			const double piece_graph_bound = solve_piece_graph_bound(selected);
			const double port_bound = solve_port_bound(selected);
			double refinement_bound = 0.0;
			double contact_bound = 0.0;
			double bound = std::max({hull_bound, piece_graph_bound, port_bound});

			const bool should_refine =
				refinement_bound_enabled
				&& selected.size() >= refinement_min_depth_value
				&& std::isfinite(bound)
				&& std::isfinite(result.final_length)
				&& result.final_length > 0.0
				&& result.final_length - bound <= refinement_gap_ratio_value * result.final_length;

			if (bound <= result.final_length && should_refine) {
				refinement_bound = solve_refinement_bound(selected, result.final_length);
				bound = std::max(bound, refinement_bound);
			}

			const bool should_contact_refine =
				contact_bound_enabled
				&& selected.size() >= contact_min_depth_value
				&& std::isfinite(bound)
				&& std::isfinite(result.final_length)
				&& result.final_length > 0.0
				&& result.final_length - bound <= contact_gap_ratio_value * result.final_length;

			if (bound <= result.final_length && should_contact_refine) {
				contact_bound = solve_contact_bound(selected, bound_instance, result.final_length);
				bound = std::max(bound, contact_bound);
			}

			if (result.convex_calls > calls_before) {
				result.bound_solves++;
			}

			if (!result.exhausted) {
				break;
			}

			if (piece_graph_bound > hull_bound + 1e-9) {
				result.piece_graph_dominates++;
			}

			if (port_bound > std::max(hull_bound, piece_graph_bound) + 1e-9) {
				result.port_dominates++;
			}

			if (refinement_bound > std::max({hull_bound, piece_graph_bound, port_bound}) + 1e-9) {
				result.refinement_dominates++;
			}

			if (contact_bound > std::max({hull_bound, piece_graph_bound, port_bound, refinement_bound}) + 1e-9) {
				result.contact_dominates++;
			}

			if (bound > result.final_length) {
				if (hull_bound > result.final_length) {
					result.hull_bound_prunes++;
				} else if (piece_graph_bound > result.final_length) {
					result.piece_graph_extra_prunes++;
				} else if (port_bound > result.final_length) {
					result.port_extra_prunes++;
				} else if (refinement_bound > result.final_length) {
					result.refinement_extra_prunes++;
				} else if (contact_bound > result.final_length) {
					result.contact_extra_prunes++;
				}

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

	result.best_path = std::move(best_path);
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

struct SvgBounds {
	double min_x = std::numeric_limits<double>::infinity();
	double min_y = std::numeric_limits<double>::infinity();
	double max_x = -std::numeric_limits<double>::infinity();
	double max_y = -std::numeric_limits<double>::infinity();
};

struct SvgViewport {
	double width = 720.0;
	double height = 520.0;
	double scale = 1.0;
	double offset_x = 0.0;
	double offset_y = 0.0;
};

void include_point(SvgBounds &bounds, const Vector2 &point) {
	bounds.min_x = std::min(bounds.min_x, point.x);
	bounds.min_y = std::min(bounds.min_y, point.y);
	bounds.max_x = std::max(bounds.max_x, point.x);
	bounds.max_y = std::max(bounds.max_y, point.y);
}

SvgViewport svg_viewport(const SvgBounds &bounds, double width, double height, double padding) {
	const double span_x = std::max(1e-9, bounds.max_x - bounds.min_x);
	const double span_y = std::max(1e-9, bounds.max_y - bounds.min_y);
	const double scale = std::min((width - 2.0 * padding) / span_x, (height - 2.0 * padding) / span_y);
	const double draw_width = span_x * scale;
	const double draw_height = span_y * scale;
	return {
		.width = width,
		.height = height,
		.scale = scale,
		.offset_x = (width - draw_width) / 2.0 - bounds.min_x * scale,
		.offset_y = (height + draw_height) / 2.0 + bounds.min_y * scale,
	};
}

std::pair<double, double> svg_point(const Vector2 &point, const SvgViewport &viewport) {
	return {
		viewport.offset_x + point.x * viewport.scale,
		viewport.offset_y - point.y * viewport.scale,
	};
}

std::string svg_points(
	const vector<Vector2> &points,
	const SvgViewport &viewport
) {
	std::ostringstream output;
	for (size_t i = 0; i < points.size(); i++) {
		const auto [x, y] = svg_point(points[i], viewport);
		if (i != 0) {
			output << ' ';
		}
		output << std::format("{:.2f},{:.2f}", x, y);
	}
	return output.str();
}

std::string svg_line(double x1, double y1, double x2, double y2, std::string_view color, double opacity, double width) {
	return std::format(
		"<line x1=\"{:.2f}\" y1=\"{:.2f}\" x2=\"{:.2f}\" y2=\"{:.2f}\" "
		"stroke=\"{}\" stroke-opacity=\"{:.2f}\" stroke-width=\"{:.2f}\"/>\n",
		x1,
		y1,
		x2,
		y2,
		color,
		opacity,
		width
	);
}

std::pair<double, int> preview_grid_metrics(double scale) {
	const double decision_value = 83.0 / scale;
	int exponent = decision_value > 0.0 ? static_cast<int>(std::ceil(std::log10(decision_value))) : 0;
	double multiplier = 1.0;
	int sub_grid_count = 4;
	const double grid_scale = std::pow(10.0, exponent);
	if (grid_scale / 5.0 > decision_value) {
		sub_grid_count = 3;
		exponent--;
		multiplier = 2.0;
	} else if (grid_scale / 2.0 > decision_value) {
		exponent--;
		multiplier = 5.0;
	}
	return {std::pow(10.0, exponent) * multiplier, sub_grid_count};
}

void write_svg_grid(std::ofstream &output, const SvgViewport &viewport) {
	const auto [grid_step, sub_grid_count] = preview_grid_metrics(viewport.scale);
	const double visible_min_x = -viewport.offset_x / viewport.scale;
	const double visible_max_x = (viewport.width - viewport.offset_x) / viewport.scale;
	const double visible_min_y = (viewport.offset_y - viewport.height) / viewport.scale;
	const double visible_max_y = viewport.offset_y / viewport.scale;

	for (double x = std::floor(visible_min_x / grid_step) * grid_step; x <= visible_max_x + grid_step; x += grid_step) {
		const double screen_x = viewport.offset_x + x * viewport.scale;
		if (0.0 <= screen_x && screen_x <= viewport.width) {
			output << svg_line(screen_x, 0.0, screen_x, viewport.height, "#515a67", 0.62, 1.0);
			for (int index = 0; index < sub_grid_count; index++) {
				const double sub_x = screen_x + static_cast<double>(index + 1) * grid_step * viewport.scale / static_cast<double>(sub_grid_count + 1);
				if (0.0 <= sub_x && sub_x <= viewport.width) {
					output << svg_line(sub_x, 0.0, sub_x, viewport.height, "#2a2f38", 0.74, 1.0);
				}
			}
		}
	}

	for (double y = std::floor(visible_min_y / grid_step) * grid_step; y <= visible_max_y + grid_step; y += grid_step) {
		const double screen_y = viewport.offset_y - y * viewport.scale;
		if (0.0 <= screen_y && screen_y <= viewport.height) {
			output << svg_line(0.0, screen_y, viewport.width, screen_y, "#515a67", 0.62, 1.0);
			for (int index = 0; index < sub_grid_count; index++) {
				const double sub_y = screen_y - static_cast<double>(index + 1) * grid_step * viewport.scale / static_cast<double>(sub_grid_count + 1);
				if (0.0 <= sub_y && sub_y <= viewport.height) {
					output << svg_line(0.0, sub_y, viewport.width, sub_y, "#2a2f38", 0.74, 1.0);
				}
			}
		}
	}

	if (0.0 <= viewport.offset_y && viewport.offset_y <= viewport.height) {
		output << svg_line(0.0, viewport.offset_y, viewport.width, viewport.offset_y, "#9aa3ad", 0.86, 1.0);
	}
	if (0.0 <= viewport.offset_x && viewport.offset_x <= viewport.width) {
		output << svg_line(viewport.offset_x, 0.0, viewport.offset_x, viewport.height, "#9aa3ad", 0.86, 1.0);
	}
}

void write_solution_preview(
	const std::filesystem::path &path,
	const Vector2 &start,
	const Vector2 &target,
	const vector<vector<Vector2>> &polygons,
	const vector<vector<vector<Vector2>>> &convex_pieces,
	const vector<Vector2> &best_path
) {
	SvgBounds bounds;
	include_point(bounds, start);
	include_point(bounds, target);
	for (const auto &polygon : polygons) {
		for (const auto &point : polygon) {
			include_point(bounds, point);
		}
	}
	for (const auto &point : best_path) {
		include_point(bounds, point);
	}

	const double width = 720.0;
	const double height = 520.0;
	const double padding = 28.0;
	const auto viewport = svg_viewport(bounds, width, height, padding);
	const std::array<std::string_view, 5> colors = {"#38bdf8", "#a3e635", "#f97316", "#f472b6", "#c084fc"};

	std::filesystem::create_directories(path.parent_path());
	std::ofstream output(path);
	if (!output) {
		return;
	}

	output << std::format(
		"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0f}\" height=\"{:.0f}\" viewBox=\"0 0 {:.0f} {:.0f}\" data-preview-version=\"7\">\n",
		width,
		height,
		width,
		height
	);
	output << "<rect width=\"100%\" height=\"100%\" fill=\"#121417\"/>\n";
	write_svg_grid(output, viewport);

	for (size_t polygon_index = 0; polygon_index < polygons.size(); polygon_index++) {
		const auto color = colors[polygon_index % colors.size()];
		output << std::format(
			"<polygon points=\"{}\" fill=\"{}\" fill-opacity=\"0.20\" stroke=\"{}\" stroke-width=\"2\"/>\n",
			svg_points(polygons[polygon_index], viewport),
			color,
			color
		);
		for (const auto &piece : convex_pieces[polygon_index]) {
			output << std::format(
				"<polygon points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-opacity=\"0.74\" stroke-width=\"1\" stroke-dasharray=\"4 3\"/>\n",
				svg_points(piece, viewport),
				color
			);
		}
	}

	vector<Vector2> full_path;
	full_path.reserve(best_path.size() + 2);
	full_path.push_back(start);
	full_path.insert(full_path.end(), best_path.begin(), best_path.end());
	full_path.push_back(target);
	output << std::format(
		"<polyline points=\"{}\" fill=\"none\" stroke=\"#facc15\" stroke-width=\"5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n",
		svg_points(full_path, viewport)
	);

	const auto marker = [&](const Vector2 &point, std::string_view label, std::string_view fill, std::string_view text_fill) {
		const auto [x, y] = svg_point(point, viewport);
		output << std::format("<circle cx=\"{:.2f}\" cy=\"{:.2f}\" r=\"5\" fill=\"{}\"/>\n", x, y, fill);
		output << std::format("<text x=\"{:.2f}\" y=\"{:.2f}\" font-size=\"14\" font-weight=\"700\" font-family=\"system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif\" fill=\"{}\">{}</text>\n", x + 10.0, y + 5.0, text_fill, label);
	};
	marker(start, "s", "#22c55e", "#f8fafc");
	marker(target, "t", "#ef4444", "#f8fafc");
	output << "</svg>\n";
}

std::filesystem::path solution_preview_path(const BenchmarkOptions &options, size_t case_index, size_t repeat_index) {
	if (!options.output_path) {
		return {};
	}
	const std::filesystem::path csv_path(*options.output_path);
	const std::filesystem::path directory = csv_path.parent_path() / (csv_path.stem().string() + "-solutions");
	return directory / std::format("case-{:04}-repeat-{:03}.svg", case_index, repeat_index);
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
		<< record.grouped_pieces << ';'
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
		<< std::format("{:.6f}", record.piece_graph_precompute_seconds) << ';'
		<< std::format("{:.6f}", record.piece_graph_bound_seconds) << ';'
		<< record.piece_graph_bound_calls << ';'
		<< std::format("{:.6f}", record.port_bound_precompute_seconds) << ';'
		<< std::format("{:.6f}", record.port_bound_seconds) << ';'
		<< record.port_bound_calls << ';'
		<< record.hull_bound_prunes << ';'
		<< record.piece_graph_extra_prunes << ';'
		<< record.piece_graph_dominates << ';'
		<< record.port_extra_prunes << ';'
		<< record.port_dominates << ';'
		<< std::format("{:.6f}", record.refinement_bound_seconds) << ';'
		<< record.refinement_bound_calls << ';'
		<< record.refinement_extra_prunes << ';'
		<< record.refinement_dominates << ';'
		<< std::format("{:.6f}", record.contact_bound_seconds) << ';'
		<< record.contact_path_calls << ';'
		<< record.contact_bound_calls << ';'
		<< record.contact_extra_prunes << ';'
		<< record.contact_dominates << ';'
		<< (record.exhausted ? "true" : "false") << ';'
		<< (record.time_limited ? "true" : "false") << ';'
		<< (record.branch_limited ? "true" : "false") << ';'
		<< record.max_observed_branching << ';'
		<< record.failed_prune_count << ';'
		<< std::format("{:.6f}", record.failed_prune_ratio_mean) << ';'
		<< std::format("{:.6f}", record.failed_prune_gap_mean) << ';'
		<< std::format("{:.3f}", record.failed_prune_depth_mean) << ';'
		<< std::format("{:.12f}", record.checksum)
		<< '\n';
}

CaseBenchmarkResult run_case_benchmark(
	size_t case_index,
	const tpp::TestCase &test_case,
	const BenchmarkOptions &options,
	const vector<vector<vector<Vector2>>> *synthetic_cover
) {
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
		for (size_t polygon_index = 0; polygon_index < polygons.size(); polygon_index++) {
			convex_hulls.push_back(convex_hull(polygons[polygon_index]));

			if (synthetic_cover != nullptr) {
				convex_pieces.push_back((*synthetic_cover)[polygon_index]);
			} else {
				convex_pieces.push_back(tpp::decompose_polygon(polygons[polygon_index]));
			}
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
	const double total_combinations = combination_count(convex_pieces);
	const double approximation_budget = approximation_work_budget(total_combinations);
	const vector<size_t> approximation_point_counts = choose_approximation_point_counts(polygons, approximation_budget);
	const bool use_legacy_sampler = use_legacy_approximation_sampler();
	vector<vector<Vector2>> interpolated_polygons;
	interpolated_polygons.reserve(polygons.size());

	for (size_t i = 0; i < polygons.size(); i++) {
		if (use_legacy_sampler) {
			interpolated_polygons.push_back(make_fixed_edge_sampled_polygon(polygons[i], 10));
		} else {
			interpolated_polygons.push_back(make_evenly_spaced_polygon(polygons[i], approximation_point_counts[i]));
		}
	}

	const vector<Vector2> approximate_path = tpp_approximation(start, target, interpolated_polygons);
	const auto approximation_end_time = std::chrono::steady_clock::now();
	const double approximation_seconds = std::chrono::duration<double>(approximation_end_time - approximation_start_time).count();

	order_pieces_from_approximation(polygons, approximate_path, convex_pieces, start, target);

	size_t total_pieces = 0;

	for (const auto &pieces : convex_pieces) {
		total_pieces += pieces.size();
	}

	vector<vector<PieceGroup>> piece_groups;
	size_t total_groups = total_pieces;

	if (use_piece_grouping()) {
		piece_groups.reserve(convex_pieces.size());
		total_groups = 0;
		const double max_excess_ratio = piece_group_max_excess_ratio();
		const size_t max_group_size = piece_group_max_size();
		const bool adaptive_grouping = use_adaptive_piece_grouping();

		for (size_t polygon_index = 0; polygon_index < convex_pieces.size(); polygon_index++) {
			if (adaptive_grouping) {
				const std::optional<Vector2> previous_point = polygon_index == 0
					? std::optional<Vector2>(start)
					: std::nullopt;
				const vector<Vector2> *previous_polygon = polygon_index == 0
					? nullptr
					: &convex_hulls[polygon_index - 1];
				const std::optional<Vector2> next_point = polygon_index + 1 == convex_pieces.size()
					? std::optional<Vector2>(target)
					: std::nullopt;
				const vector<Vector2> *next_polygon = polygon_index + 1 == convex_pieces.size()
					? nullptr
					: &convex_hulls[polygon_index + 1];

				piece_groups.push_back(make_adaptive_piece_groups_for_polygon(
					convex_pieces[polygon_index],
					previous_point,
					previous_polygon,
					next_point,
					next_polygon,
					max_excess_ratio,
					max_group_size
				));
			} else {
				piece_groups.push_back(make_piece_groups_for_polygon(convex_pieces[polygon_index], max_excess_ratio, max_group_size));
			}

			total_groups += piece_groups.back().size();
		}
	}

	result.records.reserve(options.repeat_count);

	for (size_t repeat_index = 0; repeat_index < options.repeat_count; repeat_index++) {
		const auto bnb_start_time = std::chrono::steady_clock::now();
		const BranchAndBoundResult bnb = run_branch_and_bound(
			start,
			target,
			convex_pieces,
			convex_hulls,
			piece_groups.empty() ? nullptr : &piece_groups,
			approximate_path,
			options.max_calls_per_instance,
			options.max_branching,
			options.max_seconds_per_instance,
			options.solver,
			options.length_solver,
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
		record.grouped_pieces = total_groups;
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
		record.piece_graph_precompute_seconds = bnb.piece_graph_precompute_seconds;
		record.piece_graph_bound_seconds = bnb.piece_graph_bound_seconds;
		record.piece_graph_bound_calls = bnb.piece_graph_bound_calls;
		record.port_bound_precompute_seconds = bnb.port_bound_precompute_seconds;
		record.port_bound_seconds = bnb.port_bound_seconds;
		record.port_bound_calls = bnb.port_bound_calls;
		record.hull_bound_prunes = bnb.hull_bound_prunes;
		record.piece_graph_extra_prunes = bnb.piece_graph_extra_prunes;
		record.piece_graph_dominates = bnb.piece_graph_dominates;
		record.port_extra_prunes = bnb.port_extra_prunes;
		record.port_dominates = bnb.port_dominates;
		record.refinement_bound_seconds = bnb.refinement_bound_seconds;
		record.refinement_bound_calls = bnb.refinement_bound_calls;
		record.refinement_extra_prunes = bnb.refinement_extra_prunes;
		record.refinement_dominates = bnb.refinement_dominates;
		record.contact_bound_seconds = bnb.contact_bound_seconds;
		record.contact_path_calls = bnb.contact_path_calls;
		record.contact_bound_calls = bnb.contact_bound_calls;
		record.contact_extra_prunes = bnb.contact_extra_prunes;
		record.contact_dominates = bnb.contact_dominates;
		record.exhausted = bnb.exhausted;
		record.time_limited = bnb.time_limited;
		record.branch_limited = bnb.branch_limited;
		record.max_observed_branching = bnb.max_observed_branching;
		record.failed_prune_count = bnb.failed_prune_count;
		record.failed_prune_ratio_mean = bnb.failed_prune_count == 0 ? 0.0 : bnb.failed_prune_ratio_sum / static_cast<double>(bnb.failed_prune_count);
		record.failed_prune_gap_mean = bnb.failed_prune_count == 0 ? 0.0 : bnb.failed_prune_gap_sum / static_cast<double>(bnb.failed_prune_count);
		record.failed_prune_depth_mean = bnb.failed_prune_count == 0 ? 0.0 : bnb.failed_prune_depth_sum / static_cast<double>(bnb.failed_prune_count);
		record.instance_seconds = record.decomposition_seconds + record.approximation_seconds + bnb_seconds;
		record.checksum = bnb.checksum;
		const auto preview_path = solution_preview_path(options, case_index, repeat_index);
		if (!preview_path.empty() && !bnb.best_path.empty()) {
			write_solution_preview(preview_path, start, target, polygons, convex_pieces, bnb.best_path);
			record.solution_preview_path = preview_path.string();
		}

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
	std::println(stderr, "  {} <input_file> <max_polygons> <max_instances> <max_calls_per_instance> <max_branching> <max_seconds_per_instance> [repeat_count] [csv_output_file] [summary_md_file]", program);
	std::println(stderr, "");
	std::println(stderr, "Example:");
	std::println(stderr, "  {} benchmarks/suites/canonical-v1.bin 40 6 512 6", program);
	std::println(stderr, "  {} benchmarks/suites/canonical-v1.bin 40 6 512 6 10", program);
	std::println(stderr, "  {} benchmarks/suites/canonical-v1.bin -1 -1 1000000 -1 results.csv", program);
	std::println(stderr, "  {} benchmarks/suites/canonical-v1.bin 40 6 512 6 5 results.csv", program);
	std::println(stderr, "  {} benchmarks/suites/canonical-v1.bin 40 6 512 6 5 results.csv summary.md", program);
	std::println(stderr, "");
	std::println(stderr, "Arguments:");
	std::println(stderr, "  input_file              Binary test case file.");
	std::println(stderr, "  max_polygons            Skip instances with more polygons than this.");
	std::println(stderr, "  max_instances           Stop after benchmarking this many accepted instances.");
	std::println(stderr, "  max_calls_per_instance  Cap actual convex solver calls per instance.");
	std::println(stderr, "  max_branching           Cap explored children per polygon during B&B.");
	std::println(stderr, "  max_seconds_per_instance Optional B&B elapsed-time cap per instance. Use -1 for unlimited.");
	std::println(stderr, "  repeat_count            Optional repeated runs per accepted instance.");
	std::println(stderr, "  csv_output_file         Optional file path for per-instance CSV rows.");
	std::println(stderr, "  summary_md_file         Optional file path for the markdown summary.");
	std::println(stderr, "");
	std::println(stderr, "All numeric arguments must be non-negative integers or -1 for unlimited.");
	std::println(stderr, "Set TPP_BENCH_THREADS to override the default hardware thread count.");
	std::println(stderr, "Set TPP_BENCH_MAX_SECONDS to override the default per-instance time cap.");
	std::println(stderr, "Set TPP_BENCH_SOLVER to one of linear_search_lazy, linear_search_disjoint, binary_search_lazy, binary_search_disjoint, binary_search_eager, tan_jiang, gurobi.");
	std::println(stderr, "Set TPP_GROUP_PIECES=1 to branch first on safe almost-convex piece groups.");
	std::println(stderr, "Set TPP_GROUP_MAX_EXCESS_RATIO and TPP_GROUP_MAX_SIZE to control piece grouping.");
	std::println(stderr, "Set TPP_GROUP_REQUIRE_TOUCH=1 and TPP_GROUP_ORDER_PENALTY to make grouping more local.");
	std::println(stderr, "Set TPP_GROUP_ADAPTIVE=1 and TPP_GROUP_ADAPTIVE_MAX_LOCAL_SLACK_RATIO to use neighbor-aware grouping.");
	std::println(stderr, "Set TPP_SYNTHETIC_COVER_CASES to generate synthetic merged-cover instances instead of loading input.");
	std::println(stderr, "Set TPP_USE_SYNTHETIC_COVER=1 with TPP_SYNTHETIC_COVER_CASES to use the known cover pieces.");
	std::println(stderr, "Set TPP_SYNTHETIC_COVER_PATTERN to stair, cross, overlap, comb, or longstair.");
	std::println(stderr, "Set TPP_PIECE_GRAPH_BOUND=1 to enable the experimental layered piece-distance lower bound.");
	std::println(stderr, "Set TPP_PORT_BOUND=1 to enable the experimental covering-port DP lower bound.");
	std::println(stderr, "Set TPP_REFINEMENT_BOUND=1 to enable the experimental one-step piece refinement lower bound.");
	std::println(stderr, "Set TPP_REFINEMENT_GAP_RATIO and TPP_REFINEMENT_MIN_DEPTH to gate the refinement bound.");
	std::println(stderr, "Set TPP_REFINEMENT_WINDOW_SIZE and TPP_REFINEMENT_MAX_COMBINATIONS for bounded multi-polygon refinement.");
	std::println(stderr, "Set TPP_CONTACT_BOUND=1 to enable the experimental contact-aware fake-hull refinement bound.");
	std::println(stderr, "Set TPP_CONTACT_GAP_RATIO, TPP_CONTACT_MIN_DEPTH, TPP_CONTACT_MAX_POLYGONS, and TPP_CONTACT_MAX_COMBINATIONS to gate it.");
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

std::optional<double> parse_seconds_arg(const char *text) {

	const std::string value = text;

	if (value == "-1") {
		return std::numeric_limits<double>::infinity();
	}

	if (value.empty() || value.front() == '-') {
		return std::nullopt;
	}

	size_t consumed = 0;

	try {
		const double parsed = std::stod(value, &consumed);
		if (consumed != value.size() || !std::isfinite(parsed)) {
			return std::nullopt;
		}

		return parsed;
	} catch (...) {
		return std::nullopt;
	}
}

bool set_solver(BenchmarkOptions &options, const std::string &name) {
	if (name == "linear_search_lazy" || name == "linear") {
		options.solver_name = "linear_search_lazy";
		options.solver = tpp::tpp_convex_solve_linear_search_lazy;
		options.length_solver = tpp::tpp_convex_solve_length_linear_search_lazy;
		return true;
	}

	if (name == "linear_search_disjoint" || name == "linear_disjoint") {
		options.solver_name = "linear_search_disjoint";
		options.solver = tpp::tpp_convex_solve_linear_search_disjoint;
		options.length_solver = tpp::tpp_convex_solve_length_linear_search_disjoint;
		return true;
	}

	if (name == "binary_search_lazy" || name == "binary" || name == "default") {
		options.solver_name = "binary_search_lazy";
		options.solver = tpp::tpp_convex_solve_binary_search_lazy;
		options.length_solver = tpp::tpp_convex_solve_length_binary_search_lazy;
		return true;
	}

	if (name == "binary_search_disjoint" || name == "binary_disjoint") {
		options.solver_name = "binary_search_disjoint";
		options.solver = tpp::tpp_convex_solve_binary_search_disjoint;
		options.length_solver = tpp::tpp_convex_solve_length_binary_search_disjoint;
		return true;
	}

	if (name == "binary_search_eager" || name == "binary_search_dp" || name == "binary_dp") {
		options.solver_name = "binary_search_eager";
		options.solver = tpp::tpp_convex_solve_binary_search_eager;
		options.length_solver = tpp::tpp_convex_solve_length_binary_search_eager;
		return true;
	}

	if (name == "tan_jiang" || name == "tan-jiang" || name == "tamc") {
		options.solver_name = "tan_jiang";
		options.solver = tpp::tpp_convex_solve_tan_jiang;
		options.length_solver = tpp::tpp_convex_solve_length_tan_jiang;
		return true;
	}

	if (name == "gurobi") {
#if defined(TPP_ENABLE_GUROBI)
		options.solver_name = "gurobi";
		options.solver = tpp::tpp_convex_solve_gurobi;
		options.length_solver = tpp::tpp_convex_solve_length_gurobi;
		return true;
#else
		return false;
#endif
	}

	return false;
}

std::optional<BenchmarkOptions> parse_options(int argc, char **argv) {

	BenchmarkOptions options;

	if (argc != 1 && (argc < 6 || argc > 10)) {
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

	int next_arg = 6;
	if (argc > next_arg) {
		const std::string first_optional = argv[next_arg];
		const auto repeat_count = parse_size_arg(argv[next_arg]);
		const bool looks_like_legacy_repeat =
			repeat_count
			&& *repeat_count != 0
			&& *repeat_count != std::numeric_limits<size_t>::max()
			&& first_optional.find('.') == std::string::npos;

		if (looks_like_legacy_repeat) {
			options.repeat_count = *repeat_count;
			next_arg++;
		} else if (const auto max_seconds = parse_seconds_arg(argv[next_arg])) {
			options.max_seconds_per_instance = *max_seconds;
			next_arg++;

			if (argc > next_arg) {
				const auto explicit_repeat_count = parse_size_arg(argv[next_arg]);
				if (explicit_repeat_count) {
					if (*explicit_repeat_count == 0 || *explicit_repeat_count == std::numeric_limits<size_t>::max()) {
						print_usage(argv[0]);
						return std::nullopt;
					}

					options.repeat_count = *explicit_repeat_count;
					next_arg++;
				}
			}
		} else {
			options.output_path = argv[next_arg];
			next_arg++;
		}
	}

	if (argc > next_arg) {
		if (!options.output_path) {
			options.output_path = argv[next_arg];
			next_arg++;
		}
	}

	if (argc > next_arg) {
		options.summary_output_path = argv[next_arg];
		next_arg++;
	}

	if (argc > next_arg) {
		print_usage(argv[0]);
		return std::nullopt;
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

	if (const char *max_seconds_text = std::getenv("TPP_BENCH_MAX_SECONDS")) {
		const auto max_seconds = parse_seconds_arg(max_seconds_text);

		if (!max_seconds) {
			std::println(stderr, "Invalid TPP_BENCH_MAX_SECONDS value: {}", max_seconds_text);
			print_usage(argv[0]);
			return 2;
		}

		options.max_seconds_per_instance = *max_seconds;
	}

	if (const char *solver_text = std::getenv("TPP_BENCH_SOLVER")) {
		if (!set_solver(options, solver_text)) {
			std::println(stderr, "Invalid or unavailable TPP_BENCH_SOLVER value: {}", solver_text);
			print_usage(argv[0]);
			return 2;
		}
	}

	const auto program_start_time = std::chrono::steady_clock::now();
	SyntheticCoverSuite synthetic_cover_suite;
	vector<tpp::TestCase> loaded_test_cases;
	const vector<tpp::TestCase> *test_cases = nullptr;
	const bool synthetic_cover_enabled = use_synthetic_cover();

	if (const char *synthetic_case_count_text = std::getenv("TPP_SYNTHETIC_COVER_CASES")) {
		const size_t synthetic_case_count = std::strtoull(synthetic_case_count_text, nullptr, 10);
		synthetic_cover_suite = make_synthetic_cover_suite(synthetic_case_count);
		test_cases = &synthetic_cover_suite.cases;
		std::println(stderr, "Generated {} synthetic merged-cover cases", synthetic_cover_suite.cases.size());
	} else {
		loaded_test_cases = tpp::load_test_cases(options.input_path);
		test_cases = &loaded_test_cases;
	}

	BenchmarkSummary summary;
	summary.total_instances = test_cases->size();
	vector<InstanceRecord> records;
	vector<size_t> total_visited_depth_histogram;
	vector<size_t> total_bound_depth_histogram;
	vector<size_t> total_leaf_depth_histogram;
	std::array<size_t, BRANCH_BUCKET_COUNT> total_branching_histogram = {};

	constexpr const char *csv_header =
		"source;case_index;repeat_index;polygons;decomposed_pieces;grouped_pieces;total_combinations;calls;incumbent_solves;bound_solves;leaf_solves;visited_nodes;pruned_nodes;best_updates;mean_selected;total_vertices_min;total_vertices_max;initial_length;incumbent_length;final_length;initial_gap_percent;incumbent_gap_percent;prune_rate_percent;calls_per_visited_node;bound_calls_per_leaf;decomposition_seconds;decomposition_percent;approximation_seconds;approximation_percent;bnb_seconds;bnb_percent;solver_seconds;solver_percent;incumbent_solver_seconds;bound_solver_seconds;leaf_solver_seconds;seconds_per_call;piece_graph_precompute_seconds;piece_graph_bound_seconds;piece_graph_bound_calls;port_bound_precompute_seconds;port_bound_seconds;port_bound_calls;hull_bound_prunes;piece_graph_extra_prunes;piece_graph_dominates;port_extra_prunes;port_dominates;refinement_bound_seconds;refinement_bound_calls;refinement_extra_prunes;refinement_dominates;contact_bound_seconds;contact_path_calls;contact_bound_calls;contact_extra_prunes;contact_dominates;exhausted;time_limited;branch_limited;max_observed_branching;failed_prune_count;failed_prune_ratio_mean;failed_prune_gap_mean;failed_prune_depth_mean;checksum";

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
	case_jobs.reserve(std::min(test_cases->size(), options.max_instances));

	for (size_t case_index = 0; case_index < test_cases->size() && case_jobs.size() < options.max_instances; case_index++) {
		const auto &[start, target, raw_polygons, _] = (*test_cases)[case_index];

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
				const vector<vector<vector<Vector2>>> *synthetic_cover =
					synthetic_cover_enabled && !synthetic_cover_suite.covers.empty()
						? &synthetic_cover_suite.covers[case_index]
						: nullptr;
				case_results[job_index] = run_case_benchmark(case_index, (*test_cases)[case_index], options, synthetic_cover);
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
		summary.skipped_intersecting_hulls += case_result.skipped_intersecting_hulls;
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
			summary.capped_by_calls_instances += !record.exhausted && !record.time_limited ? 1 : 0;
			summary.capped_by_time_instances += record.time_limited ? 1 : 0;
			summary.branch_limited_instances += record.branch_limited ? 1 : 0;
			summary.grouped_pieces += record.grouped_pieces;
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
			summary.piece_graph_precompute_seconds += record.piece_graph_precompute_seconds;
			summary.piece_graph_bound_seconds += record.piece_graph_bound_seconds;
			summary.piece_graph_bound_calls += record.piece_graph_bound_calls;
			summary.port_bound_precompute_seconds += record.port_bound_precompute_seconds;
			summary.port_bound_seconds += record.port_bound_seconds;
			summary.port_bound_calls += record.port_bound_calls;
			summary.hull_bound_prunes += record.hull_bound_prunes;
			summary.piece_graph_extra_prunes += record.piece_graph_extra_prunes;
			summary.piece_graph_dominates += record.piece_graph_dominates;
			summary.port_extra_prunes += record.port_extra_prunes;
			summary.port_dominates += record.port_dominates;
			summary.refinement_bound_seconds += record.refinement_bound_seconds;
			summary.refinement_bound_calls += record.refinement_bound_calls;
			summary.refinement_extra_prunes += record.refinement_extra_prunes;
			summary.refinement_dominates += record.refinement_dominates;
			summary.contact_bound_seconds += record.contact_bound_seconds;
			summary.contact_path_calls += record.contact_path_calls;
			summary.contact_bound_calls += record.contact_bound_calls;
			summary.contact_extra_prunes += record.contact_extra_prunes;
			summary.contact_dominates += record.contact_dominates;
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
	const double measured_parallelism = total_seconds == 0.0 ? 0.0 : measured_work_seconds / total_seconds;
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

	auto format_count_with_percent = [](size_t count, size_t total) {
		return std::format("{} ({})", format_count(count), format_percent(static_cast<double>(count), static_cast<double>(total)));
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

	emit("| Timing | Value |");
	emit("|---|---:|");
	emitf("| Decomposition | {} |", format_seconds_with_percent(summary.decomposition_seconds, measured_work_seconds));
	emitf("| Approximation | {} |", format_seconds_with_percent(summary.approximation_seconds, measured_work_seconds));
	emitf("| B&B | {} |", format_seconds_with_percent(summary.bnb_seconds, measured_work_seconds));
	emitf("| Convex solver | {} of measured work |", format_seconds_with_percent(summary.solver_seconds, measured_work_seconds));
	emitf("| Piece graph precompute | {:.6f}s (inside B&B) |", summary.piece_graph_precompute_seconds);
	emitf("| Piece graph bound DP | {:.6f}s (inside B&B) |", summary.piece_graph_bound_seconds);
	emitf("| Port bound precompute | {:.6f}s (inside B&B) |", summary.port_bound_precompute_seconds);
	emitf("| Port bound DP | {:.6f}s (inside B&B) |", summary.port_bound_seconds);
	emitf("| Refinement bound | {:.6f}s (inside B&B) |", summary.refinement_bound_seconds);
	emitf("| Contact-aware bound | {:.6f}s (inside B&B) |", summary.contact_bound_seconds);
	emitf("| Measured work | {:.6f}s (100.00%) |", measured_work_seconds);
	emitf("| Wall-clock total | {:.6f}s |", total_seconds);
	emitf("| Measured parallelism | {:.2f}x |", measured_parallelism);
	emitf("| Mean seconds per call | {:.6f}us |", mean_seconds_per_call * 1000000.0);
	emitf("| Checksum | {:.12f} |", summary.checksum);
	emit("");
	emit("| Metric | Value |");
	emit("|---|---:|");
	emitf("| Total instances | {} |", format_count(summary.total_instances));
	emitf("| Benchmarked instances | {} |", format_count_with_percent(summary.benchmarked_instances, summary.total_instances));
	emitf("| Benchmark runs | {} |", format_count(records.size()));
	emitf("| Repeat count | {} |", format_count(options.repeat_count));
	emitf("| Worker threads | {} |", format_count(worker_count));
	emitf("| Convex solver name | {} |", options.solver_name);
	emitf("| Max seconds per instance | {} |", std::isfinite(options.max_seconds_per_instance) ? std::format("{:.6f}s", options.max_seconds_per_instance) : std::string("unlimited"));
	emitf("| Fully solved runs | {} |", format_count_with_percent(summary.fully_covered_instances, records.size()));
	emitf("| Capped by calls runs | {} |", format_count_with_percent(summary.capped_by_calls_instances, records.size()));
	emitf("| Capped by time runs | {} |", format_count_with_percent(summary.capped_by_time_instances, records.size()));
	emitf("| Branch limited runs | {} |", format_count_with_percent(summary.branch_limited_instances, records.size()));
	emitf("| Mean grouped pieces | {:.3f} |", records.empty() ? 0.0 : static_cast<double>(summary.grouped_pieces) / static_cast<double>(records.size()));
	emitf("| Skipped by max polygons | {} |", format_count_with_percent(summary.skipped_max_polygons, summary.total_instances));
	emitf("| Skipped decomposition | {} |", format_count_with_percent(summary.skipped_decomposition, summary.total_instances));
	emitf("| Skipped intersecting convex hulls | {} |", format_count_with_percent(summary.skipped_intersecting_hulls, summary.total_instances));
	emitf("| Skipped no calls | {} |", format_count_with_percent(summary.skipped_no_calls, summary.total_instances));
	emitf("| Max observed branching | {} |", format_count(summary.max_observed_branching));
	emit("");
	emit("| B&B Counter | Value |");
	emit("|---|---:|");
	emitf("| Total convex calls | {} |", format_count(summary.total_calls));
	emitf("| Incumbent solves | {} |", format_count_with_percent(summary.total_incumbent_solves, summary.total_calls));
	emitf("| Bound solves | {} |", format_count_with_percent(summary.total_bound_solves, summary.total_calls));
	emitf("| Leaf solves | {} |", format_count_with_percent(summary.total_leaf_solves, summary.total_calls));
	emitf("| Piece graph bound calls | {} |", format_count(summary.piece_graph_bound_calls));
	emitf("| Port bound calls | {} |", format_count(summary.port_bound_calls));
	emitf("| Refinement bound calls | {} |", format_count(summary.refinement_bound_calls));
	emitf("| Contact path calls | {} |", format_count(summary.contact_path_calls));
	emitf("| Contact bound calls | {} |", format_count(summary.contact_bound_calls));
	emitf("| Visited nodes | {} |", format_count_with_percent(summary.total_visited_nodes, summary.total_visited_nodes + summary.total_pruned_nodes));
	emitf("| Pruned nodes | {} |", format_count_with_percent(summary.total_pruned_nodes, summary.total_visited_nodes + summary.total_pruned_nodes));
	emitf("| Hull-bound prunes | {} |", format_count_with_percent(summary.hull_bound_prunes, summary.total_pruned_nodes));
	emitf("| Piece-graph extra prunes | {} |", format_count_with_percent(summary.piece_graph_extra_prunes, summary.total_pruned_nodes));
	emitf("| Piece graph dominated hull bound | {} |", format_count_with_percent(summary.piece_graph_dominates, summary.piece_graph_bound_calls));
	emitf("| Port extra prunes | {} |", format_count_with_percent(summary.port_extra_prunes, summary.total_pruned_nodes));
	emitf("| Port dominated previous bounds | {} |", format_count_with_percent(summary.port_dominates, summary.port_bound_calls));
	emitf("| Refinement extra prunes | {} |", format_count_with_percent(summary.refinement_extra_prunes, summary.total_pruned_nodes));
	emitf("| Refinement dominated previous bounds | {} |", format_count_with_percent(summary.refinement_dominates, summary.refinement_bound_calls));
	emitf("| Contact extra prunes | {} |", format_count_with_percent(summary.contact_extra_prunes, summary.total_pruned_nodes));
	emitf("| Contact dominated previous bounds | {} |", format_count_with_percent(summary.contact_dominates, summary.contact_path_calls));
	emitf("| Best updates | {} |", format_count_with_percent(summary.total_best_updates, summary.total_visited_nodes));
	emit("");
	emit("## Distributions");
	emit("");
	emit("| Metric | Min | Median | P90 | P99 | Max | Mean |");
	emit("|---|---:|---:|---:|---:|---:|---:|");
	print_distribution("Seconds per call", distribution(values([](const InstanceRecord &r) { return r.seconds_per_call; })), false);
	print_distribution("Piece graph precompute seconds", distribution(values([](const InstanceRecord &r) { return r.piece_graph_precompute_seconds; })), false);
	print_distribution("Piece graph bound seconds", distribution(values([](const InstanceRecord &r) { return r.piece_graph_bound_seconds; })), false);
	print_distribution("Piece graph extra prunes", distribution(values([](const InstanceRecord &r) { return r.piece_graph_extra_prunes; })), true);
	print_distribution("Piece graph dominates", distribution(values([](const InstanceRecord &r) { return r.piece_graph_dominates; })), true);
	print_distribution("Port bound precompute seconds", distribution(values([](const InstanceRecord &r) { return r.port_bound_precompute_seconds; })), false);
	print_distribution("Port bound seconds", distribution(values([](const InstanceRecord &r) { return r.port_bound_seconds; })), false);
	print_distribution("Port extra prunes", distribution(values([](const InstanceRecord &r) { return r.port_extra_prunes; })), true);
	print_distribution("Port dominates", distribution(values([](const InstanceRecord &r) { return r.port_dominates; })), true);
	print_distribution("Refinement bound seconds", distribution(values([](const InstanceRecord &r) { return r.refinement_bound_seconds; })), false);
	print_distribution("Refinement extra prunes", distribution(values([](const InstanceRecord &r) { return r.refinement_extra_prunes; })), true);
	print_distribution("Refinement dominates", distribution(values([](const InstanceRecord &r) { return r.refinement_dominates; })), true);
	print_distribution("Contact bound seconds", distribution(values([](const InstanceRecord &r) { return r.contact_bound_seconds; })), false);
	print_distribution("Contact extra prunes", distribution(values([](const InstanceRecord &r) { return r.contact_extra_prunes; })), true);
	print_distribution("Contact dominates", distribution(values([](const InstanceRecord &r) { return r.contact_dominates; })), true);
	print_distribution("Polygons", distribution(values([](const InstanceRecord &r) { return r.polygons; })), true);
	print_distribution("Calls", distribution(values([](const InstanceRecord &r) { return r.calls; })), true);
	print_distribution("Best updates", distribution(values([](const InstanceRecord &r) { return r.best_updates; })), true);
	print_distribution("Initial gap %", distribution(values([](const InstanceRecord &r) { return initial_gap_percent(r); })), false);
	print_distribution("Incumbent gap %", distribution(values([](const InstanceRecord &r) { return incumbent_gap_percent(r); })), false);
	print_distribution("Max branching", distribution(values([](const InstanceRecord &r) { return r.max_observed_branching; })), true);
	print_distribution("Decomposed pieces", distribution(values([](const InstanceRecord &r) { return r.decomposed_pieces; })), true);
	print_distribution("Grouped pieces", distribution(values([](const InstanceRecord &r) { return r.grouped_pieces; })), true);
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
	emitf("| Piece graph bound time per call | {:.12f}s |", summary.piece_graph_bound_calls == 0 ? 0.0 : summary.piece_graph_bound_seconds / static_cast<double>(summary.piece_graph_bound_calls));
	emitf("| Port bound time per call | {:.12f}s |", summary.port_bound_calls == 0 ? 0.0 : summary.port_bound_seconds / static_cast<double>(summary.port_bound_calls));
	emitf("| Refinement bound time per call | {:.12f}s |", summary.refinement_bound_calls == 0 ? 0.0 : summary.refinement_bound_seconds / static_cast<double>(summary.refinement_bound_calls));
	emitf("| Contact bound time per call | {:.12f}s |", summary.contact_bound_calls == 0 ? 0.0 : summary.contact_bound_seconds / static_cast<double>(summary.contact_bound_calls));
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
	print_top("By Piece-Graph Extra Prunes", [](const InstanceRecord &r) { return r.piece_graph_extra_prunes; }, "Extra Prunes", true);
	print_top("By Port Extra Prunes", [](const InstanceRecord &r) { return r.port_extra_prunes; }, "Extra Prunes", true);
	print_top("By Refinement Extra Prunes", [](const InstanceRecord &r) { return r.refinement_extra_prunes; }, "Extra Prunes", true);
	print_top("By Contact Extra Prunes", [](const InstanceRecord &r) { return r.contact_extra_prunes; }, "Extra Prunes", true);
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
