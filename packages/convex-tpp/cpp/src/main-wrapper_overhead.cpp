#include "common.h"
#include "tests.h"
#include "tpp_convex.h"
#include "tpp_convex_common.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <print>
#include <string>
#include <vector>

using std::vector;

namespace {

class ForcedDisjointBinarySearch : public tpp::Solution {

	using tpp::Solution::Solution;

	size_t _locate_point(const Vector2& point, size_t i) {

		const auto polygon_index = i - 1;
		const auto &polygon = polygons[polygon_index];
		const auto vertex_count = polygon.size();

		const auto &first_vertex = polygon[0];
		const auto &[first_ray1, first_ray2] = get_cone(polygon_index, 0);

		if (tpp::point_in_cone_plus(point, first_vertex, first_ray1, first_ray2)) {
			return 0;
		}

		size_t left = 0;
		size_t right = vertex_count - 1;

		while (left != right) {
			const auto mid = left + (right - left) / 2;
			const auto mid_vertex_index = mid + 1;
			const auto &mid_vertex = polygon[mid_vertex_index];
			const auto &[mid_ray1, mid_ray2] = get_cone(polygon_index, mid_vertex_index);

			if (tpp::point_in_cone_plus(point, mid_vertex, mid_ray1, mid_ray2)) {
				return 2 * mid_vertex_index;
			}

			const auto &left_vertex = polygon[left];
			const auto &left_ray2 = get_cone(polygon_index, left).second;

			if (tpp::point_in_edge_plus(point, left_vertex, mid_vertex, left_ray2, mid_ray1)) {
				right = mid;
			} else {
				left = mid + 1;
			}
		}

		return 2 * left + 1;
	}

	int64_t locate_point(const Vector2& point, size_t i) override {

		size_t location = _locate_point(point, i);
		const auto polygon_index = i - 1;
		const auto vertex_count = polygons[polygon_index].size();
		size_t previous_index = location == 0 ? vertex_count - 1 : (location - 1) / 2;

		if (is_first_contact(polygon_index, location / 2) || is_first_contact(polygon_index, previous_index)) {
			return location;
		}

		return -1;
	}
};

vector<Vector2> solve_forced_disjoint(
	const Vector2 &start,
	const Vector2 &target,
	const vector<vector<Vector2>> &polygons
) {
	return ForcedDisjointBinarySearch(start, target, polygons).solve(tpp::PreloadPolicy::Lazy);
}

double checksum(const vector<Vector2> &path) {
	double result = 0.0;
	for (size_t i = 0; i < path.size(); i++) {
		result += path[i].x * static_cast<double>(i + 1);
		result += path[i].y * static_cast<double>((i + 1) * 17);
	}
	return result;
}

bool point_in_convex_polygon_closed(const Vector2 &point, const vector<Vector2> &polygon) {
	bool has_positive = false;
	bool has_negative = false;

	for (size_t i = 0; i < polygon.size(); i++) {
		const auto cross = (polygon[(i + 1) % polygon.size()] - polygon[i]).cross(point - polygon[i]);

		if (cross > 1e-16) {
			has_positive = true;
		} else if (cross < -1e-16) {
			has_negative = true;
		}

		if (has_positive && has_negative) {
			return false;
		}
	}

	return true;
}

bool polygons_intersect_or_touch(const vector<Vector2> &a, const vector<Vector2> &b) {
	for (size_t i = 0; i < a.size(); i++) {
		for (size_t j = 0; j < b.size(); j++) {
			if (tpp::segment_segment_intersection_safe(a[i], a[(i + 1) % a.size()], b[j], b[(j + 1) % b.size()]).is_finite()) {
				return true;
			}
		}
	}

	return point_in_convex_polygon_closed(a.front(), b) || point_in_convex_polygon_closed(b.front(), a);
}

bool pairwise_disjoint(const vector<vector<Vector2>> &polygons) {
	for (size_t i = 0; i < polygons.size(); i++) {
		for (size_t j = i + 1; j < polygons.size(); j++) {
			if (polygons_intersect_or_touch(polygons[i], polygons[j])) {
				return false;
			}
		}
	}

	return true;
}

template <typename Solver>
double benchmark(
	const vector<std::tuple<Vector2, Vector2, vector<vector<Vector2>>>> &instances,
	size_t repeats,
	Solver &&solver,
	double &checksum_output
) {
	const auto start_time = std::chrono::steady_clock::now();
	double total_checksum = 0.0;

	for (size_t repeat = 0; repeat < repeats; repeat++) {
		for (const auto &[start, target, polygons] : instances) {
			total_checksum += checksum(solver(start, target, polygons));
		}
	}

	const auto end_time = std::chrono::steady_clock::now();
	checksum_output = total_checksum;
	return std::chrono::duration<double>(end_time - start_time).count();
}

void run_case(
	const std::string &name,
	const vector<size_t> &polygon_sizes,
	size_t instance_count,
	size_t repeats
) {
	vector<std::tuple<Vector2, Vector2, vector<vector<Vector2>>>> instances;
	instances.reserve(instance_count);

	for (size_t i = 0; i < instance_count; i++) {
		auto instance = tpp::generate_test_bad(polygon_sizes, true);

		if (!pairwise_disjoint(std::get<2>(instance))) {
			throw std::runtime_error(name + ": generated intersecting polygons");
		}

		instances.push_back(std::move(instance));
	}

	double forced_checksum = 0.0;
	double wrapper_checksum = 0.0;
	const double forced_seconds = benchmark(instances, repeats, solve_forced_disjoint, forced_checksum);
	const double wrapper_seconds = benchmark(
		instances,
		repeats,
		[](const auto &start, const auto &target, const auto &polygons) {
			return tpp::tpp_convex_solve_binary_search_lazy(start, target, polygons);
		},
		wrapper_checksum
	);

	std::cout << std::format(
		"{},{},{},{:.6f},{:.6f},{:.2f}%,{:.12f},{:.12f}\n",
		name,
		polygon_sizes.size(),
		polygon_sizes.empty() ? 0 : polygon_sizes.front(),
		forced_seconds,
		wrapper_seconds,
		(wrapper_seconds / forced_seconds - 1.0) * 100.0,
		forced_checksum,
		wrapper_checksum
	) << std::flush;
}
}

int main() {
	tpp::set_rng_seed(12345);
	std::cout << "case,k,vertices_per_polygon,forced_seconds,wrapper_seconds,overhead,forced_checksum,wrapper_checksum\n" << std::flush;
	run_case("many_triangles", vector<size_t>(150, 3), 80, 5);
	run_case("many_quads", vector<size_t>(150, 4), 80, 5);
	run_case("medium_10", vector<size_t>(30, 10), 120, 5);
	run_case("medium_30", vector<size_t>(30, 30), 80, 5);
	run_case("few_large", vector<size_t>(5, 1000), 30, 5);
}
