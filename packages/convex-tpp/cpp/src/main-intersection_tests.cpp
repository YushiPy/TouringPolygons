#include "tests.h"
#include "tpp_convex.h"

#include <cmath>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

using std::vector;

namespace {

	double path_length(const vector<Vector2> &path) {
		double result = 0.0;

		for (size_t i = 1; i < path.size(); i++) {
			result += path[i - 1].distance_to(path[i]);
		}

		return result;
	}

	void expect_close(double actual, double expected, const std::string &name) {
		if (std::fabs(actual - expected) > 1e-8) {
			throw std::runtime_error(
				std::format("{}: expected length {}, got {}", name, expected, actual)
			);
		}
	}

	void verify_case(
		const std::string &name,
		const Vector2 &start,
		const Vector2 &target,
		const vector<vector<Vector2>> &polygons,
		double expected_length
	) {
		const auto lazy = tpp::tpp_convex_solve_binary_search_lazy(start, target, polygons);
		const auto eager = tpp::tpp_convex_solve_binary_search_eager(start, target, polygons);
		const auto lazy_length = tpp::tpp_convex_solve_length_binary_search_lazy(start, target, polygons);
		const auto eager_length = tpp::tpp_convex_solve_length_binary_search_eager(start, target, polygons);

		if (!tpp::is_valid_solution(start, target, polygons, lazy)) {
			throw std::runtime_error(name + ": lazy solver returned an invalid path");
		}

		if (!tpp::is_valid_solution(start, target, polygons, eager)) {
			throw std::runtime_error(name + ": eager solver returned an invalid path");
		}

		expect_close(path_length(lazy), expected_length, name + " lazy materialized");
		expect_close(path_length(eager), expected_length, name + " eager materialized");
		expect_close(lazy_length, expected_length, name + " lazy length");
		expect_close(eager_length, expected_length, name + " eager length");
	}
}

int main() {
	verify_case(
		"straight line crosses overlapping squares",
		Vector2(-2, 0),
		Vector2(3, 0),
		{
			{Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)},
			{Vector2(0, -1), Vector2(2, -1), Vector2(2, 1), Vector2(0, 1)},
		},
		5.0
	);

	verify_case(
		"target inside last polygon",
		Vector2(-2, 0),
		Vector2(0.5, 0),
		{
			{Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)},
			{Vector2(0, -1), Vector2(2, -1), Vector2(2, 1), Vector2(0, 1)},
		},
		2.5
	);

	verify_case(
		"overlap skips middle visit",
		Vector2(-3, 0),
		Vector2(4, 0),
		{
			{Vector2(-2, -1), Vector2(0, -1), Vector2(0, 1), Vector2(-2, 1)},
			{Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)},
			{Vector2(0, -1), Vector2(2, -1), Vector2(2, 1), Vector2(0, 1)},
		},
		7.0
	);

	std::println("All intersection tests passed.");
}
