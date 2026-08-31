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
		const auto linear_lazy = tpp::tpp_convex_solve_linear_search_lazy(start, target, polygons);
		const auto linear_eager = tpp::tpp_convex_solve_linear_search_eager(start, target, polygons);
		const auto lazy_length = tpp::tpp_convex_solve_length_binary_search_lazy(start, target, polygons);
		const auto eager_length = tpp::tpp_convex_solve_length_binary_search_eager(start, target, polygons);
		const auto linear_lazy_length = tpp::tpp_convex_solve_length_linear_search_lazy(start, target, polygons);
		const auto linear_eager_length = tpp::tpp_convex_solve_length_linear_search_eager(start, target, polygons);

		if (!tpp::is_valid_solution(start, target, polygons, lazy)) {
			throw std::runtime_error(name + ": lazy solver returned an invalid path");
		}

		if (!tpp::is_valid_solution(start, target, polygons, eager)) {
			throw std::runtime_error(name + ": eager solver returned an invalid path");
		}

		if (!tpp::is_valid_solution(start, target, polygons, linear_lazy)) {
			throw std::runtime_error(name + ": linear lazy solver returned an invalid path");
		}

		if (!tpp::is_valid_solution(start, target, polygons, linear_eager)) {
			throw std::runtime_error(name + ": linear eager solver returned an invalid path");
		}

		expect_close(path_length(lazy), expected_length, name + " lazy materialized");
		expect_close(path_length(eager), expected_length, name + " eager materialized");
		expect_close(path_length(linear_lazy), expected_length, name + " linear lazy materialized");
		expect_close(path_length(linear_eager), expected_length, name + " linear eager materialized");
		expect_close(lazy_length, expected_length, name + " lazy length");
		expect_close(eager_length, expected_length, name + " eager length");
		expect_close(linear_lazy_length, expected_length, name + " linear lazy length");
		expect_close(linear_eager_length, expected_length, name + " linear eager length");
	}
}

int main() {
	verify_case(
		"straight line visits separated convex polygons",
		Vector2(5.6599944320712705, 41.90941119153675),
		Vector2(27.21288557906459, -0.2783727728285078),
		{
			{
				Vector2(4.0033866926503325, 29.702672605790646),
				Vector2(17.283386692650346, 34.76267260579065),
				Vector2(13.983386692650333, 42.422672605790645),
				Vector2(1.0833866926503326, 36.80267260579065),
			},
			{
				Vector2(7.9916286191536745, 18.635407850779508),
				Vector2(21.181628619153678, 23.79540785077951),
				Vector2(18.251628619153678, 31.54540785077951),
				Vector2(5.071628619153675, 26.02540785077951),
			},
			{
				Vector2(11.88102171492205, 9.621930679287306),
				Vector2(24.231021714922047, 14.571930679287306),
				Vector2(21.86102171492205, 22.151930679287307),
				Vector2(8.671021714922048, 16.991930679287307),
			},
			{
				Vector2(16.895345211581294, 0.6087778396436541),
				Vector2(29.525345211581293, 5.8387778396436545),
				Vector2(26.225345211581292, 13.128777839643654),
				Vector2(13.875345211581294, 8.178777839643654),
			},
		},
		47.37442593444375
	);

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
