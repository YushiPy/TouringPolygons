
#include "vector2.h"
#include "tpp_convex_linear_search.h"
#include "tpp_convex_binary_search.h"
#include "tpp_convex_tamc.h"
#include "tests.h"

#include <print>
#include <chrono>

#include <print>
#include <iostream>
#include <functional>

using Solver = std::function<std::vector<Vector2>(const Vector2&, const Vector2&, const std::vector<std::vector<Vector2>>&)>;

int main() {
	
	std::vector<Solver> solvers = {
		// tpp::tpp_convex_solve_linear_search,
		tpp::tpp_convex_solve_binary_search,
		tpp::tpp_convex_solve_tamc,
	};
	
	std::vector<std::vector<double>> timings(solvers.size());
	
	size_t k = 10;

	for (size_t i = 0; i < solvers.size(); i++) {

		const auto &solver = solvers[i];

		for (size_t m = 3; m <= 2000; m++) {

			const std::vector<size_t> polygon_sizes(k, m);
			const auto [start, target, polygons] = tpp::generate_test_bad(polygon_sizes);

			const auto start_time = std::chrono::high_resolution_clock::now();
			const auto result = solver(start, target, polygons);
			const auto end_time = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double> elapsed = end_time - start_time;

			if (!tpp::is_valid_solution(start, target, polygons, result)) {
				std::println("Invalid solution for solver {}", i);
				return 1;
			}

			timings[i].push_back(elapsed.count());
		}
	}

	for (size_t i = 0; i < solvers.size(); i++) {
		
		const auto &timing = timings[i];
		std::print("{}", timing);

		if (i < solvers.size() - 1) {
			std::println(",");
		}
	}
	std::println();
}