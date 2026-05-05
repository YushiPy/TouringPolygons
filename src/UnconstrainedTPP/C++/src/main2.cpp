
#include "vector2.h"
#include "tpp_convex_linear_search.h"
#include "tpp_convex_binary_search.h"
#include "tests.h"

#include <print>
#include <chrono>

int main() {

	size_t m = 5;
	std::vector<double> times;

	for (size_t k = 3; k < 300; k += 10) {

		std::vector<size_t> polygon_sizes(m, k);

		auto [start, target, polygons] = tpp::generate_test(polygon_sizes);
		auto start_time = std::chrono::high_resolution_clock::now();
		auto solution = tpp::tpp_convex_solve_binary_search(start, target, polygons);
		auto end_time = std::chrono::high_resolution_clock::now();
		
		double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
		times.push_back(elapsed_seconds);
		
		// tpp::plot_solution(start, target, polygons, solution);
	}

	for (const auto &time : times) {
		std::print("{}, ", time);
	}

	std::println();
}