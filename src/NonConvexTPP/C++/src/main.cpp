
#include "common.h"
#include "tests.h"
#include "tpp_convex.h"

#include <print>
#include <fstream>

int main() {

	// Open file "test_cases.bin"
	auto test_cases = tpp::load_test_cases("tests/test_cases_simplified2.bin");
	const auto &[start, target, polygons, _] = test_cases[0];

	auto solution = tpp::tpp_convex_solve(start, target, polygons);
	
	solution.push_back(target);
	solution.insert(solution.begin(), start);

	tpp::plot_solution(start, target, polygons, solution);
}