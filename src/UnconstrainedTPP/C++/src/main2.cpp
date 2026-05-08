
#include "tpp_convex.h"
#include "tests.h"

#include <print>

int main() {

	const auto [start, target, polygons] = tpp::generate_test({100, 100});
	const auto path = tpp::tpp_convex_solve_tamc(start, target, polygons);

	tpp::plot_solution(start, target, polygons, path);
}