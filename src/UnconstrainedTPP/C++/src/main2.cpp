
#include "tpp_convex.h"
#include "tests.h"

#include <print>

int main() {

	tpp::set_rng_seed(42);

	const auto [start, target, polygons] = tpp::generate_test({2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000});
	const auto path = tpp::tpp_convex_solve_tamc(start, target, polygons);

	auto is_valid = tpp::is_valid_solution(start, target, polygons, path);

	std::print("Is valid: {}\n", is_valid);

	tpp::plot_solution(start, target, polygons, path);
}
