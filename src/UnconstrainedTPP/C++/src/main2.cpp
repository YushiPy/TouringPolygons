
#include "vector2.h"
#include "tpp_convex_naive.h"
#include "tests.h"

#include <print>

int main() {

	bool failed = false;

	for (size_t i = 0; i < 1000; i++) {

		auto [start, target, polygons] = tpp::generate_random_test({3, 4, 5, 6, 7});
		auto solution = tpp::tpp_convex_solve(start, target, polygons);
	
		if (!tpp::is_valid_solution(start, target, polygons, solution)) {
			failed = true;
			break;
		}
	}

	if (failed) {
		std::print("A test failed.\n");
	} else {
		std::print("All tests passed.\n");
	}
}