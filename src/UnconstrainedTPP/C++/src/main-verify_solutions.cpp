
#include "tests.h"
#include "tpp_convex.h"

#include <print>
#include <chrono>
#include <filesystem>
#include <fstream>

int main() {

	std::vector<tpp::Solver> solvers = {
		tpp::tpp_convex_solve_linear_search,
		tpp::tpp_convex_solve_binary_search,
		tpp::tpp_convex_solve_tamc
	};


	for (const auto &entry : std::filesystem::directory_iterator("tests/")) {
		
		if (entry.path().extension() != ".bin") {
			continue;
		}

		std::ifstream file(entry.path(), std::ios::binary);
		std::string filename = entry.path().filename().string();
		size_t test_num = 1;

		while (file.peek() != EOF) {

			const auto [start, target, polygons, expected_solution] = tpp::decode_test(file);

			for (const auto &solver : solvers) {

				const auto solution = solver(start, target, polygons);

				bool is_valid = tpp::is_valid_solution(start, target, polygons, solution);
				bool equals_expected = tpp::solutions_equal(solution, expected_solution);

				if (is_valid != equals_expected) {
					std::println("❌ Verifier and expected solutions disagree on test {} in file {}: valid={}, expected={}", test_num, filename, is_valid, equals_expected);
					return 1;
				} else if (!is_valid) {
					std::println("❌ Invalid solution for test {} in file {}.", test_num, filename);
					return 1;
				}
			}

			test_num++;
		}

		std::println("✅ All tests in file {} passed.", filename);
	}
}