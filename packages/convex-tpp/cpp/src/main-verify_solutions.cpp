
#include "tests.h"
#include "tpp_convex.h"

#include <print>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstdlib>

template <typename T>
struct std::formatter<std::vector<T>> {
	constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
	auto format(const std::vector<T> &v, std::format_context &ctx) const {
		auto out = ctx.out();
		out = std::format_to(out, "{{");
		for (size_t i = 0; i < v.size(); i++) {
			if (i) out = std::format_to(out, ", ");
			out = std::format_to(out, "{}", v[i]);
		}
		return std::format_to(out, "}}");
	}
};

void print_test(const Vector2 &start, const Vector2 &target, const std::vector<std::vector<Vector2>> &polygons, const std::vector<Vector2> &solution, const std::vector<Vector2> &expected_solution) {
	std::print("Vector2 start({}, {});\n", start.x, start.y);
	std::print("Vector2 target({}, {});\n", target.x, target.y);
	std::print("std::vector<std::vector<Vector2>> polygons = {};\n", polygons);
	std::print("std::vector<Vector2> solution = {};\n", solution);
	std::print("std::vector<Vector2> expected_solution = {};\n", expected_solution);
}

int main() {

	std::vector<tpp::Solver> solvers = {
		[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_linear_search_lazy(s, t, p); },
		[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_linear_search_eager(s, t, p); },
		[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_binary_search_lazy(s, t, p); },
		[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_binary_search_eager(s, t, p); },
		[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_tan_jiang(s, t, p); },
		// tpp::tpp_convex_solve_gurobi,
	};

	const char *test_dir_env = std::getenv("TPP_TEST_DIR");
	const std::filesystem::path test_dir = test_dir_env != nullptr ? test_dir_env : "tests/";

	std::vector<std::filesystem::directory_entry> entries(
		std::filesystem::directory_iterator(test_dir), {}
	);

	std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
		return a.file_size() < b.file_size();
	});

	for (const auto &entry : entries) {
		
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
				bool equals_expected = tpp::solutions_equal(solution, expected_solution, 1e-8);

				if (is_valid != equals_expected) {
					std::println("❌ Verifier and expected solutions disagree on test {} in file {}: valid={}, expected={}", test_num, filename, is_valid, equals_expected);
					print_test(start, target, polygons, solution, expected_solution);
					return 1;
				} else if (!is_valid) {
					std::println("❌ Invalid solution for test {} in file {}.", test_num, filename);
					print_test(start, target, polygons, solution, expected_solution);
					return 1;
				}
			}

			test_num++;
		}

		std::println("✅ All tests in file {} passed.", filename);
	}
}
