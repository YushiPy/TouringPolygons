#include "tests.h"
#include "tpp_convex.h"

#include <filesystem>
#include <fstream>
#include <print>
#include <string>
#include <vector>

using std::vector;

namespace {

struct SuiteSpec {
	std::string filename;
	vector<vector<size_t>> polygon_sizes;
	unsigned int seed;
	bool use_good_generator = false;
};

void write_suite(const std::filesystem::path &output_dir, const SuiteSpec &suite) {
	const auto output_path = output_dir / suite.filename;
	std::ofstream file(output_path, std::ios::binary);

	if (!file) {
		throw std::runtime_error("Could not open generated test suite: " + output_path.string());
	}

	tpp::set_rng_seed(suite.seed);

	for (const auto &sizes : suite.polygon_sizes) {
		const auto [start, target, polygons] = suite.use_good_generator
			? tpp::generate_test_good(sizes, true)
			: tpp::generate_test(sizes);
		const auto solution = tpp::tpp_convex_solve_binary_search_lazy(start, target, polygons);
		const auto encoded = tpp::encode_test(start, target, polygons, solution);
		file.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
	}

	std::println("generated {}", output_path.string());
}

vector<vector<size_t>> repeat(vector<vector<size_t>> cases, size_t count) {
	vector<vector<size_t>> repeated;
	repeated.reserve(cases.size() * count);

	for (size_t i = 0; i < count; ++i) {
		repeated.insert(repeated.end(), cases.begin(), cases.end());
	}

	return repeated;
}

} // namespace

int main(int argc, char **argv) {
	const std::filesystem::path output_dir = argc > 1
		? argv[1]
		: std::filesystem::path("generated-tests");

	std::filesystem::create_directories(output_dir);

	const vector<SuiteSpec> suites = {
		{"small_tests.bin", repeat({
			{3, 3, 3, 3},
			{4, 4, 4, 4},
			{5, 5, 5, 5},
			{6, 6, 6, 6},
			{7, 7, 7, 7},
		}, 10), 1},
		{"medium_tests.bin", repeat({
			{3, 4, 5, 8, 6, 7, 4, 4, 10, 7, 3, 5},
			vector<size_t>(10, 5),
			vector<size_t>(10, 30),
			{4, 5, 6, 4, 5, 6, 4, 5, 6},
		}, 10), 2},
		{"many_small_tests.bin", repeat({
			vector<size_t>(150, 3),
			vector<size_t>(150, 4),
			vector<size_t>(150, 5),
			vector<size_t>(150, 6),
			vector<size_t>(150, 7),
		}, 10), 3, true},
		{"many_medium_tests.bin", repeat({
			vector<size_t>(30, 10),
			vector<size_t>(30, 20),
			vector<size_t>(30, 30),
			vector<size_t>(30, 40),
			vector<size_t>(30, 50),
		}, 10), 4, true},
		{"large_tests.bin", repeat({
			vector<size_t>(1, 100),
			vector<size_t>(2, 200),
			vector<size_t>(3, 300),
			vector<size_t>(4, 400),
			vector<size_t>(5, 1000),
			vector<size_t>(2, 10000),
		}, 10), 5},
		{"many_large_tests.bin", repeat({
			vector<size_t>(40, 100),
			vector<size_t>(40, 200),
			vector<size_t>(40, 300),
			vector<size_t>(40, 400),
			vector<size_t>(40, 500),
		}, 10), 6, true},
	};

	for (const auto &suite : suites) {
		write_suite(output_dir, suite);
	}
}
