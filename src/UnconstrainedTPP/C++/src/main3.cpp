
#include "tests.h"
#include "tpp_convex.h"

#include <print>
#include <fstream>

int main() {

	std::ifstream ifs("tests/edge_cases.bin", std::ios::binary);

	if (!ifs) {
		std::println("Failed to open file");
		return 1;
	}
	
	std::vector<tpp::TestCase> tests;

	while (ifs.peek() != EOF) {
		tests.push_back(tpp::decode_test(ifs));

		const auto &test = tests.back();
		std::println("Decoded test case: start={}, target={}, num_polygons={}, solution_length={}", 
			test.start, test.target, test.polygons.size(), test.solution.size());
	}
}