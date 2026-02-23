
#include "vector2.h"
#include "u_tpp_naive.h"

#include <vector>
#include <print>

#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

typedef std::tuple<Vector2, Vector2, std::vector<std::vector<Vector2>>> TestCase;
typedef std::tuple<double, std::vector<Vector2>> TestResult;

std::vector<TestCase> read_test_cases(const std::string& filename) {
	/*
	Reads test cases from `filename`. The file should be in the following format for each test case:
	
	- `8` bytes: number of test cases (`m`)
	- For each test case:
		- `16` bytes: `start`
		- `16` bytes: `target`
		- `8` bytes: number of polygons (`k`)
		- For each polygon `P_i`:
			- `8` bytes: number of vertices (`|P_i|`)
			- For each `vertex`:
				- `16` bytes: `vertex`
	
	Returns a vector of `m` test cases, where each test case is a tuple of `start`, `target`, and a vector of polygons.
	*/

	std::vector<TestCase> test_cases;
	std::ifstream file(filename);

	size_t num_cases;
	file.read((char*) &num_cases, sizeof(size_t));

	for (size_t i = 0; i < num_cases; i++) {

		Vector2 start, target;
		uint64_t k;

		file.read((char*) &start, sizeof(Vector2));
		file.read((char*) &target, sizeof(Vector2));
		file.read((char*) &k, sizeof(uint64_t));

		std::vector<std::vector<Vector2>> polygons;

		for (size_t i = 0; i < k; i++) {
			uint64_t num_vertices;
			file.read((char*) &num_vertices, sizeof(uint64_t));

			std::vector<Vector2> polygon;

			for (size_t j = 0; j < num_vertices; j++) {
				Vector2 vertex;
				file.read((char*) &vertex, sizeof(Vector2));
				polygon.push_back(vertex);
			}

			polygons.push_back(polygon);
		}

		test_cases.emplace_back(start, target, polygons);
	}

	while (file.peek() != EOF) {
		std::cerr << "Warning: Extra data found in file after reading all test cases. Check if the file format is correct." << std::endl;
		break;
	}

	return test_cases;
}

std::vector<TestResult> run_tests(const std::vector<TestCase>& test_cases) {
	/*
	Runs `tpp_solve` on each test case and returns a vector of tuples containing
	the ammount of seconds taken to solve each test case and the solution path.
	*/

	std::vector<std::tuple<double, std::vector<Vector2>>> results;

	for (const auto& [start, target, polygons] : test_cases) {
		
		auto start_time = std::chrono::high_resolution_clock::now();
		auto path = tpp_solve(start, target, polygons);
		auto end_time = std::chrono::high_resolution_clock::now();
		
		std::chrono::duration<double> elapsed = end_time - start_time;
		results.emplace_back(elapsed.count(), path);
	}

	return results;
}

void export_reults(std::vector<TestResult>& results, const std::string& filename) {
	/*
	Exports the results to a file in the following format for each test case:
	
	- `8` bytes: number of test cases (`m`)
	- For each test case:
		- `8` bytes: time taken in seconds (double)
		- `8` bytes: number of vertices in the solution path (`n`)
		- For each vertex in the solution path:
			- `16` bytes: vertex
	*/

	std::ofstream file(filename);

	size_t num_cases = results.size();
	file.write((char*) &num_cases, sizeof(size_t));

	for (const auto& [time, path] : results) {
		
		uint64_t num_vertices = path.size();
		file.write((char*) &time, sizeof(double));
		file.write((char*) &num_vertices, sizeof(uint64_t));

		for (const auto& vertex : path) {
			file.write((char*) &vertex, sizeof(Vector2));
		}
	}
}


std::string format_path(const std::vector<Vector2>& path, int precision = 6) {

	if (path.empty()) {
		return "{}";
	}

	std::ostringstream oss;

	oss << "{";

	for (size_t i = 0; i < path.size() - 1; i++) {
		oss << path[i].round(precision) << ", ";
	}

	oss << path.back().round(precision) << "}";

	return oss.str();
}

int main() {
	std::string filename = "test_cases.bin";
	auto test_cases = read_test_cases(filename);
	auto test_results = run_tests(test_cases);

	export_reults(test_results, "test_results.bin");
}