
#include "tests.h"
#include "tpp_convex.h"

#include <print>
#include <chrono>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <functional>

#include <omp.h>

using std::tuple;
using std::vector;

vector<vector<double>> run_tests(
	const vector<tpp::Solver> &solvers, 
	const vector<vector<size_t>> &polygon_size_configs, 
	size_t num_trials, 
	const std::function<tuple<Vector2, Vector2, vector<vector<Vector2>>>(const vector<size_t>&)> &generator,
	bool verbose = true,
	bool average_over_trials = true
) {
	
	vector<vector<double>> times(solvers.size(), vector<double>(polygon_size_configs.size(), 0.0));

	const auto global_start_time = std::chrono::high_resolution_clock::now();
	double total_test_creation_time = 0.0;
	size_t last_printed_second = 0;

	for (size_t config_index = 0; config_index < polygon_size_configs.size(); config_index++) {
		
		if (verbose) {

			const auto ellapsed_time = std::chrono::high_resolution_clock::now() - global_start_time;
			size_t elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(ellapsed_time).count();
	
			if (elapsed_seconds > last_printed_second) {
				std::println("Running benchmarks for config {} / {} ({}%)...", config_index + 1, polygon_size_configs.size(), (config_index + 1) * 100 / polygon_size_configs.size());
				std::println("Elapsed time: {} seconds", elapsed_seconds);
				last_printed_second = elapsed_seconds;
			}
		}

		const auto &polygon_sizes = polygon_size_configs[config_index];
		
		vector<tuple<Vector2, Vector2, vector<vector<Vector2>>>> test_cases(num_trials);

		for (size_t trial = 0; trial < num_trials; trial++) {
			
			auto start_time = std::chrono::high_resolution_clock::now();
			test_cases[trial] = generator(polygon_sizes);
			auto end_time = std::chrono::high_resolution_clock::now();
			
			std::chrono::duration<double> elapsed = end_time - start_time;
			total_test_creation_time += elapsed.count();
		}

		for (const auto &[start, target, polygons] : test_cases) {
			for (size_t solver_index = 0; solver_index < solvers.size(); solver_index++) {
				
				const auto &solver = solvers[solver_index];
				
				auto start_time = std::chrono::high_resolution_clock::now();
				solver(start, target, polygons);
				auto end_time = std::chrono::high_resolution_clock::now();
				
				std::chrono::duration<double> elapsed = end_time - start_time;
				times[solver_index][config_index] += elapsed.count();
			}
		}

		if (average_over_trials) {

			for (size_t solver_index = 0; solver_index < solvers.size(); solver_index++) {
				times[solver_index][config_index] /= num_trials;
			}
		}
	}

	if (verbose) {

		const auto total_elapsed_time = std::chrono::high_resolution_clock::now() - global_start_time;
		double total_elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(total_elapsed_time).count() / 1000.0;

		std::println("Total elapsed time: {} seconds", total_elapsed_seconds);
		std::println("Total test creation time: {:.6} seconds ({:.2}%)", total_test_creation_time, total_test_creation_time / total_elapsed_seconds * 100);
	}

	return times;
}


vector<vector<double>> run_tests_parallel(
	const vector<tpp::Solver> &solvers, 
	const vector<vector<size_t>> &polygon_size_configs, 
	size_t num_trials, 
	const std::function<tuple<Vector2, Vector2, vector<vector<Vector2>>>(const vector<size_t>&)> &generator,
	bool average_over_trials = true
) {
	
	vector<vector<double>> times(solvers.size(), vector<double>(polygon_size_configs.size(), 0.0));
	
	#pragma omp parallel for schedule(dynamic)
	for (size_t config_index = 0; config_index < polygon_size_configs.size(); config_index++) {
		
		const auto &polygon_sizes = polygon_size_configs[config_index];
		
		vector<tuple<Vector2, Vector2, vector<vector<Vector2>>>> test_cases(num_trials);
		
		for (size_t trial = 0; trial < num_trials; trial++) {
			test_cases[trial] = generator(polygon_sizes);
		}

		for (const auto &[start, target, polygons] : test_cases) {
			for (size_t solver_index = 0; solver_index < solvers.size(); solver_index++) {
				const auto &solver = solvers[solver_index];

				auto start_time = std::chrono::high_resolution_clock::now();
				solver(start, target, polygons);
				auto end_time = std::chrono::high_resolution_clock::now();

				std::chrono::duration<double> elapsed = end_time - start_time;
				times[solver_index][config_index] += elapsed.count();  // no race: each thread owns its config_index column
			}
		}

		if (average_over_trials) {
			for (size_t solver_index = 0; solver_index < solvers.size(); solver_index++) {
				times[solver_index][config_index] /= num_trials;
			}
		}
	}

	return times;
}


std::string timings_to_string(const vector<vector<double>> &times, const vector<vector<size_t>> &polygon_size_configs) {
	
	// std::string result = "k,n,linear_search,binary_search,tamc\n";
	std::string result = "k,n,binary_search,tamc\n";

	for (size_t config_index = 0; config_index < polygon_size_configs.size(); config_index++) {

		const auto &config = polygon_size_configs[config_index];

		size_t k = config.size();
		size_t n = 0;

		for (size_t size : config) {
			n += size;
		}

		result += std::format("{},", k);
		result += std::format("{},", n);

		for (size_t solver_index = 0; solver_index < times.size(); solver_index++) {
			result += std::format("{}", times[solver_index][config_index]);

			if (solver_index < times.size() - 1) {
				result += ",";
			}
		}

		result += "\n";
	}

	return result;
}


int main() {

	vector<tpp::Solver> solvers = {
		// tpp::tpp_convex_solve_linear_search,
		tpp::tpp_convex_solve_binary_search,
		// tpp::tpp_convex_solve_tamc
	};

	auto generator_normal = tpp::generate_test;

	auto generator_bad = [&](const vector<size_t> &polygon_sizes) {
		return tpp::generate_test_bad(polygon_sizes, true);
	};

	auto generator_good = [&](const vector<size_t> &polygon_sizes) {
		return tpp::generate_test_good(polygon_sizes, true);
	};

	const auto generator = generator_bad;
	const size_t num_trials = 1;
	const bool parallel = true;

	auto do_test = [&](const vector<vector<size_t>> &polygon_size_configs, const std::string &filename) {
		
		vector<vector<double>> times;
		
		if (parallel) {
			
			const auto start_time = std::chrono::high_resolution_clock::now();
			times = run_tests_parallel(solvers, polygon_size_configs, num_trials, generator);
			const auto end_time = std::chrono::high_resolution_clock::now();
			
			std::chrono::duration<double> elapsed = end_time - start_time;
			std::println("Total elapsed time: {} seconds", elapsed.count());

		} else {
			times = run_tests(solvers, polygon_size_configs, num_trials, generator);
		}

		// Write results in CSV format
		std::string result = timings_to_string(times, polygon_size_configs);

		std::ofstream file(filename);
		file << result;
	};

	auto test_over_k = [&](size_t m, size_t max_k, size_t step = 1) {
		
		vector<vector<size_t>> polygon_size_configs;

		for (size_t k = 1; k <= max_k; k += step) {
			polygon_size_configs.push_back(vector<size_t>(k, m));
		}

		std::string filename = std::format("benchmark_results_k_1_to_{}_m_{}.csv", max_k, m);

		do_test(polygon_size_configs, filename);
	};

	auto test_over_m = [&](size_t k, size_t max_m, size_t step = 1) {
		
		vector<vector<size_t>> polygon_size_configs;

		for (size_t m = 3; m <= max_m; m += step) {
			polygon_size_configs.push_back(vector<size_t>(k, m));
		}

		std::string filename = std::format("benchmark_results_k_{}_m_1_to_{}.csv", k, max_m);

		do_test(polygon_size_configs, filename);
	};

	auto test_over_n = [&](size_t k, size_t max_n, size_t step = 1) {
		
		vector<vector<size_t>> polygon_size_configs;

		for (size_t n = k * 3; n <= max_n; n += step) {

			size_t m = n / k;
			size_t remainder = n % k;
			
			vector<size_t> polygon_sizes(k, m);

			for (size_t i = 0; i < remainder; i++) {
				polygon_sizes[i]++;
			}

			polygon_size_configs.push_back(polygon_sizes);
		}

		std::string filename = std::format("benchmark_results_k_{}_n_1_to_{}.csv", k, max_n);
		do_test(polygon_size_configs, filename);
	};

	//test_over_k(10, 1000);
	test_over_m(10, 200000, 10000);
	// test_over_n(10, 5000, 10);
}