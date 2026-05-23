
#include "tests.h"
#include "tpp_convex.h"

#include <print>
#include <chrono>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <functional>

#include <string>

#include <omp.h>

using std::tuple;
using std::vector;
using std::string;

vector<std::pair<std::string, tpp::Solver>> solvers = {
	{"Linear Search", tpp::tpp_convex_solve_linear_search},
	{"Binary Search", tpp::tpp_convex_solve_binary_search},
	{"TAMC", tpp::tpp_convex_solve_tamc},
};


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
		
		for (size_t trial = 0; trial < num_trials; trial++) {

			const auto [start, target, polygons] = generator(polygon_sizes);

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

	return times;
}


std::string timings_to_string(const vector<string> &names, const vector<vector<double>> &times, const vector<vector<size_t>> &polygon_size_configs) {
	
	std::string result = "k,n,";

	for (size_t i = 0; i < names.size(); i++) {
		result += names[i];
		if (i < names.size() - 1) {
			result += ",";
		}
	}

	result += "\n";

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

	const vector<size_t> solver_indices_to_test = {1};
	const size_t chunk_count = 1;
	const size_t num_trials = 1;
	const bool parallel = true;


	vector<string> solver_names;
	vector<tpp::Solver> solver_functions;

	for (size_t index : solver_indices_to_test) {
		if (index < solvers.size()) {
			solver_names.push_back(solvers[index].first);
			solver_functions.push_back(solvers[index].second);
		} else {
			std::println("Solver index {} is out of range. Skipping.", index);
		}
	}

	auto generator_normal = tpp::generate_test;

	auto generator_bad = [&](const vector<size_t> &polygon_sizes) {
		return tpp::generate_test_bad(polygon_sizes, true);
	};

	auto generator_good = [&](const vector<size_t> &polygon_sizes) {
		return tpp::generate_test_good(polygon_sizes, true);
	};

	const auto generator = generator_bad;

	// Returns (n, k) for a config
	auto config_nk = [](const vector<size_t> &config) -> std::pair<double, double> {
		double n = std::accumulate(config.begin(), config.end(), 0.0);
		double k = config.size();
		return {n, k};
	};

	auto complexity = [](size_t solver_index, double n, double k) -> double {
		switch (solver_index) {
			case 0: return n * n;
			case 1: return n * k * std::log(n / k);
			case 2: return 6.0 * n * k;
			default: return n * n;
		}
	};
	
	auto estimate_remaining = [&](
		const vector<vector<size_t>> &all_configs,
		size_t completed_end,
		const vector<vector<double>> &times_so_far
	) {
		vector<double> C(solver_functions.size(), 0.0);
		vector<double> f2_sum(solver_functions.size(), 0.0);

		for (size_t ci = 0; ci < completed_end; ci++) {
			auto [n, k] = config_nk(all_configs[ci]);
			for (size_t si = 0; si < solver_functions.size(); si++) {
				double f = complexity(solver_indices_to_test[si], n, k);
				if (f > 0) {
					C[si]    += f * times_so_far[si][ci];
					f2_sum[si] += f * f;
				}
			}
		}
		for (size_t si = 0; si < solver_functions.size(); si++)
			if (f2_sum[si] > 0) C[si] /= f2_sum[si];

		vector<double> remaining(solver_functions.size(), 0.0);
		for (size_t ci = completed_end; ci < all_configs.size(); ci++) {
			auto [n, k] = config_nk(all_configs[ci]);
			for (size_t si = 0; si < solver_functions.size(); si++)
				remaining[si] += C[si] * complexity(solver_indices_to_test[si], n, k);
		}

		double total = std::accumulate(remaining.begin(), remaining.end(), 0.0);
		std::println("Estimated remaining time:");
		for (size_t si = 0; si < solver_functions.size(); si++)
			std::println("  {}: {:.1f}s", solver_names[si], remaining[si]);
		std::println("  Total: {:.1f}s", total);
	};

	auto do_test = [&](const vector<vector<size_t>> &polygon_size_configs, const std::string &filename) {

		const size_t part_size = (polygon_size_configs.size() + chunk_count - 1) / chunk_count;

		// accumulated_times[solver][config] for all completed configs so far
		vector<vector<double>> accumulated_times(solver_functions.size());

		// Write header
		{
			std::ofstream file(filename);
			file << "k,n";
			for (const auto &name : solver_names) file << "," << name;
			file << "\n";
		}

		const auto global_start_time = std::chrono::high_resolution_clock::now();

		for (size_t part = 0; part * part_size < polygon_size_configs.size(); part++) {

			size_t begin = part * part_size;
			size_t end = std::min(begin + part_size, polygon_size_configs.size());

			vector<vector<size_t>> chunk(
				polygon_size_configs.begin() + begin,
				polygon_size_configs.begin() + end
			);

			vector<vector<double>> times;

			if (parallel) {

				const auto start_time = std::chrono::high_resolution_clock::now();
				times = run_tests_parallel(solver_functions, chunk, num_trials, generator);
				const auto end_time = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> elapsed = end_time - start_time;

				const auto total_elapsed_time = std::chrono::high_resolution_clock::now() - global_start_time;
				double total_elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(total_elapsed_time).count() / 1000.0;

				std::println("Part {}/{} done in {:.2f} seconds ({:.2f}s so far)", part + 1, chunk_count, elapsed.count(), total_elapsed_seconds);
			} else {
				times = run_tests(solver_functions, chunk, num_trials, generator);
			}

			// Accumulate timings
			for (size_t si = 0; si < solver_functions.size(); si++)
				for (double t : times[si])
					accumulated_times[si].push_back(t);

			// Append chunk results to file
			std::ofstream file(filename, std::ios::app);
			for (size_t ci = 0; ci < chunk.size(); ci++) {
				const auto &config = chunk[ci];
				size_t k = config.size();
				size_t n = std::accumulate(config.begin(), config.end(), 0UZ);
				file << std::format("{},{}", k, n);
				for (size_t si = 0; si < times.size(); si++)
					file << std::format(",{}", times[si][ci]);
				file << "\n";
			}

			if (end < polygon_size_configs.size())
				estimate_remaining(polygon_size_configs, end, accumulated_times);
		}

		double total_elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - global_start_time).count() / 1000.0;
		std::println("✅ All tests completed in {:.2f} seconds", total_elapsed_seconds);
		std::println("Results saved to {}", filename);
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

	vector<size_t> ks = {1, 3, 5, 10, 20, 50, 100};
	vector<size_t> ms = {3, 5, 10, 20, 50, 100};
	const size_t max_m = 40000;
	const size_t max_k = 3000;
	const size_t step = 10;

	for (auto k : ks) {
		try {
			test_over_m(k, max_m, step);
		} catch (const std::exception &e) {
			std::println("Error during test_over_k: {}", e.what());
		}
	}
}