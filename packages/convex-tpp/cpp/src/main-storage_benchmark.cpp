#include "tests.h"
#include "tpp_convex.h"

#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <print>
#include <string>
#include <vector>

using std::vector;

struct SolverEntry {
	std::string name;
	std::function<std::vector<Vector2>(const Vector2&, const Vector2&, const vector<vector<Vector2>>&)> convenience;
	std::function<void(const Vector2&, const Vector2&, const vector<vector<Vector2>>&, tpp::ConvexTppWorkspaceView, vector<Vector2>&)> with_workspace;
};

struct Config {
	std::string name;
	vector<size_t> polygon_sizes;
};

double path_length(const vector<Vector2> &path) {
	double total = 0.0;

	for (size_t i = 1; i < path.size(); i++) {
		total += path[i].distance_to(path[i - 1]);
	}

	return total;
}

int main() {
	const vector<SolverEntry> solvers = {
		{
			"linear_lazy",
			[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_linear_search_lazy(s, t, p); },
			[](const auto &s, const auto &t, const auto &p, auto w, auto &out) { tpp::tpp_convex_solve_linear_search_lazy(s, t, p, w, out); },
		},
		{
			"linear_eager",
			[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_linear_search_eager(s, t, p); },
			[](const auto &s, const auto &t, const auto &p, auto w, auto &out) { tpp::tpp_convex_solve_linear_search_eager(s, t, p, w, out); },
		},
		{
			"binary_lazy",
			[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_binary_search_lazy(s, t, p); },
			[](const auto &s, const auto &t, const auto &p, auto w, auto &out) { tpp::tpp_convex_solve_binary_search_lazy(s, t, p, w, out); },
		},
		{
			"binary_eager",
			[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_binary_search_eager(s, t, p); },
			[](const auto &s, const auto &t, const auto &p, auto w, auto &out) { tpp::tpp_convex_solve_binary_search_eager(s, t, p, w, out); },
		},
		{
			"tan_jiang",
			[](const auto &s, const auto &t, const auto &p) { return tpp::tpp_convex_solve_tan_jiang(s, t, p); },
			[](const auto &s, const auto &t, const auto &p, auto w, auto &out) { tpp::tpp_convex_solve_tan_jiang(s, t, p, w, out); },
		},
	};

	const vector<Config> configs = {
		{"few_large", vector<size_t>(5, 120)},
		{"many_small", vector<size_t>(80, 6)},
		{"balanced", vector<size_t>(30, 24)},
		{"many_medium", vector<size_t>(60, 16)},
	};

	const size_t trials = 100;
	const size_t warmups = 5;

	std::println("mode,config,k,n,solver,seconds_per_instance,checksum");

	for (size_t config_index = 0; config_index < configs.size(); config_index++) {
		const auto &config = configs[config_index];
		const size_t k = config.polygon_sizes.size();
		const size_t n = std::accumulate(config.polygon_sizes.begin(), config.polygon_sizes.end(), 0UZ);

		vector<std::tuple<Vector2, Vector2, vector<vector<Vector2>>>> instances;
		instances.reserve(trials + warmups);

		tpp::set_rng_seed(static_cast<unsigned int>(12345 + config_index));

		for (size_t trial = 0; trial < trials + warmups; trial++) {
			instances.push_back(tpp::generate_test_bad(config.polygon_sizes, true));
		}

		for (const auto &solver : solvers) {
			auto benchmark = [&](const std::string &mode, auto &&solve) {
				double checksum = 0.0;
				vector<Vector2> output;
				output.reserve(k + 2);

				for (size_t trial = 0; trial < warmups; trial++) {
					const auto &[start, target, polygons] = instances[trial];
					solve(start, target, polygons, output);
					checksum += path_length(output);
				}

				const auto start_time = std::chrono::steady_clock::now();

				for (size_t trial = warmups; trial < instances.size(); trial++) {
					const auto &[start, target, polygons] = instances[trial];
					solve(start, target, polygons, output);
					checksum += path_length(output);
				}

				const auto end_time = std::chrono::steady_clock::now();
				const std::chrono::duration<double> elapsed = end_time - start_time;
				const double seconds_per_instance = elapsed.count() / static_cast<double>(trials);

				std::println(
					"{},{},{},{},{},{:.9f},{:.12f}",
					mode,
					config.name,
					k,
					n,
					solver.name,
					seconds_per_instance,
					checksum
				);
			};

			benchmark("convenience", [&](const auto &start, const auto &target, const auto &polygons, auto &output) {
				output = solver.convenience(start, target, polygons);
			});

			tpp::DynamicConvexTppWorkspace dynamic_workspace;
			dynamic_workspace.reserve(k, n);

			benchmark("dynamic_workspace", [&](const auto &start, const auto &target, const auto &polygons, auto &output) {
				solver.with_workspace(start, target, polygons, dynamic_workspace.prepare(polygons.size(), tpp::total_vertex_count(polygons)), output);
			});

			tpp::StaticConvexTppWorkspace<100, 1200> static_workspace;

			benchmark("static_workspace", [&](const auto &start, const auto &target, const auto &polygons, auto &output) {
				solver.with_workspace(start, target, polygons, static_workspace.view(), output);
			});
		}
	}
}
