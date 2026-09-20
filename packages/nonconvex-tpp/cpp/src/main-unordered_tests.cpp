#include "tpp/nonconvex/unordered.h"
#include "tpp/convex/certified.h"
#include "common.h"
#include "solvers/unordered_geometry.h"
#include "solvers/unordered_bounds.h"
#include <functional>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace tpp;
using Polygon = std::vector<Vector2>;

double length(const Polygon &p) {
	double value = 0;
	for (size_t i = 1; i < p.size(); ++i) value += p[i - 1].distance_to(p[i]);
	return value;
}

void check(Vector2 s, Vector2 t, const std::vector<Polygon> &polygons) {
	const auto result = tpp::tpp_nonconvex_unordered_solve(s, t, polygons);
	if (result.fallback_calls != result.fallback_geometric_path_invalid_calls + result.fallback_certificate_gap_calls
		|| result.extended_precision_calls > result.fallback_calls
		|| result.repaired_geometric_path_calls > result.calls
		|| result.calls != result.relaxation_calls + result.refinement_calls
		|| result.branch_events != result.insertion_branches + result.decomposition_branches
		|| result.partial_states_created < 1
		|| result.nodes > result.partial_states_created
		|| result.children_queued > result.children_generated
		|| result.insertion_positions_pruned > result.insertion_positions_considered
		|| result.pruned_nodes > result.pruned_states
		|| result.pruned_states != result.bound_prunes + result.incumbent_prunes
		|| result.best_updates > result.incumbent_updates
		|| (result.best_updates > 0 && !std::isfinite(result.first_best_update_length))
		|| (result.best_updates == 0 && std::isfinite(result.first_best_update_length))
		|| (result.best_updates > 0 && result.first_best_update_length > result.incumbent_length)
		|| result.final_length != result.upper_bound
		|| result.initial_length != result.initial_upper_bound
		|| result.incumbent_length != result.initial_upper_bound
		|| !std::isfinite(result.order_space_log2)
		|| result.convex_oracle_seconds + result.decomposition_seconds + result.search_visit_check_seconds
			+ result.search_maintenance_seconds > result.search_seconds + 1e-9
		|| result.heuristic_visit_check_seconds + result.search_visit_check_seconds
			+ result.finalization_visit_check_seconds > result.visit_check_seconds + 1e-9)
		{
		throw std::runtime_error("Inconsistent unordered profiling metrics.");
		}
	std::vector<size_t> order(polygons.size());
	std::iota(order.begin(), order.end(), 0);
	std::vector<std::vector<Polygon>> pieces;
	for (auto p : polygons) {
		double area = 0;
		for (size_t i = 0; i < p.size(); ++i) area += p[i].cross(p[(i + 1) % p.size()]);
		if (area < 0) std::reverse(p.begin(), p.end());
		pieces.push_back(tpp::decompose_polygon(p));
	}
	double best = std::numeric_limits<double>::infinity();
	tpp::DynamicConvexTppWorkspace workspace;
	do {
		std::vector<Polygon> ordered;
		std::function<void(size_t)> enumerate = [&](size_t i) {
			if (i == order.size()) {
				const auto fixed = tpp::tpp_convex_solve_certified(s, t, ordered, workspace, 1e-7);
				if (fixed.upper_bound - fixed.lower_bound > 1e-6) {
					std::cerr << "Oracle " << fixed.lower_bound << ' ' << fixed.upper_bound << " polygons " << ordered << " endpoints " << s << ' ' << t << "\n";
					throw std::runtime_error("Enumeration oracle gap.");
				}
				best = std::min(best, fixed.upper_bound);
				return;
			}
			for (const auto &piece : pieces[order[i]]) {
				ordered.push_back(piece);
				enumerate(i + 1);
				ordered.pop_back();
			}
		};
		enumerate(0);
	} while (std::next_permutation(order.begin(), order.end()));
	if (!result.exact || result.lower_bound > best + 1e-6 || result.lower_bound > result.upper_bound + 1e-8
		|| std::abs(best - result.upper_bound) > 1e-6 * (1 + best)) {
		std::cerr << "Expected " << best << ", got " << result.upper_bound << '\n';
		throw std::runtime_error("Permutation enumeration mismatch.");
	}
	for (size_t cap : {0, 1, 3, 10}) {
		tpp::UnorderedTppSolveOptions options;
		options.max_calls = cap;
		if (cap == 10) options.oracle_relative_gap = .01;
		const auto limited = tpp::tpp_nonconvex_unordered_solve(s, t, polygons, options);
		if (limited.calls > cap || limited.lower_bound > best + 1e-6 || limited.upper_bound < best - 1e-6)
			throw std::runtime_error("Invalid interrupted search bounds.");
	}
}

void check_oracle_certificates() {
	const Vector2 s{0, 0}, t{10, 0};
	const std::vector<Polygon> polygons = {
		{{6, 1}, {8, 1}, {8, 3}, {6, 3}},
		{{2, 2}, {7, 2}, {7, 4}, {2, 4}}
	};
	const double optimum = std::sqrt(40.) + std::sqrt(20.);
	DynamicConvexTppWorkspace workspace;
	for (double cutoff : {0., 10.5, std::numeric_limits<double>::infinity()}) {
		const auto r = tpp_convex_solve_certified(s, t, polygons, workspace, 1e-7, cutoff);
		if (r.lower_bound > optimum + 1e-8 || r.upper_bound < optimum - 1e-8
			|| (r.lower_bound < cutoff && r.upper_bound - r.lower_bound > 1e-7))
			throw std::runtime_error("Invalid convex cutoff certificate.");
	}
	const auto interrupted = tpp_convex_solve_certified(
		s, t, polygons, workspace, 0.0, std::numeric_limits<double>::infinity(), 0.0
	);
	if (!interrupted.time_limited || interrupted.lower_bound > optimum + 1e-8
		|| interrupted.upper_bound < optimum - 1e-8)
		throw std::runtime_error("Invalid deadline-interrupted convex bounds.");
	std::mt19937 rng(9162026);
	std::uniform_real_distribution<double> coordinate(-100, 100);
	const Polygon inserted{{4, -2}, {5, -2}, {5, -1}, {4, -1}};
	std::vector<double> optima;
	for (size_t j = 0; j <= polygons.size(); ++j) {
		auto sequence = polygons;
		sequence.insert(sequence.begin() + j, inserted);
		optima.push_back(tpp_convex_solve_certified(s, t, sequence, workspace, 1e-7).upper_bound);
	}
	for (size_t trial = 0; trial < 200; ++trial) {
		// Dual screening must remain valid even for completely infeasible hints.
		Polygon q{s, {coordinate(rng), coordinate(rng)}, {coordinate(rng), coordinate(rng)}, t};
		if (trial % 3 == 0) q[2] = q[1];
		const auto bounds = unordered_detail::insertion_lower_bounds(q, {&polygons[0], &polygons[1]}, inserted);
		for (size_t j = 0; j < bounds.size(); ++j) if (bounds[j] > optima[j] + 1e-8)
			throw std::runtime_error("Invalid incremental insertion bound.");
	}
	const Polygon rectangle{{4, 2}, {6, 2}, {6, 4}, {4, 4}};
	const auto point = unordered_detail::best_contact(s, t, rectangle, rectangle.front());
	if (point.distance_to({5, 2}) > 1e-12)
		throw std::runtime_error("Analytic edge contact missed reflection point.");
	const auto crossing = unordered_detail::best_contact({0, 3}, {10, 3}, rectangle, rectangle.front());
	if (std::abs(Vector2{0, 3}.distance_to(crossing) + crossing.distance_to({10, 3}) - 10) > 1e-12)
		throw std::runtime_error("Analytic edge contact missed pass-through.");
}

void check_coordinate_normalization() {
	const Vector2 start{-3, -2}, target{13, 8};
	const std::vector<Polygon> polygons = {
		{{0, 0}, {2, 0}, {2, .6}, {.6, .6}, {.6, 2}, {0, 2}},
		{{5, 3}, {7, 3}, {7, 5}, {5, 5}},
	};
	const auto base = tpp_nonconvex_unordered_solve(start, target, polygons);
	for (double factor : {1e-9, 1e9}) {
		const Vector2 offset{700 * factor, -400 * factor};
		auto transformed = polygons;
		for (auto &polygon : transformed) for (auto &point : polygon) point = point * factor + offset;
		const auto scaled = tpp_nonconvex_unordered_solve(
			start * factor + offset, target * factor + offset, transformed
		);
		if (!scaled.exact || !std::isfinite(scaled.upper_bound)
			|| std::abs(scaled.upper_bound - factor * base.upper_bound) > 1e-7 * std::max(1.0, factor * base.upper_bound)
			|| std::any_of(transformed.begin(), transformed.end(), [&](const auto &polygon) {
				return unordered_detail::contact(scaled.path, polygon, 1e-8).distance > 1e-8;
			}))
			throw std::runtime_error("Coordinate normalization regression.");
	}
}

int main() {
	try {
		check_oracle_certificates();
		check_coordinate_normalization();
		check({0, 0}, {10, 0}, {});
		check({0, 0}, {10, 0}, {{{2, -1}, {3, -1}, {3, 1}, {2, 1}}});
		check({0, 0}, {0, 0}, {{{2, -1}, {3, -1}, {3, 1}, {2, 1}}});
		check({0, 0}, {0, 0}, {{{-2, -2}, {2, -2}, {2, 2}, {-2, 2}}});
		check({1, 1.2}, {1.8, 1.3}, {{{0, 0}, {2, 0}, {2, .6}, {.6, .6}, {.6, 2}, {0, 2}}});
		check({0, 0}, {0, 1}, {{{-2, -2}, {2, -2}, {2, 2}, {1, 2}, {1, -1}, {-1, -1}, {-1, 2}, {-2, 2}}});
		std::mt19937 rng(342026);
		std::uniform_real_distribution<double> offset(-1, 1);
		for (size_t trial = 0; trial < 80; ++trial) {
			std::vector<Polygon> polygons;
			const size_t count = 2 + trial % 4;
			for (size_t i = 0; i < count; ++i) {
				const double x = (trial % 2 ? 1.5 : 5) * double(i % 3) + offset(rng);
				const double y = 5 * double(i / 3) + offset(rng);
				Polygon p = {{0, 0}, {2, 0}, {2, .6}, {.6, .6}, {.6, 2}, {0, 2}};
				if (trial % 3 == 0) p = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
				for (auto &v : p) v += Vector2{x, y};
				if (trial % 4 == 0) std::reverse(p.begin(), p.end());
				polygons.push_back(p);
			}
			
			check({-3, -2}, trial % 5 ? Vector2{13, 8} : Vector2{-3, -2}, polygons);
		}
		std::cout << "Passed oracle certificate/contact regressions, 86 exhaustive-order cases, and 344 interrupted-search checks.\n";
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
