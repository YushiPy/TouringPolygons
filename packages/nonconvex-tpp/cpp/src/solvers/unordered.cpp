#include "tpp/nonconvex/unordered.h"
#include "tpp/convex/certified.h"
#include "common.h"
#include "unordered_geometry.h"
#include "unordered_bounds.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <queue>
#include <stdexcept>

namespace {
	using namespace tpp::unordered_detail;
	constexpr size_t none = std::numeric_limits<size_t>::max();

	struct Element { size_t polygon; size_t piece = none; };
	struct Node {
		std::vector<Element> sequence;
		Polygon path;
		double bound = 0;
		size_t serial = 0;
		bool refined = false;
	};
	struct Later {
		bool operator()(const Node &a, const Node &b) const {
			return std::tie(a.bound, a.serial) > std::tie(b.bound, b.serial);
		}
	};
}

namespace tpp {
	UnorderedTppSolveResult tpp_nonconvex_unordered_solve(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &input,
		const UnorderedTppSolveOptions &options
	) {
		const auto began = std::chrono::steady_clock::now();
		auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count(); };
		auto duration = [](auto since) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count(); };
		UnorderedTppSolveResult result;
		const auto preprocessing_began = std::chrono::steady_clock::now();
		if (!start.is_finite() || !target.is_finite() || std::isnan(options.max_seconds) || options.max_seconds < 0
			|| !std::isfinite(options.absolute_gap) || options.absolute_gap < 0
			|| !std::isfinite(options.relative_gap) || options.relative_gap < 0
			|| !std::isfinite(options.oracle_relative_gap) || options.oracle_relative_gap < 0
			|| !std::isfinite(options.feasibility_tolerance) || options.feasibility_tolerance <= 0)
			throw std::invalid_argument("Invalid endpoints or unordered TPP options.");
		std::vector<Polygon> polygons = input, hulls;
		double normalization_error = 0;
		for (auto &p : polygons) {
			const double duplicate_tolerance = options.feasibility_tolerance * 1e-4;
			Polygon cleaned;
			for (auto v : p) {
				if (!cleaned.empty() && cleaned.back().distance_to(v) <= duplicate_tolerance)
					normalization_error += 2 * cleaned.back().distance_to(v);
				else cleaned.push_back(v);
			}
			p = std::move(cleaned);
			if (p.size() > 1 && p.front().distance_to(p.back()) <= duplicate_tolerance) {
				normalization_error += 2 * p.front().distance_to(p.back());
				p.pop_back();
			}
			if (p.size() < 3 || !std::all_of(p.begin(), p.end(), [](auto v) { return v.is_finite(); }))
				throw std::invalid_argument("Expected finite, nondegenerate simple polygons.");
			double area = 0;
			for (size_t i = 0; i < p.size(); ++i) area += (p[i] - p[0]).cross(p[(i + 1) % p.size()] - p[0]);
			if (area == 0) throw std::invalid_argument("Zero-area polygon.");
			if (area < 0) std::reverse(p.begin(), p.end());
			hulls.push_back(convex_hull(p));
		}
		result.preprocessing_seconds = duration(preprocessing_began);
		const auto heuristic_began = std::chrono::steady_clock::now();
		const size_t n = polygons.size();
		std::vector<size_t> initial_order;
		const double eps = options.feasibility_tolerance;
		enum class Phase { Heuristic, Search, Finalization };
		Phase phase = Phase::Heuristic;
		auto covered = [&](const Polygon &path) {
			const auto check_began = std::chrono::steady_clock::now();
			const bool covered_result = std::all_of(polygons.begin(), polygons.end(), [&](const auto &p) { return contact(path, p, eps).distance <= eps; });
			const double seconds = duration(check_began);
			if (phase == Phase::Heuristic) result.heuristic_visit_check_seconds += seconds;
			else if (phase == Phase::Search) result.search_visit_check_seconds += seconds;
			else result.finalization_visit_check_seconds += seconds;
			return covered_result;
		};
		auto improve = [&](const Polygon &path) {
			const double value = path_length(path);
			if (std::isfinite(value) && value < result.upper_bound && covered(path)) {
				result.path = path;
				result.upper_bound = value;
			}
		};
		result.lower_bound = start.distance_to(target);
		improve({start, target});
		if (!std::isfinite(result.upper_bound)) {
			Polygon initial{start};
			std::vector<bool> used(n);
			for (size_t k = 0; k < n; ++k) {
				double best = std::numeric_limits<double>::infinity();
				size_t selected = none;
				Vector2 point;
				for (size_t j = 0; j < n; ++j) if (!used[j]) for (auto v : polygons[j]) {
					const double distance = initial.back().distance_to(v);
					if (distance < best) { best = distance; selected = j; point = v; }
				}
				used[selected] = true;
				initial_order.push_back(selected);
				initial.push_back(point);
			}
			initial.push_back(target);
			for (size_t pass = 0; pass < 10; ++pass) {
				bool changed = false;
				for (size_t i = 1; i < n; ++i) for (size_t j = i + 1; j <= n; ++j) {
					const double delta = initial[i - 1].distance_to(initial[j]) + initial[i].distance_to(initial[j + 1])
						- initial[i - 1].distance_to(initial[i]) - initial[j].distance_to(initial[j + 1]);
					if (delta < -eps) {
						std::reverse(initial.begin() + i, initial.begin() + j + 1);
						std::reverse(initial_order.begin() + i - 1, initial_order.begin() + j);
						changed = true;
					}
				}
				if (!changed) break;
			}
			for (size_t pass = 0; pass < 8 && elapsed() < options.max_seconds; ++pass) {
				for (size_t k = 0; k < n; ++k) {
					const auto &p = polygons[initial_order[k]];
					const auto left = initial[k], right = initial[k + 2];
					initial[k + 1] = best_contact(left, right, p, initial[k + 1]);
				}
				for (size_t i = 1; i < n; ++i) for (size_t j = i + 1; j <= n; ++j) {
					const double delta = initial[i - 1].distance_to(initial[j]) + initial[i].distance_to(initial[j + 1])
						- initial[i - 1].distance_to(initial[i]) - initial[j].distance_to(initial[j + 1]);
					if (delta < -eps) {
						std::reverse(initial.begin() + i, initial.begin() + j + 1);
						std::reverse(initial_order.begin() + i - 1, initial_order.begin() + j);
					}
				}
			}
			// Every heuristic contact stays on its original polygon.
			improve(initial);
		}
		result.initial_heuristic_seconds = duration(heuristic_began);
		phase = Phase::Search;
		const auto search_began = std::chrono::steady_clock::now();
		auto gap = [&] { return options.absolute_gap + options.relative_gap * std::abs(result.upper_bound); };
		auto limited = [&] { return result.calls >= options.max_calls || elapsed() >= options.max_seconds; };
		std::vector<std::vector<Polygon>> pieces(n);
		DynamicConvexTppWorkspace workspace;
		auto solve = [&](Node &node, bool precise = false) {
			std::vector<Polygon> selected;
			for (auto e : node.sequence) selected.push_back(e.piece == none ? hulls[e.polygon] : pieces[e.polygon][e.piece]);
			++result.calls;
			result.refinement_calls += precise;
			const double tolerance = precise ? gap() * .25
				: std::max(gap() * .25, options.oracle_relative_gap * result.upper_bound);
			const double cutoff = result.upper_bound - gap();
			const auto certified = tpp_convex_solve_certified(start, target, selected, workspace, tolerance, cutoff);
			result.oracle_cutoff_calls += certified.lower_bound >= cutoff;
			node.refined = precise || options.oracle_relative_gap == 0;
			node.path = certified.path;
			result.fallback_calls += certified.used_fallback;
			result.fallback_geometric_path_invalid_calls += certified.fallback_geometric_path_invalid;
			result.fallback_certificate_gap_calls += certified.fallback_certificate_gap;
			result.extended_precision_calls += certified.used_extended_precision;
			result.repaired_geometric_path_calls += certified.repaired_geometric_path;
			result.convex_oracle_seconds += certified.seconds;
			result.convex_geometric_solver_seconds += certified.geometric_solver_seconds;
			result.convex_certificate_verification_seconds += certified.certificate_verification_seconds;
			result.convex_fallback_seconds += certified.fallback_seconds;
			result.convex_fallback_long_double_seconds += certified.fallback_long_double_seconds;
			result.convex_fallback_extended_precision_seconds += certified.fallback_extended_precision_seconds;
			node.bound = std::max(node.bound, certified.lower_bound);
			if (node.path.size() < 2 || !std::all_of(node.path.begin(), node.path.end(), [](auto v) { return v.is_finite(); }))
				throw std::runtime_error("Convex oracle returned an invalid path.");
		};
		double settled_bound = result.upper_bound;
		std::priority_queue<Node, std::vector<Node>, Later> queue;
		queue.push({{}, {start, target}, result.lower_bound, 0});
		size_t serial = 1;
		std::optional<Node> dive;
		auto frontier_bound = [&] {
			return std::min(queue.empty() ? result.upper_bound : queue.top().bound, dive ? dive->bound : result.upper_bound);
		};
		while (!queue.empty() || dive) {
			result.peak_queue = std::max(result.peak_queue, queue.size() + size_t(dive.has_value()));
			result.lower_bound = std::min(result.upper_bound, frontier_bound());
			if (result.upper_bound - result.lower_bound <= gap() || limited()) break;
			const bool diving = dive.has_value() || (options.dive_interval && result.nodes % options.dive_interval == 0);
			Node node;
			if (dive) { node = std::move(*dive); dive.reset(); }
			else { node = queue.top(); queue.pop(); }
			++result.nodes;
			if (node.path.empty()) solve(node);
			if (node.bound >= result.upper_bound - gap()) { settled_bound = std::min(settled_bound, node.bound); continue; }
			improve(node.path);
			if (node.bound >= result.upper_bound - gap()) { settled_bound = std::min(settled_bound, node.bound); continue; }
			size_t chosen = none;
			double farthest = eps;
			const auto visit_began = std::chrono::steady_clock::now();
			for (size_t j = 0; j < n; ++j) {
				const double distance = contact(node.path, polygons[j], eps).distance;
				if (distance > farthest) { farthest = distance; chosen = j; }
			}
			result.search_visit_check_seconds += duration(visit_began);
			if (chosen == none) {
				if (!node.refined && !limited()) {
					solve(node, true);
					improve(node.path);
					queue.push(std::move(node));
					continue;
				}
				// An unresolved numerical oracle gap must remain in the global certificate.
				queue.push(std::move(node));
				break;
			}
			auto found = std::find_if(node.sequence.begin(), node.sequence.end(), [&](auto e) { return e.polygon == chosen; });
			std::vector<Node> children;
			if (found != node.sequence.end()) {
				if (found->piece != none) throw std::runtime_error("Certified oracle failed to visit an assigned piece.");
				++result.decomposition_branches;
				if (pieces[chosen].empty()) {
					const auto decomposition_began = std::chrono::steady_clock::now();
					for (auto piece : decompose_polygon(polygons[chosen])) {
						piece = convex_hull(std::move(piece));
						if (piece.size() >= 3) pieces[chosen].push_back(std::move(piece));
					}
					result.decomposition_seconds += duration(decomposition_began);
				}
				if (pieces[chosen].empty()) throw std::runtime_error("Empty convex decomposition.");
				const size_t position = found - node.sequence.begin();
				for (size_t j = 0; j < pieces[chosen].size(); ++j) {
					auto sequence = node.sequence;
					sequence[position].piece = j;
					children.push_back({std::move(sequence), {}, node.bound, serial++});
				}
			} else {
				++result.insertion_branches;
				std::vector<const Polygon *> regions;
				for (auto e : node.sequence) regions.push_back(e.piece == none ? &hulls[e.polygon] : &pieces[e.polygon][e.piece]);
				const auto bounds = insertion_lower_bounds(node.path, regions, hulls[chosen]);
				for (size_t j = 0; j <= node.sequence.size(); ++j) {
					const double bound = std::max(node.bound, bounds[j]);
					if (bound >= result.upper_bound - gap()) {
						++result.screened_nodes;
						settled_bound = std::min(settled_bound, bound);
						continue;
					}
					auto sequence = node.sequence;
					sequence.insert(sequence.begin() + j, {chosen});
					children.push_back({std::move(sequence), {}, bound, serial++});
				}
			}
			for (auto &child : children) {
				if (!limited()) { solve(child); improve(child.path); }
				if (child.bound < result.upper_bound - gap()) {
					if (diving && (!dive || child.bound < dive->bound)) {
						if (dive) queue.push(std::move(*dive));
						dive = std::move(child);
					} else queue.push(std::move(child));
				} else settled_bound = std::min(settled_bound, child.bound);
			}
		}
		result.search_seconds = duration(search_began);
		result.search_maintenance_seconds = std::max(0.0, result.search_seconds - result.convex_oracle_seconds
			- result.decomposition_seconds - result.search_visit_check_seconds);
		phase = Phase::Finalization;
		const auto finalization_began = std::chrono::steady_clock::now();
		result.lower_bound = std::min({result.upper_bound, settled_bound, frontier_bound()});
		result.lower_bound = std::max(start.distance_to(target), result.lower_bound - normalization_error);
		result.exact = result.upper_bound - result.lower_bound <= gap();
		result.termination = result.exact ? UnorderedTppTermination::Optimal
			: result.calls >= options.max_calls ? UnorderedTppTermination::CallLimit
			: elapsed() >= options.max_seconds ? UnorderedTppTermination::TimeLimit
			: UnorderedTppTermination::NumericalLimit;
		std::vector<std::pair<double, size_t>> visits;
		const auto final_visits_began = std::chrono::steady_clock::now();
		for (size_t j = 0; j < n; ++j) visits.emplace_back(contact(result.path, polygons[j], eps).position, j);
		result.finalization_visit_check_seconds += duration(final_visits_began);
		std::sort(visits.begin(), visits.end());
		for (auto [position, j] : visits) result.order.push_back(j);
		result.finalization_seconds = duration(finalization_began);
		result.visit_check_seconds = result.heuristic_visit_check_seconds + result.search_visit_check_seconds
			+ result.finalization_visit_check_seconds;
		result.seconds = elapsed();
		return result;
	}
}
