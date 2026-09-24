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
		size_t parent = none;
		size_t branch_polygon = none;
		size_t branch_piece = none;
		size_t branch_position = none;
	};
	struct Later {
		bool operator()(const Node &a, const Node &b) const {
			return std::tie(a.bound, a.serial) > std::tie(b.bound, b.serial);
		}
	};
}

namespace tpp {
	static UnorderedTppSolveResult solve_normalized_unordered_tpp(
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
			result.polygon_vertices_total += p.size();
			result.polygon_vertices_min = std::min(result.polygon_vertices_min, p.size());
			result.polygon_vertices_max = std::max(result.polygon_vertices_max, p.size());
		}
		result.preprocessing_seconds = duration(preprocessing_began);
		const auto heuristic_began = std::chrono::steady_clock::now();
		const size_t n = polygons.size();
		result.order_space_log2 = std::lgamma(static_cast<double>(n) + 1.0) / std::log(2.0);
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
		auto trace_event = [&](UnorderedTppTraceEvent event) {
			if (options.trace) result.trace.push_back(std::move(event));
		};
		auto improve = [&](const Polygon &path, const std::string &source, const std::vector<size_t> &order = std::vector<size_t>{}) {
			const double value = path_length(path);
			if (std::isfinite(value) && value < result.upper_bound && covered(path)) {
				result.path = path;
				result.upper_bound = value;
				++result.incumbent_updates;
				if (phase == Phase::Search) {
					++result.best_updates;
					if (!std::isfinite(result.first_best_update_length)) result.first_best_update_length = value;
				}
				if (!std::isfinite(result.first_incumbent_seconds)) result.first_incumbent_seconds = elapsed();
				trace_event({
					.kind = "incumbent",
					.order = order,
					.path = path,
					.upper_bound = value,
					.length = value,
					.source = source,
				});
			}
		};
		result.lower_bound = start.distance_to(target);
		result.initial_lower_bound = result.lower_bound;
		if (options.initial_path) {
			improve(*options.initial_path, "provided_initial_path");
			if (!std::isfinite(result.upper_bound))
				throw std::invalid_argument("Initial path does not visit every polygon.");
		} else {
			improve({start, target}, "direct");
		}
		if (!options.initial_path && !std::isfinite(result.upper_bound)) {
			auto initialize = [&](Vector2 source, Vector2 destination, const std::string &direction) {
				Polygon initial{source};
				std::vector<size_t> order;
				std::vector<bool> used(n);
				trace_event({.kind = "heuristic_start", .source = direction, .path = initial});
				for (size_t k = 0; k < n; ++k) {
					double best = std::numeric_limits<double>::infinity();
					size_t selected = none;
					Vector2 point;
					for (size_t j = 0; j < n; ++j) if (!used[j]) for (auto v : polygons[j]) {
						const double distance = initial.back().distance_to(v);
						if (distance < best) { best = distance; selected = j; point = v; }
					}
					used[selected] = true;
					order.push_back(selected);
					initial.push_back(point);
					trace_event({
						.kind = "heuristic_greedy_step",
						.polygon = selected,
						.order = order,
						.path = initial,
						.length = path_length(initial),
						.source = direction,
					});
				}
				initial.push_back(destination);
				trace_event({
					.kind = "heuristic_greedy_complete",
					.order = order,
					.path = initial,
					.length = path_length(initial),
					.source = direction,
				});
				for (size_t pass = 0; pass < 10; ++pass) {
					bool changed = false;
					for (size_t i = 1; i < n; ++i) for (size_t j = i + 1; j <= n; ++j) {
						const double delta = initial[i - 1].distance_to(initial[j]) + initial[i].distance_to(initial[j + 1])
							- initial[i - 1].distance_to(initial[i]) - initial[j].distance_to(initial[j + 1]);
						if (delta < -eps) {
							std::reverse(initial.begin() + i, initial.begin() + j + 1);
							std::reverse(order.begin() + i - 1, order.begin() + j);
							changed = true;
						}
					}
					trace_event({
						.kind = "heuristic_2opt",
						.order = order,
						.path = initial,
						.pass = pass,
						.length = path_length(initial),
						.source = direction,
						.reason = changed ? "changed" : "stable",
					});
					if (!changed) break;
				}
				for (size_t pass = 0; pass < 8 && elapsed() < options.max_seconds; ++pass) {
					for (size_t k = 0; k < n; ++k) {
						const auto &p = polygons[order[k]];
						initial[k + 1] = best_contact(initial[k], initial[k + 2], p, initial[k + 1]);
					}
					for (size_t i = 1; i < n; ++i) for (size_t j = i + 1; j <= n; ++j) {
						const double delta = initial[i - 1].distance_to(initial[j]) + initial[i].distance_to(initial[j + 1])
							- initial[i - 1].distance_to(initial[i]) - initial[j].distance_to(initial[j + 1]);
						if (delta < -eps) {
							std::reverse(initial.begin() + i, initial.begin() + j + 1);
							std::reverse(order.begin() + i - 1, order.begin() + j);
						}
					}
					trace_event({
						.kind = "heuristic_contact_pass",
						.order = order,
						.path = initial,
						.pass = pass,
						.length = path_length(initial),
						.source = direction,
					});
				}
				return std::pair{std::move(initial), std::move(order)};
			};
			auto forward = initialize(start, target, "forward");
			improve(forward.first, "heuristic", forward.second);
			if (options.bidirectional_initial_heuristic && elapsed() < options.max_seconds) {
				auto reverse = initialize(target, start, "reverse");
				std::reverse(reverse.first.begin(), reverse.first.end());
				std::reverse(reverse.second.begin(), reverse.second.end());
				improve(reverse.first, "heuristic_reverse", reverse.second);
			}
		}
		result.initial_heuristic_seconds = duration(heuristic_began);
		result.initial_upper_bound = result.upper_bound;
		result.initial_length = result.initial_upper_bound;
		result.incumbent_length = result.initial_upper_bound;
		if (std::isfinite(result.initial_upper_bound)) {
			result.initial_gap_percent = 100.0 * (result.initial_upper_bound - result.initial_lower_bound)
				/ std::max(std::abs(result.initial_upper_bound), 1e-30);
		}
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
			if (precise) ++result.refinement_calls;
			else ++result.relaxation_calls;
			if (node.sequence.size() == n) {
				++result.complete_order_oracle_calls;
				if (std::all_of(node.sequence.begin(), node.sequence.end(), [](auto e) { return e.piece != none; }))
					++result.complete_piece_oracle_calls;
			}
			const double tolerance = precise ? gap() * .25
				: std::max(gap() * .25, options.oracle_relative_gap * result.upper_bound);
			const double cutoff = result.upper_bound - gap();
			const double remaining_seconds = std::max(0.0, options.max_seconds - elapsed());
			const auto certified = tpp_convex_solve_certified(
				start, target, selected, workspace, tolerance, cutoff, remaining_seconds
			);
			result.oracle_cutoff_calls += certified.lower_bound >= cutoff;
			result.oracle_dual_cutoff_prunes += certified.dual_cutoff_pruned;
			node.refined = precise || options.oracle_relative_gap == 0;
			node.path = certified.path;
			result.fallback_calls += certified.used_fallback;
			result.fallback_geometric_path_invalid_calls += certified.fallback_geometric_path_invalid;
			result.fallback_certificate_gap_calls += certified.fallback_certificate_gap;
			switch (certified.fallback_reason) {
				case ConvexFallbackReason::LocatorOrRefoldingException: ++result.fallback_locator_exception_calls; break;
				case ConvexFallbackReason::Nonfinite: ++result.fallback_nonfinite_calls; break;
				case ConvexFallbackReason::ContactConstruction: ++result.fallback_contact_construction_calls; break;
				case ConvexFallbackReason::MembershipOrOrdering: ++result.fallback_membership_ordering_calls; break;
				case ConvexFallbackReason::LocalOptimality: ++result.fallback_local_optimality_calls; break;
				case ConvexFallbackReason::CoincidentContact: ++result.fallback_coincident_contact_calls; break;
				default: break;
			}
			result.predicate_exact_evaluations += certified.predicate_exact_evaluations;
			result.extended_precision_calls += certified.used_extended_precision;
			result.oracle_time_limit_calls += certified.time_limited;
			result.repaired_geometric_path_calls += certified.repaired_geometric_path;
			result.convex_oracle_seconds += certified.seconds;
			result.convex_geometric_solver_seconds += certified.geometric_solver_seconds;
			result.convex_certificate_verification_seconds += certified.certificate_verification_seconds;
			result.convex_contact_materialization_seconds += certified.contact_materialization_seconds;
			result.convex_fallback_seconds += certified.fallback_seconds;
			result.convex_fallback_long_double_seconds += certified.fallback_long_double_seconds;
			result.convex_fallback_extended_precision_seconds += certified.fallback_extended_precision_seconds;
			node.bound = std::max(node.bound, certified.lower_bound);
			if (node.path.size() < 2 || !std::all_of(node.path.begin(), node.path.end(), [](auto v) { return v.is_finite(); }))
				throw std::runtime_error("Convex oracle returned an invalid path.");
			trace_event({
				.kind = "oracle",
				.node = node.serial,
				.parent = node.parent,
				.sequence = [&] {
					std::vector<size_t> sequence;
					for (auto e : node.sequence) sequence.push_back(e.polygon);
					return sequence;
				}(),
				.path = node.path,
				.lower_bound = node.bound,
				.upper_bound = result.upper_bound,
				.source = precise ? "refinement" : "convex_relaxation",
			});
		};
		double settled_bound = result.upper_bound;
		std::priority_queue<Node, std::vector<Node>, Later> queue;
		queue.push({{}, {start, target}, result.lower_bound, 0});
		result.partial_states_created = 1;
		trace_event({
			.kind = "root",
			.node = 0,
			.path = {start, target},
			.lower_bound = result.lower_bound,
			.upper_bound = result.upper_bound,
		});
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
			result.sequence_depth_sum += node.sequence.size();
			++result.sequence_depth_samples;
			result.max_sequence_depth = std::max(result.max_sequence_depth, node.sequence.size());
			if (node.path.empty()) solve(node);
			std::vector<size_t> node_sequence;
			for (auto e : node.sequence) node_sequence.push_back(e.polygon);
			trace_event({
				.kind = "expand",
				.node = node.serial,
				.parent = node.parent,
				.sequence = node_sequence,
				.path = node.path,
				.lower_bound = node.bound,
				.upper_bound = result.upper_bound,
			});
			if (node.bound >= result.upper_bound - gap()) {
				++result.pruned_nodes;
				++result.pruned_states;
				++result.bound_prunes;
				settled_bound = std::min(settled_bound, node.bound);
				trace_event({
					.kind = "prune",
					.node = node.serial,
					.parent = node.parent,
					.sequence = node_sequence,
					.lower_bound = node.bound,
					.upper_bound = result.upper_bound,
					.pruned = true,
					.reason = "bound",
				});
				continue;
			}
			improve(node.path, "oracle", node_sequence);
			if (node.bound >= result.upper_bound - gap()) {
				++result.pruned_nodes;
				++result.pruned_states;
				++result.incumbent_prunes;
				settled_bound = std::min(settled_bound, node.bound);
				trace_event({
					.kind = "prune",
					.node = node.serial,
					.parent = node.parent,
					.sequence = node_sequence,
					.lower_bound = node.bound,
					.upper_bound = result.upper_bound,
					.pruned = true,
					.reason = "incumbent",
				});
				continue;
			}
			size_t chosen = none;
			double farthest = eps;
			const auto visit_began = std::chrono::steady_clock::now();
			for (size_t j = 0; j < n; ++j) {
				const double distance = contact(node.path, polygons[j], eps).distance;
				if (distance > farthest) { farthest = distance; chosen = j; }
			}
			if (options.endpoint_sum_root && node.sequence.empty() && chosen != none) {
				const Polygon start_point_path{start, start}, target_point_path{target, target};
				double endpoint_sum = -1;
				for (size_t j = 0; j < n; ++j) {
					const double score = contact(start_point_path, polygons[j], 0).distance
						+ contact(target_point_path, polygons[j], 0).distance;
					if (score > endpoint_sum) { endpoint_sum = score; chosen = j; }
				}
			}
			result.search_visit_check_seconds += duration(visit_began);
			if (chosen == none) {
				if (!node.refined && !limited()) {
					solve(node, true);
					improve(node.path, "refinement", node_sequence);
					queue.push(std::move(node));
					continue;
				}
				// An unresolved numerical oracle gap must remain in the global certificate.
				queue.push(std::move(node));
				break;
			}
			trace_event({
				.kind = "branch",
				.node = node.serial,
				.parent = node.parent,
				.sequence = node_sequence,
				.polygon = chosen,
				.lower_bound = node.bound,
				.upper_bound = result.upper_bound,
				.reason = std::find_if(node.sequence.begin(), node.sequence.end(), [&](auto e) { return e.polygon == chosen; }) != node.sequence.end()
					? "decomposition" : "insertion",
			});
			++result.branch_events;
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
					++result.decomposed_polygons;
					result.convex_pieces_generated += pieces[chosen].size();
					result.convex_pieces_min = std::min(result.convex_pieces_min, pieces[chosen].size());
					result.convex_pieces_max = std::max(result.convex_pieces_max, pieces[chosen].size());
					result.decomposition_seconds += duration(decomposition_began);
				}
				if (pieces[chosen].empty()) throw std::runtime_error("Empty convex decomposition.");
				const size_t position = found - node.sequence.begin();
				result.total_branching += pieces[chosen].size();
				result.max_observed_branching = std::max(result.max_observed_branching, pieces[chosen].size());
				for (size_t j = 0; j < pieces[chosen].size(); ++j) {
					auto sequence = node.sequence;
					sequence[position].piece = j;
					Node child{std::move(sequence), {}, node.bound, serial++};
					child.parent = node.serial;
					child.branch_polygon = chosen;
					child.branch_piece = j;
					child.branch_position = position;
					children.push_back(std::move(child));
					++result.children_generated;
					++result.partial_states_created;
				}
			} else {
				++result.insertion_branches;
				std::vector<const Polygon *> regions;
				for (auto e : node.sequence) regions.push_back(e.piece == none ? &hulls[e.polygon] : &pieces[e.polygon][e.piece]);
				const auto bounds = insertion_lower_bounds(node.path, regions, hulls[chosen]);
				const size_t branching = node.sequence.size() + 1;
				result.total_branching += branching;
				result.max_observed_branching = std::max(result.max_observed_branching, branching);
				for (size_t j = 0; j <= node.sequence.size(); ++j) {
					++result.insertion_positions_considered;
					++result.children_generated;
					const double bound = std::max(node.bound, bounds[j]);
					auto sequence = node.sequence;
					sequence.insert(sequence.begin() + j, {chosen});
					if (bound >= result.upper_bound - gap()) {
						++result.screened_nodes;
						++result.insertion_positions_pruned;
						++result.pruned_states;
						++result.bound_prunes;
						settled_bound = std::min(settled_bound, bound);
						trace_event({
							.kind = "child",
							.parent = node.serial,
							.sequence = [&] {
								std::vector<size_t> child_sequence;
								for (auto e : sequence) child_sequence.push_back(e.polygon);
								return child_sequence;
							}(),
							.polygon = chosen,
							.position = j,
							.lower_bound = bound,
							.upper_bound = result.upper_bound,
							.pruned = true,
							.reason = "bound",
						});
						continue;
					}
					Node child{std::move(sequence), {}, bound, serial++};
					child.parent = node.serial;
					child.branch_polygon = chosen;
					child.branch_position = j;
					children.push_back(std::move(child));
					++result.partial_states_created;
				}
			}
			for (auto &child : children) {
				const bool pruned_by_new_incumbent = child.bound >= result.upper_bound - gap();
				if (pruned_by_new_incumbent) ++result.sibling_bound_prunes;
				if (!pruned_by_new_incumbent && !limited()) {
					solve(child);
					std::vector<size_t> child_sequence;
					for (auto e : child.sequence) child_sequence.push_back(e.polygon);
					improve(child.path, "oracle", child_sequence);
				}
				const bool queued = child.bound < result.upper_bound - gap();
				const auto child_sequence = [&] {
					std::vector<size_t> sequence;
					for (auto e : child.sequence) sequence.push_back(e.polygon);
					return sequence;
				}();
				trace_event({
					.kind = "child",
					.node = child.serial,
					.parent = child.parent,
					.sequence = child_sequence,
					.polygon = child.branch_polygon,
					.piece = child.branch_piece,
					.position = child.branch_position,
					.path = child.path,
					.lower_bound = child.bound,
					.upper_bound = result.upper_bound,
					.pruned = !queued,
					.reason = queued ? "queued" : "bound_after_incumbent",
				});
				if (queued) {
					++result.children_queued;
					if (diving && (!dive || child.bound < dive->bound)) {
						if (dive) queue.push(std::move(*dive));
						dive = std::move(child);
					} else queue.push(std::move(child));
				} else {
					++result.pruned_states;
					++result.incumbent_prunes;
					settled_bound = std::min(settled_bound, child.bound);
				}
			}
		}
		result.search_seconds = duration(search_began);
		result.search_maintenance_seconds = std::max(0.0, result.search_seconds - result.convex_oracle_seconds
			- result.decomposition_seconds - result.search_visit_check_seconds);
		phase = Phase::Finalization;
		const auto finalization_began = std::chrono::steady_clock::now();
		result.lower_bound = std::min({result.upper_bound, settled_bound, frontier_bound()});
		result.lower_bound = std::max(start.distance_to(target), result.lower_bound - normalization_error);
		result.final_absolute_gap = std::max(0.0, result.upper_bound - result.lower_bound);
		result.final_relative_gap = result.final_absolute_gap / std::max(std::abs(result.upper_bound), 1e-30);
		result.final_length = result.upper_bound;
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
		trace_event({
			.kind = "complete",
			.order = result.order,
			.path = result.path,
			.lower_bound = result.lower_bound,
			.upper_bound = result.upper_bound,
			.length = result.upper_bound,
			.reason = result.exact ? "optimal" : "incomplete",
		});
		return result;
	}

	UnorderedTppSolveResult tpp_nonconvex_unordered_solve(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &input,
		const UnorderedTppSolveOptions &options
	) {
		const auto began = std::chrono::steady_clock::now();
		auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count(); };
		if (!start.is_finite() || !target.is_finite() || std::isnan(options.max_seconds) || options.max_seconds < 0
			|| !std::isfinite(options.absolute_gap) || options.absolute_gap < 0
			|| !std::isfinite(options.relative_gap) || options.relative_gap < 0
			|| !std::isfinite(options.oracle_relative_gap) || options.oracle_relative_gap < 0
			|| !std::isfinite(options.feasibility_tolerance) || options.feasibility_tolerance <= 0)
			throw std::invalid_argument("Invalid endpoints or unordered TPP options.");
		Vector2 minimum = start, maximum = start;
		auto include = [&](Vector2 point) {
			if (!point.is_finite()) throw std::invalid_argument("Expected finite polygon coordinates.");
			minimum.x = std::min(minimum.x, point.x);
			minimum.y = std::min(minimum.y, point.y);
			maximum.x = std::max(maximum.x, point.x);
			maximum.y = std::max(maximum.y, point.y);
		};
		include(target);
		for (const auto &polygon : input) for (auto point : polygon) include(point);
		if (options.initial_path) {
			const auto &path = *options.initial_path;
			if (path.size() < 2 || !std::all_of(path.begin(), path.end(), [](auto point) { return point.is_finite(); })
				|| path.front().distance_to(start) > options.feasibility_tolerance
				|| path.back().distance_to(target) > options.feasibility_tolerance)
				throw std::invalid_argument("Initial path needs finite points and matching endpoints.");
			auto snapped = path;
			snapped.front() = start;
			snapped.back() = target;
			if (!std::all_of(input.begin(), input.end(), [&](const auto &polygon) {
				if (polygon.size() < 3) throw std::invalid_argument("Expected finite, nondegenerate simple polygons.");
				return contact(snapped, polygon, options.feasibility_tolerance).distance <= options.feasibility_tolerance;
			})) throw std::invalid_argument("Initial path does not visit every polygon.");
		}
		const Vector2 center{minimum.x / 2 + maximum.x / 2, minimum.y / 2 + maximum.y / 2};
		const double scale = std::max(maximum.x - minimum.x, maximum.y - minimum.y);
		if (!std::isfinite(scale)) throw std::invalid_argument("Coordinate range is too large.");
		const double divisor = scale > 0 ? scale : 1;
		auto normalize = [&](Vector2 point) { return (point - center) / divisor; };
		std::vector<Polygon> polygons = input;
		for (auto &polygon : polygons) for (auto &point : polygon) point = normalize(point);

		auto normalized_options = options;
		if (normalized_options.initial_path) {
			auto &path = *normalized_options.initial_path;
			path.front() = start;
			path.back() = target;
			for (auto &point : path) point = normalize(point);
		}
		normalized_options.absolute_gap /= divisor;
		const double numerical_floor = 64 * std::numeric_limits<double>::epsilon();
		normalized_options.feasibility_tolerance = std::max(options.feasibility_tolerance / divisor, numerical_floor);
		normalized_options.max_seconds = std::max(0.0, options.max_seconds - elapsed());
		const double normalization_seconds = elapsed();
		auto result = solve_normalized_unordered_tpp(
			normalize(start), normalize(target), polygons, normalized_options
		);
		auto scale_length = [&](double &value) {
			if (std::isfinite(value)) value *= divisor;
		};
		scale_length(result.lower_bound);
		scale_length(result.upper_bound);
		scale_length(result.initial_lower_bound);
		scale_length(result.initial_upper_bound);
		scale_length(result.initial_length);
		scale_length(result.incumbent_length);
		scale_length(result.first_best_update_length);
		scale_length(result.final_length);
		scale_length(result.final_absolute_gap);
		for (auto &point : result.path) point = {
			std::fma(point.x, divisor, center.x),
			std::fma(point.y, divisor, center.y),
		};
		for (auto &event : result.trace) {
			for (auto &point : event.path) point = {
				std::fma(point.x, divisor, center.x),
				std::fma(point.y, divisor, center.y),
			};
			if (std::isfinite(event.lower_bound)) event.lower_bound *= divisor;
			if (std::isfinite(event.upper_bound)) event.upper_bound *= divisor;
			if (std::isfinite(event.length)) event.length *= divisor;
		}
		auto covered = [&](const Polygon &path) {
			return std::all_of(input.begin(), input.end(), [&](const auto &polygon) {
				return contact(path, polygon, options.feasibility_tolerance).distance <= options.feasibility_tolerance;
			});
		};
		if (!covered(result.path)) {
			for (const auto &polygon : input) {
				const auto missing = contact(result.path, polygon, options.feasibility_tolerance);
				if (missing.distance <= options.feasibility_tolerance) continue;
				const size_t segment = std::min(size_t(std::max(0.0, std::floor(missing.position))), result.path.size() - 2);
				const double rate = std::clamp(missing.position - segment, 0.0, 1.0);
				const auto point = result.path[segment]
					+ (result.path[segment + 1] - result.path[segment]) * rate;
				Vector2 nearest = polygon.front();
				double best = (point - nearest).length_squared();
				for (size_t i = 0; i < polygon.size(); ++i) {
					const auto a = polygon[i], edge = polygon[(i + 1) % polygon.size()] - a;
					const double squared = edge.length_squared();
					const double edge_rate = squared == 0 ? 0
						: std::clamp((point - a).dot(edge) / squared, 0.0, 1.0);
					const auto candidate = a + edge * edge_rate;
					const double distance = (point - candidate).length_squared();
					if (distance < best) { best = distance; nearest = candidate; }
				}
				result.path.insert(result.path.begin() + segment + 1, {point, nearest, point});
			}
			if (!covered(result.path)) throw std::runtime_error("Failed to restore a normalized feasible path.");
			result.upper_bound = path_length(result.path);
			result.lower_bound = std::min(result.lower_bound, result.upper_bound);
			const double gap = options.absolute_gap + options.relative_gap * std::abs(result.upper_bound);
			result.exact = result.upper_bound - result.lower_bound <= gap;
			if (result.exact) result.termination = UnorderedTppTermination::Optimal;
			else if (result.termination == UnorderedTppTermination::Optimal)
				result.termination = UnorderedTppTermination::NumericalLimit;
			result.order.clear();
			std::vector<std::pair<double, size_t>> visits;
			for (size_t i = 0; i < input.size(); ++i)
				visits.emplace_back(contact(result.path, input[i], options.feasibility_tolerance).position, i);
			std::sort(visits.begin(), visits.end());
			for (auto [position, index] : visits) result.order.push_back(index);
		}
		result.final_length = result.upper_bound;
		result.final_absolute_gap = std::max(0.0, result.upper_bound - result.lower_bound);
		result.final_relative_gap = result.final_absolute_gap / std::max(std::abs(result.upper_bound), 1e-30);
		if (options.trace) {
			result.trace.push_back({
				.kind = "complete",
				.order = result.order,
				.path = result.path,
				.lower_bound = result.lower_bound,
				.upper_bound = result.upper_bound,
				.length = result.upper_bound,
				.reason = result.exact ? "optimal" : "incomplete",
			});
		}
		result.preprocessing_seconds += normalization_seconds;
		result.seconds = elapsed();
		return result;
	}
}
