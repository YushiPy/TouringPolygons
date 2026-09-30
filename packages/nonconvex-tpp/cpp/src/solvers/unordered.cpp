#include "tpp/nonconvex/unordered.h"
#include "tpp/convex/certified.h"
#include "tpp/convex/cycle.h"
#include "common.h"
#include "unordered_geometry.h"
#include "unordered_bounds.h"
#include "unordered_portfolio.h"

#include <algorithm>
#include <array>
#include <set>
#include <thread>
#include <chrono>
#include <cmath>
#include <exception>
#include <optional>
#include <omp.h>
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
		Polygon warm_start;
        double relaxed_length = std::numeric_limits<double>::infinity();
        std::vector<int> active_features;
        tpp::ConvexRationalPolygon dual;
        double learning_parent_bound = 0, learning_distance = 0;
        bool learning_pending = false;
	};
	struct Later {
		bool operator()(const Node &a, const Node &b) const {
			return std::tie(a.bound, a.serial) > std::tie(b.bound, b.serial);
		}
	};

    // One frontier abstraction for both policies. The DFS stack has a separate
    // ordered bound index: its next node need not give the global lower bound.
    class Frontier {
        bool dfs;
        std::priority_queue<Node,std::vector<Node>,Later> heap;
        std::vector<Node> stack;
        std::set<std::pair<double,size_t>> bounds;
        static bool descending(const Node &a,const Node &b) {
            return std::tie(a.bound,a.relaxed_length,a.serial)>std::tie(b.bound,b.relaxed_length,b.serial);
        }
    public:
        explicit Frontier(bool use_dfs):dfs(use_dfs) {}
        bool empty() const { return dfs?stack.empty():heap.empty(); }
        size_t size() const { return dfs?stack.size():heap.size(); }
        const Node &top() const { return dfs?stack.back():heap.top(); }
        double lower_bound() const { return dfs?bounds.begin()->first:heap.top().bound; }
        void push(Node node) {
            if(dfs) { bounds.emplace(node.bound,node.serial);stack.push_back(std::move(node)); }
            else heap.push(std::move(node));
        }
        void pop() {
            if(dfs) { bounds.erase({stack.back().bound,stack.back().serial});stack.pop_back(); }
            else heap.pop();
        }
        void restart() { if(dfs) std::sort(stack.begin(),stack.end(),descending); }
        void finish_branch(size_t first_child) {
            if(dfs) std::sort(stack.begin()+first_child,stack.end(),descending);
        }
    };

    std::vector<Element> separated_cycle_root(const std::vector<Polygon> &polygons) {
        const size_t n=polygons.size();
        if(n<=3) { std::vector<Element> sequence;for(size_t i=0;i<n;++i)sequence.push_back({i});return sequence; }
        std::vector<Polygon> outlines=polygons;
        for(auto &p:outlines)p.push_back(p.front());
        std::vector<std::vector<double>> distances(n,std::vector<double>(n));
        size_t a=0,b=1;double longest=-1;
        for(size_t i=0;i<n;++i)for(size_t j=0;j<i;++j) {
            const double d=std::min(contact(outlines[i],polygons[j],0).distance,contact(outlines[j],polygons[i],0).distance);
            distances[i][j]=distances[j][i]=d;
            if(d>=longest){longest=d;a=i;b=j;}
        }
        size_t c=none;double farthest=-1;
        for(size_t i=0;i<n;++i)if(i!=a&&i!=b) {
            const double distance=distances[a][i]+distances[b][i];
            if(distance>farthest){farthest=distance;c=i;}
        }
        return {{a},{c},{b}};
    }

	// Keep OpenMP's function-entry initialization out of the serial search.
	// Inlining this region would make even a zero-oracle solve start its runtime.
	template<class Evaluate>
	[[gnu::noinline]] void evaluate_parallel_oracles(std::ptrdiff_t count, int threads, const Evaluate &evaluate) {
		std::vector<std::exception_ptr> failures(count);
#pragma omp parallel for schedule(dynamic) num_threads(threads)
		for (std::ptrdiff_t slot = 0; slot < count; ++slot) {
			try {
				evaluate(slot, omp_get_thread_num());
			} catch (...) {
				failures[slot] = std::current_exception();
			}
		}
		for (const auto &failure : failures) if (failure) std::rethrow_exception(failure);
	}
}

namespace tpp {
	struct RelaxationResult : CertifiedConvexTppResult {
        std::vector<int> active_features;
        size_t memo_queries=0,memo_repeated=0,memo_hits=0;
        size_t certificate_cutoff_skips=0,initial_contact_checks=0,initial_contact_accepts=0;
        size_t certificate_interval_uses=0;
        RelaxationResult() = default;
        RelaxationResult(CertifiedConvexTppResult result):CertifiedConvexTppResult(std::move(result)) {}
    };
    // Adapter only: both topologies use the same search below. No second B&B.
	static RelaxationResult solve_relaxation(bool cycle, const Vector2 &start,
		const Vector2 &target, const std::vector<Polygon> &regions,
		DynamicConvexTppWorkspace &workspace, double tolerance, double cutoff, double seconds,
		const Polygon &initial_contacts = {}, ConvexCycleWorkspace *cycle_workspace = nullptr,
        const std::vector<int> &initial_features = {}, bool retain_features = false, bool bound_first = false,
        bool interval_certificate = false) {
		if (!cycle) return tpp_convex_solve_certified(start,target,regions,workspace,tolerance,cutoff,seconds);
		const auto began=std::chrono::steady_clock::now();
		RelaxationResult out;
		if (regions.size()<2) {
			const auto point=regions.empty()?start:regions.front().front();
			out.path={point,point};return out;
		}
		ConvexCycleDoubleOptions cycle_options;
		cycle_options.lower_bound_cutoff=cutoff;
		cycle_options.initial_contacts=initial_contacts;
        cycle_options.workspace=cycle_workspace;
        cycle_options.initial_features=initial_features;
        cycle_options.retain_active_features=retain_features;
        cycle_options.bound_first=bound_first;
        cycle_options.interval_certificate=interval_certificate;
		auto solved=tpp_convex_solve_cycle_double(regions,cycle_options);
		if (solved.status!=ConvexCycleStatus::Optimal && solved.status!=ConvexCycleStatus::FloatingPointLimit
			&& solved.status!=ConvexCycleStatus::CertifiedBound)
			throw std::runtime_error("Convex cycle oracle failed: "+solved.diagnostic);
		out.active_features=std::move(solved.active_features);
        out.path=std::move(solved.contacts);out.path.push_back(out.path.front());
		out.lower_bound=solved.certificate.lower_bound;out.upper_bound=solved.certificate.upper_bound;
		out.dual_cutoff_pruned=solved.status==ConvexCycleStatus::CertifiedBound;
		out.predicate_exact_evaluations=solved.certificate.exact_predicate_evaluations;
		out.used_fallback=solved.rational_cycle_recoveries+solved.rational_anchor_recoveries+solved.rational_feature_recoveries>0;
        out.certificate_cutoff_skips=solved.certificate_cutoff_skips;
        out.certificate_interval_uses=solved.certificate_interval_uses;
        out.initial_contact_checks=solved.initial_contact_checks;out.initial_contact_accepts=solved.initial_contact_accepts;
		if (out.lower_bound<cutoff && out.upper_bound-out.lower_bound>tolerance) {
			// A rounded optimum may have a weak contact-derived dual. Recover its
			// global bound while retaining the independently feasible double path.
			ConvexCycleOptions exact_options;exact_options.lower_bound_cutoff=cutoff;
            exact_options.bound_first=bound_first;
			for(size_t i=0;i<regions.size();++i)exact_options.initial_contacts.emplace_back(out.path[i]);
			const auto exact=tpp_convex_solve_cycle(regions,exact_options);
			if (exact.status!=ConvexCycleStatus::Optimal&&exact.status!=ConvexCycleStatus::CertifiedBound)
				throw std::runtime_error("Exact cycle refinement failed: "+exact.diagnostic);
			out.lower_bound=std::max(out.lower_bound,exact.certificate.lower_bound);
			out.dual_cutoff_pruned=exact.status==ConvexCycleStatus::CertifiedBound;
			out.used_fallback=true;
            out.certificate_cutoff_skips+=exact.certificate_cutoff_skips;
		}
		out.fallback_certificate_gap=out.used_fallback;
		out.fallback_reason=out.used_fallback?ConvexFallbackReason::LocalOptimality:ConvexFallbackReason::None;
		out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
		out.geometric_solver_seconds=out.seconds;
		return out;
	}
	static UnorderedTppSolveResult solve_normalized_unordered_tpp(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &input,
		const UnorderedTppSolveOptions &options, bool cycle, PortfolioControl *control = nullptr
	) {
		const auto began = std::chrono::steady_clock::now();
		auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count(); };
		auto duration = [](auto since) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count(); };
		UnorderedTppSolveResult result;
		result.threads = options.threads;
		const auto preprocessing_began = std::chrono::steady_clock::now();
		if (!start.is_finite() || !target.is_finite() || std::isnan(options.max_seconds) || options.max_seconds < 0
			|| !std::isfinite(options.absolute_gap) || options.absolute_gap < 0
			|| !std::isfinite(options.relative_gap) || options.relative_gap < 0
			|| !std::isfinite(options.oracle_relative_gap) || options.oracle_relative_gap < 0
			|| !std::isfinite(options.feasibility_tolerance) || options.feasibility_tolerance <= 0
			|| options.threads == 0 || options.threads > static_cast<size_t>(std::numeric_limits<int>::max()))
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
		result.order_space_log2 = cycle ? (n<3?0:std::lgamma(static_cast<double>(n))/std::log(2.0)-1)
			: std::lgamma(static_cast<double>(n) + 1.0) / std::log(2.0);
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
                if(control) control->publish(path,value);
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
        auto import_incumbent = [&] {
            if(!control) return;
            Polygon candidate;
            if(control->receive(result.upper_bound,candidate)) {
                const double previous=result.upper_bound;
                improve(candidate,"portfolio");
                if(result.upper_bound<previous) ++result.portfolio_incumbent_imports;
            }
        };
		std::vector<std::vector<Polygon>> pieces(n);
		auto prepare_pieces = [&](size_t polygon_index) {
			if (!pieces[polygon_index].empty()) return;
			for (auto piece : decompose_polygon(polygons[polygon_index])) {
				piece = convex_hull(std::move(piece));
				if (piece.size() >= 3) pieces[polygon_index].push_back(std::move(piece));
			}
			if (pieces[polygon_index].empty()) throw std::runtime_error("Empty convex decomposition.");
			++result.decomposed_polygons;
			result.convex_pieces_generated += pieces[polygon_index].size();
			result.convex_pieces_min = std::min(result.convex_pieces_min, pieces[polygon_index].size());
			result.convex_pieces_max = std::max(result.convex_pieces_max, pieces[polygon_index].size());
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
			std::vector<Polygon> sampled_polygons;
			std::vector<size_t> sampled_point_counts;
			if (options.sampled_perimeter_initial_heuristic) {
				result.initial_sampling_work_budget = perimeter_sampling_work_budget(result.order_space_log2);
				sampled_point_counts = choose_perimeter_sample_point_counts(
					polygons, result.initial_sampling_work_budget, PerimeterSamplingWorkModel::AllPairs);
				for (size_t i = 0; i < n; ++i)
					result.initial_sampled_extra_points += sampled_point_counts[i] - polygons[i].size();
			}
			bool sampled_polygons_ready = false;
			auto prepare_sampled_polygons = [&] {
				if (sampled_polygons_ready || sampled_point_counts.empty()) return;
				sampled_polygons.reserve(n);
				for (size_t i = 0; i < n; ++i)
					sampled_polygons.push_back(evenly_spaced_perimeter_points(polygons[i], sampled_point_counts[i]));
				sampled_polygons_ready = true;
			};
			const bool strategy_enabled = options.sampled_perimeter_initial_heuristic
				|| options.bidirectional_initial_heuristic || options.convex_initial_refinement;
			const size_t candidate_count = (1 + size_t(options.sampled_perimeter_initial_heuristic))
				* (1 + size_t(options.bidirectional_initial_heuristic));
			const double available_heuristic_seconds = std::max(0.0, options.max_seconds - elapsed());
			const double candidate_budget = strategy_enabled ? available_heuristic_seconds * 0.5 : available_heuristic_seconds;
			const auto candidate_budget_started = std::chrono::steady_clock::now();
			auto candidate_deadline = [&](size_t index) {
				if (!strategy_enabled || !std::isfinite(candidate_budget))
					return std::chrono::steady_clock::time_point::max();
				const double fraction = static_cast<double>(index + 1) / candidate_count;
				return candidate_budget_started + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
					std::chrono::duration<double>(candidate_budget * fraction));
			};
			auto initialize = [&](Vector2 source, Vector2 destination,
				const std::string &direction, const std::vector<Polygon> &candidate_regions,
				std::chrono::steady_clock::time_point deadline) {
				Polygon initial{source};
				std::vector<size_t> order;
				std::vector<bool> used(n);
				trace_event({.kind = "heuristic_start", .source = direction, .path = initial});
				for (size_t k = 0; k < n; ++k) {
					double best = std::numeric_limits<double>::infinity();
					size_t selected = none;
					Vector2 point;
					for (size_t j = 0; j < n; ++j) if (!used[j]) for (auto v : candidate_regions[j]) {
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
				if (cycle) {
					initial.erase(initial.begin());
					initial.back()=initial.front();
				}
				trace_event({
					.kind = "heuristic_greedy_complete",
					.order = order,
					.path = initial,
					.length = path_length(initial),
					.source = direction,
				});
				for (size_t pass = 0; pass < 10 && std::chrono::steady_clock::now() < deadline; ++pass) {
					bool changed = false;
					for (size_t i = 1; i < n && std::chrono::steady_clock::now() < deadline; ++i)
					for (size_t j = i + 1; j <= n-size_t(cycle) && std::chrono::steady_clock::now() < deadline; ++j) {
						const double delta = initial[i - 1].distance_to(initial[j]) + initial[i].distance_to(initial[j + 1])
							- initial[i - 1].distance_to(initial[i]) - initial[j].distance_to(initial[j + 1]);
						if (delta < -eps) {
							std::reverse(initial.begin() + i, initial.begin() + j + 1);
							std::reverse(order.begin() + i - size_t(!cycle), order.begin() + j + size_t(cycle));
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
				for (size_t pass = 0; pass < 8 && elapsed() < options.max_seconds
					&& std::chrono::steady_clock::now() < deadline; ++pass) {
					for (size_t k = 0; k < n && std::chrono::steady_clock::now() < deadline; ++k) {
						const auto &p = polygons[order[k]];
						if (cycle) {
							initial[k]=best_contact(initial[(k+n-1)%n],initial[(k+1)%n],p,initial[k]);
							initial.back()=initial.front();
						} else initial[k + 1] = best_contact(initial[k], initial[k + 2], p, initial[k + 1]);
					}
					for (size_t i = 1; i < n && std::chrono::steady_clock::now() < deadline; ++i)
					for (size_t j = i + 1; j <= n-size_t(cycle) && std::chrono::steady_clock::now() < deadline; ++j) {
						const double delta = initial[i - 1].distance_to(initial[j]) + initial[i].distance_to(initial[j + 1])
							- initial[i - 1].distance_to(initial[i]) - initial[j].distance_to(initial[j + 1]);
						if (delta < -eps) {
							std::reverse(initial.begin() + i, initial.begin() + j + 1);
							std::reverse(order.begin() + i - size_t(!cycle), order.begin() + j + size_t(cycle));
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
			Polygon best_initial_path;
			std::vector<size_t> best_initial_order;
			double best_initial_length = std::numeric_limits<double>::infinity();
			auto consider = [&](bool reverse, const std::vector<Polygon> &candidate_regions,
				bool sampled, std::chrono::steady_clock::time_point deadline) {
				if (std::chrono::steady_clock::now() >= deadline) return;
				const std::string direction = reverse ? "reverse" : "forward";
				auto candidate = initialize(reverse ? target : start, reverse ? start : target,
					direction + (sampled ? "_sampled" : ""), candidate_regions, deadline);
				if (reverse) {
					std::reverse(candidate.first.begin(), candidate.first.end());
					std::reverse(candidate.second.begin()+size_t(cycle), candidate.second.end());
				}
				const double value = path_length(candidate.first);
				if (std::isfinite(value) && value < best_initial_length && covered(candidate.first)) {
					best_initial_path = candidate.first;
					best_initial_order = candidate.second;
					best_initial_length = value;
				}
				improve(candidate.first,
					sampled ? (reverse ? "heuristic_sampled_reverse" : "heuristic_sampled")
						: (reverse ? "heuristic_reverse" : "heuristic"),
					candidate.second);
			};
			size_t candidate_index = 0;
			consider(false, polygons, false, candidate_deadline(candidate_index++));
			if (options.sampled_perimeter_initial_heuristic) {
				prepare_sampled_polygons();
				consider(false, sampled_polygons, true, candidate_deadline(candidate_index++));
			}
			if (options.bidirectional_initial_heuristic) {
				consider(true, polygons, false, candidate_deadline(candidate_index++));
				if (options.sampled_perimeter_initial_heuristic) {
					prepare_sampled_polygons();
					consider(true, sampled_polygons, true, candidate_deadline(candidate_index++));
				}
			}

			if (options.convex_initial_refinement && !best_initial_path.empty()
				&& result.calls < options.max_calls && elapsed() < options.max_seconds) {
				const auto refinement_began = std::chrono::steady_clock::now();
				const double total_budget = std::min(1.0, options.max_seconds * 0.1);
				try {
					std::vector<Polygon> ordered_pieces;
					ordered_pieces.reserve(best_initial_order.size());
					bool assigned = best_initial_path.size() == best_initial_order.size() + 2-size_t(cycle);
					for (size_t k = 0; assigned && k < best_initial_order.size(); ++k) {
						const size_t polygon_index = best_initial_order[k];
						prepare_pieces(polygon_index);
						const Polygon point_path{best_initial_path[k + size_t(!cycle)], best_initial_path[k + size_t(!cycle)]};
						const auto found = std::find_if(pieces[polygon_index].begin(), pieces[polygon_index].end(),
							[&](const Polygon &piece) { return contact(point_path, piece, eps).distance <= eps; });
						if (found == pieces[polygon_index].end()) assigned = false;
						else ordered_pieces.push_back(*found);
					}
						const double preparation_seconds = duration(refinement_began);
						const double remaining = std::min({total_budget - preparation_seconds,
						options.max_seconds - elapsed(), total_budget});
					if (assigned && remaining > 0 && result.calls < options.max_calls
                        && (!control || control->reserve_call())) {
						DynamicConvexTppWorkspace initial_workspace;
						++result.calls;
						++result.initial_convex_refinement_calls;
						const double target_gap = options.absolute_gap
							+ options.relative_gap * std::abs(best_initial_length);
						const auto polished = solve_relaxation(
							cycle, start, target, ordered_pieces, initial_workspace,
							std::max(target_gap * 0.25, std::numeric_limits<double>::epsilon()),
							best_initial_length - target_gap, remaining);
						result.initial_convex_refinement_time_limited = polished.time_limited;
						const double polished_length = path_length(polished.path);
						if (std::isfinite(polished_length) && polished_length < best_initial_length && covered(polished.path)) {
							best_initial_path = polished.path;
							best_initial_length = polished_length;
							result.initial_convex_refinement_improved = true;
							improve(polished.path, "initial_convex_refinement", best_initial_order);
						}
					}
				} catch (const std::exception &error) {
					result.initial_convex_refinement_error = error.what();
				}
				result.initial_convex_refinement_seconds += duration(refinement_began);
			}
		}
		import_incumbent();
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
		auto gap_at = [&](double upper_bound) {
			return options.absolute_gap + options.relative_gap * std::abs(upper_bound);
		};
		auto gap = [&] { return gap_at(result.upper_bound); };
		auto limited = [&] { return result.calls >= options.max_calls || elapsed() >= options.max_seconds
            || (control && (control->stopped() || control->calls.load(std::memory_order_relaxed)>=control->max_calls)); };
        CycleOneTreeWorkspace one_tree;
        auto strengthen_one_tree = [&](Node &node) {
            if(!cycle||!options.cycle_one_tree||limited())return;
            const auto began_bound=std::chrono::steady_clock::now();
            std::vector<const Polygon *> regions;for(const auto &p:hulls)regions.push_back(&p);
            for(auto e:node.sequence)if(e.piece!=none)regions[e.polygon]=&pieces[e.polygon][e.piece];
            const auto bound=one_tree.bound(regions,result.upper_bound,limited);
            ++result.one_tree_calls;result.one_tree_cache_hits+=bound.cached;
            result.one_tree_iterations+=bound.iterations;result.one_tree_distance_queries+=bound.distance_queries;
            result.one_tree_improvements+=bound.lower_bound>node.bound;
            node.bound=std::max(node.bound,bound.lower_bound);
            result.one_tree_seconds+=duration(began_bound);
        };
        struct LearningMean {
            double mean=0;size_t count=0;
            void observe(double value){++count;mean+=(value-mean)/double(count);}
        };
        std::vector<std::array<LearningMean,2>> learned(n);
        std::array<LearningMean,2> prior;
		DynamicConvexTppWorkspace workspace;
		std::vector<DynamicConvexTppWorkspace> parallel_workspaces;
        ConvexCycleWorkspace cycle_workspace;
        std::vector<ConvexCycleWorkspace> parallel_cycle_workspaces;
        CycleMemo memo_workspace;
        std::vector<CycleMemo> parallel_memo_workspaces;
        auto strengthen_shared_bound=[&](Node &node) {
            if(!cycle||!options.cycle_share_bounds||!control||!control->sharing||node.sequence.size()<2)return;
            const auto began_bound=std::chrono::steady_clock::now();
            PortfolioControl::CycleKey key;
            for(auto e:node.sequence) {
                std::vector<std::pair<double,double>> coordinates;
                for(auto v:e.piece==none?hulls[e.polygon]:pieces[e.polygon][e.piece])coordinates.emplace_back(v.x,v.y);
                key.emplace_back(e.polygon,std::move(coordinates));
            }
            const double cutoff=result.upper_bound-gap();
            const double bound=control->compatible_cycle_bound(key,cutoff,result.cycle_shared_bound_queries,result.cycle_shared_bound_hits);
            result.cycle_shared_bound_improvements+=bound>node.bound;
            result.cycle_shared_bound_prunes+=node.bound<cutoff&&bound>=cutoff;
            node.bound=std::max(node.bound,bound);
            result.cycle_shared_bound_seconds+=duration(began_bound);
        };
		auto evaluate_oracle = [&](const Node &node, bool precise, double upper_bound,
			DynamicConvexTppWorkspace &oracle_workspace, ConvexCycleWorkspace &cycle_cache, CycleMemo &memo) {
            const auto began_oracle=std::chrono::steady_clock::now();
			std::vector<Polygon> selected;
			for (auto e : node.sequence) selected.push_back(e.piece == none ? hulls[e.polygon] : pieces[e.polygon][e.piece]);
			const double node_gap = options.absolute_gap + options.relative_gap * std::abs(upper_bound);
			const double tolerance = precise ? node_gap * .25
				: std::max(node_gap * .25, options.oracle_relative_gap * upper_bound);
			const double cutoff = upper_bound - node_gap;
			const double remaining_seconds = std::max(0.0, options.max_seconds - elapsed());
            const bool shared_bounds=cycle&&options.cycle_share_bounds&&control&&control->sharing;
            const bool cache=cycle&&(options.cycle_memo||shared_bounds)&&selected.size()>1;
            CycleMemo::Key key;std::vector<size_t> order;bool repeated=false;
            const bool shared_cache=cache&&control&&control->sharing;
            PortfolioControl::CycleKey shared_key;
            if(cache) {
                CycleMemo::Key labels;for(auto e:node.sequence)labels.emplace_back(e.polygon,e.piece);
                order=canonical_cycle_indices(labels);for(size_t i:order)key.push_back(labels[i]);
                std::optional<CycleMemo::Entry> shared_entry;
                const CycleMemo::Entry *previous=nullptr;
                if(shared_cache) {
                    for(size_t i:order) {
                        std::vector<std::pair<double,double>> coordinates;
                        for(auto q:selected[i])coordinates.emplace_back(q.x,q.y);
                        shared_key.emplace_back(node.sequence[i].polygon,std::move(coordinates));
                    }
                    if(options.cycle_memo)shared_entry=control->find_cycle(shared_key);
                    if(shared_entry)previous=&*shared_entry;
                } else if(const auto found=memo.entries.find(key);found!=memo.entries.end())previous=&found->second;
                repeated=previous!=nullptr;
                if(repeated) {
                    const auto &entry=*previous;
                    if(entry.lower_bound>=cutoff||entry.upper_bound-entry.lower_bound<=tolerance) {
                        RelaxationResult out;Polygon contacts(order.size());
                        for(size_t i=0;i<order.size();++i)contacts[order[i]]=entry.contacts[i];
                        // Revalidate membership and the candidate independently.
                        // The retained bound was proved for these identical constraints.
                        const auto check=tpp_convex_verify_cycle_certificate(cycle_cache.prepare(selected,options.cycle_interval_certificate),contacts,
                            (options.cycle_bound_first||options.cycle_interval_certificate)?cutoff:INFINITY,options.cycle_interval_certificate);
                        if(check.status==ConvexCycleCertificateStatus::Optimal||check.status==ConvexCycleCertificateStatus::Feasible) {
                            out.path=contacts;out.path.push_back(contacts.front());
                            out.lower_bound=std::max(entry.lower_bound,check.lower_bound);out.upper_bound=check.upper_bound;
                            out.dual_cutoff_pruned=out.lower_bound>=cutoff;
                            out.predicate_exact_evaluations=check.exact_predicate_evaluations;
                            out.certificate_cutoff_skips=check.optimality_check_skipped;
                            out.certificate_interval_uses=check.interval_bounds_used;
                            if(entry.features.size()==order.size()) {
                                out.active_features.resize(order.size());
                                for(size_t i=0;i<order.size();++i)out.active_features[order[i]]=entry.features[i];
                            }
                            out.memo_queries=out.memo_repeated=out.memo_hits=1;
                            out.seconds=duration(began_oracle);out.geometric_solver_seconds=out.seconds;
                            return out;
                        }
                    }
                }
            }
			auto out=solve_relaxation(
				cycle, start, target, selected, oracle_workspace, tolerance, cutoff, remaining_seconds, node.warm_start, options.cycle_cache?&cycle_cache:nullptr,
                options.cycle_active_features?node.active_features:std::vector<int>{}, options.cycle_active_features,options.cycle_bound_first,
                options.cycle_interval_certificate
			);
            if(cache) {
                out.memo_queries=options.cycle_memo;out.memo_repeated=repeated;
                CycleMemo::Entry entry;entry.lower_bound=out.lower_bound;entry.upper_bound=out.upper_bound;
                for(size_t i:order)entry.contacts.push_back(out.path[i]);
                if(out.active_features.size()==order.size())for(size_t i:order)entry.features.push_back(out.active_features[i]);
                if(shared_cache)control->store_cycle(std::move(shared_key),std::move(entry));
                else {
                    if(memo.entries.size()>=4096)memo.entries.clear();
                    memo.entries[std::move(key)]=std::move(entry);
                }
            }
            return out;
		};
		auto note_oracle_call = [&](const Node &node, bool precise) {
            if(control && !control->reserve_call()) return false;
			++result.calls;
			if (precise) ++result.refinement_calls;
			else ++result.relaxation_calls;
			if (node.sequence.size() == n) {
				++result.complete_order_oracle_calls;
				if (std::all_of(node.sequence.begin(), node.sequence.end(), [](auto e) { return e.piece != none; }))
					++result.complete_piece_oracle_calls;
			}
            return true;
		};
		auto record_oracle_result = [&](Node &node, bool precise, double cutoff,
			const RelaxationResult &certified) {
			node.warm_start=Polygon{};
			result.oracle_cutoff_calls += certified.lower_bound >= cutoff;
			result.oracle_dual_cutoff_prunes += certified.dual_cutoff_pruned;
            result.cycle_memo_queries+=certified.memo_queries;result.cycle_memo_repeated+=certified.memo_repeated;
            result.cycle_memo_hits+=certified.memo_hits;result.cycle_certificate_cutoff_skips+=certified.certificate_cutoff_skips;
            result.cycle_certificate_interval_uses+=certified.certificate_interval_uses;
            result.cycle_initial_contact_checks+=certified.initial_contact_checks;result.cycle_initial_contact_accepts+=certified.initial_contact_accepts;
			node.refined = precise || options.oracle_relative_gap == 0;
			node.path = certified.path;
            node.active_features=options.cycle_active_features?certified.active_features:std::vector<int>{};
            if(cycle&&options.cycle_dual_reuse&&node.path.size()==node.sequence.size()+1) {
                Polygon contacts(node.path.begin(),node.path.end()-1);
                node.dual=tpp_convex_cycle_dual_directions(contacts,node.dual);
            }
            node.relaxed_length = certified.upper_bound;
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
            if(cycle&&options.cycle_learned_branching&&node.learning_pending) {
                const double gain=std::max(0.0,node.bound-node.learning_parent_bound)/node.learning_distance;
                if(std::isfinite(gain)) {
                    const size_t kind=node.branch_piece!=none;
                    learned[node.branch_polygon][kind].observe(gain);prior[kind].observe(gain);
                    ++result.learned_branch_observations;
                }
                node.learning_pending=false;
            }
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
		auto solve = [&](Node &node, bool precise = false) {
			import_incumbent();
            if(!note_oracle_call(node, precise)) return false;
			const double upper_bound = result.upper_bound;
			const double cutoff = upper_bound - gap_at(upper_bound);
			const auto oracle_began = std::chrono::steady_clock::now();
			const auto certified = evaluate_oracle(node, precise, upper_bound, workspace, cycle_workspace,memo_workspace);
			result.convex_oracle_wall_seconds += duration(oracle_began);
			record_oracle_result(node, precise, cutoff, certified);
            return true;
		};
		double settled_bound = result.upper_bound;
		const bool dfs=options.search_strategy==UnorderedSearchStrategy::DfsBfs;
        Frontier queue(dfs);
		// Rooting the sequence at region 0 removes rotation, without fixing a
		// geometric point. A single-region cycle has lower bound zero.
		std::vector<Element> root_sequence;
		if (cycle && n) root_sequence=(dfs||options.cycle_separated_root)?separated_cycle_root(polygons):std::vector<Element>{{0}};
        Node root{root_sequence, cycle&&(dfs||options.cycle_separated_root)&&n>1?Polygon{}:Polygon{start,target}, result.lower_bound, 0};
        strengthen_one_tree(root);result.lower_bound=std::min(result.upper_bound,root.bound);
        result.initial_lower_bound=result.lower_bound;
        if(std::isfinite(result.initial_upper_bound))result.initial_gap_percent=100*(result.initial_upper_bound-result.initial_lower_bound)
            /std::max(std::abs(result.initial_upper_bound),1e-30);
        queue.push(std::move(root));
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
			return std::min(queue.empty() ? result.upper_bound : queue.lower_bound(), dive ? dive->bound : result.upper_bound);
		};
		while (!queue.empty() || dive) {
            import_incumbent();
			result.peak_queue = std::max(result.peak_queue, queue.size() + size_t(dive.has_value()));
			result.lower_bound = std::min(result.upper_bound, frontier_bound());
			if (result.upper_bound - result.lower_bound <= gap() || limited()) break;
			const bool diving = !dfs && (dive.has_value() || (options.dive_interval && result.nodes % options.dive_interval == 0));
			Node node;
			if (dive) { node = std::move(*dive); dive.reset(); }
			else { node = queue.top(); queue.pop(); }
			++result.nodes;
			result.sequence_depth_sum += node.sequence.size();
			++result.sequence_depth_samples;
			result.max_sequence_depth = std::max(result.max_sequence_depth, node.sequence.size());
            strengthen_shared_bound(node);
			if (node.path.empty() && node.bound < result.upper_bound-gap() && !solve(node)) { queue.push(std::move(node));break; }
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
                queue.restart();
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
                queue.restart();
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
            std::vector<std::pair<double,size_t>> branch_candidates;
            std::vector<double> branch_distances(options.cycle_learned_branching?n:0);
			size_t detour_chosen = none;
			double best_detour = -1, best_detour_distance = -1;
			const auto visit_began = std::chrono::steady_clock::now();
			for (size_t j = 0; j < n; ++j) {
				const double distance = contact(node.path, polygons[j], eps).distance;
                if(options.cycle_learned_branching)branch_distances[j]=distance;
				if (distance > farthest) { farthest = distance; chosen = j; }
                if(cycle&&options.cycle_strong_branching&&distance>eps&&
                   std::none_of(node.sequence.begin(),node.sequence.end(),[&](auto e){return e.polygon==j;}))
                    branch_candidates.emplace_back(distance,j);
				if (options.detour_root && node.sequence.empty() && distance > eps) {
					const auto point = best_contact(start, target, hulls[j], hulls[j].front());
					const double detour = start.distance_to(point) + point.distance_to(target) - result.initial_lower_bound;
					if (detour > best_detour + 1e-12 || (std::abs(detour - best_detour) <= 1e-12 && distance > best_detour_distance)) {
						best_detour = detour;
						best_detour_distance = distance;
						detour_chosen = j;
					}
				}
			}
			if (options.detour_root && node.sequence.empty() && chosen != none) {
				chosen = detour_chosen;
			} else if (options.endpoint_sum_root && node.sequence.empty() && chosen != none) {
				const Polygon start_point_path{start, start}, target_point_path{target, target};
				double endpoint_sum = -1;
				for (size_t j = 0; j < n; ++j) {
					const double score = contact(start_point_path, polygons[j], 0).distance
						+ contact(target_point_path, polygons[j], 0).distance;
					if (score > endpoint_sum) { endpoint_sum = score; chosen = j; }
				}
			}
            if(cycle&&options.cycle_strong_branching&&!options.cycle_learned_branching&&chosen!=none&&
               std::none_of(node.sequence.begin(),node.sequence.end(),[&](auto e){return e.polygon==chosen;})) {
                std::sort(branch_candidates.begin(),branch_candidates.end(),std::greater<>());
                std::vector<const Polygon*> regions;
                for(auto e:node.sequence)regions.push_back(e.piece==none?&hulls[e.polygon]:&pieces[e.polygon][e.piece]);
                double strongest=-1;
                for(size_t i=0;i<std::min(size_t(3),branch_candidates.size());++i) {
                    const size_t candidate=branch_candidates[i].second;
                    const auto bounds=insertion_lower_bounds(node.path,regions,hulls[candidate],true,node.dual);
                    const double bound=*std::min_element(bounds.begin(),bounds.end());
                    if(bound>strongest){strongest=bound;chosen=candidate;}
                }
            }

            if(cycle&&options.cycle_learned_branching&&chosen!=none) {
                const size_t geometric=chosen;double best_score=-1;
                for(size_t j=0;j<n;++j)if(branch_distances[j]>eps) {
                    const size_t kind=std::any_of(node.sequence.begin(),node.sequence.end(),[&](auto e){return e.polygon==j;});
                    const auto &history=learned[j][kind];
                    // Four prior observations regularize sparse history. This
                    // ranks complete branches; it is never a pruning bound.
                    const double weight=double(history.count)/(double(history.count)+4);
                    const double factor=1+weight*history.mean+(1-weight)*prior[kind].mean;
                    const double score=branch_distances[j]*factor;
                    if(score>best_score){best_score=score;chosen=j;}
                }
                ++result.learned_branch_decisions;result.learned_branch_changes+=chosen!=geometric;
            }
			result.search_visit_check_seconds += duration(visit_began);
			if (chosen == none) {
                queue.restart();
				if (!node.refined && !limited()) {
					if(!solve(node, true)) { queue.push(std::move(node));break; }
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
					prepare_pieces(chosen);
					result.decomposition_seconds += duration(decomposition_began);
				}
				const size_t position = found - node.sequence.begin();
                std::vector<double> replacement_bounds;
                if(cycle&&options.cycle_dual_screen) {
                    const auto began_bound=std::chrono::steady_clock::now();
                    std::vector<const Polygon *> regions;
                    for(auto e:node.sequence)regions.push_back(e.piece==none?&hulls[e.polygon]:&pieces[e.polygon][e.piece]);
                    replacement_bounds=cycle_replacement_lower_bounds(node.path,regions,pieces[chosen],position,node.dual);
                    result.cycle_dual_screen_children+=replacement_bounds.size();
                    result.cycle_dual_screen_seconds+=duration(began_bound);
                }
				result.total_branching += pieces[chosen].size();
				result.max_observed_branching = std::max(result.max_observed_branching, pieces[chosen].size());
				for (size_t j = 0; j < pieces[chosen].size(); ++j) {
					auto sequence = node.sequence;
					sequence[position].piece = j;
					Node child{std::move(sequence), {}, node.bound, serial++};
                    if(!replacement_bounds.empty()) {
                        child.bound=std::max(child.bound,replacement_bounds[j]);
                        result.cycle_dual_screen_prunes+=node.bound<result.upper_bound-gap()&&child.bound>=result.upper_bound-gap();
                    }
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
				const auto bounds = insertion_lower_bounds(node.path, regions, hulls[chosen], cycle, node.dual);
				// At size two the two insertion positions are reversals of the
				// same unoriented triangle. Later, all cyclic gaps are needed.
				const size_t branching = cycle ? (node.sequence.size()==2?1:node.sequence.size()) : node.sequence.size()+1;
				result.total_branching += branching;
				result.max_observed_branching = std::max(result.max_observed_branching, branching);
				for (size_t slot = 0; slot < branching; ++slot) {
					const size_t j=slot+size_t(cycle);
					++result.insertion_positions_considered;
					++result.children_generated;
					const double bound = std::max(node.bound, bounds[slot]);
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
            if(cycle)for(auto &child:children) {
                if(child.branch_piece!=none)strengthen_one_tree(child);
                if(options.cycle_learned_branching) {
                    child.learning_parent_bound=node.bound;child.learning_distance=branch_distances[chosen];
                    child.learning_pending=true;
                }
            }
			if (cycle) for (auto &child : children) {
				const size_t m=node.sequence.size(), position=child.branch_position;
				child.warm_start.assign(node.path.begin(),node.path.end()-1);
				const bool inserting=child.branch_piece==none;
				const auto &region=inserting?hulls[chosen]:pieces[chosen][child.branch_piece];
				const auto before=node.path[(position+m-1)%m];
				const auto after=node.path[(position+size_t(!inserting))%m];
				const auto contact=best_contact(before,after,region,region.front());
                if(options.cycle_active_features&&node.active_features.size()==m) {
                    child.active_features=node.active_features;
                    if(inserting)child.active_features.insert(child.active_features.begin()+position,-2);
                    else child.active_features[position]=-2;
                }
                if(options.cycle_dual_reuse&&node.dual.size()==m) {
                    child.dual=node.dual;
                    if(inserting)child.dual.insert(child.dual.begin()+position,node.dual[(position+m-1)%m]);
                }
				if (inserting) child.warm_start.insert(child.warm_start.begin()+position,contact);
				else child.warm_start[position]=contact;
			}
			const size_t first_child=queue.size();
			for (size_t batch_begin = 0; batch_begin < children.size();) {
				import_incumbent();
				const size_t batch_end = std::min(children.size(), batch_begin + options.threads);
				const double batch_upper_bound = result.upper_bound;
				const double batch_cutoff = batch_upper_bound - gap_at(batch_upper_bound);
				std::vector<size_t> evaluation_children;
				std::vector<size_t> evaluation_slot(batch_end - batch_begin, none);
				const size_t available_calls = result.calls < options.max_calls
					? options.max_calls - result.calls : 0;
				for (size_t child_index = batch_begin; child_index < batch_end; ++child_index) {
					if ((cycle&&options.cycle_lazy) || evaluation_children.size() >= available_calls || limited()) break;
                    strengthen_shared_bound(children[child_index]);
					if (children[child_index].bound >= batch_cutoff) continue;
					if(!note_oracle_call(children[child_index], false)) break;
					evaluation_slot[child_index - batch_begin] = evaluation_children.size();
					evaluation_children.push_back(child_index);
				}

				std::vector<RelaxationResult> certified(evaluation_children.size());
				if (evaluation_children.size() == 1) {
					const auto oracle_began = std::chrono::steady_clock::now();
					certified[0] = evaluate_oracle(children[evaluation_children[0]], false,
						batch_upper_bound, workspace, cycle_workspace,memo_workspace);
					result.convex_oracle_wall_seconds += duration(oracle_began);
				} else if (evaluation_children.size() > 1) {
					const int worker_count = static_cast<int>(std::min(options.threads, evaluation_children.size()));
					result.parallel_oracle_batches++;
					result.parallel_oracle_calls += evaluation_children.size();
					if (parallel_workspaces.size() < static_cast<size_t>(worker_count))
						{ parallel_workspaces.resize(worker_count);parallel_cycle_workspaces.resize(worker_count);parallel_memo_workspaces.resize(worker_count); }
					const auto oracle_batch_began = std::chrono::steady_clock::now();
					evaluate_parallel_oracles(static_cast<std::ptrdiff_t>(evaluation_children.size()), worker_count,
						[&](std::ptrdiff_t slot, int worker) {
							certified[slot] = evaluate_oracle(children[evaluation_children[slot]], false,
								batch_upper_bound, parallel_workspaces[worker], parallel_cycle_workspaces[worker],parallel_memo_workspaces[worker]);
						});
					result.convex_oracle_wall_seconds += duration(oracle_batch_began);
				}

				for (size_t child_index = batch_begin; child_index < batch_end; ++child_index) {
					auto &child = children[child_index];
					const bool pruned_by_new_incumbent = child.bound >= result.upper_bound - gap();
					if (pruned_by_new_incumbent) ++result.sibling_bound_prunes;
					const size_t slot = evaluation_slot[child_index - batch_begin];
					if (slot != none) {
						record_oracle_result(child, false, batch_cutoff, certified[slot]);
						if (!pruned_by_new_incumbent) {
							std::vector<size_t> child_sequence;
							for (auto e : child.sequence) child_sequence.push_back(e.polygon);
							improve(child.path, "oracle", child_sequence);
						}
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
				batch_begin = batch_end;
			}
            queue.finish_branch(first_child);
		}
		result.search_seconds = duration(search_began);
		result.search_maintenance_seconds = std::max(0.0, result.search_seconds - result.convex_oracle_wall_seconds
			- result.decomposition_seconds - result.search_visit_check_seconds);
		import_incumbent();
		phase = Phase::Finalization;
		const auto finalization_began = std::chrono::steady_clock::now();
		result.lower_bound = std::min({result.upper_bound, settled_bound, frontier_bound()});
		result.lower_bound = std::max(start.distance_to(target), result.lower_bound - normalization_error);
		result.final_absolute_gap = std::max(0.0, result.upper_bound - result.lower_bound);
		result.final_relative_gap = result.final_absolute_gap / std::max(std::abs(result.upper_bound), 1e-30);
		result.final_length = result.upper_bound;
		result.exact = result.upper_bound - result.lower_bound <= gap();
		result.termination = result.exact ? UnorderedTppTermination::Optimal
			: control && control->proved() ? UnorderedTppTermination::PortfolioStopped
            : (control ? control->calls.load(std::memory_order_relaxed)>=control->max_calls : result.calls >= options.max_calls) ? UnorderedTppTermination::CallLimit
			: (elapsed() >= options.max_seconds || (control && control->elapsed()>=control->max_seconds)) ? UnorderedTppTermination::TimeLimit
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

	static UnorderedTppSolveResult solve_unordered(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &input,
		const UnorderedTppSolveOptions &options, bool cycle, PortfolioControl *control = nullptr
	) {
		const auto began = std::chrono::steady_clock::now();
		auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count(); };
		if (!start.is_finite() || !target.is_finite() || std::isnan(options.max_seconds) || options.max_seconds < 0
			|| !std::isfinite(options.absolute_gap) || options.absolute_gap < 0
			|| !std::isfinite(options.relative_gap) || options.relative_gap < 0
			|| !std::isfinite(options.oracle_relative_gap) || options.oracle_relative_gap < 0
			|| !std::isfinite(options.feasibility_tolerance) || options.feasibility_tolerance <= 0
			|| options.threads == 0 || options.threads > static_cast<size_t>(std::numeric_limits<int>::max()))
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
				|| (cycle ? path.front().distance_to(path.back()) > options.feasibility_tolerance
					: path.front().distance_to(start) > options.feasibility_tolerance || path.back().distance_to(target) > options.feasibility_tolerance))
				throw std::invalid_argument("Initial path needs finite points and matching endpoints.");
			auto snapped = path;
			if (cycle) snapped.back()=snapped.front();
			else {snapped.front() = start; snapped.back() = target;}
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
			if (cycle) path.back()=path.front();
			else {path.front() = start; path.back() = target;}
			for (auto &point : path) point = normalize(point);
		}
		normalized_options.absolute_gap /= divisor;
		const double numerical_floor = 64 * std::numeric_limits<double>::epsilon();
		normalized_options.feasibility_tolerance = std::max(options.feasibility_tolerance / divisor, numerical_floor);
		normalized_options.max_seconds = std::max(0.0, options.max_seconds - elapsed());
		const double normalization_seconds = elapsed();
		auto result = solve_normalized_unordered_tpp(
			normalize(start), normalize(target), polygons, normalized_options, cycle, control
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


    static UnorderedTppSolveResult solve_portfolio(const Vector2 &start,const Vector2 &target,
            const std::vector<Polygon> &polygons,const UnorderedTppSolveOptions &options,bool cycle) {
#ifdef __EMSCRIPTEN__
        throw std::invalid_argument("Cooperative portfolio requires a native threaded build.");
#else
        if(std::isnan(options.max_seconds) || options.max_seconds<0) throw std::invalid_argument("Invalid portfolio time limit.");
        if(options.threads!=1) throw std::invalid_argument("Portfolio uses two single-thread searches; leave threads at 1.");
        PortfolioControl control(options.max_calls,options.max_seconds,options.portfolio_share_incumbents);
        std::array<UnorderedTppSolveResult,2> runs;
        std::array<std::exception_ptr,2> errors;
        auto worker=[&](size_t index) {
            try {
                auto local=options;
                local.portfolio=false;
                local.search_strategy=index==0?UnorderedSearchStrategy::BestBoundDive:UnorderedSearchStrategy::DfsBfs;
                if(index==1 && !cycle) local.endpoint_sum_root=true;
                local.max_seconds=std::max(0.0,options.max_seconds-control.elapsed());
                runs[index]=solve_unordered(start,target,polygons,local,cycle,&control);
                if(runs[index].exact) control.finish_proof(index);
            } catch(...) { errors[index]=std::current_exception(); }
        };
        // Join even when an exception occurs; no detached worker can outlive
        // its input, incumbent, or frontier. Stop is cooperative between calls.
        std::jthread first([&]{worker(0);});
        std::jthread second([&]{worker(1);});
        first.join();second.join();
        if(errors[0] && errors[1]) std::rethrow_exception(errors[0]);
        const size_t selected=errors[0]?1:errors[1]?0:runs[1].upper_bound<runs[0].upper_bound?1:0;
        auto result=runs[selected];
        result.portfolio_workers=2;
        result.portfolio_winner=control.winner.load(std::memory_order_relaxed);
        result.portfolio_proof_seconds=control.proof_seconds;
        result.seconds=control.elapsed();
        result.portfolio_join_seconds=control.proved()?std::max(0.0,result.seconds-control.proof_seconds):0;
        result.portfolio_incumbent_publications=control.publications.load(std::memory_order_relaxed);
        result.threads=2;
        result.lower_bound=0;
        for(size_t i=0;i<2;++i) {
            UnorderedPortfolioRun stats;
            stats.strategy=i==0?"best-bound":"dfs-bfs";
            stats.calls=runs[i].calls;stats.nodes=runs[i].nodes;
            stats.incumbent_imports=runs[i].portfolio_incumbent_imports;
            stats.seconds=runs[i].seconds;stats.termination=runs[i].termination;
            stats.lower_bound=runs[i].lower_bound;stats.upper_bound=runs[i].upper_bound;
            if(errors[i]) {
                try { std::rethrow_exception(errors[i]); }
                catch(const std::exception &e) { stats.error=e.what(); }
                catch(...) { stats.error="Unknown worker failure"; }
            } else result.lower_bound=std::max(result.lower_bound,runs[i].lower_bound);
            result.portfolio_runs.push_back(std::move(stats));
        }
        // Both frontiers cover the full problem, so max(LB) is valid. Choose
        // min(UB) together with its validated path; never mix a value and tour.
        result.lower_bound=std::min(result.lower_bound,result.upper_bound);
        result.final_length=result.upper_bound;
        result.final_absolute_gap=std::max(0.0,result.upper_bound-result.lower_bound);
        result.final_relative_gap=result.final_absolute_gap/std::max(std::abs(result.upper_bound),1e-30);
        result.exact=result.final_absolute_gap<=options.absolute_gap+options.relative_gap*std::abs(result.upper_bound);
        result.termination=result.exact?UnorderedTppTermination::Optimal
            : control.calls.load(std::memory_order_relaxed)>=options.max_calls?UnorderedTppTermination::CallLimit
            : result.seconds>=options.max_seconds?UnorderedTppTermination::TimeLimit:UnorderedTppTermination::NumericalLimit;
        auto sum=[&](auto member){result.*member=runs[0].*member+runs[1].*member;};
        sum(&UnorderedTppSolveResult::calls);
        sum(&UnorderedTppSolveResult::nodes);
        sum(&UnorderedTppSolveResult::parallel_oracle_calls);
        sum(&UnorderedTppSolveResult::parallel_oracle_batches);
        sum(&UnorderedTppSolveResult::relaxation_calls);
        sum(&UnorderedTppSolveResult::refinement_calls);
        sum(&UnorderedTppSolveResult::complete_order_oracle_calls);
        sum(&UnorderedTppSolveResult::complete_piece_oracle_calls);
        sum(&UnorderedTppSolveResult::oracle_cutoff_calls);
        sum(&UnorderedTppSolveResult::oracle_dual_cutoff_prunes);
        sum(&UnorderedTppSolveResult::screened_nodes);
        sum(&UnorderedTppSolveResult::one_tree_calls);
        sum(&UnorderedTppSolveResult::one_tree_cache_hits);
        sum(&UnorderedTppSolveResult::one_tree_iterations);
        sum(&UnorderedTppSolveResult::one_tree_distance_queries);
        sum(&UnorderedTppSolveResult::one_tree_improvements);
        sum(&UnorderedTppSolveResult::one_tree_seconds);
        sum(&UnorderedTppSolveResult::learned_branch_observations);
        sum(&UnorderedTppSolveResult::learned_branch_decisions);
        sum(&UnorderedTppSolveResult::learned_branch_changes);
        sum(&UnorderedTppSolveResult::cycle_memo_queries);
        sum(&UnorderedTppSolveResult::cycle_certificate_interval_uses);
        sum(&UnorderedTppSolveResult::cycle_dual_screen_children);
        sum(&UnorderedTppSolveResult::cycle_dual_screen_prunes);
        sum(&UnorderedTppSolveResult::cycle_dual_screen_seconds);
        sum(&UnorderedTppSolveResult::cycle_shared_bound_queries);
        sum(&UnorderedTppSolveResult::cycle_shared_bound_hits);
        sum(&UnorderedTppSolveResult::cycle_shared_bound_improvements);
        sum(&UnorderedTppSolveResult::cycle_shared_bound_prunes);
        sum(&UnorderedTppSolveResult::cycle_shared_bound_seconds);
        sum(&UnorderedTppSolveResult::cycle_memo_repeated);
        sum(&UnorderedTppSolveResult::cycle_memo_hits);
        sum(&UnorderedTppSolveResult::cycle_certificate_cutoff_skips);
        sum(&UnorderedTppSolveResult::cycle_initial_contact_checks);
        sum(&UnorderedTppSolveResult::cycle_initial_contact_accepts);
        sum(&UnorderedTppSolveResult::sibling_bound_prunes);
        sum(&UnorderedTppSolveResult::partial_states_created);
        sum(&UnorderedTppSolveResult::children_generated);
        sum(&UnorderedTppSolveResult::children_queued);
        sum(&UnorderedTppSolveResult::pruned_nodes);
        sum(&UnorderedTppSolveResult::pruned_states);
        sum(&UnorderedTppSolveResult::bound_prunes);
        sum(&UnorderedTppSolveResult::incumbent_prunes);
        sum(&UnorderedTppSolveResult::insertion_positions_considered);
        sum(&UnorderedTppSolveResult::insertion_positions_pruned);
        sum(&UnorderedTppSolveResult::branch_events);
        sum(&UnorderedTppSolveResult::total_branching);
        sum(&UnorderedTppSolveResult::sequence_depth_sum);
        sum(&UnorderedTppSolveResult::sequence_depth_samples);
        sum(&UnorderedTppSolveResult::incumbent_updates);
        sum(&UnorderedTppSolveResult::best_updates);
        sum(&UnorderedTppSolveResult::decomposed_polygons);
        sum(&UnorderedTppSolveResult::convex_pieces_generated);
        sum(&UnorderedTppSolveResult::fallback_calls);
        sum(&UnorderedTppSolveResult::fallback_geometric_path_invalid_calls);
        sum(&UnorderedTppSolveResult::fallback_certificate_gap_calls);
        sum(&UnorderedTppSolveResult::fallback_locator_exception_calls);
        sum(&UnorderedTppSolveResult::fallback_nonfinite_calls);
        sum(&UnorderedTppSolveResult::fallback_contact_construction_calls);
        sum(&UnorderedTppSolveResult::fallback_membership_ordering_calls);
        sum(&UnorderedTppSolveResult::fallback_local_optimality_calls);
        sum(&UnorderedTppSolveResult::fallback_coincident_contact_calls);
        sum(&UnorderedTppSolveResult::predicate_exact_evaluations);
        sum(&UnorderedTppSolveResult::extended_precision_calls);
        sum(&UnorderedTppSolveResult::oracle_time_limit_calls);
        sum(&UnorderedTppSolveResult::repaired_geometric_path_calls);
        sum(&UnorderedTppSolveResult::insertion_branches);
        sum(&UnorderedTppSolveResult::decomposition_branches);
        sum(&UnorderedTppSolveResult::preprocessing_seconds);
        sum(&UnorderedTppSolveResult::initial_heuristic_seconds);
        sum(&UnorderedTppSolveResult::initial_sampled_extra_points);
        sum(&UnorderedTppSolveResult::initial_sampling_work_budget);
        sum(&UnorderedTppSolveResult::initial_convex_refinement_calls);
        sum(&UnorderedTppSolveResult::initial_convex_refinement_seconds);
        sum(&UnorderedTppSolveResult::search_seconds);
        sum(&UnorderedTppSolveResult::finalization_seconds);
        sum(&UnorderedTppSolveResult::convex_oracle_seconds);
        sum(&UnorderedTppSolveResult::convex_oracle_wall_seconds);
        sum(&UnorderedTppSolveResult::convex_geometric_solver_seconds);
        sum(&UnorderedTppSolveResult::convex_certificate_verification_seconds);
        sum(&UnorderedTppSolveResult::convex_contact_materialization_seconds);
        sum(&UnorderedTppSolveResult::convex_fallback_seconds);
        sum(&UnorderedTppSolveResult::convex_fallback_long_double_seconds);
        sum(&UnorderedTppSolveResult::convex_fallback_extended_precision_seconds);
        sum(&UnorderedTppSolveResult::decomposition_seconds);
        sum(&UnorderedTppSolveResult::visit_check_seconds);
        sum(&UnorderedTppSolveResult::heuristic_visit_check_seconds);
        sum(&UnorderedTppSolveResult::search_visit_check_seconds);
        sum(&UnorderedTppSolveResult::finalization_visit_check_seconds);
        sum(&UnorderedTppSolveResult::search_maintenance_seconds);
        sum(&UnorderedTppSolveResult::portfolio_incumbent_imports);
        result.max_observed_branching=std::max(runs[0].max_observed_branching,runs[1].max_observed_branching);
        result.max_sequence_depth=std::max(runs[0].max_sequence_depth,runs[1].max_sequence_depth);
        result.convex_pieces_max=std::max(runs[0].convex_pieces_max,runs[1].convex_pieces_max);
        result.peak_queue=runs[0].peak_queue+runs[1].peak_queue; // Conservative combined peak.
        result.convex_pieces_min=std::min(runs[0].convex_pieces_min,runs[1].convex_pieces_min);
        result.calls=control.calls.load(std::memory_order_relaxed); // Includes a failed in-flight call.
        if(options.trace) result.trace.push_back({.kind="portfolio_complete",.path=result.path,
            .lower_bound=result.lower_bound,.upper_bound=result.upper_bound,
            .reason=result.exact?"optimal":"incomplete"});
        return result;
#endif
    }

	UnorderedTppSolveResult tpp_nonconvex_unordered_solve(const Vector2 &start, const Vector2 &target,
		const std::vector<Polygon> &polygons, const UnorderedTppSolveOptions &options) {
		return options.portfolio?solve_portfolio(start,target,polygons,options,false)
            :solve_unordered(start,target,polygons,options,false);
	}
	UnorderedTppSolveResult tpp_nonconvex_tspn_solve(const std::vector<Polygon> &polygons,
		const UnorderedTppSolveOptions &options) {
		if (!polygons.empty() && polygons.front().empty()) throw std::invalid_argument("Empty polygon.");
		const Vector2 seed=polygons.empty()?Vector2{}:polygons.front().front();
		return options.portfolio?solve_portfolio(seed,seed,polygons,options,true)
            :solve_unordered(seed,seed,polygons,options,true);
	}
}
