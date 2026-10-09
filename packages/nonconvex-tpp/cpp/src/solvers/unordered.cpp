#include "tpp/nonconvex/unordered.h"
#include "tpp/convex/certified.h"
#include "tpp/convex/dual.h"
#include "tpp/convex/cycle.h"
#include "tpp/nonconvex/decomposition.h"
#include "unordered_geometry.h"
#include "unordered_bounds.h"
#include "unordered_portfolio.h"
#include "unordered_oracle_capture.h"
#include "unordered_cycle_oracle.h"
#include "unordered_sequence.h"

#include <algorithm>
#include <random>
#include <array>
#include <bit>
#include <cstdint>
#include <set>
#include <thread>
#include <chrono>
#include <cmath>
#include <exception>
#include <optional>
#include <memory>
#include <omp.h>
#include <queue>
#include <stdexcept>

namespace {
	using namespace tpp::unordered_detail;
	constexpr size_t none = std::numeric_limits<size_t>::max();

	using Element = SequenceElement;
	struct Node {
		NodeSequence sequence;
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
        tpp::UnorderedSequenceStorage storage;
        size_t index_bytes;
        // Must outlive every queued sequence reference (reverse member order).
        std::unique_ptr<SequenceHistory> history;
        std::vector<Node> heap;
        std::vector<Node> stack;
        std::set<std::pair<double,size_t>> bounds;
        size_t sequence_bytes = 0, dive_bytes = 0;
        void note_storage() {
            peak_sequence_bytes=std::max(peak_sequence_bytes,sequence_bytes+dive_bytes+(history?history->reserved_bytes():0));
            peak_node_bytes=std::max(peak_node_bytes,(dfs?stack.capacity():heap.capacity())*sizeof(Node));
        }
        static bool descending(const Node &a,const Node &b) {
            return std::tie(a.bound,a.relaxed_length,a.serial)>std::tie(b.bound,b.relaxed_length,b.serial);
        }
    public:
        size_t peak_sequence_bytes = 0, peak_node_bytes = 0;
        Frontier(bool use_dfs,tpp::UnorderedSequenceStorage mode,size_t bytes,size_t polygons)
            :dfs(use_dfs),storage(mode),index_bytes(bytes) {
            if(storage==tpp::UnorderedSequenceStorage::Deltas)history=std::make_unique<SequenceHistory>(bytes,polygons);
        }
        bool empty() const { return dfs?stack.empty():heap.empty(); }
        size_t size() const { return dfs?stack.size():heap.size(); }
        void note_dive(size_t bytes) { dive_bytes=bytes;note_storage(); }
        double lower_bound() const { return dfs?bounds.begin()->first:heap.front().bound; }
        void freeze_child(Node &node,const SequenceReference &parent) {
            if(history)node.sequence.save(history->child(parent,node.branch_polygon,node.branch_piece,node.branch_position));
            else if(storage==tpp::UnorderedSequenceStorage::Packed)node.sequence.pack(index_bytes);
            note_storage();
        }
        void push(Node node,const SequenceReference &parent={}) {
            if(!node.sequence.stored()) {
                if(history)node.sequence.save(parent?parent:history->snapshot(node.sequence.elements()));
                else if(storage==tpp::UnorderedSequenceStorage::Packed)node.sequence.pack(index_bytes);
            }
            sequence_bytes+=node.sequence.payload_bytes();
            if(dfs) { bounds.emplace(node.bound,node.serial);stack.push_back(std::move(node)); }
            else { heap.push_back(std::move(node));std::push_heap(heap.begin(),heap.end(),Later{}); }
            note_storage();
        }
        Node take() {
            auto &nodes=dfs?stack:heap;
            if(dfs)bounds.erase({stack.back().bound,stack.back().serial});
            else std::pop_heap(heap.begin(),heap.end(),Later{});
            sequence_bytes-=nodes.back().sequence.payload_bytes();
            Node node=std::move(nodes.back());nodes.pop_back();return node;
        }
        void report(tpp::UnorderedTppSolveResult &result) {
            note_storage();
            result.peak_sequence_storage_bytes=peak_sequence_bytes;
            result.peak_frontier_node_bytes=peak_node_bytes;
            if(history) {
                result.sequence_history_record_bytes=history->record_bytes();
                result.peak_sequence_records=history->peak_live_records;
                result.sequence_reconstructions=history->reconstructions;
            }
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
    // Adapter only: both topologies use the same search below. No second B&B.
	RelaxationResult solve_relaxation(bool cycle, const Vector2 &start,
		const Vector2 &target, const std::vector<Polygon> &regions,
		DynamicConvexTppWorkspace &workspace, double tolerance, double cutoff, double seconds,
		const Polygon &initial_contacts, ConvexCycleWorkspace *cycle_workspace,
		const std::vector<int> &initial_features, bool retain_features, bool bound_first,
		bool interval_certificate, const std::function<bool()> &stop_requested, bool proposal_bound, bool float_oracle) {
		if (!cycle) return tpp_convex_solve_certified(start,target,regions,workspace,tolerance,cutoff,seconds);
		const auto began=std::chrono::steady_clock::now();
		RelaxationResult out;
		if (regions.size()<2) {
			// The relaxed cycle is one point: zero, with no arithmetic stage.
			const auto point=regions.empty()?start:regions.front().front();
			out.path={point,point};out.used_interval_bounds=true;return out;
		}
		ConvexCycleDoubleOptions cycle_options;
		cycle_options.lower_bound_cutoff=cutoff;
        cycle_options.max_seconds=seconds;
        cycle_options.stop_requested=stop_requested;
		cycle_options.initial_contacts=initial_contacts;
        cycle_options.workspace=cycle_workspace;
        cycle_options.initial_features=initial_features;
        cycle_options.retain_active_features=retain_features;
        cycle_options.bound_first=bound_first;
        cycle_options.interval_certificate=interval_certificate;
		cycle_options.proposal_only=proposal_bound;
		// A positive tolerance first tries the binary64 stage; GapClosed and its
		// CertifiedBound are proved on these regions without rational arithmetic.
		cycle_options.max_gap=float_oracle&&tolerance>0?tolerance:0;
		auto solved=tpp_convex_solve_cycle_double(regions,cycle_options);
		if (solved.status!=ConvexCycleStatus::Optimal && solved.status!=ConvexCycleStatus::FloatingPointLimit
			&& solved.status!=ConvexCycleStatus::CertifiedBound && solved.status!=ConvexCycleStatus::Interrupted
            && solved.status!=ConvexCycleStatus::ProposalLimit && solved.status!=ConvexCycleStatus::GapClosed)
			throw std::runtime_error("Convex cycle oracle failed: "+solved.diagnostic);
		out.used_interval_bounds=solved.float_interval_closed;
		out.used_float_oracle=solved.float_polish_closed;
		out.polish_attempted=solved.float_polish_attempted;
		out.polish_newton_iterations=solved.float_newton_iterations;
		out.active_features=std::move(solved.active_features);
        out.path=std::move(solved.contacts);if(!out.path.empty())out.path.push_back(out.path.front());
        out.time_limited=solved.status==ConvexCycleStatus::Interrupted;
        out.cycle_timings=solved.timings;
		out.lower_bound=solved.certificate.lower_bound;
        out.upper_bound=out.path.empty()?INFINITY:solved.certificate.upper_bound;
		out.dual_cutoff_pruned=solved.status==ConvexCycleStatus::CertifiedBound;
		out.predicate_exact_evaluations=solved.certificate.exact_predicate_evaluations;
		out.used_fallback=solved.rational_cycle_recoveries+solved.rational_anchor_recoveries+solved.rational_feature_recoveries>0;
		// Every call that the binary64 stage did not close reached the exact
		// path (exact certificates at least) or was interrupted before closing.
		out.used_rational=!out.used_interval_bounds&&!out.used_float_oracle;
        out.certificate_cutoff_skips=solved.certificate_cutoff_skips;
        out.certificate_interval_uses=solved.certificate_interval_uses;
        out.initial_contact_checks=solved.initial_contact_checks;out.initial_contact_accepts=solved.initial_contact_accepts;
		out.proposal_calls=proposal_bound;
        if(proposal_bound && solved.status==ConvexCycleStatus::ProposalLimit &&
           (out.path.empty() || (out.lower_bound<cutoff && out.upper_bound-out.lower_bound>tolerance))) {
            // The proposal is only a source of certified bounds. An insufficient
            // interval requests the unchanged full solve, with the shared deadline.
            Polygon warm=out.path;
            if(!warm.empty())warm.pop_back();else warm=initial_contacts;
            auto full=solve_relaxation(cycle,start,target,regions,workspace,tolerance,cutoff,
                std::max(0.0,seconds-std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()),
                warm,cycle_workspace,out.active_features,retain_features,bound_first,interval_certificate,stop_requested,false,float_oracle);
            if(!out.path.empty() && (full.path.empty() || out.upper_bound<full.upper_bound)) {
                full.path=std::move(out.path);full.upper_bound=out.upper_bound;
                full.active_features=std::move(out.active_features);
            }
            full.lower_bound=std::max(full.lower_bound,out.lower_bound);
            full.predicate_exact_evaluations+=out.predicate_exact_evaluations;
            full.certificate_cutoff_skips+=out.certificate_cutoff_skips;
            full.certificate_interval_uses+=out.certificate_interval_uses;
            full.initial_contact_checks+=out.initial_contact_checks;full.initial_contact_accepts+=out.initial_contact_accepts;
            full.cycle_timings.construction_seconds+=out.cycle_timings.construction_seconds;
            full.cycle_timings.certification_seconds+=out.cycle_timings.certification_seconds;
            full.cycle_timings.rational_recovery_seconds+=out.cycle_timings.rational_recovery_seconds;
            full.cycle_timings.interval_proof_seconds+=out.cycle_timings.interval_proof_seconds;
            full.cycle_timings.polish_seconds+=out.cycle_timings.polish_seconds;
            full.polish_attempted|=out.polish_attempted;full.polish_newton_iterations+=out.polish_newton_iterations;
            full.proposal_calls=1;
            full.geometric_solver_seconds=full.cycle_timings.construction_seconds;
            full.certificate_verification_seconds=full.cycle_timings.certification_seconds;
            full.fallback_seconds=full.cycle_timings.rational_recovery_seconds;
            full.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
            return full;
        }
		if (!out.time_limited && out.lower_bound<cutoff && out.upper_bound-out.lower_bound>tolerance) {
			// A rounded optimum may have a weak contact-derived dual. Recover its
			// global bound while retaining the independently feasible double path.
			ConvexCycleOptions exact_options;exact_options.lower_bound_cutoff=cutoff;
            exact_options.bound_first=bound_first;
            exact_options.max_seconds=std::max(0.0,seconds-std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count());
            exact_options.stop_requested=stop_requested;
			for(size_t i=0;i<regions.size();++i)exact_options.initial_contacts.emplace_back(out.path[i]);
			const auto exact=tpp_convex_solve_cycle(regions,exact_options);
			if (exact.status!=ConvexCycleStatus::Optimal&&exact.status!=ConvexCycleStatus::CertifiedBound&&exact.status!=ConvexCycleStatus::Interrupted)
				throw std::runtime_error("Exact cycle refinement failed: "+exact.diagnostic);
			out.lower_bound=std::max(out.lower_bound,exact.certificate.lower_bound);
			if(exact.status==ConvexCycleStatus::Optimal&&exact.contacts.size()==regions.size()) {
				// Rounded double contacts can stay far from the optimum, notably on
				// points and segments, which have no interior to round into. The
				// exact optimum's rounded contacts are then a better path; like any
				// relaxation path it is only used through the B&B's own visit checks.
				Polygon rounded;rounded.reserve(regions.size()+1);
				for(const auto &q:exact.contacts)rounded.push_back(q.external());
				rounded.push_back(rounded.front());
				const double length=path_length(rounded);
				if(std::isfinite(length)&&(out.path.empty()||length<path_length(out.path))) {
					// The certified bound is the exact optimum's outward-rounded
					// length; the rounded contacts only represent that cycle.
					out.path=std::move(rounded);out.upper_bound=std::max(exact.certificate.upper_bound,out.lower_bound);
				}
			}
            out.time_limited=exact.status==ConvexCycleStatus::Interrupted;
            out.cycle_timings.certification_seconds+=exact.timings.certification_seconds;
            out.cycle_timings.rational_recovery_seconds+=exact.timings.construction_seconds+exact.timings.rational_recovery_seconds;
			out.dual_cutoff_pruned=exact.status==ConvexCycleStatus::CertifiedBound;
			out.used_fallback=true;
            out.certificate_cutoff_skips+=exact.certificate_cutoff_skips;
		}
		out.fallback_certificate_gap=out.used_fallback;
        out.proposal_accepts=proposal_bound&&!out.time_limited&&!out.used_fallback;
		out.fallback_reason=out.used_fallback?ConvexFallbackReason::LocalOptimality:ConvexFallbackReason::None;
		out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
		out.geometric_solver_seconds=out.cycle_timings.construction_seconds;
        out.certificate_verification_seconds=out.cycle_timings.certification_seconds;
        out.fallback_seconds=out.cycle_timings.rational_recovery_seconds;
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
        result.sequence_storage = options.sequence_storage;
        size_t input_max_vertices = 0;
        for (const auto &polygon : input) input_max_vertices = std::max(input_max_vertices,polygon.size());
        const size_t index_bytes = sequence_index_bytes(input.size(),input_max_vertices);
        result.node_index_bits = 8*(options.sequence_storage==UnorderedSequenceStorage::Native?sizeof(size_t):index_bytes);
        switch(options.sequence_storage) {
            case UnorderedSequenceStorage::Native:case UnorderedSequenceStorage::Packed:case UnorderedSequenceStorage::Deltas:break;
            default:throw std::invalid_argument("Invalid sequence storage mode.");
        }
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
			if (p.empty() || !std::all_of(p.begin(), p.end(), [](auto v) { return v.is_finite(); }))
				throw std::invalid_argument("Expected finite points, segments or simple polygons.");
			double area = 0;
			for (size_t i = 0; i < p.size(); ++i) area += (p[i] - p[0]).cross(p[(i + 1) % p.size()] - p[0]);
			if (p.size() >= 3 && area == 0) throw std::invalid_argument("Zero-area polygon.");
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
		std::vector<PreparedContactPolygon> visit_polygons;
		if(options.prepared_visit_queries) {
			visit_polygons.reserve(n);
			for(const auto &p:polygons)visit_polygons.emplace_back(p);
		}
		PreparedContactPath visit_segments;
        SegmentContactCache segment_cache;
		Polygon visit_path;
		bool visit_path_ready=false;
		std::vector<std::optional<Contact>> visit_contacts(n);
		// One point of each region near recent paths (initially a vertex).
		std::vector<Vector2> visit_anchors;
		for (const auto &p : polygons) visit_anchors.push_back(p.front());
		auto begin_visit_queries=[&](const Polygon &path) {
			if(!options.prepared_visit_queries)return;
			if(visit_path_ready&&visit_path.size()==path.size()&&std::equal(path.begin(),path.end(),visit_path.begin(),
				[](auto a,auto b){return std::bit_cast<std::uint64_t>(a.x)==std::bit_cast<std::uint64_t>(b.x)
					&&std::bit_cast<std::uint64_t>(a.y)==std::bit_cast<std::uint64_t>(b.y);}))return;
			visit_path=path;visit_segments.prepare(path);visit_path_ready=true;
			std::fill(visit_contacts.begin(),visit_contacts.end(),std::nullopt);
		};
		auto visit_contact=[&](const Polygon &path,size_t j) {
			if(options.prepared_visit_queries&&visit_contacts[j]) {
				++result.visit_query_cache_hits;return *visit_contacts[j];
			}
			++result.visit_query_evaluations;
			const auto found=options.prepared_visit_queries?(options.segment_visit_cache?segment_cache.query(visit_segments,visit_polygons[j],j,n,eps):contact(visit_segments,visit_polygons[j],eps)):contact(path,polygons[j],eps);
			if(options.prepared_visit_queries)visit_contacts[j]=found;
			if(found.distance>0)visit_anchors[j]=found.polygon_point;
			return found;
		};
		// Upper bound of the current path's distance to region j (exact if known).
		auto visit_distance_upper=[&](size_t j) {
			return visit_contacts[j]?visit_contacts[j]->distance:path_point_distance(visit_segments,visit_anchors[j]);
		};
		enum class Phase { Heuristic, Search, Finalization };
		Phase phase = Phase::Heuristic;
		// Most paths checked during the search are partial; the region found
		// uncovered last time usually still is, so test it first. Only the
		// boolean is used, so the order does not affect any decision.
		size_t uncovered_hint = 0;
		auto covered = [&](const Polygon &path) {
			const auto check_began = std::chrono::steady_clock::now();
			begin_visit_queries(path);
			bool covered_result=true;
			if(n&&options.visit_upper_bounds&&!(visit_contact(path,uncovered_hint).distance<=eps))covered_result=false;
			else for(size_t j=0;j<n;++j)if(!(visit_contact(path,j).distance<=eps)){covered_result=false;uncovered_hint=j;break;}
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
				result.incumbent_history.emplace_back(elapsed(), value);
				if (phase == Phase::Search) {
					++result.best_updates;
					if (!std::isfinite(result.first_best_update_length)) result.first_best_update_length = value;
				}
				if (!std::isfinite(result.first_incumbent_seconds)) result.first_incumbent_seconds = elapsed();
				if (options.trace) trace_event({
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
			if (polygons[polygon_index].size() <= 2) {
				pieces[polygon_index].push_back(polygons[polygon_index]);
				return;
			}
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
				|| options.bidirectional_initial_heuristic || options.convex_initial_refinement || options.relocate_initial_heuristic;
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
				if (options.trace) trace_event({.kind = "heuristic_start", .path = initial, .source = direction});
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
					if (options.trace) trace_event({
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
				if (options.trace) trace_event({
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
					if (options.trace) trace_event({
						.kind = "heuristic_2opt",
						.pass = pass,
						.order = order,
						.path = initial,
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
					if (options.trace) trace_event({
						.kind = "heuristic_contact_pass",
						.pass = pass,
						.order = order,
						.path = initial,
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
				bool sampled, std::chrono::steady_clock::time_point deadline, std::optional<Vector2> seed = std::nullopt) {
				if (std::chrono::steady_clock::now() >= deadline) return;
				const std::string direction = seed ? "cycle_multistart" : reverse ? "reverse" : "forward";
				auto candidate = initialize(seed.value_or(reverse ? target : start), reverse ? start : target,
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
					seed ? "heuristic_multistart" : sampled ? (reverse ? "heuristic_sampled_reverse" : "heuristic_sampled")
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

			if(options.relocate_initial_heuristic&&n>1&&!best_initial_path.empty()) {
				const auto began=std::chrono::steady_clock::now();
				const double budget=std::min(0.1,0.05*std::max(0.0,options.max_seconds-elapsed()));
				const auto deadline=began+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(budget));
				auto candidate=best_initial_path;
				auto order=best_initial_order;
				size_t moves=0;
				for(size_t pass=0;pass<4&&std::chrono::steady_clock::now()<deadline;++pass) {
					bool changed=false;
					for(size_t k=size_t(cycle);k<n&&std::chrono::steady_clock::now()<deadline;++k) {
						const size_t index=k+size_t(!cycle),region=order[k];
						const auto preferred=candidate[index];
						const double removal=candidate[index-1].distance_to(candidate[index+1])
							-candidate[index-1].distance_to(preferred)-preferred.distance_to(candidate[index+1]);
						auto shortened=candidate;shortened.erase(shortened.begin()+index);
						size_t selected=none;Vector2 point;double best_delta=-eps;
						for(size_t j=1;j<shortened.size()&&std::chrono::steady_clock::now()<deadline;++j) {
							const auto q=best_contact(shortened[j-1],shortened[j],polygons[region],preferred);
							const double delta=removal+shortened[j-1].distance_to(q)+q.distance_to(shortened[j])
								-shortened[j-1].distance_to(shortened[j]);
							if(delta<best_delta){best_delta=delta;selected=j;point=q;}
						}
						if(selected==none)continue;
						shortened.insert(shortened.begin()+selected,point);
						if(!(path_length(shortened)<path_length(candidate)-eps))continue;
						candidate=std::move(shortened);
						order.erase(order.begin()+k);order.insert(order.begin()+selected-size_t(!cycle),region);
						++moves;changed=true;
					}
					if(!changed)break;
					// Reoptimize contacts with a coordinate sweep after moving
					// regions. Each contact stays in its own original polygon.
					for(size_t k=0;k<n&&std::chrono::steady_clock::now()<deadline;++k) {
						const size_t index=k+size_t(!cycle);
						candidate[index]=best_contact(candidate[cycle?(k+n-1)%n:index-1],
							candidate[cycle?(k+1)%n:index+1],polygons[order[k]],candidate[index]);
						if(cycle)candidate.back()=candidate.front();
					}
				}
				const double value=path_length(candidate);
				if(value<best_initial_length&&covered(candidate)) {
					best_initial_path=std::move(candidate);best_initial_order=std::move(order);best_initial_length=value;
					result.initial_relocation_moves=moves;
					improve(best_initial_path,"initial_relocation",best_initial_order);
				}
				result.initial_relocation_seconds=duration(began);
			}
            if(cycle&&options.cycle_primal_starts&&n>1) {
                // Diversify cyclic orders, using the maintained constructor and
                // local search. Seeds are original vertices, never a discretized
                // relaxation. Keep the baseline candidate before these attempts.
                const size_t starts=std::min(n,size_t(8));
                const double allowance=std::max(0.0,std::min(options.max_seconds-elapsed(),options.max_seconds*.05));
                const auto deadline=std::isfinite(allowance)?std::chrono::steady_clock::now()+
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(allowance)):
                    std::chrono::steady_clock::time_point::max();
                for(size_t i=1;i<starts&&std::chrono::steady_clock::now()<deadline;++i) {
                    if((options.stop_requested&&options.stop_requested())||(control&&control->stopped()))break;
                    const double previous=result.upper_bound;
                    ++result.cycle_primal_start_candidates;
                    consider(false,polygons,false,deadline,polygons[i*n/starts].front());
                    result.cycle_primal_start_improvements+=result.upper_bound<previous;
                }
            }
			if(options.primal_ils_fraction>0&&n>=4&&best_initial_order.size()==n
				&&best_initial_path.size()==n+2-size_t(cycle)) {
				// Iterated local search over one contact per region, each kept in
				// its own original polygon by best_contact, so every tour visits
				// all regions; improve() still validates coverage and length.
				const auto began=std::chrono::steady_clock::now();
				const double budget=options.primal_ils_fraction*std::max(0.0,options.max_seconds-elapsed());
				// Without a time limit the ILS needs its stagnation stop.
				const auto deadline=std::isfinite(budget)
					?began+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(budget))
					:options.primal_ils_stagnation>0?std::chrono::steady_clock::time_point::max():began;
				auto running=[&]{return std::chrono::steady_clock::now()<deadline
					&&!(options.stop_requested&&options.stop_requested())&&!(control&&control->stopped());};
				struct Tour {std::vector<size_t> order;Polygon point;double length=0;};
				auto before=[&](const Tour &t,size_t k){return cycle?t.point[(k+n-1)%n]:k==0?start:t.point[k-1];};
				auto after=[&](const Tour &t,size_t k){return cycle?t.point[(k+1)%n]:k+1==n?target:t.point[k+1];};
				auto as_path=[&](const Tour &t) {
					Polygon path;path.reserve(n+2);
					if(!cycle)path.push_back(start);
					path.insert(path.end(),t.point.begin(),t.point.end());
					path.push_back(cycle?t.point.front():target);
					return path;
				};
				auto measure=[&](Tour &t){t.length=path_length(as_path(t));};
				// Candidate lists: near[r*n+s] marks the K regions closest to r by
				// boundary distance; the Or-opt only tries gaps next to them.
				std::vector<char> near;
				if(options.primal_ils_candidates>0&&options.primal_ils_candidates+1<n) {
					auto segment_distance=[](Vector2 p,Vector2 a,Vector2 b) {
						const auto edge=b-a;const double squared=edge.length_squared();
						const double rate=squared==0?0:std::clamp((p-a).dot(edge)/squared,0.0,1.0);
						return p.distance_to(a+rate*edge);
					};
					auto region_distance=[&](const Polygon &x,const Polygon &y) {
						double d=std::numeric_limits<double>::infinity();
						for(auto [from,to]:{std::pair{&x,&y},std::pair{&y,&x}})
							for(const auto &q:*from)for(size_t j=0;j<to->size();++j)
								d=std::min(d,segment_distance(q,(*to)[j],(*to)[(j+1)%to->size()]));
						if(x.size()>2&&contact(Polygon{y.front(),y.front()},x,0).distance<=0)d=0;
						if(y.size()>2&&contact(Polygon{x.front(),x.front()},y,0).distance<=0)d=0;
						return d;
					};
					std::vector<double> distance(n*n,0);
					for(size_t r=0;r<n;++r)for(size_t q=r+1;q<n;++q)distance[r*n+q]=distance[q*n+r]=region_distance(polygons[r],polygons[q]);
					near.assign(n*n,0);
					std::vector<size_t> others;
					for(size_t r=0;r<n;++r) {
						others.clear();for(size_t q=0;q<n;++q)if(q!=r)others.push_back(q);
						std::partial_sort(others.begin(),others.begin()+options.primal_ils_candidates,others.end(),
							[&](size_t x,size_t y){return distance[r*n+x]<distance[r*n+y];});
						for(size_t i=0;i<options.primal_ils_candidates;++i)near[r*n+others[i]]=1;
					}
				}
				auto local_search=[&](Tour &t) {
					for(size_t round=0;round<64&&running();++round) {
						bool changed=false;
						for(size_t k=0;k<n;++k)t.point[k]=best_contact(before(t,k),after(t,k),polygons[t.order[k]],t.point[k]);
						measure(t);
						const double tolerance=1e-12*std::max(1.0,t.length);
						// 2-opt: reverse positions i..j.
						for(size_t i=0;i+1<n&&running();++i)for(size_t j=i+1;j<n;++j) {
							if(cycle&&i==0&&j==n-1)continue;
							const Vector2 a=before(t,i),b=after(t,j);
							const double delta=a.distance_to(t.point[j])+t.point[i].distance_to(b)
								-a.distance_to(t.point[i])-t.point[j].distance_to(b);
							if(delta<-tolerance) {
								std::reverse(t.point.begin()+i,t.point.begin()+j+1);
								std::reverse(t.order.begin()+i,t.order.begin()+j+1);
								changed=true;
							}
						}
						// Or-opt: move the block of positions k..k+l-1 to its best gap,
						// optionally reversed; the contacts at both block ends are
						// reoptimized for the new neighbours (one contact when l = 1).
						const size_t longest=std::min(options.primal_ils_block,n-2);
						for(size_t l=1;l<=longest;++l)for(size_t k=0;k+l<=n&&running();++k) {
							const Vector2 a=before(t,k),b=after(t,k+l-1);
							double inside=0;for(size_t i=k+1;i<k+l;++i)inside+=t.point[i-1].distance_to(t.point[i]);
							const double removal=a.distance_to(b)-a.distance_to(t.point[k])-t.point[k+l-1].distance_to(b)-inside;
							Tour rest;
							rest.order.assign(t.order.begin(),t.order.begin()+k);rest.order.insert(rest.order.end(),t.order.begin()+k+l,t.order.end());
							rest.point.assign(t.point.begin(),t.point.begin()+k);rest.point.insert(rest.point.end(),t.point.begin()+k+l,t.point.end());
							const size_t m=n-l;
							size_t chosen=none;bool chosen_reversed=false;Polygon chosen_points;double best_delta=-tolerance;
							for(size_t gap=0;gap<=m;++gap) {
								if(cycle&&gap==m)break;
								if(!near.empty()) {
									const size_t first=t.order[k],last=t.order[k+l-1];
									auto close=[&](size_t position) {
										if(position>=m)return false;
										const size_t region=rest.order[position];
										return near[first*n+region]||near[last*n+region];
									};
									const size_t left=gap==0?(cycle?m-1:none):gap-1;
									if(!close(left)&&!close(gap))continue;
								}
								const Vector2 u=gap==0?(cycle?rest.point[m-1]:start):rest.point[gap-1];
								const Vector2 v=gap==m?target:rest.point[gap];
								for(int reversed=0;reversed<=int(options.primal_ils_reverse&&l>1);++reversed) {
									Polygon block(t.point.begin()+k,t.point.begin()+k+l);
									std::vector<size_t> regions(t.order.begin()+k,t.order.begin()+k+l);
									if(reversed){std::reverse(block.begin(),block.end());std::reverse(regions.begin(),regions.end());}
									if(l==1)block[0]=best_contact(u,v,polygons[regions[0]],block[0]);
									else {
										block[0]=best_contact(u,block[1],polygons[regions[0]],block[0]);
										block[l-1]=best_contact(block[l-2],v,polygons[regions[l-1]],block[l-1]);
									}
									double moved=0;for(size_t i=1;i<l;++i)moved+=block[i-1].distance_to(block[i]);
									const double delta=removal+u.distance_to(block[0])+moved+block[l-1].distance_to(v)-u.distance_to(v);
									if(delta<best_delta){best_delta=delta;chosen=gap;chosen_reversed=reversed;chosen_points=block;}
								}
							}
							if(chosen==none)continue;
							std::vector<size_t> regions(t.order.begin()+k,t.order.begin()+k+l);
							if(chosen_reversed)std::reverse(regions.begin(),regions.end());
							rest.order.insert(rest.order.begin()+chosen,regions.begin(),regions.end());
							rest.point.insert(rest.point.begin()+chosen,chosen_points.begin(),chosen_points.end());
							t=std::move(rest);changed=true;
						}
						// Swap two non-adjacent regions, each with an exact contact
						// between its new neighbours.
						if(options.primal_ils_swap)for(size_t i=0;i<n&&running();++i)for(size_t j=i+2;j<n;++j) {
							if(cycle&&i==0&&j==n-1)continue;
							const size_t left=cycle?(i+n-1)%n:i==0?none:i-1;
							const size_t right=cycle?(i+1)%n:i+1;
							if(!near.empty()) {
								const size_t region=t.order[j];
								if(!(left!=none&&near[region*n+t.order[left]])&&!near[region*n+t.order[right]])continue;
							}
							const Vector2 a=before(t,i),b=after(t,i),c=before(t,j),d=after(t,j);
							const Vector2 x=best_contact(a,b,polygons[t.order[j]],t.point[j]);
							const Vector2 y=best_contact(c,d,polygons[t.order[i]],t.point[i]);
							const double delta=a.distance_to(x)+x.distance_to(b)+c.distance_to(y)+y.distance_to(d)
								-a.distance_to(t.point[i])-t.point[i].distance_to(b)-c.distance_to(t.point[j])-t.point[j].distance_to(d);
							if(delta<-tolerance) {
								std::swap(t.order[i],t.order[j]);t.point[i]=x;t.point[j]=y;changed=true;
							}
						}
						measure(t);
						if(!changed)break;
					}
				};
				// Exact contacts for the tour's order on the convex pieces that hold
				// its current contacts; only a strictly shorter tour is kept.
				DynamicConvexTppWorkspace polish_workspace;
				polish_workspace.cache_disjoint_dispatch=options.oracle_dispatch_cache;
				polish_workspace.cache_interval_geometry=options.oracle_interval_geometry_cache;
				polish_workspace.borrow_hybrid_geometry=options.oracle_borrow_geometry;
				polish_workspace.bound_before_optimality=options.oracle_bound_first;
				polish_workspace.interpolated_zero_dual=options.interpolated_zero_dual;
				polish_workspace.float_recovery=options.float_recovery;polish_workspace.float_degenerate=options.float_degenerate;
				polish_workspace.trust_double=options.trust_double;
				auto piece_holding=[&](size_t region,Vector2 point)->const Polygon * {
					prepare_pieces(region);
					const Polygon point_path{point,point};
					for(const auto &piece:pieces[region])if(contact(point_path,piece,eps).distance<=eps)return &piece;
					return nullptr;
				};
				// Exact contacts for consecutive positions first..first+count-1 of the
				// tour's order, between the fixed contacts around them (the whole cycle
				// when count = n), on the convex pieces holding their current contacts.
				auto polish_positions=[&](Tour &t,size_t first,size_t count) {
					const bool whole=count==n;
					std::vector<Polygon> ordered;ordered.reserve(count);Polygon current_points;
					for(size_t i=0;i<count;++i) {
						const size_t k=(first+i)%n;
						const auto piece=piece_holding(t.order[k],t.point[k]);
						if(!piece)return;
						ordered.push_back(*piece);current_points.push_back(t.point[k]);
					}
					const Vector2 a=whole?start:before(t,first),b=whole?target:after(t,(first+count-1)%n);
					double old_length=whole?t.length:a.distance_to(current_points.front())+current_points.back().distance_to(b);
					if(!whole)for(size_t i=1;i<count;++i)old_length+=current_points[i-1].distance_to(current_points[i]);
					if(result.calls>=options.max_calls||(control&&!control->reserve_call()))return;
					++result.calls;++result.primal_ils_polish_calls;
					const auto polish_began=std::chrono::steady_clock::now();
					try {
						const double remaining=std::chrono::duration<double>(deadline-polish_began).count();
						const bool polish_cycle=whole&&cycle;
						const auto polished=solve_relaxation(polish_cycle,a,b,ordered,polish_workspace,
							std::max(1e-9*old_length,std::numeric_limits<double>::epsilon()),old_length,std::min(1.0,remaining),
							polish_cycle?current_points:Polygon{},nullptr,{},false,false,false,{},false,options.cycle_float_oracle);
						const double length=path_length(polished.path);
						const size_t offset=size_t(!polish_cycle);
						if(polished.path.size()==count+2-size_t(polish_cycle)&&std::isfinite(length)&&length<old_length*(1-1e-12)) {
							for(size_t i=0;i<count;++i)t.point[(first+i)%n]=polished.path[offset+i];
							measure(t);++result.primal_ils_polish_improvements;
						}
					} catch(const std::exception &) {}
					result.primal_ils_polish_seconds+=duration(polish_began);
				};
				// Exact free-order reoptimization of positions first..first+count-1
				// between the fixed contacts around them: the same B&B on the
				// window's regions, with the window as initial path.
				auto reorder_positions=[&](Tour &t,size_t first,size_t count) {
					std::vector<Polygon> regions;regions.reserve(count);
					std::vector<size_t> indices;indices.reserve(count);
					Polygon window{before(t,first)};
					for(size_t i=0;i<count;++i) {
						const size_t k=(first+i)%n;
						regions.push_back(polygons[t.order[k]]);indices.push_back(t.order[k]);window.push_back(t.point[k]);
					}
					window.push_back(after(t,(first+count-1)%n));
					const double old_length=path_length(window);
					auto sub=options;
					sub.primal_ils_fraction=0;sub.window_lns=false;sub.trace=false;sub.progress=nullptr;
					sub.portfolio=false;sub.threads=1;sub.initial_path=window;
					// The window is a small part of the tour: close its own gap tightly.
					sub.relative_gap=1e-9;sub.absolute_gap=1e-9*t.length;
					sub.max_seconds=std::max(0.0,std::min(0.5,std::chrono::duration<double>(deadline-std::chrono::steady_clock::now()).count()));
					if(!(sub.max_seconds>0)||result.calls>=options.max_calls)return;
					sub.max_calls=std::min<size_t>(options.max_calls-result.calls,20000);
					const auto reorder_began=std::chrono::steady_clock::now();
					try {
						const auto solved=solve_normalized_unordered_tpp(window.front(),window.back(),regions,sub,false,nullptr);
						result.calls+=solved.calls;++result.primal_ils_reorder_calls;
						if(solved.path.size()>=2&&solved.upper_bound<old_length*(1-1e-12)) {
							// The new window path may visit several regions with one contact (or
							// along a segment): give each region its first touch along the path and
							// visit them in that order, which keeps the length of the path.
							std::vector<std::pair<double,size_t>> touches;touches.reserve(count);
							Polygon touch_points(count);
							bool assigned=true;
							for(size_t i=0;i<count&&assigned;++i) {
								const auto c=contact(solved.path,regions[i],eps);
								if(!(c.distance<=eps)){assigned=false;break;}
								const size_t segment=std::min<size_t>(size_t(std::floor(c.position)),solved.path.size()-2);
								const double rate=std::clamp(c.position-double(segment),0.0,1.0);
								touch_points[i]=solved.path[segment]+rate*(solved.path[segment+1]-solved.path[segment]);
								touches.emplace_back(c.position,i);
							}
							if(assigned) {
								std::stable_sort(touches.begin(),touches.end());
								Tour changed=t;
								for(size_t i=0;i<count;++i) {
									const size_t k=(first+i)%n;
									changed.order[k]=indices[touches[i].second];changed.point[k]=touch_points[touches[i].second];
								}
								measure(changed);
								if(changed.length<t.length*(1-1e-12)){t=std::move(changed);++result.primal_ils_reorder_improvements;}
							}
						}
					} catch(const std::exception &) {}
					result.primal_ils_reorder_seconds+=duration(reorder_began);
				};
				auto reorder=[&](Tour &t) {
					const size_t width=options.primal_ils_reorder;
					if(width<2||width+2>n)return;
					const size_t step=std::max<size_t>(1,width/2),last=cycle?n:n-width+1;
					for(size_t first=0;first<last&&running();first+=step)reorder_positions(t,first,width);
				};
				auto polish=[&](Tour &t) {
					if(!(options.primal_ils_polish>0)||!running())return;
					if(options.primal_ils_reorder>=2){reorder(t);return;}
					const size_t width=options.primal_ils_window;
					if(width==0||width+2>n){polish_positions(t,0,n);return;}
					const size_t step=std::max<size_t>(1,width/2),last=cycle?n:n-width+1;
					for(size_t first=0;first<last&&running();first+=step)polish_positions(t,first,width);
				};
				Tour current;current.order=best_initial_order;
				current.point.assign(best_initial_path.begin()+size_t(!cycle),best_initial_path.begin()+size_t(!cycle)+n);
				local_search(current);
				polish(current);
				Tour best=current;
				if(best.length<best_initial_length&&covered(as_path(best))) {
					best_initial_path=as_path(best);best_initial_order=best.order;best_initial_length=best.length;
					improve(best_initial_path,"primal_ils",best_initial_order);++result.primal_ils_improvements;
				}
				std::mt19937_64 random((0x9e3779b97f4a7c15ULL^n)+options.primal_ils_seed);
				const double total=std::max(1e-9,budget);
				double eta=0.01;size_t stagnation=0;
				size_t last_best_iteration=0;
				while(running()&&!(options.primal_ils_stagnation>0
					&&result.primal_ils_iterations-last_best_iteration>=options.primal_ils_stagnation)) {
					++result.primal_ils_iterations;
					// Double bridges: A B C D -> A C B D on positions.
					Tour trial=current;
					for(size_t kick=0;kick<options.primal_ils_kicks;) {
						std::uniform_int_distribution<size_t> cut(1,n-1);
						std::array<size_t,3> c{cut(random),cut(random),cut(random)};std::sort(c.begin(),c.end());
						if(c[0]==c[1]||c[1]==c[2])continue;
						Tour bridged;bridged.order.reserve(n);bridged.point.reserve(n);
						for(auto [from,to]:{std::pair{size_t(0),c[0]},std::pair{c[1],c[2]},std::pair{c[0],c[1]},std::pair{c[2],n}})
							for(size_t k=from;k<to;++k){bridged.order.push_back(trial.order[k]);bridged.point.push_back(trial.point[k]);}
						trial=std::move(bridged);++kick;
					}
					local_search(trial);
					if(trial.length<best.length*(1+options.primal_ils_polish))polish(trial);
					if(options.primal_ils_reheat) {
						if(trial.length<current.length||trial.length<=best.length*(1+eta))current=trial;
						if(trial.length>=best.length*(1-1e-12)&&++stagnation%10==0&&(eta*=0.95)<1e-4){eta=0.01;current=best;}
					} else {
						// Record-to-record: accept within a slack that shrinks to zero.
						const double progress=std::min(1.0,duration(began)/total);
						if(trial.length<best.length*(1+0.02*(1-progress)))current=trial;
					}
					if(trial.length<best.length*(1-1e-12)) {
						if(!(options.primal_ils_polish>0))reorder(trial);
						best=trial;stagnation=0;last_best_iteration=result.primal_ils_iterations;
						if(best.length<best_initial_length&&covered(as_path(best))) {
							best_initial_path=as_path(best);best_initial_order=best.order;best_initial_length=best.length;
							improve(best_initial_path,"primal_ils",best_initial_order);++result.primal_ils_improvements;
						}
					}
				}
				result.primal_ils_seconds=duration(began);
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
						initial_workspace.cache_disjoint_dispatch=options.oracle_dispatch_cache;
						initial_workspace.cache_interval_geometry=options.oracle_interval_geometry_cache;
                        initial_workspace.borrow_hybrid_geometry=options.oracle_borrow_geometry;
                        initial_workspace.bound_before_optimality=options.oracle_bound_first;
						initial_workspace.interpolated_zero_dual=options.interpolated_zero_dual;
						initial_workspace.float_recovery=options.float_recovery;initial_workspace.float_degenerate=options.float_degenerate;
						initial_workspace.trust_double=options.trust_double;
						++result.calls;
						++result.initial_convex_refinement_calls;
						const double target_gap = options.absolute_gap
							+ options.relative_gap * std::abs(best_initial_length);
						const auto polished = solve_relaxation(
							cycle, start, target, ordered_pieces, initial_workspace,
							std::max(target_gap * 0.25, std::numeric_limits<double>::epsilon()),
							best_initial_length - target_gap, remaining, {}, nullptr, {}, false, false, false, {}, false,
							options.cycle_float_oracle);
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
		// Exact window LNS. A window is a run of consecutive incumbent contacts
		// between two fixed points a and b. The regions it must visit are those
		// not already visited by the fixed path outside it; the same B&B solves
		// that small fixed-endpoint TPP, seeded with the current window as its
		// incumbent. A strictly shorter window is spliced in and the whole path
		// is revalidated by improve(), so only upper bounds can change.
		// The sweep is resumable: its width and next window survive between
		// calls, so a stalled search keeps spending its accumulated budget. A new
		// incumbent restarts the sweep at the last successful width; a complete
		// sweep at the largest width without improvement (and without truncated
		// subproblems) exhausts the neighbourhood of the current incumbent.
		struct WindowLnsState {
			size_t first = 1, width = 0, success_width = 0;
			double upper_bound = std::numeric_limits<double>::infinity();
			bool exhausted = false, truncated = false;
			double retry_at = 0;  // backoff after an unsuccessful truncated sweep
		} lns;
		lns.width = lns.success_width = std::max<size_t>(2, options.window_lns_size);
		const size_t lns_max_width = std::max(lns.width, std::min(options.window_lns_max_size, n));
		// Proportional budget: total LNS time stays within the requested fraction
		// of the time spent so far (at least of 0.5 s), so easy instances pay
		// little and long searches keep improving the incumbent.
		auto window_lns_allowance = [&] {
			return options.window_lns_time_fraction * std::max(0.5, elapsed()) - result.window_lns_seconds;
		};
		auto window_lns = [&] {
			if (cycle || !options.window_lns || n < 3 || result.path.size() < 3 || !std::isfinite(result.upper_bound)) return;
			if (result.upper_bound < lns.upper_bound) {
				lns = {.first = 1, .width = lns.success_width, .success_width = lns.success_width, .upper_bound = result.upper_bound};
			}
			if (lns.exhausted) {
				// Truncated subproblems may hide improvements: retry with the larger
				// budget available after the elapsed time has doubled.
				if (!(lns.retry_at > 0 && elapsed() >= lns.retry_at)) return;
				lns.exhausted = false;lns.retry_at = 0;lns.first = 1;
			}
			const double allowance = window_lns_allowance();
			if (!(allowance > 0)) return;
			const auto began = std::chrono::steady_clock::now();
			auto remaining = [&] { return std::min(allowance - duration(began), options.max_seconds - elapsed()); };
			auto stopped = [&] {
				return remaining() <= 0 || result.calls >= options.max_calls
					|| (options.stop_requested && options.stop_requested()) || (control && control->stopped());
			};
			while (!stopped() && !lns.exhausted) {
				const Polygon path = result.path;
				const size_t interior = path.size() - 2;
				if (lns.first == 1) ++result.window_lns_rounds;
				bool improved = false;
				// Half-overlapping windows over the incumbent's interior contacts.
				for (; lns.first <= interior && !stopped(); lns.first += std::max<size_t>(1, lns.width / 2)) {
					const size_t first = lns.first, width = lns.width;
					const size_t last = std::min(first + width - 1, interior);
					const Polygon left(path.begin(), path.begin() + first);
					const Polygon right(path.begin() + last + 1, path.end());
					const Polygon window(path.begin() + first - 1, path.begin() + last + 2);
					std::vector<Polygon> regions;
					for (size_t j = 0; j < n; ++j) {
						const bool outside = (left.size() > 1 && contact(left, polygons[j], eps).distance <= eps)
							|| (right.size() > 1 && contact(right, polygons[j], eps).distance <= eps);
						if (!outside) regions.push_back(polygons[j]);
					}
					// Incidental visits can make a window larger than its contacts;
					// keep each subproblem small enough to stay cheap.
					if (regions.size() > width + width / 2) continue;
					const double window_length = path_length(window);
					Polygon replacement;
					double replacement_length = window_length;
					if (regions.empty()) {
						replacement = {window.front(), window.back()};
						replacement_length = path_length(replacement);
					} else {
						auto sub = options;
						sub.window_lns = false;
						sub.trace = false;
						sub.progress = nullptr;  // a sub-search must not report as the main one
						sub.portfolio = false;
						sub.threads = 1;
						sub.initial_path = window;
						sub.max_seconds = std::max(0.0, std::min(remaining(), 0.25 * allowance));
						sub.max_calls = std::min<size_t>(options.max_calls - result.calls, 20000);
						try {
							++result.window_lns_subproblems;
							const auto solved = solve_normalized_unordered_tpp(window.front(), window.back(), regions, sub, false, nullptr);
							result.calls += solved.calls;
							result.window_lns_calls += solved.calls;
							lns.truncated |= !solved.exact;
							if (solved.path.size() >= 2) { replacement = solved.path; replacement_length = solved.upper_bound; }
						} catch (const std::exception &) {
							lns.truncated = true;
							continue;  // A failed subproblem only loses this improvement attempt.
						}
					}
					if (replacement.empty() || !(replacement_length < window_length - std::max(eps, 1e-12 * window_length))) continue;
					Polygon candidate = left;
					candidate.insert(candidate.end(), replacement.begin() + 1, replacement.end() - 1);
					candidate.insert(candidate.end(), right.begin(), right.end());
					const double before = result.upper_bound;
					improve(candidate, "window_lns");
					if (result.upper_bound < before) {
						++result.window_lns_improvements;
						result.window_lns_gain += before - result.upper_bound;
						lns = {.first = 1, .width = width, .success_width = width, .upper_bound = result.upper_bound};
						improved = true;
						break;  // Indices changed; restart the sweep on the new incumbent.
					}
				}
				if (improved || lns.first <= interior) continue;  // restarted, or out of budget
				// A complete sweep without improvement: grow, or stop at the top.
				if (lns.width >= lns_max_width) {
					lns.exhausted = true;
					lns.retry_at = lns.truncated ? 2 * std::max(0.5, elapsed()) : 0;
					lns.truncated = false;
					lns.first = 1;
				} else {
					lns.width = std::min(lns_max_width, lns.width + std::max<size_t>(1, lns.width / 2));
					lns.first = 1;
				}
			}
			result.window_lns_seconds += duration(began);
		};
		window_lns();
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
			|| (options.stop_requested && options.stop_requested())
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
		workspace.cache_disjoint_dispatch=options.oracle_dispatch_cache;
		workspace.cache_interval_geometry=options.oracle_interval_geometry_cache;
        workspace.borrow_hybrid_geometry=options.oracle_borrow_geometry;
        workspace.bound_before_optimality=options.oracle_bound_first;
        workspace.retain_binary_dual=!cycle&&options.path_certificate_dual;
		workspace.interpolated_zero_dual=options.interpolated_zero_dual;
		workspace.float_recovery=options.float_recovery;workspace.float_degenerate=options.float_degenerate;
		workspace.trust_double=options.trust_double;
		std::vector<DynamicConvexTppWorkspace> parallel_workspaces;
        ConvexCycleWorkspace cycle_workspace;
        std::vector<ConvexCycleWorkspace> parallel_cycle_workspaces;
        CycleMemo memo_workspace;
        // Per-search bounded retention, never an extra vector in every node.
        // Keys are immutable node serials; no sequence/geometry is inferred.
        std::unordered_map<size_t,Polygon> path_dual_cache;
        size_t path_dual_bytes=0;
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
        OracleCapture oracle_capture(options.oracle_capture_file,options.oracle_capture_every,options.oracle_capture_min_seconds);
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
                            out.used_rational=true; // the exact certificate rechecked it
                            out.seconds=duration(began_oracle);out.geometric_solver_seconds=out.seconds;
                            return out;
                        }
                    }
                }
            }
            const auto capture_id=oracle_capture.begin(node.serial,precise,start,target,selected,node.warm_start,node.active_features,cutoff,tolerance,remaining_seconds,options);
			auto out=solve_relaxation(
				cycle, start, target, selected, oracle_workspace, tolerance, cutoff, remaining_seconds, node.warm_start, options.cycle_cache?&cycle_cache:nullptr,
                options.cycle_active_features?node.active_features:std::vector<int>{}, options.cycle_active_features,options.cycle_bound_first,
                options.cycle_interval_certificate, [control]{return control&&control->proved();},
                options.cycle_proposal_bound&&!precise, options.cycle_float_oracle
			);
            oracle_capture.end(capture_id,out);
            if(cache&&!out.path.empty()) {
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
			RelaxationResult &certified) {
			node.warm_start=Polygon{};
			result.oracle_cutoff_calls += certified.lower_bound >= cutoff;
			result.oracle_dual_cutoff_prunes += certified.dual_cutoff_pruned;
			result.oracle_dispatch_pair_queries += certified.dispatch_pair_queries;
			result.oracle_dispatch_pair_cache_hits += certified.dispatch_pair_cache_hits;
			result.oracle_dispatch_pair_exact_checks += certified.dispatch_pair_exact_checks;
			result.convex_dispatch_seconds += certified.dispatch_seconds;
			result.convex_bound_evaluation_seconds += certified.bound_evaluation_seconds;
			result.convex_proposal_preparation_seconds += certified.proposal_preparation_seconds;
            result.cycle_memo_queries+=certified.memo_queries;result.cycle_memo_repeated+=certified.memo_repeated;
            result.cycle_memo_hits+=certified.memo_hits;result.cycle_certificate_cutoff_skips+=certified.certificate_cutoff_skips;
            result.cycle_certificate_interval_uses+=certified.certificate_interval_uses;
            result.cycle_proposal_calls+=certified.proposal_calls;result.cycle_proposal_accepts+=certified.proposal_accepts;
            result.cycle_initial_contact_checks+=certified.initial_contact_checks;result.cycle_initial_contact_accepts+=certified.initial_contact_accepts;
			node.refined = !certified.time_limited && (precise || options.oracle_relative_gap == 0);
			if(!certified.path.empty())node.path = certified.path;
            node.active_features=options.cycle_active_features?certified.active_features:std::vector<int>{};
            if(cycle&&options.cycle_dual_reuse&&node.path.size()==node.sequence.size()+1) {
                Polygon contacts(node.path.begin(),node.path.end()-1);
                node.dual=tpp_convex_cycle_dual_directions(contacts,node.dual);
            }
            node.relaxed_length = certified.upper_bound;
			result.fallback_calls += certified.used_fallback;
			result.oracle_unverified_fallbacks += certified.fallback_unverified;
			result.oracle_interval_bound_calls += certified.used_interval_bounds;
			result.oracle_float_calls += certified.used_float_oracle;
			result.oracle_float_fallbacks += certified.float_oracle_fallback;
			result.oracle_rational_calls += certified.used_rational;
			result.oracle_trusted_calls += certified.used_trusted_double;
			result.oracle_exact_replay_calls += certified.used_exact_replay;
			result.oracle_filtered_calls += certified.used_filtered_recovery;
			result.oracle_touching_calls += certified.used_touching_recovery;
			result.rational_membership_predicates += certified.rational_membership_predicates;
			result.exact_polygon_preparations += certified.exact_polygon_preparations;
			result.oracle_contracted_bound_calls += certified.used_contracted_proposal;
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
            result.cycle_construction_seconds+=certified.cycle_timings.construction_seconds;
            result.cycle_certification_seconds+=certified.cycle_timings.certification_seconds;
            result.cycle_rational_recovery_seconds+=certified.cycle_timings.rational_recovery_seconds;
            result.cycle_interval_proof_seconds+=certified.cycle_timings.interval_proof_seconds;
            result.cycle_polish_seconds+=certified.cycle_timings.polish_seconds;
            result.cycle_polish_calls+=certified.polish_attempted;
            result.cycle_polish_newton_iterations+=certified.polish_newton_iterations;
			++result.oracle_profiled_calls;
			result.oracle_max_call_seconds = std::max(result.oracle_max_call_seconds, certified.seconds);
			if (certified.used_fallback) result.oracle_fallback_call_seconds += certified.seconds;
			constexpr std::array<double, 6> timing_bounds{1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0};
			const auto bucket = std::lower_bound(timing_bounds.begin(), timing_bounds.end(), certified.seconds) - timing_bounds.begin();
			++result.oracle_call_histogram[bucket];
			result.oracle_seconds_histogram[bucket] += certified.seconds;
			result.convex_geometric_solver_seconds += certified.geometric_solver_seconds;
			result.convex_certificate_verification_seconds += certified.certificate_verification_seconds;
			result.convex_contact_materialization_seconds += certified.contact_materialization_seconds;
			result.convex_fallback_seconds += certified.fallback_seconds;
			result.convex_fallback_long_double_seconds += certified.fallback_long_double_seconds;
			result.convex_fallback_extended_precision_seconds += certified.fallback_extended_precision_seconds;
			node.bound = std::max(node.bound, certified.lower_bound);
            if(!cycle&&options.path_certificate_dual&&!certified.binary_dual.empty()&&node.bound<cutoff) {
                const size_t bytes=certified.binary_dual.capacity()*sizeof(Vector2);
                constexpr size_t budget=2*1024*1024;
                if(bytes<=budget) {
                    if(path_dual_cache.size()>=4096||path_dual_bytes+bytes>budget) {
                        path_dual_cache.clear();path_dual_bytes=0;++result.path_dual_cache_evictions;
                    }
                    auto &entry=path_dual_cache[node.serial];
                    path_dual_bytes-=entry.capacity()*sizeof(Vector2);
                    entry=std::move(certified.binary_dual);path_dual_bytes+=entry.capacity()*sizeof(Vector2);
                    ++result.path_dual_retained;
                    result.path_dual_peak_bytes=std::max(result.path_dual_peak_bytes,path_dual_bytes);
                }
            }
            if(cycle&&options.cycle_learned_branching&&node.learning_pending) {
                const double gain=std::max(0.0,node.bound-node.learning_parent_bound)/node.learning_distance;
                if(std::isfinite(gain)) {
                    const size_t kind=node.branch_piece!=none;
                    learned[node.branch_polygon][kind].observe(gain);prior[kind].observe(gain);
                    ++result.learned_branch_observations;
                }
                node.learning_pending=false;
            }
			if ((!certified.time_limited && node.path.size() < 2) || !std::all_of(node.path.begin(), node.path.end(), [](auto v) { return v.is_finite(); }))
				throw std::runtime_error("Convex oracle returned an invalid path.");
			if (options.trace) trace_event({
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
			auto certified = evaluate_oracle(node, precise, upper_bound, workspace, cycle_workspace,memo_workspace);
			result.convex_oracle_wall_seconds += duration(oracle_began);
			record_oracle_result(node, precise, cutoff, certified);
            if(certified.time_limited && !node.path.empty())improve(node.path,"interrupted_oracle");
            return !certified.time_limited;
		};
		double settled_bound = result.upper_bound;
		const bool dfs=options.search_strategy==UnorderedSearchStrategy::DfsBfs;
        Frontier queue(dfs,options.sequence_storage,index_bytes,n);
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
		if (options.trace) trace_event({
			.kind = "root",
			.node = 0,
			.path = {start, target},
			.lower_bound = result.lower_bound,
			.upper_bound = result.upper_bound,
		});
		size_t serial = 1;
		// Active dive chains: at most one per node expanded in a round (one in
		// the serial search). Each continues from the best child of its node.
		std::vector<Node> dives;
		auto note_dives = [&] {
			size_t bytes = 0;
			for (const auto &d : dives) bytes += d.sequence.payload_bytes();
			queue.note_dive(bytes);
		};
		auto frontier_bound = [&] {
			double bound = queue.empty() ? result.upper_bound : queue.lower_bound();
			for (const auto &d : dives) bound = std::min(bound, d.bound);
			return std::min(bound, result.upper_bound);
		};
		struct Family {
			std::vector<Node> children;
			SequenceReference parent;
			bool diving = false;
			size_t first_child = 0;
			std::optional<Node> dive;
		};
		// Parallel rounds batch several best-bound nodes so that every thread has
		// oracles to evaluate; sibling batches alone rarely exceed one survivor.
		const size_t nodes_per_round = options.parallel_nodes && !dfs ? std::max<size_t>(1, options.threads) : 1;
		double next_progress_report = 0.0;
		auto report_progress = [&] {
			if (!options.progress || !(options.progress_interval_seconds > 0)) return;
			const double now = elapsed();
			if (now < next_progress_report) return;
			next_progress_report = now + options.progress_interval_seconds;
			options.progress({
				.worker = options.progress_worker,
				.elapsed_seconds = now,
				.lower_bound = result.lower_bound,
				.upper_bound = result.upper_bound,
				.calls = control ? control->calls.load(std::memory_order_relaxed) : result.calls,
				.nodes = result.nodes,
				.open_nodes = queue.size() + dives.size(),
				.peak_open_nodes = result.peak_queue,
				.pruned_nodes = result.pruned_nodes,
				.max_sequence_depth = result.max_sequence_depth,
				.region_count = n,
			});
		};
		while (!queue.empty() || !dives.empty()) {
            import_incumbent();
			// Resume the LNS after a new incumbent, or in bursts once half of its
			// proportional budget is available, so subproblems are not starved.
			if (options.window_lns && (result.upper_bound < lns.upper_bound
				|| ((!lns.exhausted || (lns.retry_at > 0 && elapsed() >= lns.retry_at))
					&& window_lns_allowance() >= std::max(0.02, 0.5 * options.window_lns_time_fraction * elapsed()))))
				window_lns();
			result.peak_queue = std::max(result.peak_queue, queue.size() + dives.size());
			result.lower_bound = std::min(result.upper_bound, frontier_bound());
			report_progress();
			if (result.upper_bound - result.lower_bound <= gap() || limited()) break;
			// A round expands up to nodes_per_round frontier nodes serially and
			// then evaluates all their children's oracles together. One node per
			// round is the classic best-bound search. No popped node stays in
			// flight across rounds, so the frontier bound remains a certificate.
			std::vector<Family> families;
			bool stop_search = false;
			for (size_t round_slot = 0; round_slot < nodes_per_round; ++round_slot) {
			if (round_slot && ((queue.empty() && dives.empty()) || limited())) break;
			const bool diving = !dfs && (!dives.empty() || (options.dive_interval && result.nodes % options.dive_interval == 0));
			Node node;
			if (!dives.empty()) { node = std::move(dives.back()); dives.pop_back(); note_dives(); }
			else { node = queue.take(); }
            const auto sequence_parent = node.sequence.restore(index_bytes);
			++result.nodes;
			result.sequence_depth_sum += node.sequence.size();
			++result.sequence_depth_samples;
			result.max_sequence_depth = std::max(result.max_sequence_depth, node.sequence.size());
            strengthen_shared_bound(node);
			if (node.path.empty() && node.bound < result.upper_bound-gap() && !solve(node)) { queue.push(std::move(node),sequence_parent);stop_search=true;break; }
            Polygon parent_binary_dual;
            if(!cycle&&options.path_certificate_dual) {
                if(auto found=path_dual_cache.find(node.serial);found!=path_dual_cache.end()) {
                    path_dual_bytes-=found->second.capacity()*sizeof(Vector2);
                    parent_binary_dual=std::move(found->second);path_dual_cache.erase(found);
                    ++result.path_dual_cache_hits;
                }
            }
			std::vector<size_t> node_sequence;
			if (options.trace) for (auto e : node.sequence) node_sequence.push_back(e.polygon);
			if (options.trace) trace_event({
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
				if (options.trace) trace_event({
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
				if (options.trace) trace_event({
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
            if(options.multi_insertion_bound) {
                // Valid for the whole subtree, so children inherit it.
                const auto began_multi=std::chrono::steady_clock::now();
                std::vector<const Polygon *> regions,missing;
                std::vector<char> in_sequence(n,0);
                for(auto e:node.sequence) {
                    regions.push_back(e.piece==none?&hulls[e.polygon]:&pieces[e.polygon][e.piece]);
                    in_sequence[e.polygon]=1;
                }
                for(size_t j=0;j<n;++j)if(!in_sequence[j])missing.push_back(&hulls[j]);
                ++result.multi_insertion_calls;
                const double bound=cycle?cycle_multi_insertion_bound(node.path,regions,missing)
                    :path_multi_insertion_bound(path_insertion_dual(node.path,regions),node.path,regions,missing);
                result.multi_insertion_seconds+=duration(began_multi);
                if(bound>node.bound) {
                    ++result.multi_insertion_improvements;
                    result.multi_insertion_gain+=bound-node.bound;
                    node.bound=bound;
                }
                if(node.bound>=result.upper_bound-gap()) {
                    ++result.multi_insertion_prunes;++result.pruned_nodes;++result.pruned_states;++result.bound_prunes;
                    settled_bound=std::min(settled_bound,node.bound);
                    queue.restart();
                    continue;
                }
            }
			size_t chosen = none;
			double farthest = eps;
            if(!cycle&&options.path_dual_reuse) {
                node.dual=tpp_convex_cycle_dual_directions(node.path);
                if(!node.dual.empty())node.dual.pop_back();
            }
            std::vector<std::pair<double,size_t>> branch_candidates;
            std::vector<double> branch_distances(options.cycle_learned_branching?n:0);
			size_t detour_chosen = none;
			double best_detour = -1, best_detour_distance = -1;
			const auto visit_began = std::chrono::steady_clock::now();
			begin_visit_queries(node.path);
			// Exact contacts are needed only for regions that can still become the
			// farthest, or one of the K lookahead candidates (strict comparisons
			// keep the first index on ties). A small relative margin absorbs
			// rounding between the anchor bound and the exact contact.
			const bool bound_visits = options.prepared_visit_queries && options.visit_upper_bounds
				&& !node.sequence.empty() && !options.cycle_learned_branching
				&& !(cycle?options.cycle_strong_branching:options.path_strong_branching);
			const size_t kept_candidates = !cycle && options.insertion_lookahead ? options.insertion_lookahead : 1;
			std::vector<char> present;
			if (bound_visits && kept_candidates > 1) {
				present.assign(n, 0);
				for (auto e : node.sequence) present[e.polygon] = 1;
			}
			std::priority_queue<double, std::vector<double>, std::greater<>> kept_distances;
			// With bounds, visit regions by decreasing upper bound so the farthest
			// is found first and the threshold rises at once. Ties keep the
			// smallest index, exactly like the plain index-order scan below.
			std::vector<std::pair<double,size_t>> scan;
			scan.reserve(n);
			for (size_t j = 0; j < n; ++j) scan.emplace_back(bound_visits ? visit_distance_upper(j) : 0.0, j);
			if (bound_visits) std::stable_sort(scan.begin(), scan.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
			for (const auto &[upper, j] : scan) {
				if (bound_visits) {
					double threshold = farthest;
					if (kept_candidates > 1 && !present[j])
						threshold = std::min(threshold, kept_distances.size() < kept_candidates ? eps : kept_distances.top());
					if (upper * (1 + 1e-9) < threshold) { ++result.visit_bound_skips; continue; }
				}
				const double distance = visit_contact(node.path,j).distance;
				if (bound_visits && kept_candidates > 1 && !present[j] && distance > eps) {
					kept_distances.push(distance);
					if (kept_distances.size() > kept_candidates) kept_distances.pop();
				}
                if(options.cycle_learned_branching)branch_distances[j]=distance;
				if (distance > farthest || (distance == farthest && chosen != none && j < chosen)) { farthest = distance; chosen = j; }
                if(((cycle?options.cycle_strong_branching:options.path_strong_branching)||(!cycle&&options.insertion_lookahead))&&distance>eps&&
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
            if((cycle?options.cycle_strong_branching:options.path_strong_branching)&&!options.cycle_learned_branching&&chosen!=none&&
               std::none_of(node.sequence.begin(),node.sequence.end(),[&](auto e){return e.polygon==chosen;})) {
                std::sort(branch_candidates.begin(),branch_candidates.end(),std::greater<>());
                std::vector<const Polygon*> regions;
                for(auto e:node.sequence)regions.push_back(e.piece==none?&hulls[e.polygon]:&pieces[e.polygon][e.piece]);
                double strongest=-1;
                for(size_t i=0;i<std::min(size_t(3),branch_candidates.size());++i) {
                    const size_t candidate=branch_candidates[i].second;
                    const auto bounds=insertion_lower_bounds(node.path,regions,hulls[candidate],cycle,node.dual);
                    const double bound=*std::min_element(bounds.begin(),bounds.end());
                    if(bound>strongest){strongest=bound;chosen=candidate;}
                }
            }

            // Insertion lookahead: dual insertion bounds are analytic, so screen the
            // K farthest absent regions instead of only the farthest one. A region
            // with no admissible position proves the whole node cannot improve the
            // incumbent.
            std::vector<double> lookahead_bounds;
            if(!cycle&&options.insertion_lookahead&&chosen!=none&&!branch_candidates.empty()&&
               std::none_of(node.sequence.begin(),node.sequence.end(),[&](auto e){return e.polygon==chosen;})) {
                const auto began_lookahead=std::chrono::steady_clock::now();
                std::sort(branch_candidates.begin(),branch_candidates.end(),std::greater<>());
                std::vector<const Polygon*> regions;
                for(auto e:node.sequence)regions.push_back(e.piece==none?&hulls[e.polygon]:&pieces[e.polygon][e.piece]);
                const double cutoff=result.upper_bound-gap();
                // The node's dual directions do not depend on the candidate.
                const auto node_dual=node.dual.empty()?path_insertion_dual(node.path,regions):PathInsertionDual{};
                size_t best_count=none;double best_minimum=-INFINITY,dead_minimum=-INFINITY;bool dead=false;
                for(size_t i=0;i<std::min(options.insertion_lookahead,branch_candidates.size());++i) {
                    const size_t candidate=branch_candidates[i].second;
                    if(candidate!=chosen&&node.dual.empty()) {
                        // A region is dead only if every position is inadmissible:
                        // test the path segment nearest to it first and stop there.
                        const auto nearest=static_cast<size_t>(std::clamp(visit_contact(node.path,candidate).position,0.0,double(regions.size())));
                        ++result.lookahead_candidates;
                        if(std::max(node.bound,path_insertion_bound_at(node_dual,node.path,regions,hulls[candidate],nearest))<cutoff)continue;
                    }
                    auto bounds=node.dual.empty()?path_insertion_bounds(node_dual,node.path,regions,hulls[candidate])
                        :insertion_lower_bounds(node.path,regions,hulls[candidate],false,node.dual);
                    size_t admissible=0;double minimum=INFINITY;
                    for(auto &bound:bounds) {
                        bound=std::max(node.bound,bound);
                        minimum=std::min(minimum,bound);
                        admissible+=bound<cutoff;
                    }
                    ++result.lookahead_candidates;
                    if(!admissible){dead=true;dead_minimum=std::max(dead_minimum,minimum);continue;}
                    // Branching stays on the farthest region: changing it to the most
                    // constrained one grew the tree (screen of 2026-10-05).
                    if(candidate==chosen){best_count=admissible;best_minimum=minimum;lookahead_bounds=std::move(bounds);}
                }
                result.lookahead_seconds+=duration(began_lookahead);
                if(dead) {
                    // Every completion inserts each dead region somewhere, so the
                    // largest of their position minima bounds the whole subtree.
                    ++result.lookahead_prunes;++result.pruned_nodes;++result.pruned_states;++result.bound_prunes;
                    settled_bound=std::min(settled_bound,dead_minimum);
                    result.search_visit_check_seconds+=duration(visit_began);
                    queue.restart();
                    continue;
                }
                result.lookahead_changes+=chosen!=branch_candidates.front().second;
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
					if(!solve(node, true)) { queue.push(std::move(node),sequence_parent);stop_search=true;break; }
					improve(node.path, "refinement", node_sequence);
					queue.push(std::move(node),sequence_parent);
					continue;
				}
				// An unresolved numerical oracle gap must remain in the global certificate.
				queue.push(std::move(node),sequence_parent);
				stop_search=true;
				break;
			}
			if (options.trace) trace_event({
				.kind = "branch",
				.node = node.serial,
				.parent = node.parent,
				.polygon = chosen,
				.sequence = node_sequence,
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
                const bool screen_dual=!cycle&&options.path_certificate_dual&&
                    parent_binary_dual.size()==node.sequence.size()+1;
                Polygon proposals;
				auto bounds = !lookahead_bounds.empty()&&!screen_dual ? std::move(lookahead_bounds)
                    : insertion_lower_bounds(node.path, regions, hulls[chosen], cycle,
                    node.dual,screen_dual?&proposals:nullptr);
                if(screen_dual) {
                    const auto began_screen=std::chrono::steady_clock::now();
                    const auto inherited=tpp_convex_binary_dual_insertion_bounds(start,target,node.path,regions,
                        hulls[chosen],proposals,parent_binary_dual);
                    for(size_t j=0;j<inherited.size();++j) {
                        ++result.path_dual_screen_children;
                        const double cutoff=result.upper_bound-gap();
                        result.path_dual_screen_prunes+=std::max(node.bound,bounds[j])<cutoff&&inherited[j]>=cutoff;
                        bounds[j]=std::max(bounds[j],inherited[j]);
                    }
                    result.path_dual_screen_seconds+=duration(began_screen);
                }
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
					if (bound >= result.upper_bound - gap()) {
						++result.screened_nodes;
						++result.insertion_positions_pruned;
						++result.pruned_states;
						++result.bound_prunes;
						settled_bound = std::min(settled_bound, bound);
						if (options.trace) trace_event({
							.kind = "child",
							.parent = node.serial,
							.polygon = chosen,
							.position = j,
							.sequence = [&] {
								std::vector<size_t> child_sequence;
								for (auto e : node.sequence) child_sequence.push_back(e.polygon);
                                child_sequence.insert(child_sequence.begin()+j,chosen);
								return child_sequence;
							}(),
							.lower_bound = bound,
							.upper_bound = result.upper_bound,
							.pruned = true,
							.reason = "bound",
						});
						continue;
					}
					Node child{node.sequence.inserted(j,{chosen}), {}, bound, serial++};
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
			families.push_back({std::move(children), sequence_parent, diving, queue.size()});
			}
			// All children of this round, in expansion order.
			std::vector<std::pair<size_t,size_t>> round_children;
			for (size_t f = 0; f < families.size(); ++f)
				for (size_t c = 0; c < families[f].children.size(); ++c) round_children.emplace_back(f, c);
			auto child_at = [&](size_t index) -> Node & {
				return families[round_children[index].first].children[round_children[index].second];
			};
			for (size_t batch_begin = 0; batch_begin < round_children.size();) {
				import_incumbent();
				const size_t batch_end = std::min(round_children.size(), batch_begin + options.threads);
				const double batch_upper_bound = result.upper_bound;
				const double batch_cutoff = batch_upper_bound - gap_at(batch_upper_bound);
				std::vector<size_t> evaluation_children;
				std::vector<size_t> evaluation_slot(batch_end - batch_begin, none);
				const size_t available_calls = result.calls < options.max_calls
					? options.max_calls - result.calls : 0;
				for (size_t child_index = batch_begin; child_index < batch_end; ++child_index) {
					if ((cycle?options.cycle_lazy:options.lazy_oracles) || evaluation_children.size() >= available_calls || limited()) break;
                    strengthen_shared_bound(child_at(child_index));
					if (child_at(child_index).bound >= batch_cutoff) continue;
					if(!note_oracle_call(child_at(child_index), false)) break;
					evaluation_slot[child_index - batch_begin] = evaluation_children.size();
					evaluation_children.push_back(child_index);
				}

				std::vector<RelaxationResult> certified(evaluation_children.size());
				if (evaluation_children.size() == 1) {
					const auto oracle_began = std::chrono::steady_clock::now();
					certified[0] = evaluate_oracle(child_at(evaluation_children[0]), false,
						batch_upper_bound, workspace, cycle_workspace,memo_workspace);
					result.convex_oracle_wall_seconds += duration(oracle_began);
				} else if (evaluation_children.size() > 1) {
					const int worker_count = static_cast<int>(std::min(options.threads, evaluation_children.size()));
					result.parallel_oracle_batches++;
					result.parallel_oracle_calls += evaluation_children.size();
					if (parallel_workspaces.size() < static_cast<size_t>(worker_count))
						{ parallel_workspaces.resize(worker_count);parallel_cycle_workspaces.resize(worker_count);parallel_memo_workspaces.resize(worker_count); }
					for(auto &worker_workspace:parallel_workspaces) {
						worker_workspace.cache_disjoint_dispatch=options.oracle_dispatch_cache;
						worker_workspace.cache_interval_geometry=options.oracle_interval_geometry_cache;
                        worker_workspace.borrow_hybrid_geometry=options.oracle_borrow_geometry;
                        worker_workspace.bound_before_optimality=options.oracle_bound_first;
                        worker_workspace.retain_binary_dual=!cycle&&options.path_certificate_dual;
						worker_workspace.interpolated_zero_dual=options.interpolated_zero_dual;
						worker_workspace.float_recovery=options.float_recovery;worker_workspace.float_degenerate=options.float_degenerate;
						worker_workspace.trust_double=options.trust_double;
					}
					const auto oracle_batch_began = std::chrono::steady_clock::now();
					evaluate_parallel_oracles(static_cast<std::ptrdiff_t>(evaluation_children.size()), worker_count,
						[&](std::ptrdiff_t slot, int worker) {
							certified[slot] = evaluate_oracle(child_at(evaluation_children[slot]), false,
								batch_upper_bound, parallel_workspaces[worker], parallel_cycle_workspaces[worker],parallel_memo_workspaces[worker]);
						});
					result.convex_oracle_wall_seconds += duration(oracle_batch_began);
				}

				for (size_t child_index = batch_begin; child_index < batch_end; ++child_index) {
					auto &family = families[round_children[child_index].first];
					auto &child = child_at(child_index);
					const bool pruned_by_new_incumbent = child.bound >= result.upper_bound - gap();
					if (pruned_by_new_incumbent) ++result.sibling_bound_prunes;
					const size_t slot = evaluation_slot[child_index - batch_begin];
					if (slot != none) {
						record_oracle_result(child, false, batch_cutoff, certified[slot]);
						if (!pruned_by_new_incumbent && !child.path.empty()) {
							std::vector<size_t> child_sequence;
							if (options.trace) for (auto e : child.sequence) child_sequence.push_back(e.polygon);
							improve(child.path, "oracle", child_sequence);
						}
					}
					const bool queued = child.bound < result.upper_bound - gap();
					const auto child_sequence = [&] {
						std::vector<size_t> sequence;
						if (options.trace) for (auto e : child.sequence) sequence.push_back(e.polygon);
						return sequence;
					}();
					if (options.trace) trace_event({
						.kind = "child",
						.node = child.serial,
						.parent = child.parent,
						.polygon = child.branch_polygon,
						.piece = child.branch_piece,
						.position = child.branch_position,
						.sequence = child_sequence,
						.path = child.path,
						.lower_bound = child.bound,
						.upper_bound = result.upper_bound,
						.pruned = !queued,
						.reason = queued ? "queued" : "bound_after_incumbent",
					});
					if (queued) {
						++result.children_queued;
                        queue.freeze_child(child,family.parent);
						if (family.diving && (!family.dive || child.bound < family.dive->bound)) {
							if (family.dive) queue.push(std::move(*family.dive));
							family.dive = std::move(child);
						} else queue.push(std::move(child));
					} else {
						++result.pruned_states;
						++result.incumbent_prunes;
						settled_bound = std::min(settled_bound, child.bound);
					}
				}
				batch_begin = batch_end;
			}
            for (auto &family : families) {
				queue.finish_branch(family.first_child);
				if (family.dive) dives.push_back(std::move(*family.dive));
			}
			note_dives();
            if (stop_search) break;
		}
        queue.report(result);
        result.segment_visit_queries=segment_cache.queries;result.segment_visit_hits=segment_cache.hits;
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
			: (options.stop_requested && options.stop_requested()) ? UnorderedTppTermination::Interrupted
            : (control ? control->calls.load(std::memory_order_relaxed)>=control->max_calls : result.calls >= options.max_calls) ? UnorderedTppTermination::CallLimit
			: (elapsed() >= options.max_seconds || (control && control->elapsed()>=control->max_seconds)) ? UnorderedTppTermination::TimeLimit
			: UnorderedTppTermination::NumericalLimit;
		std::vector<std::pair<double, size_t>> visits;
		const auto final_visits_began = std::chrono::steady_clock::now();
		begin_visit_queries(result.path);
		for (size_t j = 0; j < n; ++j) visits.emplace_back(visit_contact(result.path,j).position, j);
		result.finalization_visit_check_seconds += duration(final_visits_began);
		std::sort(visits.begin(), visits.end());
		for (auto [position, j] : visits) result.order.push_back(j);
		result.finalization_seconds = duration(finalization_began);
		result.visit_check_seconds = result.heuristic_visit_check_seconds + result.search_visit_check_seconds
			+ result.finalization_visit_check_seconds;
		result.seconds = elapsed();
		if (options.trace) trace_event({
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
				if (polygon.empty()) throw std::invalid_argument("Expected nonempty regions.");
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
		if (options.progress) {
			// The search runs on normalized coordinates; report original lengths.
			normalized_options.progress = [report = options.progress, divisor](UnorderedTppProgress progress) {
				if (std::isfinite(progress.lower_bound)) progress.lower_bound *= divisor;
				if (std::isfinite(progress.upper_bound)) progress.upper_bound *= divisor;
				report(progress);
			};
		}
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
		// The affine round trip may move a fixed endpoint by an ulp. Restore
		// the actual input endpoints and account for that displacement in bounds.
		double endpoint_correction=0;
		if(!cycle && result.path.size()>=2) {
			endpoint_correction=result.path.front().distance_to(start)+result.path.back().distance_to(target);
			if(endpoint_correction>0)endpoint_correction=std::nextafter(endpoint_correction,std::numeric_limits<double>::infinity());
			result.path.front()=start; result.path.back()=target;
			if(endpoint_correction>0) {
				auto upper=[&](double &v) {if(std::isfinite(v))v=std::nextafter(v+endpoint_correction,std::numeric_limits<double>::infinity());};
				result.lower_bound=std::max(start.distance_to(target),std::nextafter(result.lower_bound-endpoint_correction,-std::numeric_limits<double>::infinity()));
				result.initial_lower_bound=std::max(start.distance_to(target),std::nextafter(result.initial_lower_bound-endpoint_correction,-std::numeric_limits<double>::infinity()));
				upper(result.upper_bound); upper(result.initial_upper_bound); upper(result.initial_length);
				upper(result.incumbent_length); upper(result.first_best_update_length);
			}
		}
		for (auto &event : result.trace) {
			for (auto &point : event.path) point = {
				std::fma(point.x, divisor, center.x),
				std::fma(point.y, divisor, center.y),
			};
			if(!cycle && event.path.size()>=2) {event.path.front()=start; event.path.back()=target;}
			if (std::isfinite(event.lower_bound)) event.lower_bound *= divisor;
			if (std::isfinite(event.upper_bound)) event.upper_bound *= divisor;
			if (std::isfinite(event.length)) event.length *= divisor;
			if(endpoint_correction>0) {
				if(std::isfinite(event.lower_bound))event.lower_bound=std::max(start.distance_to(target),std::nextafter(event.lower_bound-endpoint_correction,-std::numeric_limits<double>::infinity()));
				if(std::isfinite(event.upper_bound))event.upper_bound=std::nextafter(event.upper_bound+endpoint_correction,std::numeric_limits<double>::infinity());
				if(std::isfinite(event.length) && event.path.size()>=2)event.length=path_length(event.path);
			}
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
		if(endpoint_correction>0) {
			result.exact=result.upper_bound-result.lower_bound<=options.absolute_gap+options.relative_gap*std::abs(result.upper_bound);
			if(result.exact)result.termination=UnorderedTppTermination::Optimal;
			else if(result.termination==UnorderedTppTermination::Optimal)result.termination=UnorderedTppTermination::NumericalLimit;
		}
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
                local.progress_worker=index;
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
			: options.stop_requested && options.stop_requested()?UnorderedTppTermination::Interrupted
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
        sum(&UnorderedTppSolveResult::oracle_dispatch_pair_queries);
        sum(&UnorderedTppSolveResult::oracle_dispatch_pair_cache_hits);
        sum(&UnorderedTppSolveResult::oracle_dispatch_pair_exact_checks);
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
        sum(&UnorderedTppSolveResult::cycle_proposal_calls);
        sum(&UnorderedTppSolveResult::cycle_proposal_accepts);
        sum(&UnorderedTppSolveResult::cycle_primal_start_candidates);
        sum(&UnorderedTppSolveResult::cycle_primal_start_improvements);
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
        sum(&UnorderedTppSolveResult::path_dual_retained);
        sum(&UnorderedTppSolveResult::path_dual_cache_hits);
        sum(&UnorderedTppSolveResult::path_dual_cache_evictions);
        sum(&UnorderedTppSolveResult::path_dual_peak_bytes);
        sum(&UnorderedTppSolveResult::path_dual_screen_children);
        sum(&UnorderedTppSolveResult::path_dual_screen_prunes);
        sum(&UnorderedTppSolveResult::path_dual_screen_seconds);
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
        sum(&UnorderedTppSolveResult::peak_sequence_storage_bytes);
        sum(&UnorderedTppSolveResult::peak_frontier_node_bytes);
        sum(&UnorderedTppSolveResult::peak_sequence_records);
        sum(&UnorderedTppSolveResult::sequence_reconstructions);
        sum(&UnorderedTppSolveResult::sequence_depth_sum);
        sum(&UnorderedTppSolveResult::sequence_depth_samples);
        sum(&UnorderedTppSolveResult::incumbent_updates);
        sum(&UnorderedTppSolveResult::best_updates);
        sum(&UnorderedTppSolveResult::decomposed_polygons);
        sum(&UnorderedTppSolveResult::convex_pieces_generated);
        sum(&UnorderedTppSolveResult::fallback_calls);
        sum(&UnorderedTppSolveResult::oracle_unverified_fallbacks);
        sum(&UnorderedTppSolveResult::oracle_interval_bound_calls);
        sum(&UnorderedTppSolveResult::oracle_float_calls);
        sum(&UnorderedTppSolveResult::oracle_float_fallbacks);
        sum(&UnorderedTppSolveResult::oracle_rational_calls);
        sum(&UnorderedTppSolveResult::oracle_trusted_calls);
        sum(&UnorderedTppSolveResult::oracle_exact_replay_calls);
        sum(&UnorderedTppSolveResult::oracle_filtered_calls);
        sum(&UnorderedTppSolveResult::oracle_touching_calls);
        sum(&UnorderedTppSolveResult::rational_membership_predicates);
        sum(&UnorderedTppSolveResult::exact_polygon_preparations);
        sum(&UnorderedTppSolveResult::oracle_contracted_bound_calls);
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
		sum(&UnorderedTppSolveResult::initial_relocation_seconds);
		sum(&UnorderedTppSolveResult::initial_relocation_moves);
        sum(&UnorderedTppSolveResult::visit_bound_skips);
        sum(&UnorderedTppSolveResult::lookahead_candidates);
        sum(&UnorderedTppSolveResult::lookahead_prunes);
        sum(&UnorderedTppSolveResult::lookahead_changes);
        sum(&UnorderedTppSolveResult::lookahead_seconds);
        sum(&UnorderedTppSolveResult::multi_insertion_calls);
        sum(&UnorderedTppSolveResult::multi_insertion_improvements);
        sum(&UnorderedTppSolveResult::multi_insertion_prunes);
        sum(&UnorderedTppSolveResult::multi_insertion_seconds);
        sum(&UnorderedTppSolveResult::multi_insertion_gain);
        sum(&UnorderedTppSolveResult::window_lns_rounds);
        sum(&UnorderedTppSolveResult::window_lns_subproblems);
        sum(&UnorderedTppSolveResult::window_lns_improvements);
        sum(&UnorderedTppSolveResult::window_lns_calls);
        sum(&UnorderedTppSolveResult::window_lns_seconds);
        sum(&UnorderedTppSolveResult::window_lns_gain);
        sum(&UnorderedTppSolveResult::initial_sampled_extra_points);
        sum(&UnorderedTppSolveResult::initial_sampling_work_budget);
        sum(&UnorderedTppSolveResult::initial_convex_refinement_calls);
        sum(&UnorderedTppSolveResult::initial_convex_refinement_seconds);
        sum(&UnorderedTppSolveResult::search_seconds);
        sum(&UnorderedTppSolveResult::finalization_seconds);
        sum(&UnorderedTppSolveResult::convex_oracle_seconds);
        sum(&UnorderedTppSolveResult::convex_dispatch_seconds);
        sum(&UnorderedTppSolveResult::convex_bound_evaluation_seconds);
        sum(&UnorderedTppSolveResult::convex_proposal_preparation_seconds);
        sum(&UnorderedTppSolveResult::oracle_profiled_calls);
        sum(&UnorderedTppSolveResult::cycle_construction_seconds);
        sum(&UnorderedTppSolveResult::cycle_certification_seconds);
        sum(&UnorderedTppSolveResult::cycle_rational_recovery_seconds);
        sum(&UnorderedTppSolveResult::cycle_interval_proof_seconds);
        sum(&UnorderedTppSolveResult::cycle_polish_seconds);
        sum(&UnorderedTppSolveResult::cycle_polish_calls);
        sum(&UnorderedTppSolveResult::cycle_polish_newton_iterations);
        sum(&UnorderedTppSolveResult::oracle_fallback_call_seconds);
        result.oracle_max_call_seconds=std::max(runs[0].oracle_max_call_seconds,runs[1].oracle_max_call_seconds);
        for(size_t i=0;i<result.oracle_call_histogram.size();++i) {
            result.oracle_call_histogram[i]=runs[0].oracle_call_histogram[i]+runs[1].oracle_call_histogram[i];
            result.oracle_seconds_histogram[i]=runs[0].oracle_seconds_histogram[i]+runs[1].oracle_seconds_histogram[i];
        }
        sum(&UnorderedTppSolveResult::convex_oracle_wall_seconds);
        sum(&UnorderedTppSolveResult::convex_geometric_solver_seconds);
        sum(&UnorderedTppSolveResult::convex_certificate_verification_seconds);
        sum(&UnorderedTppSolveResult::convex_contact_materialization_seconds);
        sum(&UnorderedTppSolveResult::convex_fallback_seconds);
        sum(&UnorderedTppSolveResult::convex_fallback_long_double_seconds);
        sum(&UnorderedTppSolveResult::convex_fallback_extended_precision_seconds);
        sum(&UnorderedTppSolveResult::decomposition_seconds);
        sum(&UnorderedTppSolveResult::visit_check_seconds);
		sum(&UnorderedTppSolveResult::visit_query_evaluations);
		sum(&UnorderedTppSolveResult::visit_query_cache_hits);
        sum(&UnorderedTppSolveResult::segment_visit_queries);
        sum(&UnorderedTppSolveResult::segment_visit_hits);
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
		// A tour visits every point region, so it passes through the point.
		// Rotated to start there it is a closed endpoint path p -> p through
		// the other regions, of the same length, and every such path is a tour.
		// The endpoint search, which also accepts points and segments, therefore
		// solves these instances exactly; the others keep the positive-area
		// cycle contract below.
		std::optional<size_t> anchor;
		for(size_t i=0;i<polygons.size()&&!anchor;++i)
			if(!polygons[i].empty()&&polygons[i].front().is_finite()&&std::all_of(polygons[i].begin(),polygons[i].end(),
				[&](Vector2 v){return v.x==polygons[i].front().x&&v.y==polygons[i].front().y;}))anchor=i;
		if(anchor&&polygons.size()>=2&&options.cycle_point_anchor) {
			const Vector2 point=polygons[*anchor].front();
			std::vector<Polygon> rest;rest.reserve(polygons.size()-1);
			for(size_t i=0;i<polygons.size();++i)if(i!=*anchor)rest.push_back(polygons[i]);
			auto anchored=options;
			if(options.initial_path) {
				// Rotate the closed initial tour so that it starts and ends at the point.
				const auto &tour=*options.initial_path;
				if(tour.size()<2||tour.front().distance_to(tour.back())>options.feasibility_tolerance)
					throw std::invalid_argument("Initial path needs finite points and matching endpoints.");
				const Polygon single{point};
				std::optional<size_t> through;
				for(size_t i=0;i+1<tour.size()&&!through;++i)
					if(contact(Polygon{tour[i],tour[i+1]},single,options.feasibility_tolerance).distance<=options.feasibility_tolerance)through=i;
				if(!through)throw std::invalid_argument("Initial path does not visit every polygon.");
				Polygon rotated{point};
				for(size_t i=*through+1;i+1<tour.size();++i)rotated.push_back(tour[i]);
				for(size_t i=0;i<=*through;++i)rotated.push_back(tour[i]);
				rotated.push_back(point);
				anchored.initial_path=std::move(rotated);
			}
			auto result=options.portfolio?solve_portfolio(point,point,rest,anchored,false)
				:solve_unordered(point,point,rest,anchored,false);
			auto remap=[&](std::vector<size_t> &indices) {
				for(auto &j:indices)if(j<rest.size()&&j>=*anchor)++j;
			};
			remap(result.order);
			result.order.insert(result.order.begin(),*anchor);
			for(auto &event:result.trace){remap(event.sequence);remap(event.order);}
			result.cycle_point_anchor=*anchor;
			return result;
		}
		const Vector2 seed=polygons.empty()?Vector2{}:polygons.front().front();
		return options.portfolio?solve_portfolio(seed,seed,polygons,options,true)
            :solve_unordered(seed,seed,polygons,options,true);
	}
}
