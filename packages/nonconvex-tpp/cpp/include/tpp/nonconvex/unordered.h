#pragma once

#include "tpp/geometry/vec2.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tpp {

	enum class UnorderedSearchStrategy { BestBoundDive, DfsBfs };
	enum class UnorderedSequenceStorage { Native, Packed, Deltas };

	// A snapshot of a running search, for long-run monitoring. Reading it never
	// changes the search.
	struct UnorderedTppProgress {
		// 0 unless the portfolio runs two searches; each reports on its own.
		size_t worker = 0;
		double elapsed_seconds = 0.0;
		double lower_bound = 0.0;
		double upper_bound = std::numeric_limits<double>::infinity();
		// Oracle calls, summed over both portfolio workers.
		size_t calls = 0;
		size_t nodes = 0;
		// Frontier nodes and dive chains still to expand, and their peak so far.
		size_t open_nodes = 0;
		size_t peak_open_nodes = 0;
		size_t pruned_nodes = 0;
		// Deepest partial visiting sequence expanded so far.
		size_t max_sequence_depth = 0;
		size_t region_count = 0;
	};

	struct UnorderedTppSolveOptions {
		size_t max_calls = std::numeric_limits<size_t>::max();
		double max_seconds = std::numeric_limits<double>::infinity();
		// Number of worker threads for evaluating children of one search node.
		size_t threads = 1;
		double absolute_gap = 1e-7;
		double relative_gap = 1e-9;
		double feasibility_tolerance = 1e-8;
		size_t dive_interval = 1;
		// Try Fekete et al.'s endpoint-distance sum for the first branch.
		bool endpoint_sum_root = false;
		// Start from the region with the largest one-region convex detour.
		bool detour_root = false;
		bool bidirectional_initial_heuristic = false;
		// Add perimeter-spaced candidates using the approximation work budget.
		bool sampled_perimeter_initial_heuristic = false;
		// Polish the best initial route with one convex piece per visited region.
		bool convex_initial_refinement = false;
		// Relocate one region at a time and optimize its contact at the new slot.
		bool relocate_initial_heuristic = false;
		// Iterated local search on the initial tour before the B&B, for this
		// fraction of the remaining time (0 disables). Local search: contact
		// sweeps, 2-opt and Or-opt; perturbation: double bridges; tours near the
		// best are reoptimized exactly in windows. It only supplies validated
		// upper bounds. The defaults below were chosen on Paula's TSPN instances
		// (docs/algorithms/unordered-tpp.md).
		double primal_ils_fraction = 0.0;
		// Or-opt: relocate blocks of 1..L consecutive regions (L = 1 is
		// single-region relocation), optionally also reversed.
		size_t primal_ils_block = 3;
		bool primal_ils_reverse = false;
		// Or-opt candidate lists: only gaps next to one of the K regions
		// closest to a block end are tried (0: every gap).
		size_t primal_ils_candidates = 10;
		// Also swap pairs of non-adjacent regions in the local search.
		bool primal_ils_swap = false;
		// Random double-bridge moves applied per perturbation.
		size_t primal_ils_kicks = 1;
		// Threshold acceptance with restarts (Paula's ILS-BCD): accept within
		// eta of the best, eta *= 0.95 after every 10 iterations without a new
		// best, and below 1e-4 reset eta to 0.01 and the current tour to the
		// best. Off: record-to-record slack of 2% shrinking linearly with time.
		bool primal_ils_reheat = true;
		// Reoptimize tours within this fraction of the best (0: only new best
		// tours, and only by primal_ils_reorder).
		double primal_ils_polish = 0.01;
		// Reoptimize windows of this many consecutive regions, order and
		// contacts, with this exact search between the fixed contacts around
		// them, half a window apart (0: off).
		size_t primal_ils_reorder = 8;
		// Without primal_ils_reorder, polish contacts only (fixed order, exact
		// convex oracle on the pieces holding them) in windows of this many
		// regions; 0 polishes the whole tour at once.
		size_t primal_ils_window = 0;
		// Stop the ILS after this many consecutive iterations without a new
		// best tour (0: only the time fraction stops it). Required for the ILS
		// to run without a time limit.
		size_t primal_ils_stagnation = 0;
		// Added to the ILS random seed (independent runs).
		uint64_t primal_ils_seed = 0;
		// A feasible start-to-target path, including both endpoints. When present,
		// it replaces the initial heuristic and supplies only an upper bound.
		std::optional<std::vector<Vector2>> initial_path;
		// Internal relaxations may stop at this relative oracle gap. Feasible
		// leaves are refined to the requested global gap before certification.
		double oracle_relative_gap = 1e-6;
		// Reuse exact pair classification, independent of visit order/endpoints.
		bool oracle_dispatch_cache = true;
		bool oracle_interval_geometry_cache = true;
		// Reuse immutable edges/segments and contacts for the last binary path.
		bool prepared_visit_queries = true;
		// Skip exact contacts of regions that cannot be the farthest (or among
		// the lookahead candidates): a point of each region from its previous
		// contact bounds its distance from above. Same branching decisions.
		bool visit_upper_bounds = true;
		bool interpolated_zero_dual = false;
		// Fixed-endpoint oracle calls: when the interval proof fails, a binary64
		// interior-point polish with interval-certified bounds replaces the
		// exact replay and recoveries (DynamicConvexTppWorkspace::float_recovery);
		// rational arithmetic remains for open calls and zero gaps.
		bool float_recovery = true;
		// Diagnostic only, unsafe: fixed-endpoint oracle values are the
		// uncertified binary64 trace length (no lower-bound proof), to measure
		// the cost of certification against numerical solvers.
		bool trust_double = false;
        bool oracle_borrow_geometry = true;
        bool oracle_bound_first = false;
        bool lazy_oracles = false;
        bool segment_visit_cache = true;
        bool path_dual_reuse = false;
        bool path_certificate_dual = false;
        bool path_strong_branching = false;
        // Experimental exact large-neighbourhood search (open paths only).
        // Windows of `window_lns_size` consecutive incumbent contacts are
        // re-solved by this same B&B with both window endpoints fixed, after the
        // initial heuristic and whenever the search finds a better incumbent.
        // Only upper bounds change; all LNS oracle calls count in `calls` and
        // the LNS uses at most `window_lns_time_fraction` of the elapsed time.
        // Experimental: screen this many farthest absent regions with the dual
        // insertion bounds; prune the node if one has no admissible position.
        // Branching is unchanged. Zero disables (default).
        size_t insertion_lookahead = 0;
        // Experimental: raise each expanded node's bound with the insertion
        // gains of all absent regions under the node's contact-direction dual,
        // priced over every assignment of regions to gaps (paths and cycles;
        // see docs/algorithms/unordered-tpp.md).
        bool multi_insertion_bound = false;
        // Experimental: with threads > 1, expand up to `threads` best-bound
        // nodes per round and evaluate all their children in parallel, instead
        // of parallelizing only the siblings of one node (best-bound search only).
        bool parallel_nodes = false;
        bool window_lns = false;
        size_t window_lns_size = 8;
        size_t window_lns_max_size = 32;
        double window_lns_time_fraction = 0.1;
		// Record an explanatory execution trace. Disabled by default so normal
		// benchmark runs keep the same memory and timing behavior.
		bool trace = false;
		// Cooperative interruption checked between search operations. The active
		// oracle/decomposition call is allowed to finish before the frontier stops.
		std::function<bool()> stop_requested;
		// Periodic status for long runs: `progress` is called from the search loop
		// at most every `progress_interval_seconds` (checked between search
		// operations, so a single long oracle call delays it). Disabled when the
		// interval is not positive or no callback is set. With the portfolio, both
		// searches call it from their own threads.
		double progress_interval_seconds = 0.0;
		std::function<void(const UnorderedTppProgress &)> progress;
		size_t progress_worker = 0;
        UnorderedSearchStrategy search_strategy = UnorderedSearchStrategy::BestBoundDive;
        // Frontier sequences only. Packed automatically selects 8/16/32/64-bit
        // indices; Deltas retains insertion/piece changes in a recycled arena.
        UnorderedSequenceStorage sequence_storage = UnorderedSequenceStorage::Packed;
        // Two independent searches, one oracle thread each. max_calls is shared;
        // max_seconds is one wall deadline. Requires threads == 1.
        bool portfolio = false;
        bool portfolio_share_incumbents = true;
        // Independently selectable cycle accelerations for reproducible ablation.
        bool cycle_cache = false;
        bool cycle_dual_reuse = false;
        bool cycle_active_features = false;
        bool cycle_lazy = false;
        bool cycle_separated_root = false;
        bool cycle_strong_branching = false;
        bool cycle_one_tree = false;
        bool cycle_learned_branching = false;
        bool cycle_memo = false;
        bool cycle_bound_first = false;
        bool cycle_dual_screen = false;
        bool cycle_interval_certificate = false;
        // Try a finite floating proposal before exact recovery. Its certified
        // interval must meet the existing oracle gap or pruning cutoff.
        bool cycle_proposal_bound = false;
        // Diversify the existing greedy/2-opt/contact heuristic from up to
        // seven additional original vertices; only validated upper bounds.
        bool cycle_primal_starts = false;
        bool cycle_share_bounds = false;
        // TSPN with a point region: solve the closed endpoint path through it.
        bool cycle_point_anchor = true;
        // Diagnostic JSONL, including every in-flight cycle input; empty disables I/O.
        std::string oracle_capture_file;
        // Sampling for long runs: keep every k-th call (0: none) plus the calls
        // that took at least this long. The defaults keep every call, written
        // before it runs; any sampling writes a call only after it returns.
        std::size_t oracle_capture_every = 1;
        double oracle_capture_min_seconds = 0;
	};

	enum class UnorderedTppTermination { Optimal, CallLimit, TimeLimit, NumericalLimit, PortfolioStopped, Interrupted };

	struct UnorderedTppTraceEvent {
		std::string kind;
		size_t node = std::numeric_limits<size_t>::max();
		size_t parent = std::numeric_limits<size_t>::max();
		size_t polygon = std::numeric_limits<size_t>::max();
		size_t piece = std::numeric_limits<size_t>::max();
		size_t position = std::numeric_limits<size_t>::max();
		size_t pass = std::numeric_limits<size_t>::max();
		std::vector<size_t> sequence;
		std::vector<size_t> order;
		std::vector<Vector2> path;
		double lower_bound = std::numeric_limits<double>::infinity();
		double upper_bound = std::numeric_limits<double>::infinity();
		double length = std::numeric_limits<double>::infinity();
		bool pruned = false;
		std::string source;
		std::string reason;
	};

    struct UnorderedPortfolioRun {
        std::string strategy;
        size_t calls = 0, nodes = 0, incumbent_imports = 0;
        double lower_bound = 0, upper_bound = std::numeric_limits<double>::infinity();
        double seconds = 0;
        UnorderedTppTermination termination = UnorderedTppTermination::NumericalLimit;
        std::string error;
    };

	struct UnorderedTppSolveResult {
		std::vector<Vector2> path;
		std::vector<size_t> order;
		std::vector<UnorderedTppTraceEvent> trace;
        size_t portfolio_workers = 1;
        size_t portfolio_winner = std::numeric_limits<size_t>::max();
        size_t portfolio_incumbent_publications = 0, portfolio_incumbent_imports = 0;
        double portfolio_proof_seconds = 0, portfolio_join_seconds = 0;
        std::vector<UnorderedPortfolioRun> portfolio_runs;
		double lower_bound = 0.0;
		double upper_bound = std::numeric_limits<double>::infinity();
		double initial_lower_bound = 0.0;
		double initial_upper_bound = std::numeric_limits<double>::infinity();
		// Comparison metrics: initial_length is the heuristic approximation;
		// incumbent_length is the best feasible value before B&B improvements.
		double initial_length = std::numeric_limits<double>::infinity();
		double incumbent_length = std::numeric_limits<double>::infinity();
		double first_best_update_length = std::numeric_limits<double>::infinity();
		double final_length = std::numeric_limits<double>::infinity();
		double first_incumbent_seconds = std::numeric_limits<double>::infinity();
		double initial_gap_percent = std::numeric_limits<double>::infinity();
		double final_absolute_gap = std::numeric_limits<double>::infinity();
		double final_relative_gap = std::numeric_limits<double>::infinity();
		bool exact = false;
		UnorderedTppTermination termination = UnorderedTppTermination::NumericalLimit;
		size_t threads = 1;
		size_t calls = 0;
		size_t parallel_oracle_calls = 0;
		size_t parallel_oracle_batches = 0;
		size_t relaxation_calls = 0;
		size_t refinement_calls = 0;
		size_t complete_order_oracle_calls = 0;
		size_t complete_piece_oracle_calls = 0;
		size_t oracle_cutoff_calls = 0;
		size_t oracle_dual_cutoff_prunes = 0;
		size_t oracle_dispatch_pair_queries = 0;
		size_t oracle_dispatch_pair_cache_hits = 0;
		size_t oracle_dispatch_pair_exact_checks = 0;
		size_t oracle_interval_bound_calls = 0;
		size_t oracle_float_calls = 0;
		size_t oracle_float_fallbacks = 0;
		// Calls closed by an exact-rational stage (paths: exact replay/KKT,
		// filtered recovery, touching-disjoint recovery, full fallback =
		// fallback_calls); for cycles, calls with a rational recovery.
		size_t oracle_rational_calls = 0;
		size_t oracle_trusted_calls = 0;
		size_t oracle_exact_replay_calls = 0;
		size_t oracle_filtered_calls = 0;
		size_t oracle_touching_calls = 0;
		size_t rational_membership_predicates = 0;
		size_t exact_polygon_preparations = 0;
		size_t oracle_contracted_bound_calls = 0;
		size_t screened_nodes = 0;
        size_t path_dual_retained = 0, path_dual_cache_hits = 0, path_dual_cache_evictions = 0;
        size_t path_dual_screen_children = 0, path_dual_screen_prunes = 0;
        size_t path_dual_peak_bytes = 0;
        double path_dual_screen_seconds = 0;
        size_t one_tree_calls = 0, one_tree_cache_hits = 0, one_tree_iterations = 0;
        size_t one_tree_distance_queries = 0, one_tree_improvements = 0;
        double one_tree_seconds = 0;
        size_t learned_branch_observations = 0, learned_branch_decisions = 0, learned_branch_changes = 0;
        size_t cycle_memo_queries = 0, cycle_memo_repeated = 0, cycle_memo_hits = 0;
        size_t cycle_certificate_cutoff_skips = 0, cycle_initial_contact_checks = 0, cycle_initial_contact_accepts = 0;
        size_t cycle_certificate_interval_uses = 0;
        size_t cycle_proposal_calls = 0, cycle_proposal_accepts = 0;
        size_t cycle_primal_start_candidates = 0, cycle_primal_start_improvements = 0;
        size_t cycle_dual_screen_children = 0, cycle_dual_screen_prunes = 0;
        double cycle_dual_screen_seconds = 0;
        size_t cycle_shared_bound_queries = 0, cycle_shared_bound_hits = 0, cycle_shared_bound_improvements = 0, cycle_shared_bound_prunes = 0;
        double cycle_shared_bound_seconds = 0;
		// Generated children skipped before the oracle after a sibling improved the incumbent.
		size_t sibling_bound_prunes = 0;
		size_t nodes = 0;
		size_t partial_states_created = 0;
		size_t children_generated = 0;
		size_t children_queued = 0;
		size_t pruned_nodes = 0;
		size_t pruned_states = 0;
		size_t bound_prunes = 0;
		size_t incumbent_prunes = 0;
		size_t insertion_positions_considered = 0;
		size_t insertion_positions_pruned = 0;
		size_t branch_events = 0;
		size_t total_branching = 0;
		size_t max_observed_branching = 0;
		size_t max_sequence_depth = 0;
		size_t sequence_depth_sum = 0;
		size_t sequence_depth_samples = 0;
		// Total improvements includes heuristic/direct updates. best_updates is
		// comparable to the fixed-order benchmark and excludes those initial updates.
		size_t incumbent_updates = 0;
		size_t best_updates = 0;
		size_t decomposed_polygons = 0;
		size_t convex_pieces_generated = 0;
		size_t convex_pieces_min = std::numeric_limits<size_t>::max();
		size_t convex_pieces_max = 0;
		size_t polygon_vertices_total = 0;
		// TSPN only: index of the point region the tour was anchored at (the
		// cycle was solved as a closed endpoint path through it), or max.
		size_t cycle_point_anchor = std::numeric_limits<size_t>::max();
		size_t polygon_vertices_min = std::numeric_limits<size_t>::max();
		size_t polygon_vertices_max = 0;
		double order_space_log2 = 0.0;
		size_t fallback_calls = 0;
		// Full rational solves whose path failed the exact KKT certificate; their
		// bounds remain certified but need not close the gap.
		size_t oracle_unverified_fallbacks = 0;
		size_t fallback_geometric_path_invalid_calls = 0;
		size_t fallback_certificate_gap_calls = 0;
		size_t fallback_locator_exception_calls = 0;
		size_t fallback_nonfinite_calls = 0;
		size_t fallback_contact_construction_calls = 0;
		size_t fallback_membership_ordering_calls = 0;
		size_t fallback_local_optimality_calls = 0;
		size_t fallback_coincident_contact_calls = 0;
		size_t predicate_exact_evaluations = 0;
		size_t extended_precision_calls = 0;
		size_t oracle_time_limit_calls = 0;
		size_t repaired_geometric_path_calls = 0;
		size_t insertion_branches = 0;
		size_t decomposition_branches = 0;
		size_t peak_queue = 0;
        size_t node_index_bits = 0;
        UnorderedSequenceStorage sequence_storage = UnorderedSequenceStorage::Packed;
        // Reserved sequence buffers/arena bytes, excluding inline node headers,
        // current siblings, allocator metadata, paths and oracle caches.
        size_t peak_sequence_storage_bytes = 0;
        size_t peak_frontier_node_bytes = 0;
        size_t sequence_history_record_bytes = 0;
        size_t peak_sequence_records = 0;
        size_t sequence_reconstructions = 0;
		double seconds = 0.0;
		double preprocessing_seconds = 0.0;
		double initial_heuristic_seconds = 0.0;
		double initial_relocation_seconds = 0.0;
		size_t initial_relocation_moves = 0;
		size_t primal_ils_iterations = 0, primal_ils_improvements = 0;
		double primal_ils_seconds = 0.0;
		size_t primal_ils_polish_calls = 0, primal_ils_polish_improvements = 0;
		double primal_ils_polish_seconds = 0.0;
		// (seconds, length) of every accepted incumbent, in order.
		std::vector<std::pair<double, double>> incumbent_history;
		size_t primal_ils_reorder_calls = 0, primal_ils_reorder_improvements = 0;
		double primal_ils_reorder_seconds = 0.0;
		// Exact window LNS: sweeps, solved windows, accepted improvements,
		// oracle calls spent inside windows and total length removed.
		size_t visit_bound_skips = 0;
		size_t lookahead_candidates = 0;
		size_t lookahead_prunes = 0;
		size_t lookahead_changes = 0;
		double lookahead_seconds = 0.0;
		size_t multi_insertion_calls = 0;
		size_t multi_insertion_improvements = 0;
		size_t multi_insertion_prunes = 0;
		double multi_insertion_seconds = 0.0;
		double multi_insertion_gain = 0.0;
		size_t window_lns_rounds = 0;
		size_t window_lns_subproblems = 0;
		size_t window_lns_improvements = 0;
		size_t window_lns_calls = 0;
		double window_lns_seconds = 0.0;
		double window_lns_gain = 0.0;
		double initial_sampling_work_budget = 0.0;
		size_t initial_sampled_extra_points = 0;
		size_t initial_convex_refinement_calls = 0;
		double initial_convex_refinement_seconds = 0.0;
		bool initial_convex_refinement_improved = false;
		bool initial_convex_refinement_time_limited = false;
		std::string initial_convex_refinement_error;
		double search_seconds = 0.0;
		double finalization_seconds = 0.0;
		double convex_oracle_seconds = 0.0;
		// Completed search/refinement requests, including memo hits; excludes
		// optional initial polishing and failed/in-flight requests.
		size_t oracle_profiled_calls = 0;
		double oracle_max_call_seconds = 0.0;
		// Entire calls that used a fallback, NOT exclusive recovery time.
		double oracle_fallback_call_seconds = 0.0;
		// Exclusive buckets: <=10us, 100us, 1ms, 10ms, 100ms, 1s, >1s.
		std::array<size_t, 7> oracle_call_histogram{};
		std::array<double, 7> oracle_seconds_histogram{};
		// Sum of wall time across batches; unlike convex_oracle_seconds, parallel
		// evaluations in one batch are counted once here.
		double convex_oracle_wall_seconds = 0.0;
		double convex_geometric_solver_seconds = 0.0;
		double convex_dispatch_seconds = 0.0;
		double convex_bound_evaluation_seconds = 0.0;
		double convex_proposal_preparation_seconds = 0.0;
		double convex_certificate_verification_seconds = 0.0;
        // Exclusive cycle work, including cooperatively interrupted requests.
        double cycle_construction_seconds = 0, cycle_certification_seconds = 0, cycle_rational_recovery_seconds = 0;
		double convex_contact_materialization_seconds = 0.0;
		double convex_fallback_seconds = 0.0;
		double convex_fallback_long_double_seconds = 0.0;
		double convex_fallback_extended_precision_seconds = 0.0;
		double decomposition_seconds = 0.0;
		double visit_check_seconds = 0.0;
		size_t visit_query_evaluations = 0, visit_query_cache_hits = 0;
        size_t segment_visit_queries = 0, segment_visit_hits = 0;
		double heuristic_visit_check_seconds = 0.0;
		double search_visit_check_seconds = 0.0;
		double finalization_visit_check_seconds = 0.0;
		double search_maintenance_seconds = 0.0;
	};

	// Simple polygons, either orientation. Endpoints are fixed, including start == target.
	// Exact means the requested floating-point optimality gap was closed.
	UnorderedTppSolveResult tpp_nonconvex_unordered_solve(
		const Vector2 &start, const Vector2 &target,
		const std::vector<std::vector<Vector2>> &polygons,
		const UnorderedTppSolveOptions &options = {}
	);
	// TSPN: free cyclic visit order, no fixed point. The path repeats its first
	// point at the end. Shares the insertion/decomposition B&B and gap contract
	// above; convex relaxations use the independently certified cycle solver.
	UnorderedTppSolveResult tpp_nonconvex_tspn_solve(
		const std::vector<std::vector<Vector2>> &polygons,
		const UnorderedTppSolveOptions &options = {}
	);
}
