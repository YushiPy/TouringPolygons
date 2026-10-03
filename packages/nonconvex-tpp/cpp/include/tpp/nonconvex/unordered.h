#pragma once

#include "vector2.h"
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tpp {

	enum class UnorderedSearchStrategy { BestBoundDive, DfsBfs };
	enum class UnorderedSequenceStorage { Native, Packed, Deltas };

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
		bool interpolated_zero_dual = false;
		// Record an explanatory execution trace. Disabled by default so normal
		// benchmark runs keep the same memory and timing behavior.
		bool trace = false;
		// Cooperative interruption checked between search operations. The active
		// oracle/decomposition call is allowed to finish before the frontier stops.
		std::function<bool()> stop_requested;
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
        bool cycle_share_bounds = false;
        // Diagnostic JSONL, including every in-flight cycle input; empty disables I/O.
        std::string oracle_capture_file;
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
		size_t oracle_contracted_bound_calls = 0;
		size_t screened_nodes = 0;
        size_t one_tree_calls = 0, one_tree_cache_hits = 0, one_tree_iterations = 0;
        size_t one_tree_distance_queries = 0, one_tree_improvements = 0;
        double one_tree_seconds = 0;
        size_t learned_branch_observations = 0, learned_branch_decisions = 0, learned_branch_changes = 0;
        size_t cycle_memo_queries = 0, cycle_memo_repeated = 0, cycle_memo_hits = 0;
        size_t cycle_certificate_cutoff_skips = 0, cycle_initial_contact_checks = 0, cycle_initial_contact_accepts = 0;
        size_t cycle_certificate_interval_uses = 0;
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
		size_t polygon_vertices_min = std::numeric_limits<size_t>::max();
		size_t polygon_vertices_max = 0;
		double order_space_log2 = 0.0;
		size_t fallback_calls = 0;
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
