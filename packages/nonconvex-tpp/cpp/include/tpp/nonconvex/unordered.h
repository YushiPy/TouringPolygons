#pragma once

#include "vector2.h"
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tpp {

	struct UnorderedTppSolveOptions {
		size_t max_calls = std::numeric_limits<size_t>::max();
		double max_seconds = std::numeric_limits<double>::infinity();
		double absolute_gap = 1e-7;
		double relative_gap = 1e-9;
		double feasibility_tolerance = 1e-8;
		size_t dive_interval = 1;
		// Try Fekete et al.'s endpoint-distance sum for the first branch.
		bool endpoint_sum_root = false;
		bool bidirectional_initial_heuristic = false;
		// A feasible start-to-target path, including both endpoints. When present,
		// it replaces the initial heuristic and supplies only an upper bound.
		std::optional<std::vector<Vector2>> initial_path;
		// Internal relaxations may stop at this relative oracle gap. Feasible
		// leaves are refined to the requested global gap before certification.
		double oracle_relative_gap = 1e-6;
		// Record an explanatory execution trace. Disabled by default so normal
		// benchmark runs keep the same memory and timing behavior.
		bool trace = false;
	};

	enum class UnorderedTppTermination { Optimal, CallLimit, TimeLimit, NumericalLimit };

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

	struct UnorderedTppSolveResult {
		std::vector<Vector2> path;
		std::vector<size_t> order;
		std::vector<UnorderedTppTraceEvent> trace;
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
		size_t calls = 0;
		size_t relaxation_calls = 0;
		size_t refinement_calls = 0;
		size_t complete_order_oracle_calls = 0;
		size_t complete_piece_oracle_calls = 0;
		size_t oracle_cutoff_calls = 0;
		size_t oracle_dual_cutoff_prunes = 0;
		size_t screened_nodes = 0;
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
		double seconds = 0.0;
		double preprocessing_seconds = 0.0;
		double initial_heuristic_seconds = 0.0;
		double search_seconds = 0.0;
		double finalization_seconds = 0.0;
		double convex_oracle_seconds = 0.0;
		double convex_geometric_solver_seconds = 0.0;
		double convex_certificate_verification_seconds = 0.0;
		double convex_contact_materialization_seconds = 0.0;
		double convex_fallback_seconds = 0.0;
		double convex_fallback_long_double_seconds = 0.0;
		double convex_fallback_extended_precision_seconds = 0.0;
		double decomposition_seconds = 0.0;
		double visit_check_seconds = 0.0;
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
}
