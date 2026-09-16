#pragma once

#include "vector2.h"
#include <cstddef>
#include <limits>
#include <vector>

namespace tpp {

	struct UnorderedTppSolveOptions {
		size_t max_calls = std::numeric_limits<size_t>::max();
		double max_seconds = std::numeric_limits<double>::infinity();
		double absolute_gap = 1e-7;
		double relative_gap = 1e-9;
		double feasibility_tolerance = 1e-8;
		size_t dive_interval = 128;
		bool bidirectional_initial_heuristic = false;
		// Internal relaxations may stop at this relative oracle gap. Feasible
		// leaves are refined to the requested global gap before certification.
		double oracle_relative_gap = 1e-6;
	};

	enum class UnorderedTppTermination { Optimal, CallLimit, TimeLimit, NumericalLimit };

	struct UnorderedTppSolveResult {
		std::vector<Vector2> path;
		std::vector<size_t> order;
		double lower_bound = 0.0;
		double upper_bound = std::numeric_limits<double>::infinity();
		bool exact = false;
		UnorderedTppTermination termination = UnorderedTppTermination::NumericalLimit;
		size_t calls = 0;
		size_t refinement_calls = 0;
		size_t oracle_cutoff_calls = 0;
		size_t screened_nodes = 0;
		size_t nodes = 0;
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
