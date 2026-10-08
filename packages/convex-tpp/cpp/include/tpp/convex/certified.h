#pragma once
#include "tpp/convex/solver.h"
#include "tpp/convex/hybrid.h"
#include <limits>

namespace tpp {
	struct CertifiedConvexTppResult {
		std::vector<Vector2> path;
        std::vector<Vector2> binary_dual;
		double lower_bound = 0;
		double upper_bound = 0;
		bool used_fallback = false;
		bool used_interval_bounds = false;
		bool used_float_oracle = false;
		bool float_oracle_fallback = false;
		bool used_contracted_proposal = false;
		bool fallback_geometric_path_invalid = false;
		bool fallback_certificate_gap = false;
		bool used_extended_precision = false;
		bool repaired_geometric_path = false;
		bool dual_cutoff_pruned = false;
		bool time_limited = false;
		ConvexFallbackReason fallback_reason = ConvexFallbackReason::None;
		size_t predicate_exact_evaluations = 0;
		size_t dispatch_pair_queries = 0;
		size_t dispatch_pair_cache_hits = 0;
		size_t dispatch_pair_exact_checks = 0;
		double dispatch_seconds = 0;
		double bound_evaluation_seconds = 0;
		double proposal_preparation_seconds = 0;
		double contact_materialization_seconds = 0;
		double seconds = 0;
		double geometric_solver_seconds = 0;
		double certificate_verification_seconds = 0;
		double fallback_seconds = 0;
		double fallback_long_double_seconds = 0;
		double fallback_extended_precision_seconds = 0;
	};

	// Delegates to the safe hybrid oracle. A positive tolerance permits a
	// certified primal-dual gap; otherwise exact optimality is required.
	// Interval bounds use exact predicates on ambiguity, with rational recovery
	// when the candidate cannot meet the requested gap or cutoff.
	// Point and finite segment regions use rational directional construction
	// followed by the exact KKT certificate with singleton endpoint anchors.
	// A finite cutoff allows early return once lower_bound >= cutoff, even if
	// the primal-dual gap is still open. The returned path remains feasible.
	CertifiedConvexTppResult tpp_convex_solve_certified(
		const Vector2 &start, const Vector2 &target,
		const std::vector<std::vector<Vector2>> &polygons,
		DynamicConvexTppWorkspace &workspace, double tolerance,
		double cutoff = std::numeric_limits<double>::infinity(),
		double max_seconds = std::numeric_limits<double>::infinity()
	);
}
