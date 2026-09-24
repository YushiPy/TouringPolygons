#pragma once
#include "tpp/convex/solver.h"
#include "tpp/convex/hybrid.h"
#include <limits>

namespace tpp {
	struct CertifiedConvexTppResult {
		std::vector<Vector2> path;
		double lower_bound = 0;
		double upper_bound = 0;
		bool used_fallback = false;
		bool fallback_geometric_path_invalid = false;
		bool fallback_certificate_gap = false;
		bool used_extended_precision = false;
		bool repaired_geometric_path = false;
		bool dual_cutoff_pruned = false;
		bool time_limited = false;
		ConvexFallbackReason fallback_reason = ConvexFallbackReason::None;
		size_t predicate_exact_evaluations = 0;
		double contact_materialization_seconds = 0;
		double seconds = 0;
		double geometric_solver_seconds = 0;
		double certificate_verification_seconds = 0;
		double fallback_seconds = 0;
		double fallback_long_double_seconds = 0;
		double fallback_extended_precision_seconds = 0;
	};

	// Verifies the geometric oracle against a support-function dual bound.
	// Uses a smooth interior-point fallback if the certificate does not close.
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
