#include "tpp/convex/certified.h"
#include "certified_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace tpp {
	CertifiedConvexTppResult tpp_convex_solve_certified(
		const Vector2 &start, const Vector2 &target,
		const std::vector<std::vector<Vector2>> &polygons,
		DynamicConvexTppWorkspace &workspace, double tolerance, double cutoff, double max_seconds
	) {
		ConvexHybridOptions options;
		options.cutoff=cutoff;
		options.max_gap=tolerance;
		options.interpolated_zero_dual=workspace.interpolated_zero_dual;
		const auto hybrid=tpp_convex_solve_hybrid(start,target,polygons,options,workspace);
		CertifiedConvexTppResult result;
		// The unordered solver consumes the explicit contact-coordinate chain;
		// retain endpoints and all duplicate contacts in this compatibility API.
		result.path=reconstruct_convex_polyline(start,target,hybrid.contacts,false);
		result.lower_bound=hybrid.lower_bound;
		result.upper_bound=hybrid.upper_bound;
		result.used_fallback=hybrid.stats.rational_fallback;
		result.used_interval_bounds=hybrid.stats.interval_bounds_certified;
		result.used_contracted_proposal=hybrid.stats.interval_bounds_contracted;
		result.dual_cutoff_pruned=hybrid.cutoff_pruned;
		result.fallback_reason=hybrid.fallback_reason;
		result.fallback_geometric_path_invalid=result.used_fallback &&
			(hybrid.fallback_reason==ConvexFallbackReason::LocatorOrRefoldingException ||
			 hybrid.fallback_reason==ConvexFallbackReason::Nonfinite ||
			 hybrid.fallback_reason==ConvexFallbackReason::ContactConstruction ||
			 hybrid.fallback_reason==ConvexFallbackReason::MembershipOrOrdering);
		result.fallback_certificate_gap=result.used_fallback && !result.fallback_geometric_path_invalid;
		result.predicate_exact_evaluations=hybrid.stats.predicate_exact_evaluations;
		result.dispatch_pair_queries=hybrid.stats.dispatch_pair_queries;
		result.dispatch_pair_cache_hits=hybrid.stats.dispatch_pair_cache_hits;
		result.dispatch_pair_exact_checks=hybrid.stats.dispatch_pair_exact_checks;
		result.dispatch_seconds=hybrid.stats.dispatch_seconds;
		result.bound_evaluation_seconds=hybrid.stats.bound_evaluation_seconds;
		result.proposal_preparation_seconds=hybrid.stats.proposal_preparation_seconds;
		result.seconds=hybrid.stats.total_seconds;
		result.geometric_solver_seconds=hybrid.stats.double_solver_seconds;
		result.contact_materialization_seconds=hybrid.stats.contact_materialization_seconds;
		result.certificate_verification_seconds=hybrid.stats.certificate_seconds;
		result.fallback_seconds=hybrid.stats.rational_fallback_seconds;
		result.time_limited=max_seconds<=0.0;
		return result;
	}
}
