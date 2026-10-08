#include "tpp/convex/certified.h"
#include "certified_internal.h"
#include "tpp/convex/float_oracle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace tpp {
	CertifiedConvexTppResult tpp_convex_solve_certified(
		const Vector2 &start, const Vector2 &target,
		const std::vector<std::vector<Vector2>> &polygons,
		DynamicConvexTppWorkspace &workspace, double tolerance, double cutoff, double max_seconds
	) {
		bool float_fallback=false;
		const bool bounded=tolerance>0||std::isfinite(cutoff);
		std::optional<ConvexHybridResult> interval_only;
		if(workspace.float_recovery&&!workspace.float_oracle&&bounded) {
			ConvexHybridOptions first;
			first.cutoff=cutoff;first.max_gap=tolerance;first.stop_after_interval=true;
			first.interpolated_zero_dual=workspace.interpolated_zero_dual;
			first.retain_binary_dual=workspace.retain_binary_dual;
			interval_only=tpp_convex_solve_hybrid(start,target,polygons,first,workspace);
		}
		const bool try_float=bounded&&(workspace.float_oracle||(interval_only&&interval_only->stopped_after_interval));
		if(try_float) {
			ConvexFloatOracleOptions float_options;
			float_options.cutoff=cutoff;float_options.max_gap=tolerance;
			float_options.certify_trace=workspace.float_oracle;
			const auto attempt=tpp_convex_solve_float_certified(start,target,polygons,float_options);
			if(attempt.status==ConvexFloatOracleStatus::GapClosed||attempt.status==ConvexFloatOracleStatus::CutoffReached) {
				CertifiedConvexTppResult result;
				result.path=reconstruct_convex_polyline(start,target,attempt.contacts,false);
				result.lower_bound=attempt.lower_bound;result.upper_bound=attempt.upper_bound;
				result.dual_cutoff_pruned=attempt.lower_bound>=cutoff;
				result.used_float_oracle=true;
				result.seconds=attempt.total_seconds;
				result.geometric_solver_seconds=attempt.trace_seconds+attempt.polish_seconds;
				result.certificate_verification_seconds=attempt.trace_certificate_seconds;
				result.time_limited=max_seconds<=0.0;
				if(interval_only)result.seconds+=interval_only->stats.total_seconds;
				return result;
			}
			float_fallback=true;
		}
		ConvexHybridOptions options;
		options.cutoff=cutoff;
		options.max_gap=tolerance;
		options.interpolated_zero_dual=workspace.interpolated_zero_dual;
        options.retain_binary_dual=workspace.retain_binary_dual;
		auto hybrid=interval_only&&!interval_only->stopped_after_interval?std::move(*interval_only)
			:tpp_convex_solve_hybrid(start,target,polygons,options,workspace);
		CertifiedConvexTppResult result;
		// The unordered solver consumes the explicit contact-coordinate chain;
		// retain endpoints and all duplicate contacts in this compatibility API.
		result.path=reconstruct_convex_polyline(start,target,hybrid.contacts,false);
		result.lower_bound=hybrid.lower_bound;
		result.upper_bound=hybrid.upper_bound;
        result.binary_dual=std::move(hybrid.binary_dual);
		result.used_fallback=hybrid.stats.rational_fallback;
		result.used_interval_bounds=hybrid.stats.interval_bounds_certified;
		result.float_oracle_fallback=float_fallback;
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
