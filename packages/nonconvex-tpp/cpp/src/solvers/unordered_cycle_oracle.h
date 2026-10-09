#pragma once

#include "tpp/convex/certified.h"
#include "tpp/convex/cycle.h"
#include "tpp/convex/workspace.h"

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace tpp {

// Shared single-relaxation adapter used by the maintained TSPN search and the
// captured-call replay CLI. Keeping one implementation preserves its double
// solve, optional exact bound recovery, warm starts, cutoffs and timing rules.
struct RelaxationResult : CertifiedConvexTppResult {
	std::vector<int> active_features;
	std::size_t memo_queries = 0, memo_repeated = 0, memo_hits = 0;
	std::size_t certificate_cutoff_skips = 0, initial_contact_checks = 0, initial_contact_accepts = 0;
	std::size_t certificate_interval_uses = 0;
	std::size_t proposal_calls = 0, proposal_accepts = 0;
	// Binary64 stage: whether the polish ran and its Newton iterations.
	bool polish_attempted = false;
	std::size_t polish_newton_iterations = 0;
	ConvexCycleTimings cycle_timings;
	RelaxationResult() = default;
	RelaxationResult(CertifiedConvexTppResult result) : CertifiedConvexTppResult(std::move(result)) {}
};

RelaxationResult solve_relaxation(bool cycle, const Vector2 &start, const Vector2 &target,
	const std::vector<std::vector<Vector2>> &regions, DynamicConvexTppWorkspace &workspace,
	double tolerance, double cutoff, double seconds, const std::vector<Vector2> &initial_contacts = {},
	ConvexCycleWorkspace *cycle_workspace = nullptr, const std::vector<int> &initial_features = {},
	bool retain_features = false, bool bound_first = false, bool interval_certificate = false,
	const std::function<bool()> &stop_requested = {}, bool proposal_bound = false, bool float_oracle = false);

} // namespace tpp
