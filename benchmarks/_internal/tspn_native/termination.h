#pragma once

namespace tspn_adapter {

enum class TerminationReason { FrontierExhausted, TimeLimit, GapCriterion };

// Mirrors the order of exit checks in the pinned solver's optimize() loop.
constexpr TerminationReason classify_termination(bool frontier_exhausted_observed, bool gap_reached) {
	if (frontier_exhausted_observed) return TerminationReason::FrontierExhausted;
	if (gap_reached) return TerminationReason::GapCriterion;
	return TerminationReason::TimeLimit;
}

constexpr const char *to_string(TerminationReason reason) {
	switch (reason) {
	case TerminationReason::FrontierExhausted: return "frontier_exhausted";
	case TerminationReason::TimeLimit: return "time_limit";
	case TerminationReason::GapCriterion: return "gap_criterion";
	}
	return "unknown";
}

} // namespace tspn_adapter
