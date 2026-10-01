#include "termination.h"
#include <stdexcept>
#include <string_view>

int main() {
	using tspn_adapter::TerminationReason;
	auto require = [](bool condition) {
		if (!condition) throw std::runtime_error("Fekete termination classification regression");
	};
	require(tspn_adapter::classify_termination(true, false) == TerminationReason::FrontierExhausted);
	require(tspn_adapter::classify_termination(true, true) == TerminationReason::FrontierExhausted);
	require(tspn_adapter::classify_termination(false, true) == TerminationReason::GapCriterion);
	// Includes a timeout on the final node: has_next() after optimize may be
	// false, but optimize never observed exhaustion before its timer break.
	require(tspn_adapter::classify_termination(false, false) == TerminationReason::TimeLimit);
	require(std::string_view(tspn_adapter::to_string(TerminationReason::FrontierExhausted)) == "frontier_exhausted");
	require(std::string_view(tspn_adapter::to_string(TerminationReason::TimeLimit)) == "time_limit");
	require(std::string_view(tspn_adapter::to_string(TerminationReason::GapCriterion)) == "gap_criterion");
}
