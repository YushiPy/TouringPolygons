#pragma once

#include "vector2.h"
#include "tpp/convex/rational.h"

#include <cstddef>
#include <vector>

namespace tpp {

// This checker validates a cyclic sequence of contacts independently of any
// cycle solver.  Input coordinates and contacts are interpreted as exact
// binary64 values for all predicates and bounds.
enum class ConvexCycleCertificateStatus {
	InvalidInput,
	InvalidCandidate,
	Feasible,
	Optimal
};

struct ConvexCycleCertificateResult {
	ConvexCycleCertificateStatus status = ConvexCycleCertificateStatus::InvalidInput;
	// Outward-rounded bounds for the global optimum.  They are populated when
	// the contact sequence is feasible; for an exact optimality certificate they
	// enclose the candidate cycle's length.
	double lower_bound = 0;
	double upper_bound = 0;
	std::size_t exact_predicate_evaluations = 0;
};

// Checks a cycle with one contact per polygon, in input order, followed by the
// closing edge from contacts.back() to contacts.front().  Polygons must be
// finite convex polygonal regions; either winding is accepted. Point/segment
// regions are also accepted for restricted anchor certificates.
// A feasible candidate is globally certified optimal when all its cyclic KKT
// support conditions hold, using complete disk/cone reachability at zero links.
// Otherwise the result still contains a valid
// primal/dual interval, useful for checking epsilon-optimal candidates.
ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
	const std::vector<std::vector<Vector2>> &polygons,
	const std::vector<Vector2> &contacts
);

// Same exact predicates, without rounding rational inputs or contacts to double.
ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
    const ConvexRationalPolygons &polygons,
    const ConvexRationalPolygon &contacts
);

} // namespace tpp
