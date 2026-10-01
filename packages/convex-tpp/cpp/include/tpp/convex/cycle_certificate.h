#pragma once

#include "vector2.h"
#include "tpp/convex/rational.h"

#include <cstddef>
#include <vector>
#include <map>
#include <utility>
#include <limits>

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
    // A finite requested bound was proved without running the KKT test.
    bool optimality_check_skipped = false;
    bool interval_bounds_used = false;
};

// Immutable, independently validated geometry. Reusing it skips only input
// conversion/validation; contact feasibility and optimality are checked anew.
class ConvexCycleCertificateGeometry {
    ConvexRationalPolygons polygons_;
    std::vector<std::vector<Vector2>> binary_polygons_;
    friend class ConvexCycleWorkspace;
public:
    explicit ConvexCycleCertificateGeometry(const ConvexRationalPolygons &);
    explicit ConvexCycleCertificateGeometry(const std::vector<std::vector<Vector2>> &, bool binary_geometry = false);
    const ConvexRationalPolygons &polygons() const { return polygons_; }
    const std::vector<std::vector<Vector2>> &binary_polygons() const { return binary_polygons_; }
};

// Per-worker, content-keyed cache. Input mutation or reordering cannot reuse
// stale geometry. Clear between unrelated instance families to bound storage.
class ConvexCycleWorkspace {
    using Key = std::vector<std::pair<double,double>>;
    std::map<Key,ConvexRationalPolygon> polygons_;
    std::map<Key,std::vector<Vector2>> binary_polygons_;
public:
    ConvexCycleCertificateGeometry prepare(const std::vector<std::vector<Vector2>> &, bool binary_geometry = false);
    void clear() { polygons_.clear(); binary_polygons_.clear(); }
    std::size_t size() const { return polygons_.size(); }
};
ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
    const ConvexCycleCertificateGeometry &, const ConvexRationalPolygon &,
    double lower_bound_cutoff = std::numeric_limits<double>::infinity());
ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
    const ConvexCycleCertificateGeometry &, const std::vector<Vector2> &,
    double lower_bound_cutoff = std::numeric_limits<double>::infinity(), bool interval_filter = false);

// Any unit-disk vectors define a valid dual bound, independently of contacts.
// Invalid dimensions or vectors are rejected, never clipped with a tolerance.
double tpp_convex_cycle_dual_bound(const ConvexCycleCertificateGeometry &,
    const ConvexRationalPolygon &directions);
ConvexRationalPolygon tpp_convex_cycle_dual_directions(
    const std::vector<Vector2> &contacts, const ConvexRationalPolygon &inherited = {});

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
	const std::vector<Vector2> &contacts,
    double lower_bound_cutoff = std::numeric_limits<double>::infinity(), bool interval_filter = false
);

// Same exact predicates, without rounding rational inputs or contacts to double.
ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
    const ConvexRationalPolygons &polygons,
    const ConvexRationalPolygon &contacts,
    double lower_bound_cutoff = std::numeric_limits<double>::infinity()
);

} // namespace tpp
