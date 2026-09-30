#pragma once
#include "tpp/convex/cycle_certificate.h"
#include <limits>
#include <string>

namespace tpp {
enum class ConvexCycleStatus { InvalidInput, UnsupportedIntersection, Optimal, FloatingPointLimit, OracleFailure, CertifiedBound };

struct ConvexCycleResult {
    ConvexCycleStatus status = ConvexCycleStatus::InvalidInput;
    // One EXACT contact per polygon in original order; closing link is implicit.
    ConvexRationalPolygon contacts;
    // Exact objective: sum_i sqrt(squared_link_lengths[i]).
    // A Euclidean length need not itself be rational.
    std::vector<ConvexRational> squared_link_lengths;
    ConvexCycleCertificateResult certificate;
    std::size_t anchor_polygon = 0;
    std::size_t oracle_calls = 0;
    std::size_t certificate_checks = 0;
    std::size_t certificate_cutoff_skips = 0;
    std::string diagnostic;
};

struct ConvexCycleDoubleResult {
    ConvexCycleStatus status = ConvexCycleStatus::InvalidInput;
    std::vector<Vector2> contacts;
    // Bounds on the original input: the lower bound may additionally retain
    // an independently verified rational recovery before contact rounding.
    ConvexCycleCertificateResult certificate;
    std::size_t anchor_polygon = 0;
    std::size_t oracle_calls = 0;
    std::size_t certificate_checks = 0;
    std::size_t certificate_cutoff_skips = 0;
    std::size_t initial_contact_checks = 0, initial_contact_accepts = 0;
    std::size_t rational_anchor_recoveries = 0;
    std::size_t rational_cycle_recoveries = 0;
    std::size_t rational_feature_recoveries = 0;
    std::string diagnostic;
    // A proposal for subsequent solves; never itself a certificate.
    std::vector<int> active_features;
};

struct ConvexCycleDoubleOptions {
    // Permit rational reconstruction of a feature tuple, anchor, or general
    // cycle after a failed double construction. Every recovery is counted.
    // Set false for a double-only constructor (certificates remain exact).
    bool recover_arithmetic_failures = true;
    // Acceleration only; false exercises the shared boundary-anchor reduction.
    bool refine_contacts = true;
    // Optional B&B cut: stop after an independently certified global lower
    // bound reaches this value. CertifiedBound does not assert optimality.
    double lower_bound_cutoff = std::numeric_limits<double>::infinity();
    // Optional initial proposal, one finite contact per region. It supplies
    // no bound or optimality claim; every constructed cycle is still checked.
    std::vector<Vector2> initial_contacts;
    // Optional per-worker prepared-geometry cache.
    ConvexCycleWorkspace *workspace = nullptr;
    // Canonical CCW feature ids: vertex 2*j, edge 2*j+1, skipped -1, changed -2.
    std::vector<int> initial_features;
    // Export warm-start metadata only when the caller will reuse it.
    bool retain_active_features = false;
    // Try a certified bound before full KKT, and check inherited contacts first.
    bool bound_first = false;
};

// No fixed endpoint, no tolerance or discretization. Input must contain >=2
// closed, positive-area, pairwise disjoint convex polygons. Either winding and
// redundant collinear/closing vertices are accepted. Self-crossing cycles are
// valid; touching, overlapping or nested polygons are unsupported.
// Optimal is returned ONLY after the rational cyclic support certificate passes.
ConvexCycleResult tpp_convex_solve_cycle_disjoint(const ConvexRationalPolygons &polygons);
// Binary64 inputs are interpreted exactly as binary rationals. Output remains
// rational; external() is an explicitly rounded presentation conversion.
ConvexCycleResult tpp_convex_solve_cycle_disjoint(const std::vector<std::vector<Vector2>> &polygons);
// The same search and Dror recurrence instantiated with double. No epsilon
// stopping test. Stops at an exact certificate or representable/stationary
// arithmetic limits. FloatingPointLimit carries a feasible candidate and its
// measured certificate, and never asserts exact optimality. Input validation
// and final certificates remain exact. Rational feature/anchor/whole-cycle
// recovery is explicitly configurable and counted.
ConvexCycleDoubleResult tpp_convex_solve_cycle_disjoint_double(
    const std::vector<std::vector<Vector2>> &polygons,
    const ConvexCycleDoubleOptions &options = {});
struct ConvexCycleOptions {
    // Acceleration only. Disable to exercise the complete boundary-anchor search.
    bool refine_contacts = true;
    ConvexRationalPolygon initial_contacts;
    // Optional independently certified B&B cut, as in the double API.
    // The default infinity requests the exact optimum; CertifiedBound only
    // proves that the global lower bound reaches this threshold.
    double lower_bound_cutoff = std::numeric_limits<double>::infinity();
    bool bound_first = false;
};
// General closed convex regions, including touching, overlap and containment.
// Optimal requires an exact global certificate, including coincident contacts.
// OracleFailure reports a failed fixed-source construction or certificate.
ConvexCycleResult tpp_convex_solve_cycle(const ConvexRationalPolygons &polygons,
    const ConvexCycleOptions &options = {});
ConvexCycleResult tpp_convex_solve_cycle(const std::vector<std::vector<Vector2>> &polygons,
    const ConvexCycleOptions &options = {});
ConvexCycleDoubleResult tpp_convex_solve_cycle_double(
    const std::vector<std::vector<Vector2>> &polygons,
    const ConvexCycleDoubleOptions &options = {});
} // namespace tpp
