#pragma once
#include "tpp/convex/cycle_certificate.h"
#include <limits>
#include <functional>
#include <string>

namespace tpp {
// GapClosed: binary64 bounds proved on the input close the requested
// max_gap, without the exact KKT test; it does not assert exact optimality.
enum class ConvexCycleStatus { InvalidInput, UnsupportedIntersection, Optimal, FloatingPointLimit, OracleFailure, CertifiedBound, Interrupted, ProposalLimit, GapClosed };

struct ConvexCycleTimings {
    // Exclusive work: certification includes checks during rational recovery.
    double construction_seconds = 0;
    double certification_seconds = 0;
    double rational_recovery_seconds = 0;
    // Binary64 stage (max_gap > 0): interval proofs, and the polish's Newton
    // iterations (its proofs count as interval proofs).
    double interval_proof_seconds = 0;
    double polish_seconds = 0;
};

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
    ConvexCycleTimings timings;
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
    std::size_t certificate_interval_uses = 0;
    std::size_t rational_anchor_recoveries = 0;
    std::size_t rational_cycle_recoveries = 0;
    std::size_t rational_feature_recoveries = 0;
    // Binary64 stage (max_gap > 0): closed by an interval proof of a
    // construction candidate, or by the polish; otherwise the exact path ran.
    bool float_interval_closed = false, float_polish_attempted = false, float_polish_closed = false;
    std::size_t float_candidates = 0, float_newton_iterations = 0, float_barrier_levels = 0;
    ConvexCycleTimings timings;
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
    // Outward binary64 interval filters, with exact predicates on ambiguity.
    bool interval_certificate = false;
    // Run only the finite floating feature proposal, without rational recovery
    // or the complete boundary search. ProposalLimit makes no optimality claim;
    // contacts/bounds, when present, still have an independent certificate.
    // No tolerance enters this constructor; the caller decides how to use bounds.
    bool proposal_only = false;
    // A positive gap first runs tpp_convex_solve_cycle_float_certified: the
    // same binary64 construction, bounds proved with directed rounding and,
    // if needed, an interior-point polish. GapClosed (or CertifiedBound) is
    // returned when U-L <= max_gap (or L >= cutoff); its contacts are proved
    // in their polygons and points, and on a segment they approximate an
    // exact segment point whose cycle U bounds. Otherwise the exact path runs
    // unchanged and keeps the larger proved lower bound. Zero keeps it off.
    double max_gap = 0;
    bool float_polish = true;
    // Cooperative checkpoints; default standalone solves have no deadline.
    // Interrupted retains only completed certificates; contacts may be empty.
    double max_seconds = std::numeric_limits<double>::infinity();
    std::function<bool()> stop_requested;
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
    double max_seconds = std::numeric_limits<double>::infinity();
    std::function<bool()> stop_requested;
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
