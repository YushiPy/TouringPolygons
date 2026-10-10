#pragma once

#include "tpp/geometry/vec2.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace tpp {

// Experimental fixed-order convex oracle that never uses rational arithmetic.
// It proves L <= OPT <= U on the original binary64 polygons with directed
// rounding: U is the length of a chain whose contacts are proved inside their
// polygons (interval orientation, then an exact 128-bit integer determinant),
// and L = D(u) for disk vectors u proved to have norm at most one. Candidates
// come from the binary64 directional trace and, if their bounds do not close,
// from an interior-point polish. It needs a positive gap or a finite cutoff;
// zero-gap algebraic optimality stays with tpp_convex_solve_hybrid.
struct ConvexFloatOracleOptions {
    double cutoff = std::numeric_limits<double>::infinity();
    double max_gap = 0;
    bool polish = true;
    // False: the trace only warm-starts the polish (its bounds were already
    // tried by the caller, e.g. the hybrid interval proof).
    bool certify_trace = true;
    // Optional candidate (one contact per polygon) used instead of computing
    // the binary64 trace; it is only a proposal.
    const std::vector<Vector2> *initial_contacts = nullptr;
    std::size_t max_newton_iterations = 600;
    // Warm start: the first barrier level is warm_mu_ratio times the final
    // one, and each contact moves this fraction toward its polygon's mean.
    double warm_mu_ratio = 1e4;
    double warm_interior_fraction = 0x1p-7;
};

enum class ConvexFloatOracleStatus { GapClosed, CutoffReached, Open, Unsupported };

struct ConvexFloatOracleResult {
    // One contact per polygon (proved inside it) when upper_bound is finite.
    std::vector<Vector2> contacts;
    double lower_bound = 0;
    double upper_bound = std::numeric_limits<double>::infinity();
    ConvexFloatOracleStatus status = ConvexFloatOracleStatus::Open;
    bool trace_attempted = false, trace_failed = false, trace_closed = false;
    bool polish_attempted = false, polish_closed = false;
    std::size_t newton_iterations = 0, barrier_levels = 0;
    double trace_seconds = 0, trace_certificate_seconds = 0, polish_seconds = 0, total_seconds = 0;
};

ConvexFloatOracleResult tpp_convex_solve_float_certified(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    const ConvexFloatOracleOptions &options = {});

const char *to_string(ConvexFloatOracleStatus status);

// The same oracle for an ordered convex cycle (no fixed endpoint, closing link
// included): two or more closed convex regions, each a point, a segment or a
// polygon of positive area. Proposals come from the shared binary64 cycle
// construction (cycle_refinement.h) and, if their bounds do not close, from
// an interior-point polish of the cyclic chain. Bounds are proved on the
// input regions with directed rounding: U encloses the length of a cycle of
// proved contacts, L = D(u) for disk vectors u. No rational arithmetic is
// used; an undecided normalization or membership sign returns Unsupported or
// an open interval, never an unproved bound. Needs max_gap > 0 or a finite cutoff.
class ConvexCycleWorkspace;
struct ConvexCycleFloatOptions {
    double cutoff = std::numeric_limits<double>::infinity();
    double max_gap = 0;
    bool construct = true;
    bool polish = true;
    // Proposals only: the construction's starting contacts and feature hints.
    const std::vector<Vector2> *initial_contacts = nullptr;
    const std::vector<int> *initial_features = nullptr;
    std::size_t max_newton_iterations = 600;
    double warm_mu_ratio = 1e4;
    double warm_interior_fraction = 0x1p-7;
    // Optional per-worker cache of the normalized regions.
    ConvexCycleWorkspace *workspace = nullptr;
};

struct ConvexCycleFloatResult {
    // One contact per region when upper_bound is finite: inside a polygon or
    // at a point (proved), or a binary64 approximation of the exact segment
    // point a+t(b-a) that the upper bound encloses.
    std::vector<Vector2> contacts;
    double lower_bound = 0;
    double upper_bound = std::numeric_limits<double>::infinity();
    ConvexFloatOracleStatus status = ConvexFloatOracleStatus::Open;
    bool construction_closed = false, polish_attempted = false, polish_closed = false;
    // The cooperative cycle deadline stopped the stage; bounds stay proved.
    bool interrupted = false;
    // Feature tuple of the construction candidate that closed (a proposal).
    std::vector<int> active_features;
    std::size_t candidates = 0, newton_iterations = 0, barrier_levels = 0;
    double total_seconds = 0;
};

ConvexCycleFloatResult tpp_convex_solve_cycle_float_certified(
    const std::vector<std::vector<Vector2>> &polygons,
    const ConvexCycleFloatOptions &options = {});

// Diagnostic only, never a bound: the binary64 directional trace, replayed and
// repaired, with its plain floating-point length. A wrong sign decision can make
// it suboptimal (too long) or slightly infeasible; nothing is certified. Returns
// false when the trace fails. Used to measure what certification costs.
bool tpp_convex_solve_double_trusted(const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons, std::vector<Vector2> &contacts, double &length);

}
