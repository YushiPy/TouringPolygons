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

}
