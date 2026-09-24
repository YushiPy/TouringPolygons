#pragma once

#include "vector2.h"

#include <cstddef>
#include <array>
#include <limits>
#include <vector>

namespace tpp {

enum class ConvexHybridMode { SafeCertified, Unchecked };
enum class ConvexHybridBackend { DoubleDisjoint, DoubleIntersection, RationalDisjoint, RationalIntersection };
enum class ConvexFallbackReason {
    None,
    LocatorOrRefoldingException,
    Nonfinite,
    ContactConstruction,
    MembershipOrOrdering,
    LocalOptimality,
    CoincidentContact,
    ShadowMismatch
};

struct ConvexHybridOptions {
    ConvexHybridMode mode = ConvexHybridMode::SafeCertified;
    // A search may return early with a certified dual bound above this value.
    double cutoff = std::numeric_limits<double>::infinity();
    bool shadow_rational = false;
    // Callers requesting only a diagnostic length may skip final contact
    // reconstruction. Safe mode always materializes contacts for certification.
    bool materialize_contacts = true;
    // Diagnostic only: preserve a fully materialized double candidate when
    // certification rejects it and safe mode replaces it with rational output.
    bool retain_rejected_double_candidate = false;
};

struct ConvexHybridStats {
    bool disjoint = false;
    bool double_attempted = false;
    bool double_certified = false;
    bool rational_fallback = false;
    bool rational_disjoint_directional_recovery = false;
    std::size_t predicate_exact_evaluations = 0;
    std::size_t zero_link_witnesses = 0;
    double dispatch_seconds = 0;
    double double_solver_seconds = 0;
    double contact_materialization_seconds = 0;
    double certificate_seconds = 0;
    double rational_fallback_seconds = 0;
    double total_seconds = 0;
};

struct ConvexHybridResult {
    // Exactly one contact per input polygon. Endpoints are never included
    // implicitly and duplicate contacts are retained.
    std::vector<Vector2> contacts;
    std::vector<Vector2> rejected_double_contacts;
    // Diagnostic certified bounds for the internally exact-replayed candidate.
    // The lower bound is a feasible convex-dual value; the upper bound is its
    // outward-rounded path length.  exact_feasible describes the exported
    // binary64 contacts, whose boundary rounding can differ from the replay.
    double rejected_double_lower_bound = 0;
    double rejected_double_upper_bound = 0;
    bool rejected_double_exact_feasible = false;
    double lower_bound = 0;
    double upper_bound = 0;
    bool cutoff_pruned = false;
    ConvexHybridBackend backend = ConvexHybridBackend::DoubleDisjoint;
    ConvexFallbackReason fallback_reason = ConvexFallbackReason::None;
    ConvexHybridStats stats;
};

struct ConvexHybridAggregate {
    std::size_t total_calls = 0;
    std::size_t disjoint_calls = 0;
    std::size_t certified_double_disjoint_calls = 0;
    std::size_t certified_double_intersection_calls = 0;
    std::size_t rational_disjoint_fallbacks = 0;
    std::size_t rational_disjoint_directional_recoveries = 0;
    std::size_t rational_intersection_fallbacks = 0;
    std::array<std::size_t,8> fallback_reasons{};
    std::size_t predicate_exact_evaluations = 0;
    std::size_t zero_link_witnesses = 0;
    double dispatch_seconds = 0;
    double double_solver_seconds = 0;
    double contact_materialization_seconds = 0;
    double certificate_seconds = 0;
    double rational_fallback_seconds = 0;
    double total_seconds = 0;
};

void reset_convex_hybrid_aggregate();
ConvexHybridAggregate convex_hybrid_aggregate();

ConvexHybridResult tpp_convex_solve_hybrid(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    const ConvexHybridOptions &options = {});

std::vector<Vector2> tpp_convex_solve_hybrid_safe(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);
double tpp_convex_solve_length_hybrid_safe(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);
std::vector<Vector2> tpp_convex_solve_hybrid_unchecked(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);
double tpp_convex_solve_length_hybrid_unchecked(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);

std::vector<Vector2> reconstruct_convex_polyline(
    const Vector2 &start, const Vector2 &target,
    const std::vector<Vector2> &contacts,
    bool remove_redundant_collinear = true);

const char *to_string(ConvexFallbackReason reason);

}
