#pragma once

#include "vector2.h"
#include "tpp/convex/rational.h"

#include <vector>

namespace tpp::detail {

struct RationalDisjointResult {
    std::vector<Vector2> contacts;
    double lower_bound = 0;
    double upper_bound = 0;
};

struct RationalDisjointExactResult {
    ConvexRationalPolygon contacts;
    double lower_bound = 0;
    double upper_bound = 0;
};

// Same recurrence, retaining rational coordinates through source, queries and
// contact materialization. No binary64 conversion occurs before the result.
RationalDisjointExactResult solve_rational_disjoint_exact(
    const ConvexRationalPoint &start, const ConvexRationalPoint &target,
    const ConvexRationalPolygons &polygons);

// Exact-rational form of the established binary-search disjoint last-step
// recurrence. The caller must have proved the polygons pairwise disjoint.
RationalDisjointResult solve_rational_disjoint(
    const Vector2 &start,const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);

}
