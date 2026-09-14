#pragma once

#include "tpp/convex/options.h"
#include "vector2.h"
#include <vector>

namespace tpp::detail {

// Corrected directional last-step maps. Shared by the general solver entry
// points for intersections; the established disjoint core is separate.
std::vector<Vector2> solve_intersecting_maps(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

double length_intersecting_maps(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

}
