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

std::vector<Vector2> solve_intersecting_map_contacts(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    bool use_last_contact,
    PreloadPolicy preload = PreloadPolicy::Lazy);

// Native-binary64 construction used by the unchecked path and as the first
// attempt of the certified hybrid.  These entry points deliberately make no
// correctness claim; callers must certify the result or label it unchecked.
std::vector<Vector2> solve_intersecting_maps_unchecked_double(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

double length_intersecting_maps_unchecked_double(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

std::vector<Vector2> solve_intersecting_map_contacts_unchecked_double(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    bool use_last_contact,
    PreloadPolicy preload = PreloadPolicy::Lazy);

}
