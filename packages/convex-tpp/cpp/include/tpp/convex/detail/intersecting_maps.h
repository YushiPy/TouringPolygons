#pragma once

#include "tpp/convex/options.h"
#include "vector2.h"
#include <vector>

namespace tpp::detail {

struct DirectionalMapContact {
    Vector2 point;
    Vector2 segment_start;
    Vector2 segment_end;
    Vector2 edge_start;
    Vector2 edge_end;
    bool has_edge = false;
};

enum class DirectionalTraceRegion { Crossing, Vertex, Edge };

// One final-query decision per polygon, ordered from the last polygon back to
// the first.  A vertex is either an original input point or the exact
// intersection of its original edge with an earlier original edge.
struct DirectionalTraceStep {
    std::size_t level = 0;
    DirectionalTraceRegion region = DirectionalTraceRegion::Crossing;
    std::size_t original_edge = 0;
    Vector2 defining_point;
    bool vertex_is_edge_intersection = false;
    std::size_t defining_polygon = 0;
    std::size_t defining_edge = 0;
};

struct RationalMapResult {
    std::vector<Vector2> contacts;
    double lower_bound = 0;
    double upper_bound = 0;
};

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

// Exact directional last-step maps specialized for a caller-proved pairwise-
// disjoint sequence.  They retain exact predicates and refolding but omit all
// previous-polygon boundary splitting.
std::vector<Vector2> solve_disjoint_map_contacts(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

double length_disjoint_maps(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

RationalMapResult solve_intersecting_map_contacts_with_bounds(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    bool use_last_contact,
    PreloadPolicy preload = PreloadPolicy::Lazy);

RationalMapResult solve_disjoint_map_contacts_with_bounds(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
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

std::vector<DirectionalMapContact> solve_intersecting_map_contact_details_unchecked_double(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    bool use_last_contact,
    PreloadPolicy preload = PreloadPolicy::Lazy);

std::vector<DirectionalTraceStep> solve_intersecting_map_trace_unchecked_double(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

std::vector<DirectionalTraceStep> solve_binary_search_disjoint_trace_unchecked(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

bool pairwise_disjoint_unchecked_double(
    const std::vector<std::vector<Vector2>> &polygons);

}
