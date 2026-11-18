#pragma once

#include "tpp/convex/options.h"
#include "tpp/geometry/vec2.h"
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

// Counts of Tan and Jiang's construction on the split maps (the *_tan_jiang
// entry points below), per thread, for tests and measurements. A batch is the
// set of queries located together in one map.
struct TanJiangScanCounters {
    std::size_t batches = 0;          // batches in maps with at least three vertices
    std::size_t queries = 0;          // their queries
    std::size_t inside = 0;           // crossings decided by a pseudo-edge inside the polygon
    std::size_t scanned = 0;          // located by a scan of the cells
    std::size_t searched = 0;         // located by binary search: first of a batch, or a fallback
    std::size_t scan_steps = 0;       // cells tested by the scans
    std::size_t scan_bound = 0;       // sum over the batches of cells plus queries
    std::size_t scan_failures = 0;    // scans that tested every cell in vain
    std::size_t guard_rejections = 0; // cells found but not proved outside the polygon
    std::size_t budget_exhausted = 0; // batches that left their scan budget
    std::size_t small_maps = 0;       // queries in maps of a point or a segment (binary search)
    double max_batch_ratio = 0;       // largest scan steps / (cells + queries) of one batch
    bool verify = false;              // compare every location with the binary search
    std::size_t mismatches = 0;       // locations that differ from it
};
inline TanJiangScanCounters &tan_jiang_scan_counters() {
    thread_local TanJiangScanCounters counters;
    return counters;
}

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

// Tan and Jiang's construction on the same split maps: every cone, map by
// map, the two sides of the vertices of each map located together, level by
// level, by scans in the circular order of the cells (their Lemma 2). Exact
// rationals; the binary64 versions follow the unchecked contract below.
std::vector<Vector2> solve_intersecting_maps_tan_jiang(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);

double length_intersecting_maps_tan_jiang(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);

double length_intersecting_maps_tan_jiang_unchecked_double(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);

std::vector<DirectionalTraceStep> solve_intersecting_map_trace_tan_jiang_unchecked_double(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);

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

// Same directional construction with interval predicates and lazy rational
// evaluation. Its combinatorial trace still requires the hybrid certificate.
std::vector<DirectionalTraceStep> solve_intersecting_map_trace_filtered(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

std::vector<DirectionalTraceStep> solve_binary_search_disjoint_trace_unchecked(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload = PreloadPolicy::Lazy);

// Tan and Jiang's construction (every cone, by monotone scans), with the same
// trace. Pairwise disjoint polygons only (otherwise std::invalid_argument).
std::vector<DirectionalTraceStep> solve_tan_jiang_trace_unchecked(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);

bool pairwise_disjoint_unchecked_double(
    const std::vector<std::vector<Vector2>> &polygons);

// The lengths of the lazy or eager binary-search construction and of Tan and
// Jiang's, without the pairwise-disjointness check of the public entry points
// (O(k^2) pairs of polygons). For benchmarks on inputs known to be disjoint;
// on other inputs the result is undefined.
double length_binary_search_disjoint_unchecked(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons,
    PreloadPolicy preload);
double length_tan_jiang_unchecked(
    const Vector2 &start, const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);

}

#include "tpp/convex/rational.h"
namespace tpp::detail {
ConvexRationalPolygon solve_intersecting_map_contacts_exact(
    const ConvexRationalPoint &start,const ConvexRationalPoint &target,
    const ConvexRationalPolygons &polygons);
// Exact contacts of solve_disjoint_map_contacts (last contact on each visit).
ConvexRationalPolygon solve_disjoint_map_contacts_exact(
    const ConvexRationalPoint &start,const ConvexRationalPoint &target,
    const ConvexRationalPolygons &polygons);
}
