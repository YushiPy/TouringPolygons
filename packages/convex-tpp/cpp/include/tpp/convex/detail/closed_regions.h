#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace tpp::detail {

// Closed convex regions in dimensions 0, 1 and 2. A segment needs both its
// supporting line and endpoint bounds; two opposing halfplanes alone do not
// describe the segment. Scalars may be exact or filtered rational numbers.
template<class Point>
bool closed_region_contains(const Point &q, const std::vector<Point> &p) {
    if (p.empty()) return false;
    if (p.size() == 1) return q == p.front();
    if (p.size() == 2) {
        const auto edge = p[1] - p[0], delta = q - p[0];
        return edge.cross(delta) == 0 && edge.dot(delta) >= 0
            && edge.dot(delta) <= edge.dot(edge);
    }
    for (size_t i = 0; i < p.size(); ++i)
        if ((p[(i + 1) % p.size()] - p[i]).cross(q - p[i]) < 0) return false;
    return true;
}

template<class Point, class Scalar>
bool clip_closed_region(const Point &a, const Point &b, const std::vector<Point> &p,
                        Scalar floor, Scalar &lo, Scalar &hi) {
    lo = floor; hi = 1;
    const auto d = b - a;
    auto constrain = [&](Scalar constant, Scalar slope) {
        if (slope > 0) lo = std::max(lo, Scalar(-constant / slope));
        else if (slope < 0) hi = std::min(hi, Scalar(-constant / slope));
        else if (constant < 0) return false;
        return lo <= hi;
    };
    if (p.empty()) return false;
    if (p.size() == 1) {
        const auto delta = a - p[0];
        if (!constrain(delta.x, d.x) || !constrain(-delta.x, -d.x)
            || !constrain(delta.y, d.y) || !constrain(-delta.y, -d.y)) return false;
    } else if (p.size() == 2) {
        const auto edge = p[1] - p[0], delta = a - p[0];
        const Scalar cross = edge.cross(delta), slope = edge.cross(d);
        const Scalar projection = edge.dot(delta), rate = edge.dot(d);
        if (!constrain(cross, slope) || !constrain(-cross, -slope)
            || !constrain(projection, rate)
            || !constrain(Scalar(edge.dot(edge) - projection), -rate)) return false;
    } else {
        for (size_t i = 0; i < p.size(); ++i) {
            const auto edge = p[(i + 1) % p.size()] - p[i];
            if (!constrain(edge.cross(a - p[i]), edge.cross(d))) return false;
        }
    }
    return lo <= hi && hi >= floor && lo <= 1;
}

} // namespace tpp::detail
