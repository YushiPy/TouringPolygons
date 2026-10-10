#pragma once
#include "tpp/geometry/vec2.h"
#include <cmath>
#include <utility>
#include <vector>

namespace tpp {
// Certified bounds for inserting one region in every gap of an open path.
// Contacts only propose new dual directions; they need not be feasible.
// An empty result declines unsupported arithmetic or a dual hint whose
// unit-disk membership cannot be proved with the interval filter.
std::vector<double> tpp_convex_binary_dual_insertion_bounds(
    Vector2 start,Vector2 target,const std::vector<Vector2> &contacts,
    const std::vector<const std::vector<Vector2> *> &regions,
    const std::vector<Vector2> &inserted,const std::vector<Vector2> &proposals,
    const std::vector<Vector2> &dual);
// The same bounds for a closed cycle: link i runs from contacts[i] to
// contacts[(i+1)%n], and bounds[i] is for inserting the region between regions
// i and i+1, with proposals[i] proposing its contact. Every link direction is a
// binary64 vector proved to lie in the unit disk (zero for a zero link), and
// supports are bounded with an a priori rounding-error bound (see
// tpp_convex_binary_path_dual), so each bound is a weak-duality value on the
// child cycle. An empty result declines unsupported arithmetic.
std::vector<double> tpp_convex_binary_cycle_insertion_bounds(
    const std::vector<Vector2> &contacts,const std::vector<const std::vector<Vector2> *> &regions,
    const std::vector<Vector2> &inserted,const std::vector<Vector2> &proposals);

// The contact-direction dual of a reference chain, proved with directed
// rounding. An open path runs from contacts[0] through regions 0..n-1 (at
// contacts[1..n]) to contacts[n+1], with n+1 links; a cycle visits regions
// 0..k-1 at contacts[0..k-1], link i ending at region (i+1)%k. Each link takes
// the binary64 unit vector of its contacts, proved in the unit disk; a zero
// link takes a neighbouring link's vector (the best of the forward and
// backward fills and of none). D(u) is the weak-duality value with these
// vectors; supports and widths are taken along each region's normal (incoming
// minus outgoing link vector), relative to contacts[0]. Each support is the
// plain binary64 minimum widened by an a priori bound of its rounding error
// (standard model, gradual underflow), and sums are rounded outward. Empty
// directions mean unsupported arithmetic; contacts need not be feasible.
struct ConvexBinaryChainDual {
    bool cyclic = false;
    std::vector<Vector2> directions;
    std::vector<double> support_lower, support_upper, width_upper;
    double lower = -INFINITY, upper = -INFINITY; // enclosure of D(u)
    bool valid() const { return !directions.empty(); }
};
ConvexBinaryChainDual tpp_convex_binary_path_dual(const std::vector<Vector2> &contacts,
    const std::vector<const std::vector<Vector2> *> &regions);
ConvexBinaryChainDual tpp_convex_binary_cycle_dual(const std::vector<Vector2> &contacts,
    const std::vector<const std::vector<Vector2> *> &regions);
// Enclosure of the change of D when region `inserted` is visited in gap j
// (on link j) through `proposal`: the link is replaced by two whose proved
// directions point to and from the proposal (a zero one is zero or link j's
// vector, whichever bounds more), and only the terms of the inserted region
// and of the regions at both ends of link j change. Any proposal gives a
// valid bound.
std::pair<double,double> tpp_convex_binary_insertion_gain(const ConvexBinaryChainDual &dual,
    const std::vector<Vector2> &contacts,const std::vector<const std::vector<Vector2> *> &regions,
    const std::vector<Vector2> &inserted,std::size_t gap,Vector2 proposal);
// A lower bound of the distance between two binary64 points.
double tpp_convex_distance_lower(Vector2 a,Vector2 b);
}
