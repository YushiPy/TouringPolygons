#pragma once

// Contacts between the boundaries of two convex polygons in linear time.
//
// Edge j of a polygon joins vertex j to vertex j+1 (cyclically). In the order
// of (x, y), every point of an edge lies between the endpoints of the edge, so
// two edges with a common point have overlapping ranges in that order. The
// boundary of a convex polygon splits, at its smallest and largest vertices in
// that order, into two chains whose edges have consecutive ranges. Merging a
// chain of one polygon with a chain of the other, as in the slab method of
// Hoey (described by O'Rourke, Chien, Olson and Naddor, 1982), lists the pairs
// of edges with overlapping ranges in O(|A|+|B|) time, and there are
// O(|A|+|B|) of them. Polygons with fewer than three vertices, or whose
// boundary does not split this way, fall back to all pairs of edges.

#include "tpp/geometry/common.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace tpp::detail {

template<class P>
bool lexicographic_less(const P &a,const P &b) {
    return a.x<b.x || (a.x==b.x && a.y<b.y);
}

// The edges of each chain, in increasing order of their ranges. Without a
// valid split, every pair of edges is reported.
struct LexicographicChains {
    std::vector<size_t> increasing,decreasing;
    bool valid=false;
};

template<class Polygon>
LexicographicChains lexicographic_chains(const Polygon &v) {
    LexicographicChains chains;
    const size_t m=v.size();
    if(m<3) return chains;
    size_t s=0;
    for(size_t j=1;j<m;++j) if(lexicographic_less(v[j],v[s])) s=j;
    size_t j=s,steps=0;
    while(steps<m && lexicographic_less(v[j],v[(j+1)%m])) {chains.increasing.push_back(j);j=(j+1)%m;++steps;}
    while(steps<m && lexicographic_less(v[(j+1)%m],v[j])) {chains.decreasing.push_back(j);j=(j+1)%m;++steps;}
    if(steps!=m || chains.increasing.empty() || chains.decreasing.empty()) return {};
    std::reverse(chains.decreasing.begin(),chains.decreasing.end());
    chains.valid=true;
    return chains;
}

namespace contacts_internal {

template<class Polygon>
const auto &edge_low(const Polygon &v,size_t e) {
    const auto &p=v[e],&q=v[(e+1)%v.size()];
    return lexicographic_less(q,p) ? q : p;
}

template<class Polygon>
const auto &edge_high(const Polygon &v,size_t e) {
    const auto &p=v[e],&q=v[(e+1)%v.size()];
    return lexicographic_less(q,p) ? p : q;
}

// Reports each pair of edges of the two chains with overlapping closed ranges
// once. Returns true when the callback asked to stop.
template<class PolygonA,class PolygonB,class F>
bool merge(const PolygonA &a,const std::vector<size_t> &ca,
           const PolygonB &b,const std::vector<size_t> &cb,F &f) {
    size_t start=0;
    for(size_t e:ca) {
        const auto &low=edge_low(a,e),&high=edge_high(a,e);
        while(start<cb.size() && lexicographic_less(edge_high(b,cb[start]),low)) ++start;
        for(size_t t=start;t<cb.size() && !lexicographic_less(high,edge_low(b,cb[t]));++t)
            if(f(e,cb[t])) return true;
    }
    return false;
}

}  // namespace contacts_internal

// Calls f(j,k) for edges j of a and k of b, at most once per pair and at least
// for every pair of closed edges with a common point. With valid chains it
// makes O(|a|+|b|) calls. f returns true to stop, and the function then
// returns true.
template<class PolygonA,class PolygonB,class F>
bool for_each_edge_pair_that_may_touch(const PolygonA &a,const LexicographicChains &ca,
                                       const PolygonB &b,const LexicographicChains &cb,F &&f) {
    if(!ca.valid || !cb.valid) {
        for(size_t j=0;j<a.size();++j)
            for(size_t k=0;k<b.size();++k)
                if(f(j,k)) return true;
        return false;
    }
    using contacts_internal::merge;
    return merge(a,ca.increasing,b,cb.increasing,f) || merge(a,ca.increasing,b,cb.decreasing,f) ||
           merge(a,ca.decreasing,b,cb.increasing,f) || merge(a,ca.decreasing,b,cb.decreasing,f);
}

template<class PolygonA,class PolygonB,class F>
bool for_each_edge_pair_that_may_touch(const PolygonA &a,const PolygonB &b,F &&f) {
    return for_each_edge_pair_that_may_touch(a,lexicographic_chains(a),b,lexicographic_chains(b),f);
}

// Whether the closed segments ab and cd have a common point, in exact
// arithmetic.
template<class P>
bool closed_segments_meet(const P &a,const P &b,const P &c,const P &d) {
    auto orientation=[](const P &p,const P &q,const P &r) {
        const auto value=(q-p).cross(r-p);
        return value>0 ? 1 : (value<0 ? -1 : 0);
    };
    // With p, q and r collinear: whether r lies on the segment pq.
    auto between=[](const P &p,const P &q,const P &r) {return (r-p).dot(r-q)<=0;};
    const int o1=orientation(a,b,c),o2=orientation(a,b,d),o3=orientation(c,d,a),o4=orientation(c,d,b);
    if(o1*o2<0 && o3*o4<0) return true;
    return (o1==0 && between(a,b,c)) || (o2==0 && between(a,b,d)) ||
           (o3==0 && between(c,d,a)) || (o4==0 && between(c,d,b));
}

// Whether the boundaries of two closed convex polygons have a common point, in
// exact arithmetic and O(|a|+|b|) time.
template<class Polygon>
bool convex_boundaries_meet(const Polygon &a,const LexicographicChains &ca,
                            const Polygon &b,const LexicographicChains &cb) {
    return for_each_edge_pair_that_may_touch(a,ca,b,cb,[&](size_t j,size_t k) {
        return closed_segments_meet(a[j],a[(j+1)%a.size()],b[k],b[(k+1)%b.size()]);
    });
}

// The binary64 test of the floating-point constructions, with their tolerance:
// boxes apart by more than 1e-8, edges that intersect in binary64, or a vertex
// of one polygon in the other within the tolerance of the membership test. The
// edges tested are those whose ranges overlap, which include every pair that
// meets exactly.
// As in the constructions: the square of 1e-8, computed in binary64.
inline constexpr double membership_epsilon=1e-8;
inline constexpr double membership_tolerance=membership_epsilon*membership_epsilon;

// No edge with the point clearly on its right, or none with the point clearly
// on its left, which accepts either orientation. O(m) time.
inline bool point_in_convex_polygon_closed_linear(const Vector2 &point,const std::vector<Vector2> &polygon) {
    if(polygon.empty()) return false;
    constexpr double tolerance=membership_tolerance;
    bool has_positive=false,has_negative=false;
    for(size_t j=0;j<polygon.size();j++) {
        const auto &v1=polygon[j];
        const auto &v2=polygon[(j+1)%polygon.size()];
        const double cross=(v2-v1).cross(point-v1);
        if(cross>tolerance) has_positive=true;
        else if(cross<-tolerance) has_negative=true;
        if(has_positive && has_negative) return false;
    }
    return true;
}

// The same test in O(log m) time: the two edges at vertex 0 and, by binary
// search on the fan from vertex 0, the edge that faces the point, each with the
// tolerance of the linear test. Polygons with fewer than three vertices, or
// whose first three vertices are collinear, use the linear test.
inline bool point_in_convex_polygon_closed_logarithmic(const Vector2 &point,const std::vector<Vector2> &polygon) {
    const size_t m=polygon.size();
    if(m<3) return point_in_convex_polygon_closed_linear(point,polygon);
    const Vector2 &origin=polygon[0];
    const double turn=(polygon[1]-origin).cross(polygon[2]-origin);
    if(turn==0) return point_in_convex_polygon_closed_linear(point,polygon);
    // Signed so that the inside of every edge is positive.
    const double orientation=turn>0 ? 1.0 : -1.0;
    auto edge_cross=[&](size_t j) {
        const auto &v1=polygon[j];
        const auto &v2=polygon[(j+1)%m];
        return orientation*(v2-v1).cross(point-v1);
    };
    if(edge_cross(0)<-membership_tolerance || edge_cross(m-1)<-membership_tolerance) return false;
    size_t low=1,high=m-1;
    while(high-low>1) {
        const size_t mid=low+(high-low)/2;
        if(orientation*(polygon[mid]-origin).cross(point-origin)>=0) low=mid;
        else high=mid;
    }
    return edge_cross(low)>=-membership_tolerance;
}

inline bool convex_polygons_intersect_or_touch_double(const std::vector<Vector2> &a,const LexicographicChains &ca,
                                                      const std::vector<Vector2> &b,const LexicographicChains &cb) {
    constexpr double tolerance=1e-8;
    auto bounds=[](const std::vector<Vector2> &polygon) {
        double min_x=polygon.front().x,max_x=polygon.front().x,min_y=polygon.front().y,max_y=polygon.front().y;
        for(const auto &point:polygon) {
            min_x=std::min(min_x,point.x);max_x=std::max(max_x,point.x);
            min_y=std::min(min_y,point.y);max_y=std::max(max_y,point.y);
        }
        return std::array<double,4>{min_x,max_x,min_y,max_y};
    };
    const auto ab=bounds(a),bb=bounds(b);
    if(ab[1]<bb[0]-tolerance || bb[1]<ab[0]-tolerance || ab[3]<bb[2]-tolerance || bb[3]<ab[2]-tolerance)
        return false;
    if(for_each_edge_pair_that_may_touch(a,ca,b,cb,[&](size_t i,size_t j) {
        return segment_segment_intersection_safe(a[i],a[(i+1)%a.size()],b[j],b[(j+1)%b.size()]).is_finite();
    })) return true;
    return point_in_convex_polygon_closed_linear(a.front(),b) || point_in_convex_polygon_closed_linear(b.front(),a);
}

// A point or a segment: fewer than three distinct vertices in cyclic order.
// The memoized disjoint construction (SolutionBinarySearchDisjoint) builds
// cones from two distinct edges at every vertex, so these regions go to the
// directional maps, which treat them (intersecting_maps.cpp).
inline bool lower_dimensional_region(const std::vector<Vector2> &polygon) {
    size_t count=0;
    for(size_t i=0;i<polygon.size();++i)
        if(i==0||polygon[i]!=polygon[i-1]) ++count;
    if(polygon.size()>1&&polygon.front()==polygon.back()) --count;
    return count<3;
}
inline bool has_lower_dimensional_region(const std::vector<std::vector<Vector2>> &polygons) {
    for(const auto &p:polygons) if(lower_dimensional_region(p)) return true;
    return false;
}

// O(kn) in all: each polygon meets k-1 others in O(|P_i|+|P_h|) time.
inline bool pairwise_disjoint_double(const std::vector<std::vector<Vector2>> &polygons) {
    std::vector<LexicographicChains> chains;chains.reserve(polygons.size());
    for(const auto &p:polygons) chains.push_back(lexicographic_chains(p));
    for(size_t i=0;i<polygons.size();i++)
        for(size_t j=i+1;j<polygons.size();j++)
            if(convex_polygons_intersect_or_touch_double(polygons[i],chains[i],polygons[j],chains[j]))
                return false;
    return true;
}

}  // namespace tpp::detail
