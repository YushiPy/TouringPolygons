#pragma once
#include "cycle_internal.h"
#include "cycle_execution.h"
#include "binary_certificate.h"
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace tpp::detail {
// Complete planar disk/cone reachability: O(b^3 + N_block) exact operations
// and O(b) stored halfplanes for b coincident contacts. No dual sampling or cap.
inline bool cycle_zero_block_certificate(const ConvexRationalPolygons &polygons,
        const ConvexRationalPolygon &q,const std::vector<size_t> &block,
        ConvexRationalPoint before,ConvexRationalPoint after,size_t &predicates) {
    ConvexDualReachability reachable(before,predicates,cycle_checkpoint);
    if(reachable.reaches(after))return true;
    for(size_t i:block) {
        reachable.advance(polygons[i],q[i]);
        // Keeping a dual unchanged is allowed at every later contact.
        if(reachable.reaches(after))return true;
    }
    return false;
}
// Homogeneous integer coordinates (x/w, y/w) with w > 0, from the numerators
// and the (positive) denominators of a rational point, without any gcd.
struct HomogeneousPoint {ConvexInteger x,y,w;};
inline HomogeneousPoint homogeneous_point(const ConvexRationalPoint &p) {
    using boost::multiprecision::numerator;
    using boost::multiprecision::denominator;
    const auto &dx=denominator(p.x),&dy=denominator(p.y);
    if(dx==dy)return {numerator(p.x),numerator(p.y),dx};
    return {numerator(p.x)*dy,numerator(p.y)*dx,dx*dy};
}
// a - b times the positive scalar a.w*b.w (or a.w when the w are equal).
inline ConvexArithmeticPoint<ConvexInteger> homogeneous_difference(const HomogeneousPoint &a,const HomogeneousPoint &b) {
    if(a.w==b.w)return {a.x-b.x,a.y-b.y};
    return {a.x*b.w-b.x*a.w,a.y*b.w-b.y*a.w};
}
// Membership has already been proved. The tangent cone of a convex region
// generates all v-q, so its generators suffice for the same support inequality.
// sign(before.d/|before| - after.d/|after|) and every zero test below are
// unchanged when before, after, d or a cross product's factors are multiplied
// by positive scalars, so all of them use homogeneous integer differences:
// u and w are positive multiples of before and after, and hq is q.
inline bool cycle_nonzero_contact_support(const ConvexRationalPolygon &p,const ConvexRationalPoint &q,
        const HomogeneousPoint &hq,const ConvexArithmeticPoint<ConvexInteger> &u,
        const ConvexArithmeticPoint<ConvexInteger> &w,size_t &predicates,const std::vector<Vector2> *binary=nullptr,
        bool filtered=false) {
    using I=ConvexInteger;
    using H=ConvexArithmeticPoint<I>;
    if(p.size()==1)return true;
    const I a2=u.dot(u),b2=w.dot(w);
    auto sign=[&](const H &d) {
        ++predicates;
        return convex_normalized_difference_sign_integer(u.dot(d),a2,w.dot(d),b2);
    };
    // Vertices in homogeneous coordinates, converted when first needed.
    std::vector<std::optional<HomogeneousPoint>> converted(p.size());
    auto v=[&](size_t i)->const HomogeneousPoint & {
        if(!converted[i])converted[i]=homogeneous_point(p[i]);
        return *converted[i];
    };
    if(p.size()==2)return sign(homogeneous_difference(v(0),hq))>=0&&sign(homogeneous_difference(v(1),hq))>=0;
    // A straight-through contact has zero gradient in every feasible cone.
    if(u.cross(w)==0&&u.dot(w)>0)return true;
    for(size_t i=0;i<p.size();++i)if(q==p[i]) {
        const H left=homogeneous_difference(v((i+p.size()-1)%p.size()),hq),right=homogeneous_difference(v((i+1)%p.size()),hq);
        if(left.cross(right)!=0)return sign(left)>=0&&sign(right)>=0;
        // A redundant collinear vertex is an edge-interior contact, not a
        // pointed cone: the inward condition is necessary as well as tangency.
        return sign(right)==0&&sign(H{-right.y,right.x})>=0;
    }
    // Only an edge whose binary64 enclosure leaves its side undecided can
    // contain q; the exact test decides it.
    // binary, when given, holds the same vertices as binary64 values.
    // filtered: the thread's floating-point environment was verified.
    const auto qi=enclose_rational(q,filtered);
    const bool exact_binary=binary&&binary->size()==p.size();
    auto vertex=[&](size_t i){return exact_binary?IntervalPoint((*binary)[i]):enclose_rational(p[i],filtered);};
    for(size_t i=0;i<p.size();++i) {
        cycle_checkpoint();
        if(filtered_side(vertex(i),vertex((i+1)%p.size()),qi))continue;
        const H edge=homogeneous_difference(v((i+1)%p.size()),v(i));
        if(edge.cross(homogeneous_difference(hq,v(i)))==0)
            return sign(edge)==0&&sign(H{-edge.y,edge.x})>=0;
    }
    // In the strict interior, the tangent cone is the whole plane and only
    // the zero gradient (tested above) satisfies every support direction.
    return false;
}
// The same test from rational before and after.
inline bool cycle_nonzero_contact_support(const ConvexRationalPolygon &p,
        const ConvexRationalPoint &q,const ConvexRationalPoint &before,
        const ConvexRationalPoint &after,size_t &predicates,const std::vector<Vector2> *binary=nullptr) {
    const HomogeneousPoint zero{0,0,1};
    return cycle_nonzero_contact_support(p,q,homogeneous_point(q),homogeneous_difference(homogeneous_point(before),zero),
                                         homogeneous_difference(homogeneous_point(after),zero),predicates,binary,
                                         cycle_interval_environment());
}
inline bool cycle_support_certificate(const ConvexRationalPolygons &p,
        const ConvexRationalPolygon &q,size_t &predicates,
        const std::vector<std::vector<Vector2>> *binary=nullptr) {
    const size_t k=q.size();size_t first=0;
    while(first<k&&q[first]==q[(first+1)%k])++first;
    if(first==k)return true; // Feasible zero cycle attains the universal lower bound.
    const bool filtered=cycle_interval_environment();
    std::vector<HomogeneousPoint> h;h.reserve(k);
    for(const auto &point:q)h.push_back(homogeneous_point(point));
    size_t start=(first+1)%k,used=0;
    while(used<k) {
        cycle_checkpoint();
        std::vector<size_t> block{start};
        while(block.size()+used<k&&q[block.back()]==q[(block.back()+1)%k])block.push_back((block.back()+1)%k);
        const size_t end=block.back(),previous=(start+k-1)%k,next=(end+1)%k;
        if(block.size()==1) {
            if(!cycle_nonzero_contact_support(p[start],q[start],h[start],homogeneous_difference(h[start],h[previous]),
                    homogeneous_difference(h[next],h[end]),predicates,binary?&(*binary)[start]:nullptr,filtered))return false;
        } else if(!cycle_zero_block_certificate(p,q,block,q[start]-q[previous],q[next]-q[end],predicates))return false;
        used+=block.size();start=(end+1)%k;
    }
    return true;
}
} // namespace tpp::detail
