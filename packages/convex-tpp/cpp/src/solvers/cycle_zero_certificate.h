#pragma once
#include "cycle_internal.h"
#include "cycle_execution.h"
#include <algorithm>
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
// Membership has already been proved. The tangent cone of a convex region
// generates all v-q, so its generators suffice for the same support inequality.
inline bool cycle_nonzero_contact_support(const ConvexRationalPolygon &p,
        const ConvexRationalPoint &q,const ConvexRationalPoint &before,
        const ConvexRationalPoint &after,size_t &predicates) {
    using P=ConvexRationalPoint;
    if(p.size()==1)return true;
    const auto a2=before.dot(before),b2=after.dot(after);
    auto sign=[&](const P &direction) {
        ++predicates;
        return cycle_normalized_difference_sign(before.dot(direction),a2,after.dot(direction),b2);
    };
    if(p.size()==2)return sign(p[0]-q)>=0&&sign(p[1]-q)>=0;
    // A straight-through contact has zero gradient in every feasible cone.
    if(before.cross(after)==0&&before.dot(after)>0)return true;
    for(size_t i=0;i<p.size();++i)if(q==p[i]) {
        const P left=p[(i+p.size()-1)%p.size()]-q,right=p[(i+1)%p.size()]-q;
        if(left.cross(right)!=0)return sign(left)>=0&&sign(right)>=0;
        // A redundant collinear vertex is an edge-interior contact, not a
        // pointed cone: the inward condition is necessary as well as tangency.
        return sign(right)==0&&sign(P{-right.y,right.x})>=0;
    }
    for(size_t i=0;i<p.size();++i) {
        cycle_checkpoint();
        const P edge=p[(i+1)%p.size()]-p[i];
        if(edge.cross(q-p[i])==0)
            return sign(edge)==0&&sign(P{-edge.y,edge.x})>=0;
    }
    // In the strict interior, the tangent cone is the whole plane and only
    // the zero gradient (tested above) satisfies every support direction.
    return false;
}
inline bool cycle_support_certificate(const ConvexRationalPolygons &p,
        const ConvexRationalPolygon &q,size_t &predicates) {
    const size_t k=q.size();size_t first=0;
    while(first<k&&q[first]==q[(first+1)%k])++first;
    if(first==k)return true; // Feasible zero cycle attains the universal lower bound.
    size_t start=(first+1)%k,used=0;
    while(used<k) {
        cycle_checkpoint();
        std::vector<size_t> block{start};
        while(block.size()+used<k&&q[block.back()]==q[(block.back()+1)%k])block.push_back((block.back()+1)%k);
        const size_t end=block.back();
        const auto before=q[start]-q[(start+k-1)%k],after=q[(end+1)%k]-q[end];
        if(block.size()==1) {
            if(!cycle_nonzero_contact_support(p[start],q[start],before,after,predicates))return false;
        } else if(!cycle_zero_block_certificate(p,q,block,before,after,predicates))return false;
        used+=block.size();start=(end+1)%k;
    }
    return true;
}
} // namespace tpp::detail
