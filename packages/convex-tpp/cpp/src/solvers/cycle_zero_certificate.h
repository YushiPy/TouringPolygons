#pragma once
#include "cycle_internal.h"
#include <algorithm>
#include <stdexcept>

namespace tpp::detail {
// Reachable subgradients at a zero-link block. R is the unit disk intersected
// with halfplanes n.u <= n.d/|d|. Every extreme point of R lies on the circle:
// adding a cone retains only old extreme points, and clipping by the disk
// creates new extreme points only on the circle. Each circular vertex has a
// rational direction d; its actual normalization never needs to be evaluated.
class CycleDualReachability {
    using R=ConvexRational;
    using P=ConvexRationalPoint;
    struct Halfplane {
        P normal,direction;
        R dot,squared;
        Halfplane(P n,P d):normal(std::move(n)),direction(std::move(d)),
            dot(normal.dot(direction)),squared(direction.dot(direction)) {}
    };
    std::vector<Halfplane> planes;
    size_t &predicates;
    bool contains(const P &d)const {
        const R squared=d.dot(d);
        for(const auto &h:planes) {
            ++predicates;
            if(cycle_normalized_difference_sign(h.normal.dot(d),squared,h.dot,h.squared)>0)return false;
        }
        return true;
    }
    P support(const P &normal)const {
        if(contains(normal))return normal; // Unconstrained disk maximum.
        P best;R best_dot=0,best_squared=0;bool found=false;
        auto consider=[&](const P &d) {
            if(!contains(d))return;
            const R dot=normal.dot(d),squared=d.dot(d);
            if(!found||cycle_normalized_difference_sign(dot,squared,best_dot,best_squared)>0) {
                best=d;best_dot=dot;best_squared=squared;found=true;
            }
        };
        for(const auto &h:planes) {
            consider(h.direction);
            // The other intersection of a chord line with the unit circle is
            // the reflection in its normal axis. It also has rational direction.
            consider(h.normal*(R(2)*h.normal.dot(h.direction)/h.normal.dot(h.normal))-h.direction);
        }
        if(!found)throw std::logic_error("Empty reachable dual set");
        return best;
    }
public:
    CycleDualReachability(const P &incoming,size_t &count):predicates(count) {
        // The disk and its reverse tangent halfplane intersect in one point.
        planes.emplace_back(-incoming,incoming);
    }
    void advance(const ConvexRationalPolygon &p,const P &q) {
        if(p.size()==1){planes.clear();return;} // A fixed contact imposes no dual constraint.
        std::vector<P> active,tangents;
        auto append_direction=[](std::vector<P> &directions,const P &d) {
            if(d.zero())return;
            if(std::none_of(directions.begin(),directions.end(),[&](const P &e) {
                return d.cross(e)==0&&d.dot(e)>0;
            }))directions.push_back(d);
        };
        if(p.size()==2) {
            append_direction(tangents,p[0]-q);append_direction(tangents,p[1]-q);
        } else {
            for(size_t j=0;j<p.size();++j) {
                const P edge=p[(j+1)%p.size()]-p[j];
                ++predicates;
                if(edge.cross(q-p[j])==0)append_direction(active,edge);
            }
            if(active.empty())return; // Interior: the normal cone is {0}.
        }
        auto allowed=[&](const P &n) {
            if(p.size()==2) {
                const P e=p[1]-p[0];
                if(e.cross(n)!=0)return false;
                if(q==p[0])return e.dot(n)>=0;
                if(q==p[1])return e.dot(n)<=0;
                return true;
            }
            for(const P &e:active)if(e.cross(n)<0)return false;
            return true;
        };
        for(const P &e:active) {
            if(allowed(e))append_direction(tangents,e);
            if(allowed(-e))append_direction(tangents,-e);
        }
        // R' = D intersect (R + N_P(q)), where N is the outward normal cone.
        // Keep normals in its polar (the feasible tangent cone), and add its
        // extreme rays with the OLD set's support values. Circular constraints
        // remain redundant because D is imposed throughout.
        std::vector<Halfplane> next;
        for(const auto &h:planes)if(allowed(h.normal))next.push_back(h);
        for(const P &n:tangents) {
            const P d=support(n);
            std::erase_if(next,[&](const Halfplane &h){return h.normal.cross(n)==0&&h.normal.dot(n)>0;});
            next.emplace_back(n,d);
        }
        planes=std::move(next);
    }
    bool reaches(const P &outgoing)const {return contains(outgoing);}
};

// Complete planar disk/cone reachability: O(b^3 + N_block) rational operations
// and O(b) stored halfplanes for b coincident contacts. No dual sampling or cap.
inline bool cycle_zero_block_certificate(const ConvexRationalPolygons &polygons,
        const ConvexRationalPolygon &q,const std::vector<size_t> &block,
        ConvexRationalPoint before,ConvexRationalPoint after,size_t &predicates) {
    CycleDualReachability reachable(before,predicates);
    if(reachable.reaches(after))return true;
    for(size_t i:block) {
        reachable.advance(polygons[i],q[i]);
        // Keeping a dual unchanged is allowed at every later contact.
        if(reachable.reaches(after))return true;
    }
    return false;
}
inline bool cycle_support_certificate(const ConvexRationalPolygons &p,
        const ConvexRationalPolygon &q,size_t &predicates) {
    const size_t k=q.size();size_t first=0;
    while(first<k&&q[first]==q[(first+1)%k])++first;
    if(first==k)return true; // Feasible zero cycle attains the universal lower bound.
    size_t start=(first+1)%k,used=0;
    while(used<k) {
        std::vector<size_t> block{start};
        while(block.size()+used<k&&q[block.back()]==q[(block.back()+1)%k])block.push_back((block.back()+1)%k);
        const size_t end=block.back();
        const auto before=q[start]-q[(start+k-1)%k],after=q[(end+1)%k]-q[end];
        if(block.size()==1) {
            const auto a2=before.dot(before),b2=after.dot(after);
            for(const auto &v:p[start]) {
                ++predicates;
                if(cycle_normalized_difference_sign(before.dot(v-q[start]),a2,after.dot(v-q[start]),b2)<0)return false;
            }
        } else if(!cycle_zero_block_certificate(p,q,block,before,after,predicates))return false;
        used+=block.size();start=(end+1)%k;
    }
    return true;
}
} // namespace tpp::detail
