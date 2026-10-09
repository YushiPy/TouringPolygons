#pragma once

#include "binary_dual.h"
#include "float_chain.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace tpp::detail {
// Rigorous binary64 pieces of the floating-point oracles: membership proofs,
// enclosed lengths and dual bounds on the original regions. Nothing here
// trusts a construction; a failed proof only loses bound strength.
using FloatRegion=std::vector<Vector2>;

// Exact sign of orient(a,b,q): intervals first, then the 128-bit determinant.
inline std::optional<int> proved_orientation(Vector2 a,Vector2 b,Vector2 q) {
    const auto side=(IntervalPoint(b)-IntervalPoint(a)).cross(IntervalPoint(q)-IntervalPoint(a));
    if(!side.finite())return {};
    if(side.lo>0)return 1;
    if(side.hi<0)return -1;
    return dyadic_orientation(a,b,q);
}

// The same closed convex region as prepare_cycle_polygons, decided without
// rational arithmetic: consecutive and closing duplicates removed, a point, a
// segment, or a strictly convex counter-clockwise polygon. Every turn sign is
// exact; an undecided sign, a nonconvex or multiply wound boundary and a
// degenerate polygon return nothing (the exact path then reports it).
inline std::optional<FloatRegion> binary_region(const FloatRegion &input) {
    FloatRegion p;
    for(const auto &v:input) {
        if(!v.is_finite())return {};
        if(p.empty()||!(v==p.back()))p.push_back(v);
    }
    while(p.size()>1&&p.front()==p.back())p.pop_back();
    if(p.empty())return {};
    if(p.size()<=2)return p;
    const size_t n=p.size();
    std::vector<int> turns(n);
    bool positive=false,negative=false;
    for(size_t i=0;i<n;++i) {
        const auto turn=proved_orientation(p[(i+n-1)%n],p[i],p[(i+1)%n]);
        if(!turn)return {};
        turns[i]=*turn;positive|=*turn>0;negative|=*turn<0;
    }
    if(positive==negative)return {}; // nonconvex, or all collinear
    if(negative){std::reverse(p.begin(),p.end());std::reverse(turns.begin(),turns.end());for(auto &t:turns)t=-t;}
    FloatRegion strict;size_t winding=0;
    for(size_t i=0;i<n;++i) {
        const auto before=p[i]-p[(i+n-1)%n],after=p[(i+1)%n]-p[i];
        // The sign of a rounded difference is the sign of the exact one.
        if(before.y<=0&&after.y>0)++winding;
        if(turns[i]) {strict.push_back(p[i]);continue;}
        const auto forward=(IntervalPoint(p[i])-IntervalPoint(p[(i+n-1)%n])).dot(IntervalPoint(p[(i+1)%n])-IntervalPoint(p[i]));
        if(!(forward.lo>0))return {}; // backtracking or undecided
    }
    if(winding!=1||strict.size()<3)return {};
    return strict;
}

// Proof that q lies in the closed CCW polygon (three or more vertices): each
// edge sign from an interval, or the 128-bit determinant if it straddles zero.
// An undecided sign counts as outside.
inline bool proved_inside(Vector2 q,const FloatRegion &p) {
    if(!q.is_finite())return false;
    for(size_t i=0;i<p.size();++i) {
        const auto &a=p[i],&b=p[(i+1)%p.size()];
        const auto side=(IntervalPoint(b)-IntervalPoint(a)).cross(IntervalPoint(q)-IntervalPoint(a));
        if(side.hi<0)return false;
        if(side.lo>=0)continue;
        const auto sign=dyadic_orientation(a,b,q);
        if(!sign||*sign<0)return false;
    }
    return true;
}

inline Vector2 vertex_mean(const FloatRegion &p) {
    Vector2 c{};
    for(const auto &v:p)c+=v/double(p.size());
    return c;
}

// A contact whose membership is proved: `box` encloses an exact point of the
// region and `point` is a binary64 representative. They coincide except on a
// segment, where a+t(b-a) is on the segment for every binary64 t in [0,1] but
// need not be representable; lengths are then bounded through the box.
struct ProvedContact {IntervalPoint box;Vector2 point;};
inline std::optional<ProvedContact> prove_contact(Vector2 q,const FloatRegion &p) {
    if(!q.is_finite()||p.empty())return {};
    if(p.size()==1)return ProvedContact{IntervalPoint(p[0]),p[0]};
    if(p.size()==2) {
        const auto a=p[0],b=p[1];
        if(q==a||q==b)return ProvedContact{IntervalPoint(q),q};
        if(const auto sign=dyadic_orientation(a,b,q);sign&&*sign==0) {
            const auto e=IntervalPoint(b)-IntervalPoint(a);
            const auto from_a=(IntervalPoint(q)-IntervalPoint(a)).dot(e),to_b=(IntervalPoint(b)-IntervalPoint(q)).dot(e);
            if(from_a.lo>=0&&to_b.lo>=0)return ProvedContact{IntervalPoint(q),q};
        }
        const auto e=b-a;
        const double t=std::clamp((q-a).dot(e)/e.dot(e),0.0,1.0);
        if(!std::isfinite(t))return {};
        if(t==0)return ProvedContact{IntervalPoint(a),a};
        if(t==1)return ProvedContact{IntervalPoint(b),b};
        const IntervalPoint A(a),E=IntervalPoint(b)-A;
        const IntervalPoint box{A.x+CycleInterval(t)*E.x,A.y+CycleInterval(t)*E.y};
        if(!box.x.finite()||!box.y.finite())return {};
        return ProvedContact{box,{a.x+t*e.x,a.y+t*e.y}};
    }
    if(proved_inside(q,p))return ProvedContact{IntervalPoint(q),q};
    // A move toward the vertex mean: tiny for a rounded boundary contact,
    // up to the mean itself for a construction proposal beyond an edge. The
    // length bound is computed afterwards, so a move only costs bound quality.
    const auto center=vertex_mean(p);
    for(double fraction:{0x1p-45,0x1p-40,0x1p-30,0x1p-20,0x1p-10,0x1p-6,0x1p-3,0.5,1.0}) {
        const auto candidate=q+(center-q)*fraction;
        if(proved_inside(candidate,p))return ProvedContact{IntervalPoint(candidate),candidate};
    }
    return {};
}

inline bool degenerate(const IntervalPoint &p) {return p.x.lo==p.x.hi&&p.y.lo==p.y.hi;}

// Upper bound for the length of the chain (closed when cyclic) through the
// exact points enclosed by the boxes; equal binary points are a zero link.
inline double enclosed_length_upper(const std::vector<IntervalPoint> &boxes,bool cyclic) {
    const size_t n=boxes.size(),links=cyclic?n:n-1;
    if(n<2)return 0;
    CycleInterval length;
    for(size_t l=0;l<links;++l) {
        const auto &a=boxes[l],&b=boxes[(l+1)%n];
        if(degenerate(a)&&degenerate(b)&&a.x.lo==b.x.lo&&a.y.lo==b.y.lo)continue;
        const auto d=b-a;
        length=length+(d.x.square()+d.y.square()).sqrt();
    }
    return length.finite()?length.hi:INFINITY;
}

// Cycle dual: for |u_i| <= 1, sum_i |q_(i+1)-q_i| >= sum_i u_i.(q_(i+1)-q_i)
// = sum_i q_i.(u_(i-1)-u_i) >= sum_i min_{v in P_i} (v-r).(u_(i-1)-u_i) for
// every feasible cycle q and any reference r (the coefficients sum to zero).
// Each proposal is replaced by a binary vector proved to lie in the disk (or
// zero), and the sum is enclosed with directed rounding.
inline double cycle_dual_lower(const std::vector<FloatRegion> &regions,const std::vector<Vector2> &proposal,Vector2 reference) {
    const size_t k=regions.size();
    if(proposal.size()!=k||!reference.is_finite())return -INFINITY;
    std::vector<IntervalPoint> u;u.reserve(k);
    for(const auto &v:proposal) {
        if(!v.is_finite()){u.emplace_back();continue;}
        u.emplace_back(binary_dual_vector(IntervalPoint(v)));
    }
    const IntervalPoint origin(reference);
    CycleInterval dual;
    for(size_t i=0;i<k;++i) {
        const auto normal=u[(i+k-1)%k]-u[i];
        CycleInterval support(INFINITY);
        for(const auto &v:regions[i]) {
            const auto term=normal.dot(IntervalPoint(v)-origin);
            support.lo=std::min(support.lo,term.lo);
            support.hi=std::min(support.hi,term.hi);
        }
        dual=dual+support;
    }
    return dual.finite()?dual.lo:-INFINITY;
}

// Link directions of a closed chain of representatives. Links no longer than
// short_link (zero ones included) have unreliable directions: they keep their
// own, borrow the nearest long link's on either side, or use zero. Length only
// selects proposals; every proposal is a valid dual.
inline std::vector<std::vector<Vector2>> cycle_dual_proposals(const std::vector<Vector2> &q,double short_link) {
    const size_t k=q.size();
    std::vector<Vector2> base;std::vector<bool> shorter;
    for(size_t i=0;i<k;++i) {
        const auto a=q[i],b=q[(i+1)%k];
        const auto d=IntervalPoint(b)-IntervalPoint(a);
        const auto norm=(d.x.square()+d.y.square()).sqrt();
        base.push_back(a==b?Vector2{}:binary_dual_direction(a,b));
        shorter.push_back(base.back()==Vector2{}||!norm.finite()||norm.hi<=short_link);
    }
    const auto longer=std::count(shorter.begin(),shorter.end(),false);
    if(longer==0||longer==std::ptrdiff_t(k))return {base};
    std::vector<std::vector<Vector2>> policies(4,base);
    for(size_t i=0;i<k;++i)if(shorter[i]) {
        size_t left=i,right=i;
        do left=(left+k-1)%k; while(shorter[left]);
        do right=(right+1)%k; while(shorter[right]);
        policies[1][i]=base[left];policies[2][i]=base[right];policies[3][i]={};
    }
    return policies;
}

// Polish node of a region (float_chain.h), in coordinates relative to
// `reference` divided by `scale`, starting near `warm`: a point is a fixed
// node, a segment a + t(b-a) has the barrier -mu(log t + log(1-t)).
inline ChainNode region_node(const FloatRegion &p,Vector2 reference,double scale,const Vector2 *warm,double fraction) {
    auto local=[&](Vector2 v){return Vector2{(v.x-reference.x)/scale,(v.y-reference.y)/scale};};
    using Kind=ChainNode::Kind;
    ChainNode node;
    if(p.size()==1){node.kind=Kind::Fixed;node.offset=local(p[0]);return node;}
    if(p.size()==2) {
        const auto a=local(p[0]);
        node.kind=Kind::Segment;node.offset=a;node.edge=local(p[1])-a;
        node.faces={{{1,0},0},{{-1,0},-1}};
        double t=.5;
        if(warm) {
            const auto d=p[1]-p[0];
            const double projected=std::clamp((*warm-p[0]).dot(d)/d.dot(d),0.0,1.0);
            if(std::isfinite(projected))t=.5+(projected-.5)*(1-fraction);
        }
        node.y={t,0};
        return node;
    }
    for(size_t j=0;j<p.size();++j) {
        const auto a=local(p[j]),b=local(p[(j+1)%p.size()]);
        Vector2 normal{a.y-b.y,b.x-a.x};
        const double length=normal.length();
        if(!(length>0))continue;
        normal=normal/length;
        node.faces.push_back({normal,normal.dot(a)});
    }
    const auto center=local(vertex_mean(p));
    node.y=center;
    if(warm&&warm->is_finite())node.y=center+(local(*warm)-center)*(1-fraction);
    auto feasible=[&]{return std::all_of(node.faces.begin(),node.faces.end(),[&](const auto &f){return f.normal.dot(node.y)-f.offset>0;});};
    if(!feasible())node.y=center;
    return node;
}

// Original-coordinate contact of a polished node.
inline Vector2 region_contact(const FloatRegion &p,const ChainLevel &level,size_t i,Vector2 reference,double scale) {
    if(p.size()==1)return p[0];
    if(p.size()==2) {
        const double t=std::clamp(level.variables[i].x,0.0,1.0);
        return {p[0].x+t*(p[1].x-p[0].x),p[0].y+t*(p[1].y-p[0].y)};
    }
    return {reference.x+scale*level.points[i].x,reference.y+scale*level.points[i].y};
}
}
