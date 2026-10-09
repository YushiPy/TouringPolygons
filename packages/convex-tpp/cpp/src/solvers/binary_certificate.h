#pragma once

#include "cycle_interval.h"
#include "tpp/convex/rational.h"
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <optional>

namespace tpp::detail {
struct IntervalPoint {
    CycleInterval x,y;
    IntervalPoint() = default;
    explicit IntervalPoint(Vector2 p):x(p.x),y(p.y) {}
    IntervalPoint(CycleInterval a,CycleInterval b):x(a),y(b) {}
    IntervalPoint operator-(const IntervalPoint &p) const {return {x-p.x,y-p.y};}
    CycleInterval dot(const IntervalPoint &p) const {return x*p.x+y*p.y;}
    CycleInterval cross(const IntervalPoint &p) const {return x*p.y-y*p.x;}
};

// Binary64 inputs are dyadic rationals. Scale each axis independently by a
// common power of two. If all signed coordinates fit within 61 magnitude bits,
// differences fit 62 and the determinant fits signed 128 bits (at most 125).
// Wider exponents and unsupported integer arithmetic retain the rational path.
inline std::optional<int> dyadic_orientation(Vector2 a,Vector2 b,Vector2 q) {
#if defined(__SIZEOF_INT128__)
    if(!a.is_finite()||!b.is_finite()||!q.is_finite())return {};
    auto axis=[](std::array<double,3> values,std::array<int64_t,3> &scaled) {
        std::array<uint64_t,3> mantissas,bits;
        std::array<int,3> exponents;
        int minimum=std::numeric_limits<int>::max();
        for(size_t i=0;i<3;++i) {
            bits[i]=std::bit_cast<uint64_t>(values[i]);
            const int exponent=int((bits[i]>>52)&0x7ff);
            mantissas[i]=bits[i]&0x000fffffffffffffULL;
            if(exponent)mantissas[i]|=uint64_t(1)<<52;
            exponents[i]=exponent?exponent-1075:-1074;
            if(mantissas[i])minimum=std::min(minimum,exponents[i]);
        }
        for(size_t i=0;i<3;++i) {
            if(!mantissas[i]){scaled[i]=0;continue;}
            const int shift=exponents[i]-minimum;
            if(int(std::bit_width(mantissas[i]))+shift>61)return false;
            const int64_t magnitude=int64_t(mantissas[i]<<shift);
            scaled[i]=(bits[i]>>63)?-magnitude:magnitude;
        }
        return true;
    };
    std::array<int64_t,3> x,y;
    if(!axis({a.x,b.x,q.x},x)||!axis({a.y,b.y,q.y},y))return {};
    const __int128 determinant=__int128(x[1]-x[0])*(y[2]-y[0])
        -__int128(y[1]-y[0])*(x[2]-x[0]);
    return determinant>0?1:determinant<0?-1:0;
#else
    return {};
#endif
}

// Membership signs that needed rational arithmetic (neither intervals nor the
// 128-bit determinant decided them), per thread; callers read differences.
inline thread_local std::size_t rational_membership_predicates=0;

// Binary and exact polygons must describe the same CCW convex boundary.
// An ambiguous interval sign uses only the original binary rational inputs.
inline bool interval_convex_contains(Vector2 q,const std::vector<Vector2> &p,
        const ConvexRationalPolygon &exact,size_t &predicates) {
    if(!q.is_finite()||p.size()<3||p.size()!=exact.size())return false;
    for(const auto &v:p)if(q==v)return true;
    std::optional<ConvexRationalPoint> rational_q;
    for(size_t i=0;i<p.size();++i) {
        const size_t next=(i+1)%p.size();
        const auto side=(IntervalPoint(p[next])-IntervalPoint(p[i])).cross(
            IntervalPoint(q)-IntervalPoint(p[i]));
        if(side.hi<0)return false;
        if(side.lo>=0)continue;
        ++predicates;
#ifdef TPP_HAS_DYADIC_MEMBERSHIP
        if(const auto sign=dyadic_orientation(p[i],p[next],q)) {
            if(*sign<0)return false;
            continue;
        }
#endif
        ++rational_membership_predicates;
        if(!rational_q)rational_q.emplace(q);
        if((exact[next]-exact[i]).cross(*rational_q-exact[i])<0)return false;
    }
    return true;
}

// One immutable prepared polygon, two points (typically proposal and repair).
// Both successful and rejected membership proofs can be reused; this never
// reuses an objective or a dual. Binary and exact geometry must stay identical.
struct BinaryContactMemo {
    const ConvexRationalPolygon *geometry=nullptr;
    struct Slot {std::array<uint64_t,2> key{};bool ready=false,value=false;};
    std::array<Slot,2> slots{};
    size_t next=0;
    size_t queries=0,hits=0;
    bool contains(Vector2 q,const std::vector<Vector2> &binary,
            const ConvexRationalPolygon &exact,size_t &predicates) {
        ++queries;
        const std::array<uint64_t,2> point{std::bit_cast<uint64_t>(q.x),std::bit_cast<uint64_t>(q.y)};
        if(geometry!=&exact) {
            for(auto &slot:slots)slot.ready=false;
            geometry=&exact;next=0;
        }
        for(const auto &slot:slots)if(slot.ready&&point==slot.key){++hits;return slot.value;}
        const bool value=interval_convex_contains(q,binary,exact,predicates);
        slots[next]={point,true,value};next^=1;
        return value;
    }
};
}
