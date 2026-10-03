#pragma once

#include "cycle_interval.h"
#include "tpp/convex/rational.h"
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
        if(!rational_q)rational_q.emplace(q);
        ++predicates;
        if((exact[next]-exact[i]).cross(*rational_q-exact[i])<0)return false;
    }
    return true;
}
}
