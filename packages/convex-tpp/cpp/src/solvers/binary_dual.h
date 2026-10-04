#pragma once
#include "binary_certificate.h"
#include <numeric>

namespace tpp::detail {
inline bool binary_dual_feasible(Vector2 u) {
    if(!u.is_finite())return false;
    const auto squared=CycleInterval(u.x).square()+CycleInterval(u.y).square();
    return squared.finite()&&squared.hi<=1;
}
// Select a nearby binary vector, then prove that this specific vector lies
// in the disk. Interval endpoints alone are not dual-feasibility proofs.
inline Vector2 binary_dual_vector(const IntervalPoint &u) {
    Vector2 q{std::midpoint(u.x.lo,u.x.hi),std::midpoint(u.y.lo,u.y.hi)};
    for(int attempt=0;attempt<8;++attempt) {
        if(binary_dual_feasible(q))return q;
        const auto squared=CycleInterval(q.x).square()+CycleInterval(q.y).square();
        const auto norm=squared.sqrt();
        if(!norm.finite()||norm.hi==0)return {};
        const double scale=CycleInterval::up(std::max(1.0,norm.hi));
        q={q.x/scale,q.y/scale};
    }
    return {}; // Zero is always dual-feasible; only bound strength is lost.
}
inline Vector2 binary_dual_direction(Vector2 a,Vector2 b) {
    const auto d=IntervalPoint(b)-IntervalPoint(a);
    const auto norm=(d.x.square()+d.y.square()).sqrt();
    if(!norm.finite()||norm.hi==0)return {};
    return binary_dual_vector({d.x.divided_by(norm.hi),d.y.divided_by(norm.hi)});
}
}
