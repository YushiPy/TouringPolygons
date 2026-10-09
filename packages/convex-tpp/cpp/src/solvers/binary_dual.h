#pragma once
#include "binary_certificate.h"
#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>

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
// One representable value outward (the rounding of one operation).
inline double below(double x) {return std::isfinite(x)?CycleInterval::down(x):x;}
inline double above(double x) {return std::isfinite(x)?CycleInterval::up(x):x;}
// Bounds of min over the region of (v-o).(a-b), and an upper bound of the
// region's width along a-b, for binary64 a, b, o and vertices. Each value
// t = dx*nx + dy*ny (dx = v.x-o.x, nx = a.x-b.x, ...) is plain binary64; in
// the standard model (each operation has relative error at most u = 2^-53,
// plus 2^-1075 absolute for an underflowing product; a fused multiply-add is
// no less accurate) it is within 4.0000001u(|dx||nx| + |dy||ny|) + 2^-1074 of
// the exact value. E below exceeds that for every vertex: 2^-50 = 8u covers
// the factor and the rounding of E itself, with mx, my the largest computed
// coordinate differences.
struct Support { double lower=-INFINITY, upper=INFINITY, width=INFINITY; };
template<bool Width=false,class Vertices>
inline Support support_bounds(const Vertices &p,Vector2 o,Vector2 a,Vector2 b) {
    const double nx=a.x-b.x,ny=a.y-b.y;
    // Equal vectors (a straight or doubly zero visit): every term is 0.
    if(nx==0&&ny==0)return {0,0,0};
    double low=INFINITY,high=-INFINITY,mx=0,my=0;
    for(const auto &v:p) {
        const double dx=v.x-o.x,dy=v.y-o.y,t=dx*nx+dy*ny;
        low=std::min(low,t);
        if constexpr(Width)high=std::max(high,t);
        mx=std::max(mx,std::abs(dx));my=std::max(my,std::abs(dy));
    }
    const double error=0x1p-50*(mx*std::abs(nx)+my*std::abs(ny))+0x1p-1070;
    if(!std::isfinite(low)||!std::isfinite(error))return {};
    Support result{below(low-error),above(low+error)};
    // A point has width exactly zero (the pricing treats zero widths apart).
    if constexpr(Width)if(std::isfinite(high))result.width=std::size(p)==1?0:std::max(0.0,above(above(high-low)+2*error));
    return result;
}
inline Support support_bounds(const std::vector<Vector2> &p,Vector2 o,Vector2 a,Vector2 b) {
    if(p.empty())throw std::invalid_argument("Empty binary dual region");
    return support_bounds<false>(p,o,a,b);
}
inline Support point_bounds(Vector2 v,Vector2 o,Vector2 a,Vector2 b) {
    return support_bounds<false>(std::array<Vector2,1>{v},o,a,b);
}
// A unit-disk vector along b-a (zero for a zero link or an unproved norm).
// It is shortened by 2^-49 so that, in the standard model, a computed
// squared norm at most 1-2^-50 proves the exact one at most 1; the interval
// check is the fallback. Rounding only changes which feasible vector is used.
inline Vector2 unit_direction(Vector2 a,Vector2 b) {
    const double dx=b.x-a.x,dy=b.y-a.y,r=std::sqrt(dx*dx+dy*dy);
    if(!(r>0)||!std::isfinite(r))return {};
    const double scale=(1-0x1p-49)/r;
    Vector2 q{dx*scale,dy*scale};
    for(int attempt=0;attempt<4;++attempt) {
        if(q.x*q.x+q.y*q.y<=1-0x1p-50||binary_dual_feasible(q))return q;
        q={q.x*(1-0x1p-50),q.y*(1-0x1p-50)};
    }
    return {};
}
}
