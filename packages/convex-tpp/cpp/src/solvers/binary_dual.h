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
// A vertex maximizing dx*v.x+dy*v.y on a counterclockwise polygon without
// collinear vertices, by binary search (KACTL's extrVertex). A guess only:
// every caller certifies or walks from it.
template<class Vertices> inline size_t extreme_vertex(const Vertices &p,double dx,double dy) {
    const long n=long(std::size(p));
    auto value=[&](long i){const auto &v=p[size_t(i%n)];return dx*v.x+dy*v.y;};
    auto cmp=[&](long i,long j){const double a=value(i),b=value(j);return (b>a)-(b<a);};
    auto extr=[&](long i){return cmp(i+1,i)>=0&&cmp(i,i-1+n)<0;};
    if(extr(0))return 0;
    long lo=0,hi=n;
    while(lo+1<hi) {
        const long mid=(lo+hi)/2;
        if(extr(mid))return size_t(mid);
        const int ls=cmp(lo+1,lo),ms=cmp(mid+1,mid);
        if(ls<ms||(ls==ms&&ls==cmp(lo,mid)))hi=mid;else lo=mid;
    }
    return size_t(lo);
}
// The same bounds as the vertex loop of support_bounds in O(log m), for a
// strictly convex counterclockwise polygon. The largest coordinate
// differences come from the coordinate extremes (exact comparisons; rounding
// is monotone). The exact values along the boundary have a single local
// minimum (a vertex, or an edge orthogonal to a-b, typical at an optimal
// edge contact). A window of one or two vertices whose outer neighbors are
// certified no lower by intervals therefore holds an exact minimizer, and
// each computed t is within E of its exact value. The reported value is the
// least computed t over the window and its neighbors, the loop's minimum in
// all but rounding ties. Anything uncertified returns nothing (use the loop).
template<class Vertices>
inline std::optional<Support> support_bounds_logarithmic(const Vertices &p,Vector2 o,double nx,double ny) {
    const size_t m=std::size(p);
    auto at=[&](size_t i)->const Vector2&{return p[i%m];};
    auto coordinate_extreme=[&](double dx,double dy)->std::optional<size_t> {
        size_t i=extreme_vertex(p,dx,dy);
        auto value=[&](size_t j){return dx*at(j).x+dy*at(j).y;};
        for(int step=0;step<64;++step) {
            if(value(i+m-1)>value(i))i=(i+m-1)%m;
            else if(value(i+1)>value(i))i=(i+1)%m;
            else return i;
        }
        return {};
    };
    const auto xmax=coordinate_extreme(1,0),xmin=coordinate_extreme(-1,0),ymax=coordinate_extreme(0,1),ymin=coordinate_extreme(0,-1);
    if(!xmax||!xmin||!ymax||!ymin)return {};
    const double mx=std::max(std::abs(at(*xmax).x-o.x),std::abs(at(*xmin).x-o.x));
    const double my=std::max(std::abs(at(*ymax).y-o.y),std::abs(at(*ymin).y-o.y));
    auto t=[&](size_t i){const double dx=at(i).x-o.x,dy=at(i).y-o.y;return dx*nx+dy*ny;};
    size_t w=extreme_vertex(p,-nx,-ny);
    for(int step=0;;++step) {
        if(step==64)return {};
        if(t(w+m-1)<t(w))w=(w+m-1)%m;
        else if(t(w+1)<t(w))w=(w+1)%m;
        else break;
    }
    const IntervalPoint n{CycleInterval(nx),CycleInterval(ny)};
    // Exact sign of t(to)-t(from) when the interval decides it.
    auto rise=[&](size_t from,size_t to)->std::optional<int> {
        const auto d=(IntervalPoint(at(to))-IntervalPoint(at(from))).dot(n);
        if(d.lo>=0)return 1;
        if(d.hi<0)return -1;
        return {};
    };
    // The window [first, last] in boundary order, at most two vertices.
    size_t first=w,last=w;
    if(const auto left=rise(w,w+m-1);!left)first=(w+m-1)%m;else if(*left<0)return {};
    if(const auto right=rise(w,w+1);!right){if(first!=w)return {};last=(w+1)%m;}else if(*right<0)return {};
    // A widened side needs its outer neighbor certified no lower.
    if(first!=w)if(const auto outer=rise(first,first+m-1);!outer||*outer<0)return {};
    if(last!=w)if(const auto outer=rise(last,last+1);!outer||*outer<0)return {};
    const double low=std::min({t(first+m-1),t(first),t(last),t(last+1)});
    const double error=0x1p-50*(mx*std::abs(nx)+my*std::abs(ny))+0x1p-1070;
    if(!std::isfinite(low)||!std::isfinite(error))return Support{};
    return Support{below(low-error),above(low+error)};
}
// The vertex loop: every vertex once.
template<bool Width=false,class Vertices>
inline Support support_bounds_linear(const Vertices &p,Vector2 o,double nx,double ny) {
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
template<bool Width=false,class Vertices>
inline Support support_bounds(const Vertices &p,Vector2 o,Vector2 a,Vector2 b) {
    const double nx=a.x-b.x,ny=a.y-b.y;
    // Equal vectors (a straight or doubly zero visit): every term is 0.
    if(nx==0&&ny==0)return {0,0,0};
    if constexpr(!Width)
        if(std::size(p)>=64)
            if(auto fast=support_bounds_logarithmic(p,o,nx,ny))return *fast;
    return support_bounds_linear<Width>(p,o,nx,ny);
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
