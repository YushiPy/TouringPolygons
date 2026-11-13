#pragma once
#include "binary_certificate.h"
#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>

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
// all but rounding ties. A vertex certified lower moves the window there.
// Anything uncertified (three vertices within rounding: nearly collinear)
// returns nothing (use the loop).
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
    // The window [first, last] in boundary order, at most two vertices. The
    // computed walk stops within the rounding of t, which is coarse when o is
    // far from the region, so at a near tie (an edge almost orthogonal to
    // a-b) an interval may still certify a neighbor lower, or the vertex just
    // beyond the tie. The walk then continues on those certified signs. Along
    // the boundary the exact values fall to their minimum and then rise, so
    // each such step strictly lowers the exact value.
    size_t first=w,last=w;
    for(int step=0;;++step) {
        if(step==64)return {};
        const auto left=rise(w,w+m-1),right=rise(w,w+1);
        if(left&&*left<0){w=(w+m-1)%m;continue;}
        if(right&&*right<0){w=(w+1)%m;continue;}
        if(!left&&!right)return {}; // three vertices within rounding
        first=left?w:(w+m-1)%m;last=right?w:(w+1)%m;
        if(first==w&&last==w)break;
        // A widened side needs its outer neighbor certified no lower.
        const size_t outer=first!=w?(first+m-1)%m:(last+1)%m;
        const auto beyond=rise(first!=w?first:last,outer);
        if(!beyond)return {};
        if(*beyond>0)break;
        w=outer;
    }
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

// A run of short links (coincident contacts, up to rounding) between the long
// links `left` and `right`: its regions share (about) one point, and a KKT
// dual splits the turn right-left among them, each taking a part in the
// normal cone of its own contact, with the partial sums in the unit disk.
// The cones are generated by the outward unit normals of the edges through
// each contact (within 1e-9 relative). A pair of generators with nonnegative
// weights that sums to the turn is a basic split; the one with the smallest
// binary64 loss on the run's regions is returned, as the directions of the
// run's links (regions.size()-1 of them). Binary64 only selects the proposal:
// callers enclose it in the unit disk and prove the bound with intervals.
// The scan is linear in the run's vertices, so regions with more than
// kSplitWitnessVertices vertices are left to the other proposals, keeping the
// oracle's logarithmic cost on large polygons.
constexpr size_t kSplitWitnessVertices=64;
inline std::optional<std::vector<Vector2>> split_witness(const std::vector<const std::vector<Vector2>*> &regions,
        const std::vector<Vector2> &contacts,Vector2 left,Vector2 right,Vector2 reference) {
    struct Generator { size_t region; Vector2 normal; };
    std::vector<Generator> generators;
    const size_t count=regions.size();
    if(count<2||contacts.size()!=count)return {};
    for(size_t k=0;k<count;++k) {
        const auto &polygon=*regions[k];
        if(polygon.size()<3||polygon.size()>kSplitWitnessVertices)return {};
        double area=0;
        for(size_t j=0;j<polygon.size();++j)area+=polygon[j].cross(polygon[(j+1)%polygon.size()]);
        const double orientation=area>0?1:-1;
        const Vector2 c=contacts[k];
        for(size_t j=0;j<polygon.size();++j) {
            const Vector2 a=polygon[j],edge=polygon[(j+1)%polygon.size()]-a;
            const double length=std::hypot(edge.x,edge.y);
            if(!(length>0))continue;
            const double scale=std::max({1.0,std::abs(c.x),std::abs(c.y),length});
            const double distance=std::abs(edge.cross(c-a))/length;
            const double along=edge.dot(c-a)/(length*length);
            if(!(distance<=1e-9*scale)||along<-1e-9*scale/length||along>1+1e-9*scale/length)continue;
            generators.push_back({k,Vector2{edge.y,-edge.x}*(orientation/length)});
        }
    }
    const Vector2 turn=right-left;
    std::optional<std::vector<Vector2>> best;double best_loss=INFINITY;
    auto consider=[&](const std::vector<std::pair<size_t,Vector2>> &parts) {
        std::vector<Vector2> share(count);
        for(const auto &[k,v]:parts)share[k]+=v;
        std::vector<Vector2> directions;Vector2 w=left;
        for(size_t k=0;k+1<count;++k) {
            w+=share[k];
            if(!(std::hypot(w.x,w.y)<=1+1e-9))return;
            directions.push_back(w);
        }
        double loss=0;Vector2 previous=left;
        for(size_t k=0;k<count;++k) {
            const Vector2 next=k+1<count?directions[k]:right;
            const Vector2 normal=previous-next;
            double support=INFINITY;
            for(const auto &v:*regions[k])support=std::min(support,normal.dot(v-reference));
            loss+=normal.dot(contacts[k]-reference)-support;
            previous=next;
        }
        if(loss<best_loss){best_loss=loss;best=std::move(directions);}
    };
    const double turn_norm=std::hypot(turn.x,turn.y);
    for(size_t a=0;a<generators.size();++a) {
        const auto &g=generators[a];
        if(std::abs(g.normal.cross(turn))<=1e-12*turn_norm&&g.normal.dot(turn)>0)
            consider({{g.region,g.normal*g.normal.dot(turn)}});
        for(size_t c=a+1;c<generators.size();++c) {
            const auto &h=generators[c];
            const double det=g.normal.cross(h.normal);
            if(std::abs(det)<=1e-12)continue;
            const double alpha=turn.cross(h.normal)/det,beta=g.normal.cross(turn)/det;
            if(alpha<-1e-12||beta<-1e-12)continue;
            consider({{g.region,g.normal*std::max(alpha,0.0)},{h.region,h.normal*std::max(beta,0.0)}});
        }
    }
    return best;
}
}
