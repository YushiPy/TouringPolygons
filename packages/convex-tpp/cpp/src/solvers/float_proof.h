#pragma once
#include <array>

#include "binary_dual.h"
#include "float_chain.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>
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

// Exact sign of orient(a,b,q): the static filter, then an interval, then the
// 128-bit determinant; nothing when undecided.
inline std::optional<int> filtered_orientation(Vector2 a,Vector2 b,Vector2 q) {
    const double l=(b.x-a.x)*(q.y-a.y),r=(b.y-a.y)*(q.x-a.x),det=l-r;
    const double bound=0x1p-50*(std::abs(l)+std::abs(r))+0x1p-1069;
    if(std::isfinite(det)&&std::isfinite(bound)) {
        if(det>bound)return 1;
        if(det<-bound)return -1;
    }
    return proved_orientation(a,b,q);
}
// Proof that q lies in the closed CCW polygon (three or more vertices): each
// edge sign from an interval, or the 128-bit determinant if it straddles zero.
// An undecided sign counts as outside.
inline bool proved_inside_linear(Vector2 q,const FloatRegion &p);
// The same proof in O(log m) for a strictly convex polygon: a binary search
// in the fan from p[0] picks a triangle (p[0], p[j], p[j+1]) of the polygon,
// and three exact signs prove q in that closed triangle, hence in the
// polygon. q exactly inside the wedge but beyond the edge p[j]p[j+1] is
// outside. Anything undecided falls back to every edge sign.
inline bool proved_inside(Vector2 q,const FloatRegion &p) {
    if(!q.is_finite())return false;
    const size_t m=p.size();
    if(m<16)return proved_inside_linear(q,p);
    const Vector2 o=p[0];
    // Wedges j = 1..m-2 between the rays o->p[j] and o->p[j+1].
    size_t lo=1,hi=m-1;
    while(hi-lo>1) {
        const size_t mid=lo+(hi-lo)/2;
        if((p[mid]-o).cross(q-o)>=0)lo=mid;else hi=mid;
    }
    for(size_t j:{lo,lo>1?lo-1:lo,lo+1<m-1?lo+1:lo}) {
        const auto left=filtered_orientation(o,p[j],q),right=filtered_orientation(o,p[j+1],q);
        if(!left||!right)break;
        if(*left<0||*right>0)continue; // not in this closed wedge
        const auto edge=filtered_orientation(p[j],p[j+1],q);
        if(!edge)break;
        return *edge>=0;
    }
    return proved_inside_linear(q,p);
}
inline bool proved_inside_linear(Vector2 q,const FloatRegion &p) {
    if(!q.is_finite())return false;
    for(size_t i=0;i<p.size();++i) {
        const auto &a=p[i],&b=p[(i+1)%p.size()];
        // Static filter first: in the standard model the binary64
        // determinant is within 2^-50(|l|+|r|) + 2^-1069 of the exact one
        // (Shewchuk's orient2d bound is 3.0000001u(|l|+|r|) without underflow).
        const double l=(b.x-a.x)*(q.y-a.y),r=(b.y-a.y)*(q.x-a.x),det=l-r;
        const double bound=0x1p-50*(std::abs(l)+std::abs(r))+0x1p-1069;
        if(std::isfinite(det)&&std::isfinite(bound)) {
            if(det>bound)continue;
            if(det<-bound)return false;
        }
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
inline std::optional<ProvedContact> prove_contact(Vector2 q,const FloatRegion &p,const Vector2 *mean=nullptr) {
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
    const auto center=mean?*mean:vertex_mean(p);
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
// zero). Each support is the binary64 minimum widened by its a priori
// rounding-error bound (support_bounds, binary_dual.h), and the sum is
// rounded down one value per addition.
// A region held directly or through a pointer.
inline const FloatRegion &region_ref(const FloatRegion &region) {return region;}
inline const FloatRegion &region_ref(const FloatRegion *region) {return *region;}
template<class Regions>
inline double cycle_dual_lower(const Regions &regions,const std::vector<Vector2> &proposal,Vector2 reference) {
    const size_t k=regions.size();
    if(proposal.size()!=k||!reference.is_finite())return -INFINITY;
    std::vector<Vector2> u;u.reserve(k);
    for(const auto &v:proposal)u.push_back(v.is_finite()?binary_dual_vector(IntervalPoint(v)):Vector2{});
    double dual=0;
    for(size_t i=0;i<k;++i) {
        const double term=support_bounds<false>(region_ref(regions[i]),reference,u[(i+k-1)%k],u[i]).lower;
        if(!std::isfinite(term))return -INFINITY;
        dual=dual==0||term==0?dual+term:below(dual+term);
    }
    return dual;
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

// The policies above give every short link the same rule, but a cycle with
// several blocks of coincident contacts usually needs a different vector in
// each. D(u) separates over the runs of short links: a run j..j+m-1 between
// the long links l = j-1 and j+m only enters the supports of regions
// j..j+m, whose other vectors are those two long links. Each run therefore
// picks its own vectors, by dynamic programming over the run, among a few
// candidates (both neighboring long directions, their midpoint, the link's
// own direction and zero) to maximize its part of the bound. The choice only
// uses the same rounded supports; any choice is a valid dual.
template<class Regions>
inline std::vector<Vector2> cycle_dual_by_runs(const Regions &regions,const std::vector<Vector2> &q,
        double short_link,Vector2 reference) {
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
    if(longer==0||longer==std::ptrdiff_t(k)||regions.size()!=k)return base;
    auto term=[&](size_t region,Vector2 previous,Vector2 current) {
        const double value=support_bounds<false>(region_ref(regions[region]),reference,previous,current).lower;
        return std::isfinite(value)?value:-INFINITY;
    };
    auto result=base;
    size_t first=0;while(shorter[first])++first;
    for(size_t step=0;step<k;) {
        const size_t left=(first+step)%k;
        size_t m=0;while(shorter[(left+1+m)%k])++m;
        step+=m+1;
        if(m==0)continue;
        const size_t right=(left+1+m)%k;
        const Vector2 in=base[left],out=base[right];
        const Vector2 middle=binary_dual_vector(IntervalPoint(Vector2{(in.x+out.x)/2,(in.y+out.y)/2}));
        constexpr size_t C=5;
        std::vector<std::array<Vector2,C>> candidates(m);
        for(size_t r=0;r<m;++r)candidates[r]={in,out,middle,base[(left+1+r)%k],Vector2{}};
        // value[r][c]: best sum of the supports of regions left+1..left+1+r with
        // link left+1+r taking candidate c; choice[r][c] its predecessor.
        std::vector<std::array<double,C>> value(m);std::vector<std::array<size_t,C>> choice(m);
        for(size_t c=0;c<C;++c)value[0][c]=term((left+1)%k,in,candidates[0][c]);
        for(size_t r=1;r<m;++r)for(size_t c=0;c<C;++c) {
            value[r][c]=-INFINITY;choice[r][c]=0;
            for(size_t b=0;b<C;++b) {
                const double v=value[r-1][b]+term((left+1+r)%k,candidates[r-1][b],candidates[r][c]);
                if(v>value[r][c]){value[r][c]=v;choice[r][c]=b;}
            }
        }
        size_t best=0;double top=-INFINITY;
        for(size_t c=0;c<C;++c) {
            const double v=value[m-1][c]+term(right,candidates[m-1][c],out);
            if(v>top){top=v;best=c;}
        }
        if(!std::isfinite(top))continue;
        for(size_t r=m;r-->0;) {
            result[(left+1+r)%k]=candidates[r][best];
            if(r>0)best=choice[r][best];
        }
    }
    return result;
}

// Inward unit normal and offset of the chord (p[a], p[b]) of a
// counterclockwise polygon in the polish's coordinates (relative to
// `reference`, divided by `scale`); nothing for a vanishing chord.
inline std::optional<ChainFace> chord_face(const FloatRegion &p,size_t i,size_t j,Vector2 reference,double scale) {
    auto local=[&](Vector2 v){return Vector2{(v.x-reference.x)/scale,(v.y-reference.y)/scale};};
    const auto a=local(p[i]),b=local(p[j]);
    Vector2 normal{a.y-b.y,b.x-a.x};
    const double length=normal.length();
    if(!(length>0))return {};
    normal=normal/length;
    return ChainFace{normal,normal.dot(a)};
}
// Regions with at least this many vertices are polished on an inner
// approximation (region_node); the others keep every face.
constexpr size_t kPolishWindowVertices=64,kPolishWindowHalf=4,kPolishSkeleton=16;
// The wedge (p[0], p[j], p[j+1]) of the fan from p[0] holding x, by binary
// search; 0 or m-1 for a point beyond the edge at p[0] on that side.
inline size_t fan_wedge(const FloatRegion &p,Vector2 x) {
    const size_t m=p.size();const Vector2 o=p[0];
    if((p[m-1]-o).cross(x-o)>0)return m-1;
    if((p[1]-o).cross(x-o)<0)return 0;
    size_t lo=1,hi=m-1;
    while(hi-lo>1){const size_t mid=lo+(hi-lo)/2;if((p[mid]-o).cross(x-o)>=0)lo=mid;else hi=mid;}
    return lo;
}
// Largest coordinate difference from `reference` over a region's vertices,
// from the four coordinate extremes of a large region. Only a scale for the
// polish's coordinates: any positive value is valid.
inline double coordinate_reach(const FloatRegion &p,Vector2 reference) {
    double reach=0;
    if(p.size()>=kPolishWindowVertices) {
        constexpr std::pair<double,double> directions[]{{1,0},{-1,0},{0,1},{0,-1}};
        for(const auto &[dx,dy]:directions) {
            const auto &v=p[extreme_vertex(p,dx,dy)];
            reach=std::max({reach,std::abs(v.x-reference.x),std::abs(v.y-reference.y)});
        }
        return reach;
    }
    for(const auto &v:p)reach=std::max({reach,std::abs(v.x-reference.x),std::abs(v.y-reference.y)});
    return reach;
}

// Polish node of a region (float_chain.h), in coordinates relative to
// `reference` divided by `scale`, starting near `warm`: a point is a fixed
// node, a segment a + t(b-a) has the barrier -mu(log t + log(1-t)).
//
// A large region is polished on the hull Q of a subset of its vertices, in
// index order: a skeleton of evenly spaced vertices, the band around the fan
// wedge holding `warm` (with p[0], so Q holds `warm`) and the band around the
// vertex extreme toward `outward` (the direction in which the proposal's
// length decreases: an outward normal at an optimal contact). Q lies in the
// region, so every iterate does. refine(faces, g), called whenever Newton
// stops, inserts the band around the vertex extreme toward g and rebuilds
// the faces, so Q follows the predicted contact; the dual bound D(u) on the
// region then minimizes over a vertex of Q, where Q and the region agree. The barrier problem changes,
// not the bounds: they are proved afterwards on the regions.
inline ChainNode region_node(const FloatRegion &p,Vector2 reference,double scale,const Vector2 *warm,double fraction,
        Vector2 outward={}) {
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
    const size_t m=p.size();
    auto feasible=[&]{return std::all_of(node.faces.begin(),node.faces.end(),[&](const auto &f){return f.normal.dot(node.y)-f.offset>0;});};
    if(m<kPolishWindowVertices) {
        for(size_t j=0;j<m;++j)if(const auto face=chord_face(p,j,(j+1)%m,reference,scale))node.faces.push_back(*face);
        const auto center=local(vertex_mean(p));
        node.y=center;
        if(warm&&warm->is_finite())node.y=center+(local(*warm)-center)*(1-fraction);
        if(!feasible())node.y=center;
        return node;
    }
    std::vector<size_t> subset;
    for(size_t j=0;j<kPolishSkeleton;++j)subset.push_back(j*m/kPolishSkeleton);
    // Into the sorted subset, vertices w-H .. w+H and, at each coarser level
    // 2^j up to the skeleton's spacing, the multiples of 2^j around w: a
    // graded band, fine at w and coarse away from it, so Q follows the region
    // over a wide stretch. The coarse vertices lie on a fixed grid, so a small
    // move of w adds only fine ones. Returns whether any vertex was new.
    auto band=[m](std::vector<size_t> &s,size_t w) {
        bool added=false;
        auto insert=[&](long index) {
            const size_t j=size_t(((index%long(m))+long(m))%long(m));
            const auto at=std::lower_bound(s.begin(),s.end(),j);
            if(at!=s.end()&&*at==j)return;
            s.insert(at,j);added=true;
        };
        for(long d=-long(kPolishWindowHalf);d<=long(kPolishWindowHalf);++d)insert(long(w)+d);
        for(size_t step=2*kPolishWindowHalf;step<=m/(2*kPolishSkeleton);step*=2) {
            const long base=long(w/step*step);
            for(const long k:{-1L,0L,1L,2L})insert(base+k*long(step));
        }
        return added;
    };
    auto rebuild=[&p,reference,scale](const std::vector<size_t> &s,std::vector<ChainFace> &faces) {
        faces.clear();
        for(size_t i=0;i<s.size();++i)if(const auto face=chord_face(p,s[i],s[(i+1)%s.size()],reference,scale))faces.push_back(*face);
    };
    const bool warmed=warm&&warm->is_finite();
    if(warmed)band(subset,fan_wedge(p,*warm));
    if(outward.is_finite()&&(outward.x!=0||outward.y!=0))band(subset,extreme_vertex(p,outward.x,outward.y));
    rebuild(subset,node.faces);
    Vector2 center{};
    for(const size_t j:subset)center+=local(p[j])/double(subset.size());
    node.y=center;
    if(warmed)node.y=center+(local(*warm)-center)*(1-fraction);
    if(!feasible())node.y=center;
    node.refine=[&p,subset=std::move(subset),band,rebuild](std::vector<ChainFace> &faces,Vector2 g) mutable {
        if(!g.is_finite()||(g.x==0&&g.y==0))return false;
        // With the extreme vertex already in Q, the region's term of D(u) is Q's.
        const size_t w=extreme_vertex(p,g.x,g.y);
        if(std::binary_search(subset.begin(),subset.end(),w)||!band(subset,w))return false;
        rebuild(subset,faces);
        return true;
    };
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
