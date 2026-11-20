#include "cycle_execution.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "tpp/convex/solution.h"
#include "tpp/convex/detail/closed_regions.h"
#include "tpp/convex/detail/convex_contacts.h"

#include <boost/multiprecision/cpp_int.hpp>
#ifdef TPP_EXPERIMENT_NATIVE_DOUBLE
#ifdef TPP_DIRECTIONAL_FILTERED
#include "filtered_rational.h"
#else
#include "native_double_experiment.h"
#endif
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace tpp::detail {
namespace {

#ifdef TPP_EXPERIMENT_NATIVE_DOUBLE
#ifdef TPP_DIRECTIONAL_FILTERED
using Scalar = FilteredRational;
#else
using Scalar = NativeDoubleExperimentScalar;
#endif
#else
using Scalar = tpp::ConvexRational;
#endif

struct Point {
    Scalar x = 0, y = 0;
    Point() = default;
    Point(Scalar x_, Scalar y_) : x(std::move(x_)), y(std::move(y_)) {}
    explicit Point(Vector2 p) : x(p.x), y(p.y) {}
#if !defined(TPP_EXPERIMENT_NATIVE_DOUBLE) || defined(TPP_DIRECTIONAL_FILTERED)
    explicit Point(const ConvexRationalPoint &p) : x(p.x),y(p.y) {}
#endif
    Point operator+(const Point &b) const { return {x+b.x,y+b.y}; }
    Point operator-(const Point &b) const { return {x-b.x,y-b.y}; }
    Point operator-() const { return {-x,-y}; }
    Point operator*(const Scalar &s) const { return {x*s,y*s}; }
    Point operator/(const Scalar &s) const { return {x/s,y/s}; }
    bool operator==(const Point &b) const { return x==b.x && y==b.y; }
    Scalar cross(const Point &b) const { return x*b.y-y*b.x; }
    Scalar dot(const Point &b) const { return x*b.x+y*b.y; }
    bool zero() const { return x==0 && y==0; }
    Vector2 external() const { return {x.convert_to<double>(),y.convert_to<double>()}; }
};

int sign(const Scalar &a) { return a>0 ? 1 : a<0 ? -1 : 0; }

// Sign of a predicate written once over a generic arithmetic `lift`. Filtered
// arithmetic first evaluates only its intervals; the expression DAG is built,
// exactly as before, when they leave the sign open.
template<class Build> int sign_of(const Build &build) {
#ifdef TPP_DIRECTIONAL_FILTERED
    if(Scalar::filtering())
        if(const auto decided=build([](const Scalar &v){return Scalar::Virtual(v);}).sign()) return *decided;
#endif
    return sign(build([](const Scalar &v) -> const Scalar & {return v;}));
}
int cross_sign(const Point &a,const Point &b) {
    return sign_of([&](auto lift){return lift(a.x)*lift(b.y)-lift(a.y)*lift(b.x);});
}
int dot_sign(const Point &a,const Point &b) {
    return sign_of([&](auto lift){return lift(a.x)*lift(b.x)+lift(a.y)*lift(b.y);});
}
// sign(axis.cross(p-origin)) and sign(axis.dot(p-origin)).
int offset_cross_sign(const Point &axis,const Point &p,const Point &origin) {
    return sign_of([&](auto lift){
        return lift(axis.x)*(lift(p.y)-lift(origin.y))-lift(axis.y)*(lift(p.x)-lift(origin.x));});
}
int offset_dot_sign(const Point &axis,const Point &p,const Point &origin) {
    return sign_of([&](auto lift){
        return lift(axis.x)*(lift(p.x)-lift(origin.x))+lift(axis.y)*(lift(p.y)-lift(origin.y));});
}

struct Query {
    Point point, side;
    // Lexicographic secondary directions disambiguate a query that lies on
    // a backwards extension of a map ray. They do not change the requested
    // one-sided limit. Reflections must carry the complete symbolic query.
    Point tie1{1,0}, tie2{0,1};
    bool operator==(const Query &) const = default;
};

// Exact sign at q + epsilon*d, for positive infinitesimal epsilon.
int cross_sign(const Point &axis, const Query &q, const Point &origin, bool break_ties=true) {
    const int constant = offset_cross_sign(axis,q.point,origin);
    if(constant) return constant;
    const int first=cross_sign(axis,q.side);
    if(first || !break_ties) return first;
    const int second=cross_sign(axis,q.tie1);
    return second ? second : cross_sign(axis,q.tie2);
}
int dot_sign(const Point &axis, const Query &q, const Point &origin) {
    const int constant = offset_dot_sign(axis,q.point,origin);
    if(constant) return constant;
    const int first=dot_sign(axis,q.side);
    if(first) return first;
    const int second=dot_sign(axis,q.tie1);
    return second ? second : dot_sign(axis,q.tie2);
}
bool same_direction(const Point &a,const Point &b) {
    return cross_sign(a,b)==0 && dot_sign(a,b)>0;
}
Point reflect_direction(const Point &v,const Point &edge) {
    return edge*(2*v.dot(edge)/edge.dot(edge))-v;
}
Point reflect_point(const Point &p,const Point &a,const Point &edge) {
    return a+reflect_direction(p-a,edge);
}
Query reflect_query(const Query &q,const Point &a,const Point &edge) {
    return {reflect_point(q.point,a,edge),reflect_direction(q.side,edge),
            reflect_direction(q.tie1,edge),reflect_direction(q.tie2,edge)};
}

struct Vertex {
    struct Definition {
        Point point;
        bool edge_intersection = false;
        size_t polygon = 0, edge = 0;
    } definition;
    Point point;
    size_t original_edge = 0;
    Scalar edge_parameter;
    Point before_ray, after_ray;
    bool before_reflects = false, after_reflects = false, ready = false;
    // Both sides reach the vertex straight from the same virtual source, so
    // its cone is the single ray before_ray (== after_ray). Decided from the
    // construction, without comparing the two rays (DirectionalMaps::build_vertex).
    bool single_ray = false;
    std::optional<long double> prefix_length;
};
struct Bounds {
    double min_x, max_x, min_y, max_y;
    Bounds(Vector2 a,Vector2 b)
        : min_x(std::min(a.x,b.x)),max_x(std::max(a.x,b.x)),
          min_y(std::min(a.y,b.y)),max_y(std::max(a.y,b.y)) {}
    bool disjoint(const Bounds &b) const {
        return max_x<b.min_x || b.max_x<min_x || max_y<b.min_y || b.max_y<min_y;
    }
};
struct Map {
    std::vector<Point> original;
    std::vector<Bounds> edge_bounds;
    // Union of edge_bounds: an edge box disjoint from it misses every edge box.
    std::optional<Bounds> bounds;
    std::vector<Point> membership_corners;
    std::vector<Vertex> vertices;
    size_t segment_end=0;
    struct CachedQuery { Query query; Point source; const void *trail; };
    std::optional<CachedQuery> last_query;
};

// Where an open pseudo-edge of a map lies with respect to an earlier polygon.
// Boundaries are split at all their contacts, so one point decides it.
enum class EdgeStatus : std::int8_t { Unknown, Inside, Outside, Boundary };

class DirectionalMaps {
    Point start, target;
    std::vector<Map> maps;
    // A virtual source depends only on the cells its query is located in,
    // level by level: the start, a vertex, or the reflection in an edge of an
    // inner source. Each such sequence (a trail) is interned once, so equal
    // trails mean equal sources; nullptr is the start. Elements of an
    // unordered set keep their addresses when it rehashes.
    using Trail=std::tuple<const void*,size_t,long long>;
    struct TrailHash {
        size_t operator()(const Trail &t) const noexcept {
            const size_t a=std::hash<const void*>{}(std::get<0>(t));
            const size_t b=std::hash<size_t>{}(std::get<1>(t));
            const size_t c=std::hash<long long>{}(std::get<2>(t));
            return a^(b+0x9e3779b97f4a7c15ULL+(a<<6)+(a>>2))^(c*0xff51afd7ed558ccdULL);
        }
    };
    std::unordered_set<Trail,TrailHash> trails;
    const void *intern(const void *inner,size_t level,long long location) {
        return &*trails.insert({inner,level,location}).first;
    }
    using Source=std::pair<Point,const void*>;
    // statuses[g][h][e]: pseudo-edge e of map g with respect to polygon h<g,
    // computed once per pair of maps (Tan and Jiang's construction only).
    std::vector<std::vector<std::vector<EdgeStatus>>> statuses;

    static bool inside(const Query &q,const std::vector<Point> &polygon) {
        if(polygon.size()<=2) {
            if(!closed_region_contains(q.point,polygon)) return false;
            if(q.side.zero()) return true;
            if(polygon.size()==1) return false;
            const Point edge=polygon[1]-polygon[0];
            if(cross_sign(edge,q.side)!=0) return false;
            const Scalar position=edge.dot(q.point-polygon[0]),direction=edge.dot(q.side);
            return (position!=0 || direction>=0) &&
                (position!=edge.dot(edge) || direction<=0);
        }
        const auto &origin=polygon.front();
        if(cross_sign(polygon[1]-origin,q,origin,false)<0 ||
           cross_sign(polygon.back()-origin,q,origin,false)>0) return false;
        size_t left=1,right=polygon.size()-1;
        while(left+1<right) {
            const size_t mid=left+(right-left)/2;
            if(cross_sign(polygon[mid]-origin,q,origin,false)>=0) left=mid;
            else right=mid;
        }
        return cross_sign(polygon[right]-polygon[left],q,polygon[left],false)>=0;
    }

    // single: r1 and r2 are known to be the same ray (Vertex::single_ray).
    static bool in_cone(const Query &q,const Point &v,const Point &r1,const Point &r2,bool single=false) {
        if(single || same_direction(r1,r2))
            return cross_sign(r1,q,v)==0 && dot_sign(r1,q,v)>=0;
        const bool c1=cross_sign(r1,q,v)>=0, c2=cross_sign(r2,q,v)<=0;
        return cross_sign(r1,r2)>=0 ? c1 && c2 : c1 || c2;
    }

    // The existing binary locator's chord predicate, evaluated on symbolic
    // queries. Rays themselves use the corrected incident limits (Dror Claim 1).
    static bool in_edge_plus(const Query &q,const Point &v1,const Point &v2,
                             const Point &r1,const Point &r2) {
        if(v1==v2) return in_cone(q,v1,r1,r2);
        const Point dv=v2-v1;
        if(same_direction(r1,dv) || same_direction(r2,-dv)) return false;
        if(cross_sign(dv,r1)<0) {
            if(cross_sign(dv,r2)<0)
                return cross_sign(r1,q,v1)>=0 && cross_sign(r2,q,v2)<=0
                    && cross_sign(dv,q,v1)<=0;
            return cross_sign(dv,q,v1)<0 ? cross_sign(r1,q,v1)>=0 : cross_sign(r2,q,v2)<=0;
        }
        if(cross_sign(dv,r2)<0)
            return cross_sign(dv,q,v2)<0 ? cross_sign(r2,q,v2)<=0 : cross_sign(r1,q,v1)>=0;
        return cross_sign(r1,q,v1)>=0 || cross_sign(r2,q,v2)<=0 || cross_sign(dv,q,v1)<=0;
    }

    // The queries of the two sides of vertex j of map i: the vertex, moved
    // toward its previous and its next vertex.
    std::pair<Query,Query> vertex_queries(size_t i,size_t j) const {
        const auto &vertices=maps[i].vertices;
        const size_t n=vertices.size();
        const Point &v=vertices[j].point;
        return {Query{v,vertices[(j+n-1)%n].point-v},Query{v,vertices[(j+1)%n].point-v}};
    }

    void build_vertex(size_t i,size_t j) {
        cycle_checkpoint();
        if(maps[i].vertices[j].ready) return;
        ++tpp::detail::cone_builds();
        const auto [before,after]=vertex_queries(i,j);
        const Source before_source=virtual_source_trail(before,i);
        const Source after_source=virtual_source_trail(after,i);
        finish_vertex(i,j,before_source,after_source);
    }

    // The cone of vertex j of map i from the virtual sources of its two sides.
    void finish_vertex(size_t i,size_t j,const Source &before_source,const Source &after_source) {
        auto &vertices=maps[i].vertices;
        auto &v=vertices[j];
        const Point before=vertices[(j+vertices.size()-1)%vertices.size()].point;
        const Point after=vertices[(j+1)%vertices.size()].point;
        auto outgoing=[&](const Source &source,Point side,Point edge,bool &reflects,bool &at_source) {
            Point incoming=v.point-source.first;
            at_source=incoming.zero();
            if(at_source) incoming=side;
            reflects=cross_sign(edge,incoming)>0;
            return reflects ? reflect_direction(incoming,edge) : incoming;
        };
        bool before_at_source=false,after_at_source=false;
        v.before_ray=outgoing(before_source,before-v.point,v.point-before,v.before_reflects,before_at_source);
        v.after_ray=outgoing(after_source,after-v.point,after-v.point,v.after_reflects,after_at_source);
        // Same source and no reflection: both rays are v minus that source.
        // Most vertices of intersecting maps are of this kind; recognizing
        // them spares the cone test an exact zero (filtered arithmetic would
        // otherwise evaluate it in rationals at every location query).
        if(before_source.second==after_source.second && !v.before_reflects && !v.after_reflects
           && !before_at_source && !after_at_source) {
            v.single_ray=true;
            v.after_ray=v.before_ray;
        }
        v.ready=true;
    }

    // A segment has no polygonal interior/fan: its two boundary edges overlap.
    // Locate its contact by the monotone one-dimensional directional derivative
    // of D_previous(z)+|z-q|, on the existing split boundary. Comparing opposite
    // signed normalized dot products uses squared rational quantities only.
    // Carry the symbolic query through this comparison, including both ties.
    int segment_derivative(const Query &q,size_t level,const Point &v,const Point &edge,bool right) {
        const Point incoming=v-virtual_source({v,right?edge:-edge},level-1);
        if(incoming.zero()) return right?1:-1;
        const Scalar a=incoming.dot(edge),squared=incoming.dot(incoming);
        const std::array<Point,4> outgoing{v-q.point,-q.side,-q.tie1,-q.tie2};
        std::array<Scalar,4> b;
        for(size_t i=0;i<4;++i)b[i]=outgoing[i].dot(edge);
        int bs=0;
        for(const auto &coefficient:b)if((bs=sign(coefficient)))break;
        const int as=sign(a);
        if(!as)return bs;
        if(!bs || as==bs)return as;
        std::array<Scalar,7> difference{};
        for(size_t i=0;i<4;++i)for(size_t j=0;j<4;++j)
            difference[i+j]+=a*a*outgoing[i].dot(outgoing[j])-squared*b[i]*b[j];
        for(const auto &coefficient:difference)
            if(const int s=sign(coefficient))return as*s;
        return 0;
    }

    long long locate_segment(const Query &q,size_t level) {
        auto &map=maps[level-1];
        const Point edge=map.original[1]-map.original[0];
        // The forward boundary is sorted by its segment parameter; the first
        // vertex of the reverse boundary supplies the final endpoint.
        const size_t end=map.segment_end;
        auto derivative=[&](size_t j,bool right) {
            return segment_derivative(q,level,map.vertices[j].point,edge,right);
        };
        if(derivative(0,true)>=0)return 0;
        if(derivative(end,false)<=0)return static_cast<long long>(2*end);
        size_t left=0,right=end;
        while(left+1<right) {
            const size_t mid=left+(right-left)/2;
            if(derivative(mid,true)>=0)right=mid;
            else left=mid;
        }
        if(derivative(right,false)<=0)return static_cast<long long>(2*right);
        const Point v=map.vertices[left].point;
        const Point source=virtual_source({v,edge},level-1);
        const int source_side=sign(edge.cross(source-v));
        const int query_side=cross_sign(edge,q,v);
        return source_side*query_side>0 ? static_cast<long long>(2*left+1) : -1;
    }

    // -1: crossing; 2*j: vertex; 2*j+1: reflection on the following pseudo-edge.
    long long locate(const Query &q,size_t level) {
        auto &map=maps[level-1];
        if(map.original.size()==1) return 0;
        if(inside(q,map.membership_corners)) return -1;
        if(map.original.size()==2) return locate_segment(q,level);
        const auto cell=locate_cell(q,level);
        // A failed locator is exposed, never repaired by a scan or another solver.
        if(!cell)
            throw std::runtime_error("Directional map locator invariant failed at level "+std::to_string(level)
                +" ("+std::to_string(map.original.size())+" region vertices, "
                +std::to_string(map.vertices.size())+" split vertices)");
        return location_of_cell(level,*cell);
    }

    // The cells of a map with at least three vertices, in circular order:
    // 2j is the cone of vertex j and 2j+1 the region of the pseudo-edge from
    // vertex j to vertex j+1. Together they cover the outside of the polygon.
    // The cell of q, outside the polygon, by binary search; none if the last
    // pseudo-edge region does not contain it.
    std::optional<size_t> locate_cell(const Query &q,size_t level) {
        auto &map=maps[level-1];
        const size_t i=level-1, n=map.vertices.size();
        auto cone=[&](size_t j) {
            build_vertex(i,j);
            const auto &v=map.vertices[j];
            return in_cone(q,v.point,v.before_ray,v.after_ray,v.single_ray);
        };
        if(cone(0)) return 0;
        size_t left=0,right=n-1;
        while(left!=right) {
            const size_t mid=left+(right-left)/2, j=mid+1;
            if(cone(j)) return 2*j;
            build_vertex(i,left);
            const auto &a=map.vertices[left], &b=map.vertices[j];
            if(in_edge_plus(q,a.point,b.point,a.after_ray,b.before_ray)) right=mid;
            else left=mid+1;
        }
        if(!in_cell(q,level,2*left+1)) return std::nullopt;
        return 2*left+1;
    }

    // Whether q lies in cell c of the map at `level`.
    bool in_cell(const Query &q,size_t level,size_t c) {
        auto &map=maps[level-1];
        const size_t i=level-1, n=map.vertices.size(), j=c/2;
        build_vertex(i,j);
        if(c%2==0) {
            const auto &v=map.vertices[j];
            return in_cone(q,v.point,v.before_ray,v.after_ray,v.single_ray);
        }
        build_vertex(i,(j+1)%n);
        const auto &a=map.vertices[j], &b=map.vertices[(j+1)%n];
        return in_edge_plus(q,a.point,b.point,a.after_ray,b.before_ray);
    }

    // What locate returns for a query in cell c: the vertex, the reflection on
    // the pseudo-edge, or a crossing when neither of its rays reflects.
    long long location_of_cell(size_t level,size_t c) {
        if(c%2==0) return static_cast<long long>(c);
        const auto &map=maps[level-1];
        const size_t j=c/2;
        const auto &a=map.vertices[j], &b=map.vertices[(j+1)%map.vertices.size()];
        if(a.after_reflects!=b.before_reflects &&
           (b.point-a.point).cross(a.after_ray)!=0 &&
           (b.point-a.point).cross(b.before_ray)!=0)
            throw std::runtime_error("Directional map pseudo-edge changes contact type at level "+std::to_string(level));
        // Keep the crossing sentinel signed on both 32-bit WASM and 64-bit hosts.
        if(a.after_reflects || b.before_reflects) return static_cast<long long>(c);
        return -1;
    }

    // ---- Tan and Jiang's construction ------------------------------------
    //
    // Every cone is built, map by map. The 2n queries of map i (the two sides
    // of each vertex, in the order of the boundary) are located together in
    // map i-1, those that continue (crossings and reflections) together in map
    // i-2, and so on (their Lemma 2). In a map, each query is first decided by
    // the pseudo-edge it stands for (its anchor, below).
    // - Anchor inside the polygon of the map, strict: a crossing. For a side
    //   of a vertex the query is then inside. For a reflected query, the
    //   contact of its reflection lies on the last segment of its unfolded
    //   path; a bend on this polygon would leave it from the boundary into the
    //   outside, which cannot pass through an inner point. A contact that may
    //   sit at an end of the pseudo-edge (not strict) goes to the binary
    //   search. docs/algorithms/tan-jiang-implementation.md has the proof.
    // - Anchor outside: a scan looks for the cell of the query, from the cell
    //   of the previous query, widening in both directions. A found cell is
    //   kept only if the query is strictly outside an edge at the cell, so it
    //   is the cell of the binary search.
    // - Otherwise, and when a scan fails, the binary search of locate.
    // If consecutive queries meet the cells in circular order, as their Lemma
    // 1 says for disjoint polygons, a batch costs O(cells+queries); a budget
    // bounds the scans when it does not hold.

    // A query, and the open pseudo-edge (map, edge) that holds the boundary
    // point it stands for: its own pseudo-edge for a side of a vertex, else
    // the pseudo-edge of its last reflection. `strict`: that point is inside
    // the open pseudo-edge for every small enough perturbation, which holds
    // for a side of a vertex and for a reflection whose query is off both
    // limiting rays of the reflection cell.
    struct Anchored {
        Query query;
        size_t map=0, edge=0;
        bool strict=true;
    };

    // Statuses of the pseudo-edges of map g with respect to polygon h<g: the
    // midpoint of each, whose wedge around an inner point of h moves at most
    // twice monotonically along the convex boundary of g. O(|g|+|h|) time.
    std::vector<EdgeStatus> compute_statuses(size_t g,size_t h) {
        const auto &vertices=maps[g].vertices;
        const auto &corners=maps[h].membership_corners;
        const size_t n=vertices.size(), m=corners.size();
        std::vector<EdgeStatus> result(n,EdgeStatus::Outside);
        // The wedges between consecutive rays from o through the corners cover
        // the plane but o, each with an angle below pi.
        const Point o=(corners[0]+corners[1]+corners[2])/Scalar(3);
        auto in_wedge=[&](const Point &d,size_t w) {
            return cross_sign(corners[w]-o,d)>=0 && cross_sign(corners[(w+1)%m]-o,d)<0;
        };
        std::optional<size_t> wedge;
        for(size_t e=0;e<n;++e) {
            const Point midpoint=(vertices[e].point+vertices[(e+1)%n].point)/Scalar(2), d=midpoint-o;
            if(d.zero()) {result[e]=EdgeStatus::Inside;continue;}
            const size_t from=wedge.value_or(0);
            std::optional<size_t> found;
            if(in_wedge(d,from)) found=from;
            for(size_t step=1;!found && 2*step<=m;++step) {
                if(in_wedge(d,(from+step)%m)) found=(from+step)%m;
                else if(in_wedge(d,(from+m-step)%m)) found=(from+m-step)%m;
            }
            if(!found) {result[e]=EdgeStatus::Unknown;continue;}
            wedge=found;
            const Point &a=corners[*wedge], &b=corners[(*wedge+1)%m];
            const int side=offset_cross_sign(b-a,midpoint,a);
            result[e]=side>0 ? EdgeStatus::Inside : side<0 ? EdgeStatus::Outside : EdgeStatus::Boundary;
        }
        return result;
    }

    EdgeStatus edge_status(size_t g,size_t h,size_t e) {
        if(maps[g].original.size()<=2) return EdgeStatus::Unknown;
        // Strictly separated boxes (exact on the binary64 input coordinates,
        // as in split_boundaries): every pseudo-edge is outside.
        if(maps[g].bounds->disjoint(*maps[h].bounds)) return EdgeStatus::Outside;
        auto &row=statuses[g];
        if(row.empty()) row.resize(g);
        if(row[h].empty()) row[h]=compute_statuses(g,h);
        return row[h][e];
    }

    // Whether q, in cell c, is strictly outside the supporting line of an edge
    // at the cell (ties left out, as in `inside`): then `inside` rejects it, and
    // the cell is the one the binary search finds.
    bool outside_at_cell(const Query &q,size_t level,size_t c) {
        const auto &vertices=maps[level-1].vertices;
        const size_t n=vertices.size(), j=c/2;
        const Point &v=vertices[j].point;
        if(c%2) return cross_sign(vertices[(j+1)%n].point-v,q,v,false)<0;
        const Point &before=vertices[(j+n-1)%n].point, &after=vertices[(j+1)%n].point;
        return cross_sign(v-before,q,before,false)<0 || cross_sign(after-v,q,v,false)<0;
    }

    // Locations of the batch in the map at `level`, as locate returns them.
    void locate_together(const std::vector<Anchored> &batch,size_t level,std::vector<long long> &locations) {
        auto &counters=tan_jiang_scan_counters();
        auto &map=maps[level-1];
        const size_t h=level-1;
        locations.resize(batch.size());
        if(map.original.size()<=2) {
            for(size_t t=0;t<batch.size();++t) locations[t]=locate(batch[t].query,level);
            counters.small_maps+=batch.size();
            return;
        }
        const size_t cells=2*map.vertices.size();
        ++counters.batches;counters.queries+=batch.size();counters.scan_bound+=cells+batch.size();
        const size_t budget=4*(cells+batch.size())+16;
        size_t steps=0;
        std::optional<size_t> pointer;
        auto scan=[&](const Query &q) -> std::optional<size_t> {
            if(!pointer) {++counters.searched;return locate_cell(q,level);}
            const size_t p=*pointer;
            ++steps;
            if(in_cell(q,level,p)) return p;
            for(size_t d=1;2*d<=cells && steps<budget;++d) {
                const size_t forward=(p+d)%cells, backward=(p+cells-d)%cells;
                ++steps;
                if(in_cell(q,level,forward)) return forward;
                if(backward==forward) continue;
                ++steps;
                if(in_cell(q,level,backward)) return backward;
            }
            if(steps<budget) ++counters.scan_failures;
            return std::nullopt;
        };
        for(size_t t=0;t<batch.size();++t) {
            cycle_checkpoint();
            const auto &[q,g,e,strict]=batch[t];
            const EdgeStatus status=edge_status(g,h,e);
            long long location=0;
            bool decided=false;
            if(status==EdgeStatus::Inside && strict) {location=-1;decided=true;++counters.inside;}
            else if(status==EdgeStatus::Outside && steps<budget) {
                const bool first=!pointer;
                if(const auto cell=scan(q)) {
                    if(outside_at_cell(q,level,*cell)) {
                        pointer=*cell;location=location_of_cell(level,*cell);decided=true;
                        if(!first) ++counters.scanned;
                    } else ++counters.guard_rejections;
                }
                if(steps>=budget) ++counters.budget_exhausted;  // once: no scan follows
            }
            if(!decided) {location=locate(q,level);++counters.searched;}
            if(counters.verify) {
                long long reference=0;
                try {reference=locate(q,level);} catch(const std::exception &) {reference=location+1;}
                if(reference!=location) ++counters.mismatches;
            }
            locations[t]=location;
        }
        counters.scan_steps+=steps;
        counters.max_batch_ratio=std::max(counters.max_batch_ratio,double(steps)/double(cells+batch.size()));
    }

    // Virtual sources of a batch at `level`: virtual_source_trail of each
    // query, with the queries located together.
    void batch_sources(const std::vector<Anchored> &batch,size_t level,std::vector<Source> &out) {
        out.assign(batch.size(),Source{start,nullptr});
        if(level==0 || batch.empty()) return;
        std::vector<long long> locations;
        locate_together(batch,level,locations);
        const auto &vertices=maps[level-1].vertices;
        const size_t n=vertices.size();
        std::vector<Anchored> next;
        std::vector<size_t> owner;
        for(size_t t=0;t<batch.size();++t) {
            const long long location=locations[t];
            if(location>=0 && location%2==0) {
                out[t]={vertices[size_t(location)/2].point,intern(nullptr,level,location)};
                continue;
            }
            owner.push_back(t);
            if(location<0) next.push_back(batch[t]);
            else {
                const size_t j=size_t(location)/2;
                const auto &from=vertices[j], &to=vertices[(j+1)%n];
                const Point a=from.point, edge=to.point-a;
                const Query &q=batch[t].query;
                const bool strict=cross_sign(from.after_ray,q,a,false)!=0
                    && cross_sign(to.before_ray,q,to.point,false)!=0;
                next.push_back({reflect_query(q,a,edge),level-1,j,strict});
            }
        }
        std::vector<Source> inner;
        batch_sources(next,level-1,inner);
        for(size_t u=0;u<owner.size();++u) {
            const size_t t=owner[u];
            const long long location=locations[t];
            if(location<0) {out[t]=inner[u];continue;}
            const size_t j=size_t(location)/2;
            const Point a=vertices[j].point, edge=vertices[(j+1)%n].point-a;
            out[t]={reflect_point(inner[u].first,a,edge),intern(inner[u].second,level,location)};
        }
    }

public:
    void build_by_scans() {
        statuses.assign(maps.size(),{});
        std::vector<Anchored> batch;
        std::vector<Source> sources;
        for(size_t i=0;i<maps.size();++i) {
            cycle_checkpoint();
            const size_t n=maps[i].vertices.size();
            batch.clear();
            batch.reserve(2*n);
            for(size_t j=0;j<n;++j) {
                const auto [before,after]=vertex_queries(i,j);
                batch.push_back({before,i,(j+n-1)%n});
                batch.push_back({after,i,j});
            }
            batch_sources(batch,i,sources);
            for(size_t j=0;j<n;++j) {
                if(maps[i].vertices[j].ready) continue;
                ++tpp::detail::cone_builds();
                finish_vertex(i,j,sources[2*j],sources[2*j+1]);
            }
        }
    }

private:

    Point virtual_source(const Query &q,size_t level) {
        return virtual_source_trail(q,level).first;
    }
    // The virtual source and its interned trail.
    std::pair<Point,const void*> virtual_source_trail(const Query &q,size_t level) {
        cycle_checkpoint();
        if(level==0) return {start,nullptr};
        auto &cache=maps[level-1].last_query;
        if(cache && cache->query==q) return {cache->source,cache->trail};
        const auto location=locate(q,level);
        Point source;
        const void *trail=nullptr;
        if(location<0) std::tie(source,trail)=virtual_source_trail(q,level-1);
        else {
            const auto &vertices=maps[level-1].vertices;
            const size_t j=size_t(location)/2;
            const Point a=vertices[j].point;
            if(location%2==0) {
                source=a;
                trail=intern(nullptr,level,location);
            } else {
                const Point edge=vertices[(j+1)%vertices.size()].point-a;
                const auto inner=virtual_source_trail(reflect_query(q,a,edge),level-1);
                source=reflect_point(inner.first,a,edge);
                trail=intern(inner.second,level,location);
            }
        }
        cache=Map::CachedQuery{q,source,trail};
        return {source,trail};
    }

    static void append(std::vector<Point> &path,const Point &p) {
        if(path.empty() || !(path.back()==p)) path.push_back(p);
    }

    void query_path(const Point &q,size_t level,std::vector<Point> &path) {
        cycle_checkpoint();
        if(level==0) { append(path,start); append(path,q); return; }
        const auto location=locate({q,{}},level);
        if(location<0) { query_path(q,level-1,path); return; }
        auto &vertices=maps[level-1].vertices;
        const size_t j=size_t(location)/2;
        const Point a=vertices[j].point;
        if(location%2==0) {
            query_path(a,level-1,path);
            append(path,q);
            return;
        }
        const Point edge=vertices[(j+1)%vertices.size()].point-a;
        const Point reflected=reflect_point(q,a,edge);
        query_path(reflected,level-1,path);
        if(path.size()<2) throw std::runtime_error("Reflection has no incoming segment");
        const Point previous=path[path.size()-2], direction=reflected-previous;
        const Scalar denominator=edge.cross(direction);
        if(denominator==0) throw std::runtime_error("Reflection segment is parallel to its edge");
        const Scalar t=edge.cross(a-previous)/denominator;
        const Point contact=previous+direction*t;
        const Scalar edge_t=(contact-a).dot(edge)/edge.dot(edge);
        if(t<0 || t>1 || edge_t<0 || edge_t>1)
            throw std::runtime_error("Refolding would remove preceding visits at level "+std::to_string(level));
        path.pop_back();
        append(path,contact);
        append(path,q);
    }

    void query_trace(const Point &q,size_t level,std::vector<DirectionalTraceStep> &trace) {
        if(level==0) return;
        const auto location=locate({q,{}},level);
        DirectionalTraceStep step;step.level=level;
        if(location<0) {
            step.region=DirectionalTraceRegion::Crossing;
            trace.push_back(step);
            query_trace(q,level-1,trace);
            return;
        }
        const auto &vertices=maps[level-1].vertices;
        const size_t j=size_t(location)/2;
        const auto &vertex=vertices[j];
        step.original_edge=vertex.original_edge;
        if(location%2==0) {
            step.region=DirectionalTraceRegion::Vertex;
            step.defining_point=vertex.definition.point.external();
            step.vertex_is_edge_intersection=vertex.definition.edge_intersection;
            step.defining_polygon=vertex.definition.polygon;
            step.defining_edge=vertex.definition.edge;
            trace.push_back(step);
            query_trace(vertex.point,level-1,trace);
            return;
        }
        step.region=DirectionalTraceRegion::Edge;
        trace.push_back(step);
        const Point edge=maps[level-1].original[(vertex.original_edge+1)%maps[level-1].original.size()]
            -maps[level-1].original[vertex.original_edge];
        query_trace(reflect_point(q,maps[level-1].original[vertex.original_edge],edge),level-1,trace);
    }

    static long double distance(const Point &a,const Point &b) {
        const Point d=a-b;
#ifdef __EMSCRIPTEN__
        // Boost does not support conversion to Emscripten's long double.
        // Construction predicates remain rational; only the reported norm is rounded.
        return std::hypot(d.x.convert_to<double>(),d.y.convert_to<double>());
#else
        return std::hypot(d.x.convert_to<long double>(),d.y.convert_to<long double>());
#endif
    }

    long double query_length(const Point &q,size_t level) {
        if(level==0) return distance(start,q);
        const auto location=locate({q,{}},level);
        if(location<0) return query_length(q,level-1);
        auto &vertices=maps[level-1].vertices;
        const size_t j=size_t(location)/2;
        auto &v=vertices[j];
        if(location%2==0) {
            if(!v.prefix_length) v.prefix_length=query_length(v.point,level-1);
            return *v.prefix_length+distance(v.point,q);
        }
        const Point edge=vertices[(j+1)%vertices.size()].point-v.point;
        return query_length(reflect_point(q,v.point,edge),level-1);
    }

    // Splits each boundary at its contacts with the earlier boundaries. Only the
    // pairs of edges that may touch are tested, O(|P_i|+|P_h|) of them for each
    // pair of polygons (convex_contacts.h), so all contacts take O(kn) time.
    // The contacts of an edge are sorted by parameter alone: in general
    // position an edge meets each earlier convex boundary at most twice, so
    // this costs O(M log k) in all.
    void split_boundaries(bool include_previous_intersections) {
        // A contact found several times keeps the definition that comes first
        // in the order (the edge's own vertex, the start, then (polygon, edge)
        // of the earlier boundary, its first endpoint before its second).
        struct Split { Scalar parameter; Vertex::Definition definition; std::array<size_t,3> order; };
        struct Candidate { size_t polygon, edge; };
        std::vector<Split> splits;  // reused across edges
        std::vector<LexicographicChains> chains;
        if(include_previous_intersections) {
            chains.reserve(maps.size());
            for(const auto &m:maps) chains.push_back(lexicographic_chains(m.original));
        }
        std::vector<std::vector<Candidate>> candidates;
        for(size_t i=0;i<maps.size();++i) {
        cycle_checkpoint();
            auto &map=maps[i];
            map.vertices.reserve(map.original.size());
            if(map.original.size()==1) {
                Vertex v; v.point=map.original.front(); v.definition.point=v.point;
                map.vertices.push_back(std::move(v));
                continue;
            }
            candidates.assign(map.original.size(),{});
            if(include_previous_intersections) for(size_t h=0;h<i;++h) {
                if(map.bounds->disjoint(*maps[h].bounds)) continue;
                for_each_edge_pair_that_may_touch(map.original,chains[i],maps[h].original,chains[h],
                    [&](size_t j,size_t k){candidates[j].push_back({h,k});return false;});
            }
            for(size_t j=0;j<map.original.size();++j) {
                if(map.original.size()==2 && j==1)map.segment_end=map.vertices.size();
                const Point a=map.original[j], edge=map.original[(j+1)%map.original.size()]-a;
                splits.clear();
                splits.push_back({Scalar(0),Vertex::Definition{.point=a},{0,0,0}});
                auto add=[&](const Scalar &t,Vertex::Definition definition,std::array<size_t,3> order) {
                    if(t>=0 && t<1) splits.push_back({t,std::move(definition),order});
                };
                auto collinear_point=[&](const Point &p,std::array<size_t,3> order) {
                    if(edge.cross(p-a)==0)
                        add((p-a).dot(edge)/edge.dot(edge),Vertex::Definition{.point=p},order);
                };
                collinear_point(start,{0,0,1});
                for(const auto [h,k]:candidates[j]) {
                    // Bounds use the original binary-double coordinates,
                    // so strict separation is an exact rejection filter.
                    if(map.edge_bounds[j].disjoint(*maps[h].bounds)) continue;
                    if(map.edge_bounds[j].disjoint(maps[h].edge_bounds[k])) continue;
                    const auto &previous=maps[h].original;
                    const Point c=previous[k], other=previous[(k+1)%previous.size()]-c;
                    const Scalar denominator=edge.cross(other);
                    if(denominator==0) {
                        collinear_point(c,{1+h,k,0});
                        collinear_point(c+other,{1+h,k,1});
                    } else {
                        const Scalar t=(c-a).cross(other)/denominator;
                        const Scalar u=(c-a).cross(edge)/denominator;
                        if(u>=0 && u<=1) add(t,Vertex::Definition{.point=a+edge*t,
                            .edge_intersection=true,.polygon=h,.edge=k},{1+h,k,0});
                    }
                }
                std::sort(splits.begin(),splits.end(),[](const Split &x,const Split &y){
                    if(!(x.parameter==y.parameter)) return x.parameter<y.parameter;
                    return x.order<y.order;
                });
                splits.erase(std::unique(splits.begin(),splits.end(),[](const Split &x,const Split &y){
                    return x.parameter==y.parameter;
                }),splits.end());
                for(const auto &split:splits) {
                    Vertex v;
                    v.point=a+edge*split.parameter;v.original_edge=j;v.edge_parameter=split.parameter;
                    v.definition=split.definition;
                    map.vertices.push_back(std::move(v));
                }
            }
        }
    }

public:
    DirectionalMaps(Vector2 s,Vector2 t,const std::vector<std::vector<Vector2>> &polygons,
                    PreloadPolicy preload,bool assume_disjoint=false)
        : DirectionalMaps(Point(s),Point(t),polygons,preload,assume_disjoint) {}
    template<class InputPoint>
    DirectionalMaps(Point s,Point t,const std::vector<std::vector<InputPoint>> &polygons,
                    PreloadPolicy preload,bool assume_disjoint=false) : start(s),target(t) {
        maps.resize(polygons.size());
        for(size_t i=0;i<polygons.size();++i) {
        cycle_checkpoint();
            auto &p=maps[i].original;
            p.reserve(polygons[i].size());
            maps[i].edge_bounds.reserve(polygons[i].size());
            maps[i].membership_corners.reserve(polygons[i].size());
            for(auto v:polygons[i]) {
                if constexpr(std::is_same_v<InputPoint,Vector2>)
                if(!std::isfinite(v.x) || !std::isfinite(v.y))
                    throw std::invalid_argument("Nonfinite polygon coordinate");
                append(p,Point(v));
            }
            if(p.size()>1 && p.front()==p.back()) p.pop_back();
            Scalar area=0;
            for(size_t j=0;j<p.size();++j) area+=p[j].cross(p[(j+1)%p.size()]);
            if(p.empty() || (p.size()>=3 && area==0))
                throw std::invalid_argument("Expected a point, segment or positive-area polygon");
            if(area<0) std::reverse(p.begin(),p.end());
            // Keep original edge provenance, but omit straight intermediate
            // corners from the auxiliary O(log m) closed-membership fan.
            for(size_t j=0;j<p.size();++j) {
                const Point before=p[(j+p.size()-1)%p.size()],after=p[(j+1)%p.size()];
                const Bounds &edge=maps[i].edge_bounds.emplace_back(p[j].external(),after.external());
                auto &all=maps[i].bounds;
                if(!all) all=edge;
                else {
                    all->min_x=std::min(all->min_x,edge.min_x);all->max_x=std::max(all->max_x,edge.max_x);
                    all->min_y=std::min(all->min_y,edge.min_y);all->max_y=std::max(all->max_y,edge.max_y);
                }
                if(p.size()<=2 || (p[j]-before).cross(after-p[j])!=0)
                    maps[i].membership_corners.push_back(p[j]);
            }
        }
        split_boundaries(!assume_disjoint);
        if(preload==PreloadPolicy::Eager)
            for(size_t i=0;i<maps.size();++i)
                for(size_t j=0;j<maps[i].vertices.size();++j) build_vertex(i,j);
    }
    std::vector<Vector2> solve() {
        std::vector<Point> exact;
        query_path(target,maps.size(),exact);
        // Remove only exactly collinear forward contacts. This changes the
        // representation of the same continuous path, never its geometry.
        std::vector<Point> compact;
        for(const auto &p:exact) {
            while(compact.size()>1) {
                const Point a=compact[compact.size()-2], b=compact.back();
                if((b-a).cross(p-b)!=0 || (b-a).dot(p-b)<0) break;
                compact.pop_back();
            }
            append(compact,p);
        }
        std::vector<Vector2> result;
        result.reserve(compact.size());
        for(const auto &p:compact) result.push_back(p.external());
        return result;
    }
    std::vector<DirectionalMapContact> contact_details_from_path(
            const std::vector<Point> &path,bool use_last_contact,std::vector<Point> *exact_contacts=nullptr) {
        std::vector<DirectionalMapContact> result;
        result.reserve(maps.size());
        if(path.empty()) throw std::runtime_error("Directional map returned an empty path");
        size_t segment=path.size()==1 ? 0 : 1;
        Scalar rate=0;
        for(size_t i=0;i<maps.size();++i) {
        cycle_checkpoint();
            bool found=false;
            if(path.size()==1) {
                if(!inside({path.front(),{}},maps[i].original))
                    throw std::runtime_error("Stationary path misses polygon "+std::to_string(i));
                if(exact_contacts){exact_contacts->push_back(path.front());continue;}
                result.push_back({.point=path.front().external(),.segment_start=path.front().external(),
                    .segment_end=path.front().external()});
                continue;
            }
            while(segment<path.size()) {
                const Point a=path[segment-1], direction=path[segment]-a;
                Scalar lo=rate,hi=1;
                std::optional<size_t> lo_edge,hi_edge;
                if(maps[i].original.size()<=2) {
                    if(!clip_closed_region(a,a+direction,maps[i].original,rate,lo,hi)) hi=-1;
                } else for(size_t j=0;j<maps[i].original.size();++j) {
                    const Point v=maps[i].original[j];
                    const Point edge=maps[i].original[(j+1)%maps[i].original.size()]-v;
                    const Scalar constant=edge.cross(a-v),slope=edge.cross(direction);
                    if(slope>0) {const Scalar crossing=-constant/slope;if(crossing>lo){lo=crossing;lo_edge=j;}}
                    else if(slope<0) {const Scalar crossing=-constant/slope;if(crossing<hi){hi=crossing;hi_edge=j;}}
                    else if(constant<0) {hi=-1;break;}
                }
                if(lo<=hi && hi>=rate && lo<=1) {
                    const bool choose_hi=use_last_contact && hi<Scalar(1);
                    rate=use_last_contact ? std::min(hi,Scalar(1)) : std::max(lo,rate);
                    const auto edge_index=choose_hi?hi_edge:lo_edge;
                    if(exact_contacts)exact_contacts->push_back(a+direction*rate);
                    else {
                    DirectionalMapContact detail{.point=(a+direction*rate).external(),
                        .segment_start=a.external(),.segment_end=(a+direction).external()};
                    if(edge_index) {
                        detail.edge_start=maps[i].original[*edge_index].external();
                        detail.edge_end=maps[i].original[(*edge_index+1)%maps[i].original.size()].external();
                        detail.has_edge=true;
                    }
                    result.push_back(detail);
                    }
                    found=true;
                    break;
                }
                ++segment;rate=0;
            }
            if(!found) throw std::runtime_error("Could not materialize ordered contact "+std::to_string(i));
        }
        return result;
    }
    std::vector<DirectionalMapContact> contact_details(bool use_last_contact) {
        std::vector<Point> path;
        query_path(target,maps.size(),path);
        return contact_details_from_path(path,use_last_contact);
    }
    std::vector<DirectionalTraceStep> trace() {
        std::vector<DirectionalTraceStep> result;result.reserve(maps.size());
        query_trace(target,maps.size(),result);
        if(result.size()!=maps.size())throw std::runtime_error("Directional trace has wrong cardinality");
        return result;
    }
    std::vector<Vector2> contacts_from_path(const std::vector<Point> &path,bool use_last_contact) {
        const auto details=contact_details_from_path(path,use_last_contact);
        std::vector<Vector2> result;result.reserve(details.size());
        for(const auto &detail:details)result.push_back(detail.point);
        return result;
    }
    std::vector<Vector2> contacts(bool use_last_contact) {
        std::vector<Point> path;
        query_path(target,maps.size(),path);
        return contacts_from_path(path,use_last_contact);
    }
#ifndef TPP_EXPERIMENT_NATIVE_DOUBLE
    ConvexRationalPolygon exact_contacts(bool use_last_contact=false) {
        std::vector<Point> path,contacts;query_path(target,maps.size(),path);
        contact_details_from_path(path,use_last_contact,&contacts);
        ConvexRationalPolygon result;for(const auto &q:contacts)result.emplace_back(q.x,q.y);
        return result;
    }
    std::pair<double,double> exact_length_bounds(const std::vector<Point> &path) {
        constexpr unsigned precision=96;
        const tpp::ConvexInteger scale=tpp::ConvexInteger(1)<<precision;
        Scalar lower=0,upper=0;
        for(size_t i=1;i<path.size();++i) {
            const Point delta=path[i]-path[i-1];const Scalar squared=delta.dot(delta);
            if(squared==0)continue;
            const auto numerator=boost::multiprecision::numerator(squared);
            const auto denominator=boost::multiprecision::denominator(squared);
            const tpp::ConvexInteger scaled=(numerator<<(2*precision))/denominator;
            const tpp::ConvexInteger root=sqrt(scaled);
            lower+=Scalar(root)/Scalar(scale);
            upper+=Scalar(root+1)/Scalar(scale);
        }
        double lo=lower.convert_to<double>();
        while(Scalar(lo)>lower)lo=std::nextafter(lo,-std::numeric_limits<double>::infinity());
        double hi=upper.convert_to<double>();
        while(Scalar(hi)<upper)hi=std::nextafter(hi,std::numeric_limits<double>::infinity());
        return {lo,hi};
    }
    RationalMapResult exact_result(bool use_last_contact) {
        std::vector<Point> path;
        query_path(target,maps.size(),path);
        RationalMapResult result;result.contacts=contacts_from_path(path,use_last_contact);
        std::tie(result.lower_bound,result.upper_bound)=exact_length_bounds(path);
        return result;
    }
#endif
#ifdef TPP_DIRECTIONAL_FILTERED
    // exact_contacts with filtered decisions: the same expressions, so the
    // same rationals, evaluated from their DAGs for the contacts alone.
    ConvexRationalPolygon exact_contacts(bool use_last_contact=false) {
        std::vector<Point> path,contacts;query_path(target,maps.size(),path);
        contact_details_from_path(path,use_last_contact,&contacts);
        ConvexRationalPolygon result;result.reserve(contacts.size());
        for(const auto &q:contacts)result.emplace_back(q.x.exact(),q.y.exact());
        return result;
    }
#endif
    double length() { return double(query_length(target,maps.size())); }
};

}

namespace {
template<class InputPoint>
DirectionalMaps tan_jiang_maps(const Point &start,const Point &target,
        const std::vector<std::vector<InputPoint>> &polygons) {
    DirectionalMaps maps(start,target,polygons,PreloadPolicy::Lazy);
    maps.build_by_scans();
    return maps;
}
}

#if defined(TPP_DIRECTIONAL_FILTERED)
std::vector<DirectionalTraceStep> solve_intersecting_map_trace_filtered(
        const Vector2 &start,const Vector2 &target,const std::vector<std::vector<Vector2>> &polygons,
        PreloadPolicy preload) {
    FilteredRational::Scope arithmetic_scope;
    return DirectionalMaps(start,target,polygons,preload).trace();
}
ConvexRationalPolygon solve_intersecting_map_contacts_exact_filtered(const ConvexRationalPoint &start,
        const ConvexRationalPoint &target,const ConvexRationalPolygons &polygons) {
    FilteredRational::Scope arithmetic_scope;
    return DirectionalMaps(Point(start),Point(target),polygons,PreloadPolicy::Lazy).exact_contacts();
}
ConvexRationalPolygon solve_disjoint_map_contacts_exact_filtered(const ConvexRationalPoint &start,
        const ConvexRationalPoint &target,const ConvexRationalPolygons &polygons) {
    FilteredRational::Scope arithmetic_scope;
    return DirectionalMaps(Point(start),Point(target),polygons,PreloadPolicy::Lazy,true).exact_contacts(true);
}
#elif defined(TPP_DIRECTIONAL_DOUBLE_VARIANT)
std::vector<Vector2> solve_intersecting_maps_unchecked_double(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload).solve();
}
double length_intersecting_maps_unchecked_double(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload).length();
}
std::vector<Vector2> solve_intersecting_map_contacts_unchecked_double(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,bool use_last_contact,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload).contacts(use_last_contact);
}
std::vector<DirectionalMapContact> solve_intersecting_map_contact_details_unchecked_double(
        const Vector2 &start,const Vector2 &target,const std::vector<std::vector<Vector2>> &polygons,
        bool use_last_contact,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload).contact_details(use_last_contact);
}
std::vector<DirectionalTraceStep> solve_intersecting_map_trace_unchecked_double(
        const Vector2 &start,const Vector2 &target,const std::vector<std::vector<Vector2>> &polygons,
        PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload).trace();
}
double length_intersecting_maps_tan_jiang_unchecked_double(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    return tan_jiang_maps(Point(start),Point(target),polygons).length();
}
std::vector<DirectionalTraceStep> solve_intersecting_map_trace_tan_jiang_unchecked_double(
        const Vector2 &start,const Vector2 &target,const std::vector<std::vector<Vector2>> &polygons) {
    return tan_jiang_maps(Point(start),Point(target),polygons).trace();
}
#else
std::vector<Vector2> solve_intersecting_maps_tan_jiang(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    return tan_jiang_maps(Point(start),Point(target),polygons).solve();
}
double length_intersecting_maps_tan_jiang(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    return tan_jiang_maps(Point(start),Point(target),polygons).length();
}
bool &exact_maps_filtered() {static bool value=true;return value;}
ConvexRationalPolygon solve_intersecting_map_contacts_exact(const ConvexRationalPoint &start,
        const ConvexRationalPoint &target,const ConvexRationalPolygons &polygons) {
#ifdef TPP_HAS_FILTERED_DIRECTIONAL
    if(exact_maps_filtered()) return solve_intersecting_map_contacts_exact_filtered(start,target,polygons);
#endif
    return DirectionalMaps(Point(start),Point(target),polygons,PreloadPolicy::Lazy).exact_contacts();
}
ConvexRationalPolygon solve_disjoint_map_contacts_exact(const ConvexRationalPoint &start,
        const ConvexRationalPoint &target,const ConvexRationalPolygons &polygons) {
#ifdef TPP_HAS_FILTERED_DIRECTIONAL
    if(exact_maps_filtered()) return solve_disjoint_map_contacts_exact_filtered(start,target,polygons);
#endif
    return DirectionalMaps(Point(start),Point(target),polygons,PreloadPolicy::Lazy,true).exact_contacts(true);
}
std::vector<Vector2> solve_intersecting_maps(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload).solve();
}
double length_intersecting_maps(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload).length();
}
std::vector<Vector2> solve_intersecting_map_contacts(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,bool use_last_contact,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload).contacts(use_last_contact);
}
std::vector<Vector2> solve_disjoint_map_contacts(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload,true).contacts(true);
}
double length_disjoint_maps(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload,true).length();
}
RationalMapResult solve_intersecting_map_contacts_with_bounds(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,bool use_last_contact,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload).exact_result(use_last_contact);
}
RationalMapResult solve_disjoint_map_contacts_with_bounds(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons,PreloadPolicy preload) {
    return DirectionalMaps(start,target,polygons,preload,true).exact_result(true);
}
#endif

}
