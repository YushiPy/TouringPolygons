#include "cycle_execution.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "tpp/convex/detail/closed_regions.h"

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
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
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
#ifndef TPP_EXPERIMENT_NATIVE_DOUBLE
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
    struct CachedQuery { Query query; Point source; };
    std::optional<CachedQuery> last_query;
};

class DirectionalMaps {
    Point start, target;
    std::vector<Map> maps;

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

    static bool in_cone(const Query &q,const Point &v,const Point &r1,const Point &r2) {
        if(same_direction(r1,r2))
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

    void build_vertex(size_t i,size_t j) {
        cycle_checkpoint();
        auto &vertices=maps[i].vertices;
        auto &v=vertices[j];
        if(v.ready) return;
        const Point before=vertices[(j+vertices.size()-1)%vertices.size()].point;
        const Point after=vertices[(j+1)%vertices.size()].point;
        auto outgoing=[&](Point side,Point edge,bool &reflects) {
            const Point source=virtual_source({v.point,side},i);
            Point incoming=v.point-source;
            if(incoming.zero()) incoming=side;
            reflects=cross_sign(edge,incoming)>0;
            return reflects ? reflect_direction(incoming,edge) : incoming;
        };
        v.before_ray=outgoing(before-v.point,v.point-before,v.before_reflects);
        v.after_ray=outgoing(after-v.point,after-v.point,v.after_reflects);
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
        const size_t i=level-1, n=map.vertices.size();
        auto cone=[&](size_t j) {
            build_vertex(i,j);
            const auto &v=map.vertices[j];
            return in_cone(q,v.point,v.before_ray,v.after_ray);
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
        build_vertex(i,left);
        build_vertex(i,(left+1)%n);
        const auto &a=map.vertices[left], &b=map.vertices[(left+1)%n];
        // A failed locator is exposed, never repaired by a scan or another solver.
        if(!in_edge_plus(q,a.point,b.point,a.after_ray,b.before_ray))
            throw std::runtime_error("Directional map locator invariant failed at level "+std::to_string(level)
                +" ("+std::to_string(map.original.size())+" region vertices, "+std::to_string(n)+" split vertices)");
        if(a.after_reflects!=b.before_reflects &&
           (b.point-a.point).cross(a.after_ray)!=0 &&
           (b.point-a.point).cross(b.before_ray)!=0)
            throw std::runtime_error("Directional map pseudo-edge changes contact type at level "+std::to_string(level));
        // Keep the crossing sentinel signed on both 32-bit WASM and 64-bit hosts.
        if(a.after_reflects || b.before_reflects) return static_cast<long long>(2*left+1);
        return -1;
    }

    Point virtual_source(const Query &q,size_t level) {
        cycle_checkpoint();
        if(level==0) return start;
        auto &cache=maps[level-1].last_query;
        if(cache && cache->query==q) return cache->source;
        const auto location=locate(q,level);
        Point source;
        if(location<0) source=virtual_source(q,level-1);
        else {
            const auto &vertices=maps[level-1].vertices;
            const size_t j=size_t(location)/2;
            const Point a=vertices[j].point;
            if(location%2==0) source=a;
            else {
                const Point edge=vertices[(j+1)%vertices.size()].point-a;
                source=reflect_point(virtual_source(reflect_query(q,a,edge),level-1),a,edge);
            }
        }
        cache=Map::CachedQuery{q,source};
        return source;
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

    void split_boundaries(bool include_previous_intersections) {
        for(size_t i=0;i<maps.size();++i) {
        cycle_checkpoint();
            auto &map=maps[i];
            if(map.original.size()==1) {
                Vertex v; v.point=map.original.front(); v.definition.point=v.point;
                map.vertices.push_back(std::move(v));
                continue;
            }
            for(size_t j=0;j<map.original.size();++j) {
                if(map.original.size()==2 && j==1)map.segment_end=map.vertices.size();
                const Point a=map.original[j], edge=map.original[(j+1)%map.original.size()]-a;
                struct Split { Scalar parameter; Vertex::Definition definition; };
                std::vector<Split> splits{{Scalar(0),Vertex::Definition{.point=a}}};
                auto add=[&](const Scalar &t,Vertex::Definition definition) {
                    if(t>=0 && t<1) splits.push_back({t,std::move(definition)});
                };
                auto collinear_point=[&](const Point &p) {
                    if(edge.cross(p-a)==0)
                        add((p-a).dot(edge)/edge.dot(edge),Vertex::Definition{.point=p});
                };
                collinear_point(start);
                if(include_previous_intersections) for(size_t h=0;h<i;++h) {
                    if(map.edge_bounds[j].disjoint(*maps[h].bounds)) continue;
                    const auto &previous=maps[h].original;
                    for(size_t k=0;k<previous.size();++k) {
                        // Bounds use the original binary-double coordinates,
                        // so strict separation is an exact rejection filter.
                        if(map.edge_bounds[j].disjoint(maps[h].edge_bounds[k])) continue;
                        const Point c=previous[k], other=previous[(k+1)%previous.size()]-c;
                        const Scalar denominator=edge.cross(other);
                        if(denominator==0) {
                            collinear_point(c);
                            collinear_point(c+other);
                        } else {
                            const Scalar t=(c-a).cross(other)/denominator;
                            const Scalar u=(c-a).cross(edge)/denominator;
                            if(u>=0 && u<=1) add(t,Vertex::Definition{.point=a+edge*t,
                                .edge_intersection=true,.polygon=h,.edge=k});
                        }
                    }
                }
                std::stable_sort(splits.begin(),splits.end(),[](const Split &x,const Split &y){
                    return x.parameter<y.parameter;
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
    ConvexRationalPolygon exact_contacts() {
        std::vector<Point> path,contacts;query_path(target,maps.size(),path);
        contact_details_from_path(path,false,&contacts);
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
    double length() { return double(query_length(target,maps.size())); }
};

}

#if defined(TPP_DIRECTIONAL_FILTERED)
std::vector<DirectionalTraceStep> solve_intersecting_map_trace_filtered(
        const Vector2 &start,const Vector2 &target,const std::vector<std::vector<Vector2>> &polygons,
        PreloadPolicy preload) {
    FilteredRational::Scope arithmetic_scope;
    return DirectionalMaps(start,target,polygons,preload).trace();
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
#else
ConvexRationalPolygon solve_intersecting_map_contacts_exact(const ConvexRationalPoint &start,
        const ConvexRationalPoint &target,const ConvexRationalPolygons &polygons) {
    return DirectionalMaps(Point(start),Point(target),polygons,PreloadPolicy::Lazy).exact_contacts();
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
