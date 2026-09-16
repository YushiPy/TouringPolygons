#include "tpp/convex/detail/intersecting_maps.h"

#include <boost/multiprecision/cpp_int.hpp>
#ifdef TPP_EXPERIMENT_NATIVE_DOUBLE
#include "native_double_experiment.h"
#endif
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace tpp::detail {
namespace {

#ifdef TPP_EXPERIMENT_NATIVE_DOUBLE
using Scalar = NativeDoubleExperimentScalar;
#else
using Scalar = boost::multiprecision::cpp_rational;
#endif

struct Point {
    Scalar x = 0, y = 0;
    Point() = default;
    Point(Scalar x_, Scalar y_) : x(std::move(x_)), y(std::move(y_)) {}
    explicit Point(Vector2 p) : x(p.x), y(p.y) {}
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
    const int constant = sign(axis.cross(q.point-origin));
    if(constant) return constant;
    const int first=sign(axis.cross(q.side));
    if(first || !break_ties) return first;
    const int second=sign(axis.cross(q.tie1));
    return second ? second : sign(axis.cross(q.tie2));
}
int dot_sign(const Point &axis, const Query &q, const Point &origin) {
    const int constant = sign(axis.dot(q.point-origin));
    if(constant) return constant;
    const int first=sign(axis.dot(q.side));
    if(first) return first;
    const int second=sign(axis.dot(q.tie1));
    return second ? second : sign(axis.dot(q.tie2));
}
bool same_direction(const Point &a,const Point &b) {
    return a.cross(b)==0 && a.dot(b)>0;
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
    std::vector<Point> membership_corners;
    std::vector<Vertex> vertices;
    struct CachedQuery { Query query; Point source; };
    std::optional<CachedQuery> last_query;
};

class DirectionalMaps {
    Point start, target;
    std::vector<Map> maps;

    static bool inside(const Query &q,const std::vector<Point> &polygon) {
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
        return r1.cross(r2)>=0 ? c1 && c2 : c1 || c2;
    }

    // The existing binary locator's chord predicate, evaluated on symbolic
    // queries. Rays themselves use the corrected incident limits (Dror Claim 1).
    static bool in_edge_plus(const Query &q,const Point &v1,const Point &v2,
                             const Point &r1,const Point &r2) {
        if(v1==v2) return in_cone(q,v1,r1,r2);
        const Point dv=v2-v1;
        if(same_direction(r1,dv) || same_direction(r2,-dv)) return false;
        if(dv.cross(r1)<0) {
            if(dv.cross(r2)<0)
                return cross_sign(r1,q,v1)>=0 && cross_sign(r2,q,v2)<=0
                    && cross_sign(dv,q,v1)<=0;
            return cross_sign(dv,q,v1)<0 ? cross_sign(r1,q,v1)>=0 : cross_sign(r2,q,v2)<=0;
        }
        if(dv.cross(r2)<0)
            return cross_sign(dv,q,v2)<0 ? cross_sign(r2,q,v2)<=0 : cross_sign(r1,q,v1)>=0;
        return cross_sign(r1,q,v1)>=0 || cross_sign(r2,q,v2)<=0 || cross_sign(dv,q,v1)<=0;
    }

    void build_vertex(size_t i,size_t j) {
        auto &vertices=maps[i].vertices;
        auto &v=vertices[j];
        if(v.ready) return;
        const Point before=vertices[(j+vertices.size()-1)%vertices.size()].point;
        const Point after=vertices[(j+1)%vertices.size()].point;
        auto outgoing=[&](Point side,Point edge,bool &reflects) {
            const Point source=virtual_source({v.point,side},i);
            Point incoming=v.point-source;
            if(incoming.zero()) incoming=side;
            reflects=edge.cross(incoming)>0;
            return reflects ? reflect_direction(incoming,edge) : incoming;
        };
        v.before_ray=outgoing(before-v.point,v.point-before,v.before_reflects);
        v.after_ray=outgoing(after-v.point,after-v.point,v.after_reflects);
        v.ready=true;
    }

    // -1: crossing; 2*j: vertex; 2*j+1: reflection on the following pseudo-edge.
    long long locate(const Query &q,size_t level) {
        auto &map=maps[level-1];
        if(inside(q,map.membership_corners)) return -1;
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
            throw std::runtime_error("Directional map locator invariant failed at level "+std::to_string(level));
        if(a.after_reflects!=b.before_reflects &&
           (b.point-a.point).cross(a.after_ray)!=0 &&
           (b.point-a.point).cross(b.before_ray)!=0)
            throw std::runtime_error("Directional map pseudo-edge changes contact type at level "+std::to_string(level));
        // Keep the crossing sentinel signed on both 32-bit WASM and 64-bit hosts.
        if(a.after_reflects || b.before_reflects) return static_cast<long long>(2*left+1);
        return -1;
    }

    Point virtual_source(const Query &q,size_t level) {
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

    void split_boundaries() {
        for(size_t i=0;i<maps.size();++i) {
            auto &map=maps[i];
            for(size_t j=0;j<map.original.size();++j) {
                const Point a=map.original[j], edge=map.original[(j+1)%map.original.size()]-a;
                std::vector<Scalar> parameters{Scalar(0)};
                auto add=[&](const Scalar &t) { if(t>=0 && t<1) parameters.push_back(t); };
                auto collinear_point=[&](const Point &p) {
                    if(edge.cross(p-a)==0) add((p-a).dot(edge)/edge.dot(edge));
                };
                collinear_point(start);
                for(size_t h=0;h<i;++h) {
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
                            if(u>=0 && u<=1) add(t);
                        }
                    }
                }
                std::sort(parameters.begin(),parameters.end());
                parameters.erase(std::unique(parameters.begin(),parameters.end()),parameters.end());
                for(const auto &t:parameters) {
                    Vertex v;
                    v.point=a+edge*t;v.original_edge=j;v.edge_parameter=t;
                    map.vertices.push_back(std::move(v));
                }
            }
        }
    }

public:
    DirectionalMaps(Vector2 s,Vector2 t,const std::vector<std::vector<Vector2>> &polygons,
                    PreloadPolicy preload) : start(s),target(t) {
        maps.resize(polygons.size());
        for(size_t i=0;i<polygons.size();++i) {
            auto &p=maps[i].original;
            for(auto v:polygons[i]) {
                if(!std::isfinite(v.x) || !std::isfinite(v.y))
                    throw std::invalid_argument("Nonfinite polygon coordinate");
                append(p,Point(v));
            }
            if(p.size()>1 && p.front()==p.back()) p.pop_back();
            Scalar area=0;
            for(size_t j=0;j<p.size();++j) area+=p[j].cross(p[(j+1)%p.size()]);
            if(p.size()<3 || area==0) throw std::invalid_argument("Polygon must have positive area");
            if(area<0) std::reverse(p.begin(),p.end());
            // Keep original edge provenance, but omit straight intermediate
            // corners from the auxiliary O(log m) closed-membership fan.
            for(size_t j=0;j<p.size();++j) {
                const Point before=p[(j+p.size()-1)%p.size()],after=p[(j+1)%p.size()];
                maps[i].edge_bounds.emplace_back(p[j].external(),after.external());
                if((p[j]-before).cross(after-p[j])!=0)
                    maps[i].membership_corners.push_back(p[j]);
            }
        }
        split_boundaries();
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
    std::vector<Vector2> contacts(bool use_last_contact) {
        std::vector<Point> path;
        query_path(target,maps.size(),path);
        std::vector<Vector2> result;
        result.reserve(maps.size());
        if(path.empty()) throw std::runtime_error("Directional map returned an empty path");
        size_t segment=path.size()==1 ? 0 : 1;
        Scalar rate=0;
        for(size_t i=0;i<maps.size();++i) {
            bool found=false;
            if(path.size()==1) {
                if(!inside({path.front(),{}},maps[i].original))
                    throw std::runtime_error("Stationary path misses polygon "+std::to_string(i));
                result.push_back(path.front().external());
                continue;
            }
            while(segment<path.size()) {
                const Point a=path[segment-1], direction=path[segment]-a;
                Scalar lo=rate,hi=1;
                for(size_t j=0;j<maps[i].original.size();++j) {
                    const Point v=maps[i].original[j];
                    const Point edge=maps[i].original[(j+1)%maps[i].original.size()]-v;
                    const Scalar constant=edge.cross(a-v),slope=edge.cross(direction);
                    if(slope>0) {const Scalar crossing=-constant/slope;if(crossing>lo)lo=crossing;}
                    else if(slope<0) {const Scalar crossing=-constant/slope;if(crossing<hi)hi=crossing;}
                    else if(constant<0) {hi=-1;break;}
                }
                if(lo<=hi && hi>=rate && lo<=1) {
                    rate=use_last_contact ? std::min(hi,Scalar(1)) : std::max(lo,rate);
                    result.push_back((a+direction*rate).external());
                    found=true;
                    break;
                }
                ++segment;rate=0;
            }
            if(!found) throw std::runtime_error("Could not materialize ordered contact "+std::to_string(i));
        }
        return result;
    }
    double length() { return double(query_length(target,maps.size())); }
};

}

#ifdef TPP_DIRECTIONAL_DOUBLE_VARIANT
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
#else
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
#endif

}
