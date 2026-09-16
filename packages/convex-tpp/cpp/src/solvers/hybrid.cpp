#include "tpp/convex/hybrid.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "tpp/convex/solver.h"
#include "common.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace tpp {
namespace {
std::mutex aggregate_mutex;
ConvexHybridAggregate aggregate;

struct AggregateRecorder {
    ConvexHybridResult *result;
    ~AggregateRecorder() {
        std::lock_guard lock(aggregate_mutex);const auto &r=*result;
        ++aggregate.total_calls;aggregate.disjoint_calls+=r.stats.disjoint;
        aggregate.certified_double_disjoint_calls+=r.backend==ConvexHybridBackend::DoubleDisjoint&&r.stats.double_certified;
        aggregate.certified_double_intersection_calls+=r.backend==ConvexHybridBackend::DoubleIntersection&&r.stats.double_certified;
        aggregate.rational_disjoint_fallbacks+=r.backend==ConvexHybridBackend::RationalDisjoint;
        aggregate.rational_intersection_fallbacks+=r.backend==ConvexHybridBackend::RationalIntersection;
        ++aggregate.fallback_reasons[static_cast<size_t>(r.fallback_reason)];
        aggregate.predicate_exact_evaluations+=r.stats.predicate_exact_evaluations;
        aggregate.dispatch_seconds+=r.stats.dispatch_seconds;
        aggregate.double_solver_seconds+=r.stats.double_solver_seconds;
        aggregate.contact_materialization_seconds+=r.stats.contact_materialization_seconds;
        aggregate.certificate_seconds+=r.stats.certificate_seconds;
        aggregate.rational_fallback_seconds+=r.stats.rational_fallback_seconds;
        aggregate.total_seconds+=r.stats.total_seconds;
    }
};
}

void reset_convex_hybrid_aggregate() {std::lock_guard lock(aggregate_mutex);aggregate={};}
ConvexHybridAggregate convex_hybrid_aggregate() {std::lock_guard lock(aggregate_mutex);return aggregate;}

namespace {
using Rational = boost::multiprecision::cpp_rational;
using Clock = std::chrono::steady_clock;

struct Point {
    Rational x=0,y=0;
    Point()=default;
    Point(Rational x_,Rational y_):x(std::move(x_)),y(std::move(y_)){}
    explicit Point(Vector2 p):x(p.x),y(p.y){}
    Point operator+(const Point &p)const{return{x+p.x,y+p.y};}
    Point operator-(const Point &p)const{return{x-p.x,y-p.y};}
    Point operator*(const Rational &s)const{return{x*s,y*s};}
    Rational cross(const Point &p)const{return x*p.y-y*p.x;}
    Rational dot(const Point &p)const{return x*p.x+y*p.y;}
    bool operator==(const Point &)const=default;
    Vector2 external()const{return{x.convert_to<double>(),y.convert_to<double>()};}
};
using Polygon=std::vector<Point>;

double elapsed(Clock::time_point began) {
    return std::chrono::duration<double>(Clock::now()-began).count();
}

std::vector<Polygon> exact_polygons(const std::vector<std::vector<Vector2>> &polygons) {
    std::vector<Polygon> result;
    result.reserve(polygons.size());
    for(const auto &input:polygons) {
        Polygon p;
        for(auto v:input) {
            if(!v.is_finite()) throw std::invalid_argument("Nonfinite polygon coordinate");
            Point q(v);if(p.empty() || !(p.back()==q))p.push_back(std::move(q));
        }
        if(p.size()>1 && p.front()==p.back())p.pop_back();
        Rational area=0;
        for(size_t i=0;i<p.size();++i)area+=p[i].cross(p[(i+1)%p.size()]);
        if(p.size()<3 || area==0)throw std::invalid_argument("Polygon must have positive area");
        if(area<0)std::reverse(p.begin(),p.end());
        result.push_back(std::move(p));
    }
    return result;
}

bool inside(const Point &q,const Polygon &p) {
    for(size_t i=0;i<p.size();++i)
        if((p[(i+1)%p.size()]-p[i]).cross(q-p[i])<0)return false;
    return true;
}

bool clip(const Point &a,const Point &b,const Polygon &p,Rational floor,Rational &lo,Rational &hi) {
    const Point d=b-a;lo=floor;hi=1;
    for(size_t i=0;i<p.size();++i) {
        const Point edge=p[(i+1)%p.size()]-p[i];
        const Rational constant=edge.cross(a-p[i]),slope=edge.cross(d);
        if(slope>0) {const Rational t=-constant/slope;if(t>lo)lo=t;}
        else if(slope<0) {const Rational t=-constant/slope;if(t<hi)hi=t;}
        else if(constant<0)return false;
    }
    return lo<=hi && hi>=floor && lo<=1;
}

bool materialize(const std::vector<Vector2> &raw_path,const std::vector<Polygon> &polygons,
                 bool last,std::vector<Point> &contacts) {
    std::vector<Point> path;path.reserve(raw_path.size());
    for(auto v:raw_path) {
        if(!v.is_finite())return false;
        Point p(v);if(path.empty() || !(path.back()==p))path.push_back(std::move(p));
    }
    if(path.empty())return false;
    contacts.clear();contacts.reserve(polygons.size());
    if(path.size()==1) {
        for(const auto &p:polygons) {if(!inside(path.front(),p))return false;contacts.push_back(path.front());}
        return true;
    }
    size_t segment=1;Rational rate=0;
    for(const auto &polygon:polygons) {
        bool found=false;
        while(segment<path.size()) {
            Rational lo,hi;
            if(clip(path[segment-1],path[segment],polygon,rate,lo,hi)) {
                rate=last?std::min(hi,Rational(1)):std::max(lo,rate);
                contacts.push_back(path[segment-1]+(path[segment]-path[segment-1])*rate);
                found=true;break;
            }
            ++segment;rate=0;
        }
        if(!found)return false;
    }
    return true;
}

bool segment_hits(const Point &a,const Point &b,const Polygon &p) {
    Rational lo,hi;return clip(a,b,p,0,lo,hi);
}

struct Bounds {Rational min_x,max_x,min_y,max_y;};
Bounds bounds(const Polygon &p) {
    Bounds b{p.front().x,p.front().x,p.front().y,p.front().y};
    for(const auto &v:p){b.min_x=std::min(b.min_x,v.x);b.max_x=std::max(b.max_x,v.x);
        b.min_y=std::min(b.min_y,v.y);b.max_y=std::max(b.max_y,v.y);}return b;
}
bool bounds_disjoint(const Bounds &a,const Bounds &b) {
    return a.max_x<b.min_x||b.max_x<a.min_x||a.max_y<b.min_y||b.max_y<a.min_y;
}

bool pairwise_disjoint(const std::vector<Polygon> &polygons) {
    std::vector<Bounds> polygon_bounds;polygon_bounds.reserve(polygons.size());
    for(const auto &p:polygons)polygon_bounds.push_back(bounds(p));
    for(size_t i=0;i<polygons.size();++i)for(size_t j=i+1;j<polygons.size();++j) {
        if(bounds_disjoint(polygon_bounds[i],polygon_bounds[j]))continue;
        if(inside(polygons[i].front(),polygons[j]) || inside(polygons[j].front(),polygons[i]))return false;
        for(size_t e=0;e<polygons[i].size();++e) {
            Polygon edge_box_points{polygons[i][e],polygons[i][(e+1)%polygons[i].size()]};
            if(bounds_disjoint(bounds(edge_box_points),polygon_bounds[j]))continue;
            if(segment_hits(polygons[i][e],polygons[i][(e+1)%polygons[i].size()],polygons[j]))return false;
        }
    }
    return true;
}

int normalized_difference_sign(const Rational &p,const Rational &a2,
                               const Rational &q,const Rational &b2) {
    if(p>=0 && q<=0)return p==0&&q==0?0:1;
    if(p<=0 && q>=0)return p==0&&q==0?0:-1;
    const Rational left=p*p*b2,right=q*q*a2;
    if(left==right)return 0;
    if(p>0)return left>right?1:-1;
    return left<right?1:-1;
}

ConvexFallbackReason certify(const Vector2 &start,const Vector2 &target,
        const std::vector<Polygon> &polygons,const std::vector<Point> &contacts,
        std::size_t &exact_predicates) {
    if(contacts.size()!=polygons.size())return ConvexFallbackReason::ContactConstruction;
    std::vector<Point> chain;chain.reserve(contacts.size()+2);
    chain.emplace_back(start);chain.insert(chain.end(),contacts.begin(),contacts.end());chain.emplace_back(target);
    for(size_t i=0;i<contacts.size();++i) {
        if(!inside(contacts[i],polygons[i]))return ConvexFallbackReason::MembershipOrOrdering;
        const Point incoming=chain[i+1]-chain[i],outgoing=chain[i+2]-chain[i+1];
        const Rational a2=incoming.dot(incoming),b2=outgoing.dot(outgoing);
        if(a2==0 || b2==0)return ConvexFallbackReason::CoincidentContact;
        for(const Point &vertex:polygons[i]) {
            const Point feasible=vertex-contacts[i];
            ++exact_predicates;
            if(normalized_difference_sign(incoming.dot(feasible),a2,
                                          outgoing.dot(feasible),b2)<0)
                return ConvexFallbackReason::LocalOptimality;
        }
    }
    return ConvexFallbackReason::None;
}

double contact_length(Vector2 start,Vector2 target,const std::vector<Vector2> &contacts) {
    long double value=0;Vector2 previous=start;
    for(auto p:contacts){value+=std::hypot((long double)p.x-previous.x,(long double)p.y-previous.y);previous=p;}
    value+=std::hypot((long double)target.x-previous.x,(long double)target.y-previous.y);
    return double(value);
}

std::pair<double,double> sqrt_bounds(const Rational &squared) {
    if(squared==0)return {0,0};
    using Decimal=boost::multiprecision::cpp_dec_float_100;
    double lower=sqrt(squared.convert_to<Decimal>()).convert_to<double>();
    auto square=[](double x){const Rational q(x);return q*q;};
    for(size_t steps=0;square(lower)>squared;++steps) {
        if(steps==8)throw std::runtime_error("Could not bracket exact square root from above");
        lower=std::nextafter(lower,0.0);
    }
    for(size_t steps=0;;++steps) {
        const double next=std::nextafter(lower,std::numeric_limits<double>::infinity());
        if(square(next)>squared)break;
        if(steps==8)throw std::runtime_error("Could not bracket exact square root from below");
        lower=next;
    }
    if(square(lower)==squared)return {lower,lower};
    return {lower,std::nextafter(lower,std::numeric_limits<double>::infinity())};
}

double rational_lower(const Rational &q) {
    double d=q.convert_to<double>();
    while(Rational(d)>q)d=std::nextafter(d,-std::numeric_limits<double>::infinity());
    return d;
}
double rational_upper(const Rational &q) {
    double d=q.convert_to<double>();
    while(Rational(d)<q)d=std::nextafter(d,std::numeric_limits<double>::infinity());
    return d;
}

void set_exact_bounds(ConvexHybridResult &result,Vector2 start,Vector2 target,
                      const std::vector<Point> &contacts) {
    std::vector<Point> chain;chain.reserve(contacts.size()+2);chain.emplace_back(start);
    chain.insert(chain.end(),contacts.begin(),contacts.end());chain.emplace_back(target);
    Rational lower=0,upper=0;
    for(size_t i=1;i<chain.size();++i) {
        const Point d=chain[i]-chain[i-1];
        const auto [lo,hi]=sqrt_bounds(d.dot(d));lower+=Rational(lo);upper+=Rational(hi);
    }
    result.lower_bound=rational_lower(lower);result.upper_bound=rational_upper(upper);
}

void set_bounds(ConvexHybridResult &result,Vector2 start,Vector2 target) {
    const double value=contact_length(start,target,result.contacts);
    result.lower_bound=std::nextafter(value,-std::numeric_limits<double>::infinity());
    result.upper_bound=std::nextafter(value,std::numeric_limits<double>::infinity());
}
}

const char *to_string(ConvexFallbackReason reason) {
    switch(reason) {
        case ConvexFallbackReason::None:return "none";
        case ConvexFallbackReason::LocatorOrRefoldingException:return "locator_or_refolding_exception";
        case ConvexFallbackReason::Nonfinite:return "nonfinite";
        case ConvexFallbackReason::ContactConstruction:return "contact_construction";
        case ConvexFallbackReason::MembershipOrOrdering:return "membership_or_ordering";
        case ConvexFallbackReason::LocalOptimality:return "local_optimality";
        case ConvexFallbackReason::CoincidentContact:return "coincident_contact";
        case ConvexFallbackReason::ShadowMismatch:return "shadow_mismatch";
    }
    return "unknown";
}

std::vector<Vector2> reconstruct_convex_polyline(const Vector2 &start,const Vector2 &target,
        const std::vector<Vector2> &contacts,bool compact) {
    std::vector<Vector2> result;result.reserve(contacts.size()+2);result.push_back(start);
    result.insert(result.end(),contacts.begin(),contacts.end());result.push_back(target);
    if(compact)remove_collinear_points_inplace(result);
    return result;
}

ConvexHybridResult tpp_convex_solve_hybrid(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &input,const ConvexHybridOptions &options) {
    const auto began=Clock::now();ConvexHybridResult result;
    AggregateRecorder recorder{&result};
    const auto dispatch_began=Clock::now();
    const auto polygons=exact_polygons(input);
    result.stats.disjoint=pairwise_disjoint(polygons);
    result.stats.dispatch_seconds=elapsed(dispatch_began);
    if(input.empty()) {set_bounds(result,start,target);result.stats.total_seconds=elapsed(began);return result;}
    std::vector<Vector2> candidate;std::vector<Point> exact_contacts;
    result.stats.double_attempted=true;
    try {
        const auto solve_began=Clock::now();
        candidate=result.stats.disjoint?tpp_convex_solve_binary_search_disjoint(start,target,input)
            :detail::solve_intersecting_maps_unchecked_double(start,target,input);
        result.stats.double_solver_seconds=elapsed(solve_began);
        const auto contact_began=Clock::now();
        if(!materialize(candidate,polygons,result.stats.disjoint,exact_contacts))
            result.fallback_reason=ConvexFallbackReason::ContactConstruction;
        result.stats.contact_materialization_seconds=elapsed(contact_began);
    } catch(const std::exception &) {
        result.fallback_reason=ConvexFallbackReason::LocatorOrRefoldingException;
    }
    if(result.fallback_reason==ConvexFallbackReason::None) {
        result.contacts.reserve(exact_contacts.size());for(const auto &p:exact_contacts)result.contacts.push_back(p.external());
        if(options.mode==ConvexHybridMode::Unchecked) {
            result.backend=result.stats.disjoint?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
            set_bounds(result,start,target);result.stats.total_seconds=elapsed(began);return result;
        }
        const auto certificate_began=Clock::now();
        result.fallback_reason=certify(start,target,polygons,exact_contacts,result.stats.predicate_exact_evaluations);
        result.stats.certificate_seconds=elapsed(certificate_began);
        result.stats.double_certified=result.fallback_reason==ConvexFallbackReason::None;
    }
    if(result.stats.double_certified && !options.shadow_rational) {
        result.backend=result.stats.disjoint?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
        set_exact_bounds(result,start,target,exact_contacts);result.stats.total_seconds=elapsed(began);return result;
    }
    const auto fast_contacts=result.contacts;
    const auto fallback_began=Clock::now();
    result.contacts=detail::solve_intersecting_map_contacts(start,target,input,result.stats.disjoint);
    result.stats.rational_fallback_seconds=elapsed(fallback_began);
    if(options.shadow_rational && result.stats.double_certified) {
        const double fast=contact_length(start,target,fast_contacts);
        const double exact=detail::length_intersecting_maps(start,target,input);
        const double tolerance=64*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(exact));
        if(std::abs(fast-exact)<=tolerance) {
            result.contacts=fast_contacts;
            result.backend=result.stats.disjoint?ConvexHybridBackend::DoubleDisjoint:ConvexHybridBackend::DoubleIntersection;
            result.fallback_reason=ConvexFallbackReason::None;
            set_exact_bounds(result,start,target,exact_contacts);result.stats.total_seconds=elapsed(began);return result;
        }
        result.fallback_reason=ConvexFallbackReason::ShadowMismatch;
    }
    result.stats.rational_fallback=true;
    result.backend=result.stats.disjoint?ConvexHybridBackend::RationalDisjoint:ConvexHybridBackend::RationalIntersection;
    set_bounds(result,start,target);result.stats.total_seconds=elapsed(began);return result;
}

std::vector<Vector2> tpp_convex_solve_hybrid_safe(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    const auto result=tpp_convex_solve_hybrid(start,target,polygons);
    return reconstruct_convex_polyline(start,target,result.contacts);
}
double tpp_convex_solve_length_hybrid_safe(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    return tpp_convex_solve_hybrid(start,target,polygons).lower_bound;
}
std::vector<Vector2> tpp_convex_solve_hybrid_unchecked(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    ConvexHybridOptions options;options.mode=ConvexHybridMode::Unchecked;
    const auto result=tpp_convex_solve_hybrid(start,target,polygons,options);
    return reconstruct_convex_polyline(start,target,result.contacts);
}
double tpp_convex_solve_length_hybrid_unchecked(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &polygons) {
    ConvexHybridOptions options;options.mode=ConvexHybridMode::Unchecked;
    return tpp_convex_solve_hybrid(start,target,polygons,options).lower_bound;
}
}
