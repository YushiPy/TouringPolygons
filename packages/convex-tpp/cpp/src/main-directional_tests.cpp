#include "tests.h"
#include "tpp/convex/certified.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "tpp/convex/hybrid.h"
#include "tpp_convex.h"
#include "solvers/filtered_rational.h"
#include "solvers/binary_certificate.h"
#include "solvers/zero_contact_certificate.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {
using Polygon=std::vector<Vector2>;
using Polygons=std::vector<Polygon>;
using tpp::TestCase;
size_t checks=0,failures=0,unresolved=0,hybrid_fast=0,hybrid_fallback=0,hybrid_shadow_mismatch=0;
bool public_api=false;

void check(bool condition,const std::string &name) {
    ++checks;
    if(!condition) { ++failures; std::cout<<"FAIL "<<name<<std::endl; }
}
Polygon box(double x,double y,double X,double Y) {return {{x,y},{X,y},{X,Y},{x,Y}};}
double length(const Polygon &p) {
    long double result=0;
    for(size_t i=1;i<p.size();++i)
        result+=std::hypot((long double)p[i].x-p[i-1].x,(long double)p[i].y-p[i-1].y);
    return double(result);
}
Polygon solve(const TestCase &c,bool eager=false) {
    if(public_api) return eager ? tpp::tpp_convex_solve_binary_search_eager(c.start,c.target,c.polygons)
                               : tpp::tpp_convex_solve_binary_search_lazy(c.start,c.target,c.polygons);
    return tpp::detail::solve_intersecting_maps(c.start,c.target,c.polygons,
                eager?tpp::PreloadPolicy::Eager:tpp::PreloadPolicy::Lazy);
}
void describe(const TestCase &c) {
    std::cout<<"s "<<c.start.x<<' '<<c.start.y<<" t "<<c.target.x<<' '<<c.target.y<<'\n';
    for(const auto &p:c.polygons) {
        std::cout<<"polygon";
        for(auto v:p)std::cout<<" ("<<v.x<<','<<v.y<<')';
        std::cout<<'\n';
    }
}

void interval_rounding_regressions() {
    using tpp::detail::CycleInterval;
    std::fenv_t saved;
    std::fegetenv(&saved);
    auto compare=[&](uint64_t bits) {
        const double x=std::bit_cast<double>(bits);
        const double low=std::isnan(x)?-INFINITY:std::nextafter(x,-INFINITY);
        const double high=std::isnan(x)?INFINITY:std::nextafter(x,INFINITY);
        check(std::bit_cast<uint64_t>(CycleInterval::down(x))==std::bit_cast<uint64_t>(low) &&
              std::bit_cast<uint64_t>(CycleInterval::up(x))==std::bit_cast<uint64_t>(high),
              "interval expansion matches nextafter bit-for-bit");
    };
    for(uint64_t exponent=0;exponent<2048;++exponent) {
        const uint64_t center=exponent<<52;
        for(int offset=-3;offset<=3;++offset) {
            if(offset<0 && center<uint64_t(-offset))continue;
            const uint64_t bits=center+uint64_t(offset);
            compare(bits);compare(bits|(uint64_t(1)<<63));
        }
    }
    for(uint64_t bits:{0x000fffffffffffffULL,0x7fefffffffffffffULL,
                       0x7ff8000000000000ULL,0xfff8000000000000ULL})compare(bits);
    std::mt19937_64 random(2026100301);
    for(size_t i=0;i<200000;++i)compare(random());
    std::fesetenv(&saved);
}

void dyadic_orientation_regressions() {
    using P=tpp::ConvexRationalPoint;
    auto compare=[&](Vector2 a,Vector2 b,Vector2 q) {
        const auto sign=tpp::detail::dyadic_orientation(a,b,q);
        if(!sign)return;
        const auto determinant=(P(b)-P(a)).cross(P(q)-P(a));
        const int reference=determinant>0?1:determinant<0?-1:0;
        check(*sign==reference,"bounded dyadic orientation matches exact rational sign");
    };
    std::mt19937_64 random(2026100302);
    std::uniform_real_distribution<double> coordinate(-1,1);
    for(size_t i=0;i<20000;++i) {
        const int exponent=int(i%2001)-1000;
        auto point=[&] {return Vector2{std::ldexp(coordinate(random),exponent),
                                      std::ldexp(coordinate(random),-exponent)};};
        const Vector2 a=point(),b=point();
        Vector2 q=i%4==0?a:i%4==1?(a+b)*.5:point();
        if(i%4==2)q.x=std::nextafter(q.x,INFINITY);
        if(i%4==3)q.y=std::nextafter(q.y,-INFINITY);
        compare(a,b,q);
    }
    for(int exponent:{-1074,-1022,-100,0,100,1000}) {
        const double unit=std::ldexp(1.,exponent);
        compare({0,0},{unit,0},{0,unit});
        compare({-unit,0},{unit,0},{0,-unit});
        compare({0,0},{unit,unit},{unit,unit});
    }
#if defined(__SIZEOF_INT128__)
    const Vector2 a{1,1},b{-1,-1},q{0x1p-8,-0x1p-8};
    compare(a,b,q);
    check(tpp::detail::dyadic_orientation(a,b,q).has_value(),"61-bit scaled coordinates use exact integer determinant");
    check(!tpp::detail::dyadic_orientation(a,b,{0x1p-9,-0x1p-9}),
          "62-bit scaled coordinates retain rational fallback before overflow");
    check(tpp::detail::dyadic_orientation({-0.,0.},{1.,0.},{0.,1.})==1,
          "dyadic orientation supports signed zeros");
    check(!tpp::detail::dyadic_orientation({std::ldexp(1.,-1000),0},
          {std::ldexp(1.,1000),0},{0,1}),"wide dyadic exponents retain rational fallback");
#endif
    check(!tpp::detail::dyadic_orientation({INFINITY,0},{1,0},{0,1}),
          "dyadic orientation declines nonfinite inputs");
}

void binary_membership_memo_regressions() {
    const Polygon polygon{{0,0},{1,1},{0,1}},shifted{{2,2},{3,3},{2,3}};
    tpp::ConvexRationalPolygon exact,other;
    for(auto p:polygon)exact.emplace_back(p);
    for(auto p:shifted)other.emplace_back(p);
    tpp::detail::BinaryContactMemo memo;
    size_t predicates=0;
    const Vector2 inside{.5,std::nextafter(.5,1.)},outside{.5,std::nextafter(.5,0.)};
    for(size_t i=0;i<100;++i) {
        const auto point=i%2?outside:inside;
        size_t reference_predicates=0;
        const bool reference=tpp::detail::interval_convex_contains(point,polygon,exact,reference_predicates);
        check(memo.contains(point,polygon,exact,predicates)==reference,
              "membership memo preserves accepted and rejected proofs");
    }
    check(memo.hits==98,"two membership slots retain proposal and repaired contact");
    check(!memo.contains(inside,shifted,other,predicates),"membership memo invalidates changed geometry identity");
    for(size_t i=0;i<20;++i) {
        const Vector2 point{double(i)/16.,.75};
        size_t reference_predicates=0;
        check(memo.contains(point,polygon,exact,predicates)==
              tpp::detail::interval_convex_contains(point,polygon,exact,reference_predicates),
              "membership memo eviction preserves proof");
    }
}

void dispatch_cache_regressions() {
    tpp::DynamicConvexTppWorkspace cached,uncached;
    cached.borrow_hybrid_geometry=true;
    uncached.borrow_hybrid_geometry=false;
    uncached.cache_disjoint_dispatch=false;
    uncached.cache_interval_geometry=false;
    tpp::ConvexHybridOptions options;options.max_gap=1e-6;
    auto compare=[&](const Polygons &polygons,bool disjoint,Vector2 start=Vector2{-4,-1},Vector2 target=Vector2{4,-1}) {
        const auto reference=tpp::tpp_convex_solve_hybrid(start,target,polygons,options,uncached);
        const auto result=tpp::tpp_convex_solve_hybrid(start,target,polygons,options,cached);
        check(result.stats.disjoint==disjoint && reference.stats.disjoint==disjoint,
              "cached dispatch preserves exact closed-set intersection");
        check(result.contacts==reference.contacts && result.lower_bound==reference.lower_bound &&
              result.upper_bound==reference.upper_bound && result.backend==reference.backend &&
              result.fallback_reason==reference.fallback_reason,
              "cached dispatch preserves contacts, bounds and recovery");
        check(uncached.hybrid_cache && reference.stats.dispatch_pair_cache_hits==0,
              "dispatch ablation retains geometry without reusing pairs");
        return result;
    };
    const auto a=box(-2,0,0,2),b=box(1,0,2,2);
    compare({a,b},true);
    const auto warm=compare({a,b},true);
    check(warm.stats.dispatch_pair_queries==1 && warm.stats.dispatch_pair_cache_hits==1 &&
          warm.stats.dispatch_pair_exact_checks==0,"warm pair avoids repeated exact dispatch");
    check(compare({b,a},true,{5,-2},{-5,-2}).stats.dispatch_pair_cache_hits==1,
          "pair certificate is independent of order and endpoints");
    compare({a,box(0,0,2,2)},false); // Shared edge is intersection.
    check(compare({box(0,0,2,2),a},false).stats.dispatch_pair_cache_hits==1,
          "intersecting pair certificates are reused too");
    compare({a,a},false);
    compare({a,box(-1,1,1,3)},false);
    compare({box(-2,-2,2,2),box(-1,-1,1,1)},false);
    compare({box(-2,0,1,2),box(std::nextafter(1.0,2.0),0,2,2)},true);
    compare({box(-2,0,1,2),box(std::nextafter(1.0,0.0),0,2,2)},false);
    auto reversed=a;std::reverse(reversed.begin(),reversed.end());
    compare({reversed,b},true);
    auto closed=a;closed.push_back(closed.front());compare({closed,b},true);
    auto copy=cached;
    const auto detached=tpp::tpp_convex_solve_hybrid({-4,-1},{4,-1},{a,b},options,copy);
    check(detached.stats.dispatch_pair_cache_hits==0,"copied workspaces detach mutable caches");
    // Fill past the geometry bound. Handles already selected for the current
    // call must survive an eviction while the remaining inputs are prepared.
    cached={};
    for(size_t i=0;i<2048;++i)
        tpp::tpp_convex_solve_hybrid({-4,-1},{4,-1},{box(10+double(i),0,11+double(i),2)},options,cached);
    compare({box(2057,0,2058,2),a,b},true);
    compare({a,b},true);
    check(compare({b,a},true).stats.dispatch_pair_cache_hits==1,
          "eviction keeps new pair identities coherent");
    // The pair table has a separate bounded capacity. Crossing it preserves
    // the same exact dispatcher, even while all geometry still fits.
    Polygons many;
    for(size_t i=0;i<365;++i)many.push_back(box(3*double(i),0,3*double(i)+1,1));
    const auto full=compare(many,true,{-2,0},{1098,0});
    check(full.stats.dispatch_pair_queries==365*364/2,"bounded pair table checks the entire sequence");
    compare(many,true,{-2,0},{1098,0});
}

void cached_contact_rotation_regressions() {
    tpp::DynamicConvexTppWorkspace workspace;workspace.borrow_hybrid_geometry=true;
    tpp::ConvexHybridOptions options; // Zero gap exercises rational materialization.
    const std::vector<Polygons> inputs{
        {},{box(-1,0,1,2)},
        {box(-2,0,-1,2),box(1,0,2,2)},
        {box(-2,0,0,2),box(0,0,2,2)},
        {{{-2,0},{0,0},{1,0},{1,2},{-2,2}},
         {{0,-1},{2,0},{1,3}},box(-1,-1,1,1)}};
    for(size_t repeat=0;repeat<2;++repeat)for(auto polygons:inputs) {
        if(repeat) {
            std::reverse(polygons.begin(),polygons.end());
            for(auto &p:polygons){std::reverse(p.begin(),p.end());p.push_back(p.front());}
        }
        for(double height:{-1.,3.}) {
            const Vector2 start{-4,height},target{4,height};
            const auto reference=tpp::tpp_convex_solve_hybrid(start,target,polygons,options);
            const auto cached=tpp::tpp_convex_solve_hybrid(start,target,polygons,options,workspace);
            check(cached.contacts==reference.contacts&&cached.lower_bound==reference.lower_bound&&
                  cached.upper_bound==reference.upper_bound&&cached.backend==reference.backend&&
                  cached.fallback_reason==reference.fallback_reason,
                  "cached contact rotations preserve exact uncached reconstruction");
        }
    }
}
Polygon verify(const TestCase &c,const std::string &name,bool oracle=false) {
    const auto old_failures=failures;
    try {
        const auto path=solve(c);
        const auto validation=tpp::validate_ordered_path(c.start,c.target,c.polygons,path);
        check(validation.valid,name+" ordered visits");
        const double value=length(path), scale=std::max(1.0,value);
        if(!c.solution.empty())
            check(std::abs(value-length(c.solution))<=2e-12*scale,name+" exact reference length");
        const auto eager=solve(c,true);
        check(tpp::validate_ordered_path(c.start,c.target,c.polygons,eager).valid,name+" eager ordered visits");
        check(std::abs(length(eager)-value)<=2e-12*scale,name+" eager/lazy agreement");
        const double api_length=public_api
            ?tpp::tpp_convex_solve_length_binary_search_lazy(c.start,c.target,c.polygons)
            :tpp::detail::length_intersecting_maps(c.start,c.target,c.polygons);
        check(std::abs(api_length-value)<=2e-12*scale,name+" length API");
        if(oracle) {
            tpp::DynamicConvexTppWorkspace workspace;
            auto oracle_polygons=c.polygons;
            for(auto &p:oracle_polygons) {
                double area=0;
                for(size_t j=0;j<p.size();++j)area+=(p[j]-p[0]).cross(p[(j+1)%p.size()]-p[0]);
                if(area<0)std::reverse(p.begin(),p.end());
            }
            const auto certified=tpp::tpp_convex_solve_certified(c.start,c.target,oracle_polygons,
                workspace,1e-10,std::numeric_limits<double>::infinity(),2.0);
            const bool resolved=tpp::validate_ordered_path(c.start,c.target,c.polygons,certified.path).valid
                &&std::isfinite(certified.lower_bound)
                &&certified.upper_bound-certified.lower_bound<2e-8*scale;
            if(resolved)
                check(value>=certified.lower_bound-2e-8*scale && value<=certified.upper_bound+2e-8*scale,
                      name+" certified optimality");
            else {++unresolved;std::cout<<"UNRESOLVED "<<name<<std::endl;}
        }
        if(old_failures!=failures)describe(c);
        return path;
    } catch(const std::exception &e) {
        check(false,name+" exception: "+e.what());describe(c);return {};
    }
}

void verify_hybrid(const TestCase &c,const std::string &name,bool shadow=true) {
    try {
        tpp::ConvexHybridOptions options;options.shadow_rational=shadow;
        options.retain_rejected_double_candidate=true;
        const auto hybrid=tpp::tpp_convex_solve_hybrid(c.start,c.target,c.polygons,options);
        if(hybrid.stats.disjoint && hybrid.stats.rational_fallback &&
           std::getenv("TPP_DEBUG_DISJOINT_FALLBACK")) {
            std::cout<<"DISJOINT_FALLBACK "<<name<<" reason="
                     <<tpp::to_string(hybrid.fallback_reason)<<'\n';describe(c);
        }
        hybrid_fast+=hybrid.stats.double_certified&&!hybrid.stats.rational_fallback;
        hybrid_fallback+=hybrid.stats.rational_fallback;
        hybrid_shadow_mismatch+=hybrid.fallback_reason==tpp::ConvexFallbackReason::ShadowMismatch;
        if(hybrid.fallback_reason==tpp::ConvexFallbackReason::ContactConstruction&&
           std::getenv("TPP_DEBUG_CONTACT_FALLBACK")) {
            std::cout<<"CONTACT_FALLBACK "<<name<<'\n';describe(c);
        }
        check(hybrid.contacts.size()==c.polygons.size(),name+" hybrid exact-k contacts");
        const auto displayed=tpp::reconstruct_convex_polyline(c.start,c.target,hybrid.contacts);
        check(tpp::validate_ordered_path(c.start,c.target,c.polygons,displayed).valid,
              name+" hybrid reconstructed visits");
        check(hybrid.lower_bound<=hybrid.upper_bound,name+" hybrid bounds ordered");
#ifdef TPP_HAS_INTERVAL_PRIMAL_DUAL
        tpp::ConvexHybridOptions bounded_options;
        bounded_options.max_gap=1e-7*std::max(1.0,hybrid.upper_bound);
        const auto bounded=tpp::tpp_convex_solve_hybrid(c.start,c.target,c.polygons,bounded_options);
        static tpp::DynamicConvexTppWorkspace bounded_workspace;
        for(bool cached_binary:{false,true}) {
            bounded_workspace.cache_interval_geometry=cached_binary;
            const auto reused=tpp::tpp_convex_solve_hybrid(c.start,c.target,c.polygons,bounded_options,bounded_workspace);
            check(reused.contacts==bounded.contacts && reused.lower_bound==bounded.lower_bound &&
                  reused.upper_bound==bounded.upper_bound && reused.backend==bounded.backend &&
                  reused.stats.interval_bounds_certified==bounded.stats.interval_bounds_certified &&
                  reused.fallback_reason==bounded.fallback_reason,
                  name+" cached normalized geometry preserves bounded oracle result");
        }
        check(bounded.lower_bound<=hybrid.upper_bound&&bounded.upper_bound>=hybrid.lower_bound,
              name+" interval bounds enclose the independent exact reference");
        check(bounded.upper_bound-bounded.lower_bound<=bounded_options.max_gap,
              name+" bounded oracle respects requested gap");
        check(bounded.contacts.size()==c.polygons.size(),name+" bounded contacts retain visit order");
        bounded_options.interpolated_zero_dual=true;
        const auto interpolated=tpp::tpp_convex_solve_hybrid(c.start,c.target,c.polygons,bounded_options);
        check(interpolated.lower_bound<=hybrid.upper_bound&&interpolated.upper_bound>=hybrid.lower_bound,
              name+" interpolated zero dual encloses exact reference");
        check(interpolated.upper_bound-interpolated.lower_bound<=bounded_options.max_gap,
              name+" interpolated zero dual respects requested gap");
        check(tpp::validate_ordered_path(c.start,c.target,c.polygons,
              tpp::reconstruct_convex_polyline(c.start,c.target,interpolated.contacts,false)).valid,
              name+" interpolated proposal retains original visits");
        if(bounded.stats.interval_bounds_certified)for(size_t i=0;i<c.polygons.size();++i) {
            tpp::ConvexRationalPoint q(bounded.contacts[i]);
            tpp::ConvexRational area=0;
            const auto &p=c.polygons[i];
            for(size_t j=0;j<p.size();++j)area+=tpp::ConvexRationalPoint(p[j]).cross(tpp::ConvexRationalPoint(p[(j+1)%p.size()]));
            bool belongs=true;
            for(size_t j=0;j<p.size();++j) {
                const tpp::ConvexRationalPoint a(p[j]),b(p[(j+1)%p.size()]);
                const auto side=(b-a).cross(q-a);
                belongs&=area>0?side>=0:side<=0;
            }
            check(belongs,name+" exported interval contact is exactly feasible");
        }
#endif
        if(!hybrid.rejected_double_contacts.empty()) {
            check(hybrid.rejected_double_lower_bound<=hybrid.upper_bound,
                  name+" rejected-candidate dual bound safe");
            check(hybrid.lower_bound<=hybrid.rejected_double_upper_bound,
                  name+" rejected-candidate upper bound safe");
            check(hybrid.rejected_double_lower_bound<=hybrid.rejected_double_upper_bound,
                  name+" rejected-candidate bounds ordered");
        }
    } catch(const std::exception &e) {check(false,name+" hybrid exception: "+e.what());}
}

void deterministic() {
    const Polygon A=box(-2,-1,-1,1), B=box(1,-1,2,1);
    std::vector<std::pair<std::string,TestCase>> cases={
        {"empty sequence",{{-3,0},{3,0},{},{{-3,0},{3,0}}}},
        {"repeated separated contacts",{{-3,0},{3,0},{A,B,A},{{-3,0},{1,0},{-1,0},{3,0}}}},
        {"identical polygons",{{-3,-2},{3,-2},{box(-1,0,1,2),box(-1,0,1,2)},{{-3,-2},{0,0},{3,-2}}}},
        {"nested interior reflection",{{-3,-2},{3,-2},{box(-2,-1,2,3),box(-1,0,1,2),box(-2,-1,2,3)},{{-3,-2},{0,0},{3,-2}}}},
        {"three polygons common point",{{-3,-2},{3,-2},{box(-2,0,0,2),box(0,0,2,2),box(-1,0,1,3)},{{-3,-2},{0,0},{3,-2}}}},
        {"vertex tangency",{{-3,-2},{3,-2},{box(-2,0,0,2),box(0,-2,2,0)},{{-3,-2},{0,0},{3,-2}}}},
        {"vertex-only contact",{{-3,0},{3,0},{{{-1,1},{0,0},{1,1}},box(-1,-1,1,1)},{{-3,0},{3,0}}}},
        {"start on shared vertex",{{0,0},{3,-2},{box(-2,0,0,2),box(0,0,2,2)},{{0,0},{3,-2}}}},
        {"start on shared edge",{{0,0},{3,-2},{box(-2,0,2,2),box(-1,0,1,3)},{{0,0},{3,-2}}}},
        {"start in all interiors",{{0,1},{3,-2},{box(-2,0,2,2),box(-1,0,1,3)},{{0,1},{3,-2}}}},
        {"target on shared vertex",{{-3,-2},{0,0},{box(-2,0,0,2),box(0,0,2,2)},{{-3,-2},{0,0}}}},
        {"target inside all",{{-3,-2},{0,1},{box(-2,0,2,2),box(-1,0,1,3)},{{-3,-2},{0,1}}}},
        {"stationary inside",{{0,1},{0,1},{box(-2,0,2,2),box(-1,0,1,3)},{{0,1}}}},
        {"stationary boundary",{{0,0},{0,0},{box(-2,0,0,2),box(0,0,2,2)},{{0,0}}}},
        {"return to start",{{-3,0},{-3,0},{A,B,A},{{-3,0},{1,0},{-3,0}}}},
        {"nonconsecutive intersections",{{-4,-4},{4,-4},{box(0,2,2,4),box(0,-2,3,2),box(-1,3,1,4)},{{-4,-4},{0,2},{.5,3},{4,-4}}}},
        {"backwards ray extension",{{0,-2},{1,3},{box(2,3,5,4),box(-3,-2,0,2),box(2,2,3,5)},{{0,-2},{2,3},{0,2},{2,8./3},{1,3}}}},
        {"thin positive area",{{-3,-2},{3,-2},{box(-2,0,0,1e-13),box(-1,0,2,2e-13)},{{-3,-2},{0,0},{3,-2}}}},
        {"floating feasible suboptimal",{{-3,-2},{3,-2},
            {box(-2,0,0,2),{{-1e-6,1},{1.999999,0},{1.999999,2}}},
            {{-3,-2},{0,0},{1.6799996719999488,0.15999966400002566},{3,-2}}}}
    };
    tpp::DynamicConvexTppWorkspace hybrid_workspace;
    for(auto &[name,c]:cases) {
        verify(c,name,name=="floating feasible suboptimal");
        verify_hybrid(c,name);
        try {
            const auto hybrid=tpp::tpp_convex_solve_hybrid(c.start,c.target,c.polygons);
            if(name=="nested interior reflection" || name=="floating feasible suboptimal" ||
               name=="three polygons common point") {
                for(int repeat=0;repeat<2;++repeat) {
                    const auto cached=tpp::tpp_convex_solve_hybrid(c.start,c.target,c.polygons,
                        {},hybrid_workspace);
                    check(cached.contacts==hybrid.contacts && cached.lower_bound==hybrid.lower_bound &&
                          cached.upper_bound==hybrid.upper_bound && cached.backend==hybrid.backend &&
                          cached.fallback_reason==hybrid.fallback_reason,
                          name+" reusable hybrid workspace");
                }
            }
            if(name=="identical polygons")
                check(hybrid.contacts.size()==2 && hybrid.contacts[0]==hybrid.contacts[1],
                      name+" preserves duplicate contacts");
            if(name=="three polygons common point")
                check(hybrid.contacts.size()==3 && hybrid.contacts[0]==hybrid.contacts[1]
                      && hybrid.contacts[1]==hybrid.contacts[2],
                      name+" preserves all common-point contacts");
            if(name=="floating feasible suboptimal") {
                const double reference=length(c.solution);
                check(hybrid.lower_bound<=reference+2e-12*reference &&
                      hybrid.upper_bound>=reference-2e-12*reference,
                      name+" rejects the suboptimal objective, including filtered construction");
#ifdef TPP_HAS_FILTERED_DIRECTIONAL
                check(hybrid.stats.filtered_attempted&&hybrid.stats.filtered_certified&&
                      !hybrid.stats.rational_fallback,
                      name+" recovers its combinatorial trace with filtered predicates");
#endif
            }
        } catch(const std::exception &e) {check(false,name+" hybrid exception: "+e.what());}
        auto reversed=c;
        for(auto &p:reversed.polygons)std::reverse(p.begin(),p.end());
        verify(reversed,name+" CW");
        // Midpoints of non-dyadic binary doubles may round off the original
        // supporting line, so this transform changes the precision fixture.
        if(name=="floating feasible suboptimal") continue;
        auto collinear=c;
        for(auto &p:collinear.polygons) {
            Polygon expanded;
            for(size_t j=0;j<p.size();++j){expanded.push_back(p[j]);expanded.push_back((p[j]+p[(j+1)%p.size()])/2);}
            p=std::move(expanded);
        }
        verify(collinear,name+" collinear");
    }
    if(public_api) {
        tpp::DynamicConvexTppWorkspace workspace;
        workspace.reserve(10,80);
        tpp::StaticConvexTppWorkspace<10,80> fixed;
        Polygon output;
        // Reuse the same buffers as input sizes grow and shrink, across both
        // dispatch paths and both preload policies.
        for(size_t repeat=0;repeat<2;++repeat)for(const auto &[name,c]:cases) {
            const auto expected=solve(c);
            auto equivalent=[&] {
                return tpp::validate_ordered_path(c.start,c.target,c.polygons,output).valid
                    &&std::abs(length(output)-length(expected))<=2e-12*std::max(1.,length(expected));
            };
            tpp::tpp_convex_solve_binary_search_lazy(c.start,c.target,c.polygons,workspace,output);
            check(equivalent(),name+" dynamic lazy workspace");
            tpp::tpp_convex_solve_binary_search_eager(c.start,c.target,c.polygons,workspace,output);
            check(equivalent(),name+" dynamic eager workspace");
            tpp::tpp_convex_solve_binary_search_lazy(c.start,c.target,c.polygons,fixed.view(),output);
            check(equivalent(),name+" fixed lazy workspace");
            tpp::tpp_convex_solve_binary_search_eager(c.start,c.target,c.polygons,fixed.view(),output);
            check(equivalent(),name+" fixed eager workspace");
        }
    }
}

void normalized_sign_predicates() {
    using R=tpp::ConvexRational;
    using I=tpp::ConvexInteger;
    auto reference=[](const R &p,const R &a2,const R &q,const R &b2) {
        if(p>=0&&q<=0)return p==0&&q==0?0:1;
        if(p<=0&&q>=0)return p==0&&q==0?0:-1;
        const R left=p*p*b2,right=q*q*a2;
        if(left==right)return 0;
        return p>0?(left>right?1:-1):(left<right?1:-1);
    };
    auto compare=[&](const R &p,const R &a2,const R &q,const R &b2) {
        check(tpp::detail::convex_normalized_difference_sign(p,a2,q,b2)==reference(p,a2,q,b2),
              "integer normalized sign agrees with rational products");
    };
    std::mt19937 random(20261003);
    for(size_t trial=0;trial<1200;++trial) {
        auto fraction=[&] {
            I numerator=1+random()%1009,denominator=1+random()%1013;
            numerator<<=random()%601;denominator<<=random()%601;
            return R(numerator)/R(denominator);
        };
        const R p=fraction(),q=fraction(),a2=fraction(),b2=fraction();
        for(int p_sign:{-1,0,1})for(int q_sign:{-1,0,1})
            compare(R(p_sign)*p,a2,R(q_sign)*q,b2);
        const R scale=fraction(),small=R(1)/R(I(1)<<700);
        // Exact equality and arbitrarily close values on each side must
        // remain distinct; no epsilon or rounded square root decides them.
        for(int sign:{-1,1})for(int side:{-1,0,1})
            compare(R(sign)*p,a2,R(sign)*p*scale*(R(1)+R(side)*small),a2*scale*scale);
    }
}

void filtered_predicates() {
    using F=tpp::detail::FilteredRational;
    using R=tpp::ConvexRational;
    const int original_rounding=std::fegetround();
    std::mt19937 random(20261002);
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD}) {
        if(std::fesetround(rounding)!=0)continue;
        F::Scope scope;
        check((F(9007199254740992.)+F(1))-F(9007199254740992.)==F(1),
              "filtered cancellation retains a lost binary64 bit");
        const double tiny=std::numeric_limits<double>::denorm_min();
        check(F(tiny)*F(tiny)>F(0),"filtered underflow retains an exact positive sign");
        const double huge=std::numeric_limits<double>::max();
        check((F(huge)*F(huge))/(F(huge)*F(huge))==F(1),
              "filtered overflow resolves through the rational DAG");
        for(size_t i=0;i<1000;++i) {
            auto coordinate=[&] {
                const int exponent=int(random()%1101)-550;
                return std::ldexp(double(int(random()%15)+1),exponent)*(random()%2?1:-1);
            };
            const double a=coordinate(),b=coordinate(),c=coordinate(),d=coordinate();
            const F A(a),B(b),C(c),D(d);
            const F left=(A*B-C*D)/(A*A+B*B),right=(A*D+C*B)/(C*C+D*D);
            const R x=(R(a)*R(b)-R(c)*R(d))/(R(a)*R(a)+R(b)*R(b));
            const R y=(R(a)*R(d)+R(c)*R(b))/(R(c)*R(c)+R(d)*R(d));
            check((left<right)==(x<y)&&(left>right)==(x>y)&&
                  (left==right)==(x==y)&&(left<=right)==(x<=y)&&(left>=right)==(x>=y),
                  "filtered predicate agrees with independent rational arithmetic");
        }
    }
    std::fesetround(original_rounding);
}

// The only intermediate subgradient at these two edge-interior contacts is
// (0,0), strictly inside the disk. Enumerating unit directions cannot certify
// this optimum. An independent lower bound is |dx| >= 1 before the first
// contact and |dy| >= 1 after the second contact.
void coincident_disk_contacts() {
    for(double scale:{1e-9,1.0,1e9})for(bool clockwise:{false,true}) {
        TestCase c{{-scale,0},{0,-scale},
            {box(0,-2*scale,2*scale,2*scale),box(-2*scale,0,2*scale,2*scale)}, {}};
        for(bool repeated:{false,true}) {
            auto regions=c.polygons;
            if(repeated) {
                regions.insert(regions.begin()+1,regions.front());
                regions.insert(regions.begin()+2,box(-scale,-scale,scale,scale));
            }
            if(clockwise)for(auto &p:regions)std::reverse(p.begin(),p.end());
            tpp::ConvexHybridOptions options;options.shadow_rational=true;
            const auto result=tpp::tpp_convex_solve_hybrid(c.start,c.target,regions,options);
            check(result.stats.double_certified&&!result.stats.rational_fallback,
                  "interior-disk zero-link witness avoids fallback");
            check(result.stats.zero_link_witnesses==1,
                  "one maximal coincident-contact block");
            check(result.contacts.size()==regions.size()&&
                  std::ranges::all_of(result.contacts,[](Vector2 q){return q==Vector2{};}),
                  "disk witness retains every ordered contact");
            check(result.lower_bound<=2*scale&&result.upper_bound>=2*scale&&
                  result.upper_bound-result.lower_bound<=1e-12*scale,
                  "disk certificate encloses independent exact optimum");
            const auto reverse=tpp::tpp_convex_solve_hybrid(c.target,c.start,
                Polygons(regions.rbegin(),regions.rend()),options);
            check(reverse.stats.double_certified&&!reverse.stats.rational_fallback&&
                  reverse.lower_bound<=2*scale&&reverse.upper_bound>=2*scale,
                  "reversed endpoint problem certifies the same zero block");
        }
    }
}

// Reduced from the case-451 diagnostic capture (three regions, twelve
// vertices). Regions 0 and 2 share an edge; the native intersection proposal
// is feasible but suboptimal. Perturbation supplies only the trace: the
// certified contacts and reference bounds use the original coordinates.
void touching_disjoint_recovery() {
    const TestCase adjacent{{2,-1},{2,1},{box(-1,-2,0,2),box(0,-2,1,2)},{{2,-1},{0,0},{2,1}}};
    verify(adjacent,"adjacent-edge reflection");
    verify_hybrid(adjacent,"adjacent-edge reflection");
    const TestCase fixture{{-.5,-.48688664345128624},{.5,.4868866434512862},{
        {{-.08158994695989587,.13196831561051192},{.07339897409960683,.14598698349019848},
         {-.06564341289376201,.18935638398990542}},
        {{-.3325970104407017,.04044627569758704},{-.40928720204092256,.07285934029040048},
         {-.5,.0673657112014923}},
        {{-.1743635331463786,.24816481518664948},{-.13404537698100738,.06623788795487584},
         {-.08158994695989587,.13196831561051192},{-.06564341289376201,.18935638398990542},
         {-.07053212110242861,.23717399195248245},{-.1436679489140717,.27098419610361024}}}, {}};
    for(double scale:{1e-9,1.0,1e9})for(bool clockwise:{false,true}) {
        auto c=fixture;c.start=c.start*scale;c.target=c.target*scale;
        for(auto &p:c.polygons) {
            for(auto &v:p)v=v*scale;
            if(clockwise)std::reverse(p.begin(),p.end());
        }
        verify_hybrid(c,"shared-edge trace recovery");
        tpp::ConvexHybridOptions options;options.shadow_rational=true;
        const auto r=tpp::tpp_convex_solve_hybrid(c.start,c.target,c.polygons,options);
        const auto reference=tpp::detail::solve_intersecting_map_contacts_with_bounds(c.start,c.target,c.polygons,false);
        check(r.lower_bound<=reference.upper_bound&&r.upper_bound>=reference.lower_bound,
              "shared-edge recovery agrees with original rational reference");
#ifdef TPP_HAS_TOUCHING_DISJOINT
        if(scale==1)check(r.stats.touching_disjoint_attempted&&r.stats.touching_disjoint_certified&&
            r.stats.touching_disjoint_perturbed&&!r.stats.rational_fallback&&
            r.fallback_reason==tpp::ConvexFallbackReason::None,
            "shared-edge contraction avoids directional recovery after exact replay");
#endif
        const auto reversed=tpp::tpp_convex_solve_hybrid(c.target,c.start,
            Polygons(c.polygons.rbegin(),c.polygons.rend()),options);
        check(reversed.lower_bound<=r.upper_bound&&reversed.upper_bound>=r.lower_bound&&
              reversed.contacts.size()==c.polygons.size(),"reversed shared-edge sequence preserves certified optimum");
    }
}

void interval_bound_regressions() {
#ifdef TPP_HAS_INTERVAL_PRIMAL_DUAL
    const TestCase reflection{{2,-1},{2,1},{box(-1,-2,0,2),box(0,-2,1,2)}, {}};
    const auto exact=tpp::tpp_convex_solve_hybrid(reflection.start,reflection.target,reflection.polygons);
    check(!exact.stats.interval_bounds_attempted,"default hybrid retains exact-optimality contract");
    tpp::ConvexHybridOptions options;options.max_gap=1e-8;
    const auto bounded=tpp::tpp_convex_solve_hybrid(reflection.start,reflection.target,reflection.polygons,options);
    check(bounded.stats.interval_bounds_certified&&bounded.lower_bound<=exact.upper_bound&&
          bounded.upper_bound>=exact.lower_bound,"interval proof encloses analytic shared-edge reflection");
    options.max_gap=0;options.cutoff=4;
    const auto pruned=tpp::tpp_convex_solve_hybrid(reflection.start,reflection.target,{reflection.polygons.front()},options);
    check(pruned.cutoff_pruned&&pruned.lower_bound>=4&&pruned.lower_bound<=exact.upper_bound,
          "interval cutoff with zero allowed gap uses a certified dual");
    options.cutoff=INFINITY;options.max_gap=1e-18;
    const auto tight=tpp::tpp_convex_solve_hybrid(reflection.start,reflection.target,reflection.polygons,options);
    check(!tight.stats.interval_bounds_certified&&tight.stats.double_certified&&
          tight.lower_bound==exact.lower_bound&&tight.upper_bound==exact.upper_bound,
          "unrepresentable gap retains original exact certificate");
    const auto original_rounding=std::fegetround();
    options.max_gap=1e-7;
    if(std::fesetround(FE_UPWARD)==0) {
        const auto unsupported=tpp::tpp_convex_solve_hybrid({-3,0},{3,0},{box(-1,-1,1,1)},options);
        check(!unsupported.stats.interval_bounds_attempted&&unsupported.lower_bound<=6&&unsupported.upper_bound>=6,
              "unsupported rounding environment retains rational certificate");
    }
    std::fesetround(original_rounding);
#endif
}

void adversarial_disjoint() {
    std::vector<std::pair<std::string,TestCase>> cases={
        {"disjoint exit contact",{{-3,0},{3,0},{box(-1,-1,1,1)}, {}}},
        {"disjoint tiny scale",{{-3e-9,0},{3e-9,0},
            {box(-2e-9,-1e-12,-1e-9,1e-12),box(1e-9,-1e-12,2e-9,1e-12)}, {}}},
        {"disjoint huge scale",{{-3e12,0},{3e12,0},
            {box(-2e12,-1,-1e12,1),box(1e12,-1,2e12,1)}, {}}},
        {"disjoint near tangency",{{-4,-2},{4,-2},
            {box(-2,0,0,2),box(std::nextafter(0.,1.),2,2,4)}, {}}},
        {"disjoint near collinear",{{-4,-1e-13},{4,-1e-13},
            {box(-3,0,-2,1e-14),box(-1,2e-14,0,3e-14),box(1,4e-14,2,5e-14)}, {}}}
    };
    for(const auto &[name,c]:cases) {
        tpp::ConvexHybridOptions options;options.shadow_rational=true;
        const auto hybrid=tpp::tpp_convex_solve_hybrid(c.start,c.target,c.polygons,options);
        if(hybrid.stats.rational_fallback && std::getenv("TPP_DEBUG_DISJOINT_FALLBACK")) {
            std::cout<<"DISJOINT_FALLBACK "<<name<<" reason="
                     <<tpp::to_string(hybrid.fallback_reason)<<'\n';describe(c);
        }
        check(hybrid.stats.disjoint,name+" classified disjoint");
        check(hybrid.fallback_reason!=tpp::ConvexFallbackReason::ShadowMismatch,
              name+" rational shadow agreement");
        check(hybrid.contacts.size()==c.polygons.size(),name+" exact-k contacts");
        check(tpp::validate_ordered_path(c.start,c.target,c.polygons,
              tpp::reconstruct_convex_polyline(c.start,c.target,hybrid.contacts)).valid,
              name+" ordered visits");
        if(name=="disjoint exit contact")
            check(hybrid.contacts[0]==Vector2{1,0},name+" uses last/exit point");
    }
}

double point_segment(Vector2 p,Vector2 a,Vector2 b) {
    const auto d=b-a;
    if(d.length_squared()==0)return p.distance_to(a);
    return p.distance_to(a+d*std::clamp((p-a).dot(d)/d.length_squared(),0.,1.));
}
// A rigorous upper bound on directed segment Hausdorff distance: distance to
// a fixed target segment is convex, so its maximum over a source segment occurs
// at an endpoint. Minimizing that bound over target segments remains an upper
// bound on distance to their union. No sampling or matching vertex indices.
double directed_distance_bound(const Polygon &a,const Polygon &b) {
    if(a.empty()||b.empty())return std::numeric_limits<double>::infinity();
    double maximum=0;
    for(size_t i=0;i<std::max(size_t(1),a.size()-1);++i) {
        double minimum=std::numeric_limits<double>::infinity();
        for(size_t j=0;j<std::max(size_t(1),b.size()-1);++j)
            minimum=std::min(minimum,std::max(point_segment(a[i],b[j],b[std::min(j+1,b.size()-1)]),
                point_segment(a[std::min(i+1,a.size()-1)],b[j],b[std::min(j+1,b.size()-1)])));
        maximum=std::max(maximum,minimum);
    }
    return maximum;
}

void continuity() {
    const std::vector<double> deltas={1e-3,1e-6,1e-9,0.,-1e-9,-1e-6,-1e-3};
    const std::vector<std::string> names={"vertex-edge","edge-edge","near-collinear",
        "containment tangency","nonconsecutive tangency","common point"};
    for(size_t family=0;family<names.size();++family) {
        std::vector<Polygon> paths;
        for(double delta:deltas) {
            TestCase c{{-3,-2},{3,-2},{},{}};
            switch(family) {
                case 0:c.polygons={box(-2,0,0,2),{{delta,1},{delta+2,0},{delta+2,2}}};break;
                case 1:c.polygons={box(-2,0,0,2),box(delta,0,2+delta,2)};break;
                case 2:c.polygons={box(-2,0,0,2),{{delta,0},{2+delta,-delta/100},{2+delta,2},{delta,2}}};break;
                case 3:c.polygons={box(-2,-1,2,3),box(-1,-1+delta,1,2+delta)};break;
                case 4:c.polygons={box(-2,0,0,2),box(-3,0,3,4),box(delta,0,2+delta,2)};break;
                default:c.polygons={box(-2,0,0,2),box(delta,0,2+delta,2),box(-1,0,1,3)};
            }
            paths.push_back(verify(c,names[family]+" delta="+std::to_string(delta),true));
        }
        for(size_t i=0;i<paths.size();++i) {
            const double d=std::abs(deltas[i]),L=length(paths[i]),L0=length(paths[3]);
            // Moving a contact by <= d changes its two incident segments by
            // at most 2d; all these families move at most one polygon by d.
            check(std::abs(L-L0)<=2*d+2e-10,names[family]+" optimum continuity");
            const double H=std::max(directed_distance_bound(paths[i],paths[3]),
                                    directed_distance_bound(paths[3],paths[i]));
            check(H<=20*std::sqrt(d)+2e-9,names[family]+" geometric convergence");
        }
    }
}

void random_boxes(size_t count) {
    std::mt19937 rng(20260914);
    for(size_t trial=0;trial<count;++trial) {
        auto coordinate=[&](){return int(rng()%7)-3;};
        TestCase c{{double(coordinate()),double(coordinate())},
                   {double(coordinate()),double(coordinate())},{},{}};
        const size_t k=2+rng()%6;
        for(size_t i=0;i<k;++i) {
            const int x=coordinate(),y=coordinate();
            auto p=box(x,y,x+1+int(rng()%4),y+1+int(rng()%4));
            if(rng()%2)std::reverse(p.begin(),p.end());
            c.polygons.push_back(p);
        }
        if(trial%5==0)c.polygons.push_back(c.polygons.front());
        verify(c,"integer boxes "+std::to_string(trial),true);
        verify_hybrid(c,"integer boxes "+std::to_string(trial));
    }
}

void random_convex(size_t count) {
    std::mt19937 rng(20260915);
    std::uniform_real_distribution<double> unit(0,1);
    for(size_t trial=0;trial<count;++trial) {
        auto coordinate=[&](){return 10*unit(rng)-5;};
        TestCase c{{coordinate(),coordinate()},{coordinate(),coordinate()},{},{}};
        const size_t k=2+rng()%11;
        for(size_t i=0;i<k;++i) {
            const Vector2 center{coordinate(),coordinate()};
            const double angle=6.283185307179586*unit(rng), radius=.2+4*unit(rng);
            const double aspect=trial%3==0 ? 1./32 : .5+2*unit(rng);
            const size_t n=3+rng()%18;
            Polygon p;
            for(size_t j=0;j<n;++j) {
                const double theta=6.283185307179586*j/n;
                const double x=radius*std::cos(theta),y=aspect*radius*std::sin(theta);
                p.push_back(center+Vector2{x*std::cos(angle)-y*std::sin(angle),
                                           x*std::sin(angle)+y*std::cos(angle)});
            }
            if(rng()%2)std::reverse(p.begin(),p.end());
            c.polygons.push_back(std::move(p));
        }
        verify(c,"affine convex "+std::to_string(trial),true);
        verify_hybrid(c,"affine convex "+std::to_string(trial));
    }
}

void corpus(const std::string &directory) {
    for(const auto &entry:std::filesystem::directory_iterator(directory)) {
        if(entry.path().extension()!=".bin")continue;
        std::ifstream file(entry.path(),std::ios::binary);
        size_t n=0;
        while(file.peek()!=EOF) {
            auto c=tpp::decode_test(file);
            verify(c,entry.path().filename().string()+" "+std::to_string(++n));
        }
    }
}
}

int main(int argc,char **argv) {
    std::cout<<std::setprecision(17);
    size_t random_count=0,convex_count=0;
    std::vector<std::string> corpora;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--public")public_api=true;
        else if(arg=="--random-boxes"&&i+1<argc)random_count=std::stoul(argv[++i]);
        else if(arg=="--random-convex"&&i+1<argc)convex_count=std::stoul(argv[++i]);
        else if(arg=="--corpus"&&i+1<argc)corpora.push_back(argv[++i]);
        else throw std::invalid_argument("Unknown argument: "+arg);
    }
    interval_rounding_regressions();dyadic_orientation_regressions();binary_membership_memo_regressions();normalized_sign_predicates();filtered_predicates();dispatch_cache_regressions();cached_contact_rotation_regressions();deterministic();coincident_disk_contacts();touching_disjoint_recovery();interval_bound_regressions();adversarial_disjoint();continuity();random_boxes(random_count);random_convex(convex_count);
    for(const auto &directory:corpora)corpus(directory);
    const auto aggregate=tpp::convex_hybrid_aggregate();
    std::cout<<"Checks="<<checks<<", failures="<<failures<<", unresolved="<<unresolved
             <<", hybrid_total_calls="<<aggregate.total_calls
             <<", hybrid_disjoint_calls="<<aggregate.disjoint_calls
             <<", certified_double_disjoint_calls="<<aggregate.certified_double_disjoint_calls
             <<", interval_bound_calls="<<aggregate.interval_bound_calls
             <<", interval_contracted_calls="<<aggregate.interval_contracted_calls
             <<", hybrid_fast="<<hybrid_fast<<", hybrid_fallback="<<hybrid_fallback
             <<", hybrid_shadow_mismatch="<<hybrid_shadow_mismatch
             <<", rational_disjoint_directional_recoveries="
             <<aggregate.rational_disjoint_directional_recoveries
             <<", rational_disjoint_fallbacks="<<aggregate.rational_disjoint_fallbacks
             <<", rational_intersection_fallbacks="<<aggregate.rational_intersection_fallbacks
             <<", reasons(locator/contact/membership/local/zero)="
             <<aggregate.fallback_reasons[size_t(tpp::ConvexFallbackReason::LocatorOrRefoldingException)]<<'/'
             <<aggregate.fallback_reasons[size_t(tpp::ConvexFallbackReason::ContactConstruction)]<<'/'
             <<aggregate.fallback_reasons[size_t(tpp::ConvexFallbackReason::MembershipOrOrdering)]<<'/'
             <<aggregate.fallback_reasons[size_t(tpp::ConvexFallbackReason::LocalOptimality)]<<'/'
             <<aggregate.fallback_reasons[size_t(tpp::ConvexFallbackReason::CoincidentContact)]<<std::endl;
    return failures?1:0;
}
