#include "tests.h"
#include "tpp_convex.h"
#include "tpp/convex/detail/intersecting_maps.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
template<class F> double benchmark(size_t repeats,F &&f,double &checksum) {
    for(size_t i=0;i<5;++i)checksum+=f();
    const auto began=std::chrono::steady_clock::now();
    for(size_t i=0;i<repeats;++i)checksum+=f();
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-began).count()/repeats;
}
double length(const std::vector<Vector2> &p) {
    double result=0;for(size_t i=1;i<p.size();++i)result+=p[i-1].distance_to(p[i]);return result;
}
}

int main(int argc,char **argv) {
    const size_t repeats=argc>1?std::stoull(argv[1]):100;
    tpp::set_rng_seed(1729);
    auto [start,target,polygons]=tpp::generate_test(std::vector<size_t>(20,8));
    double checksum=0;
    auto show=[&](const char *name,auto &&f){
        std::cout<<name<<' '<<benchmark(repeats,f,checksum)<<" us/call\n";
    };
    std::cout<<std::setprecision(17)<<"disjoint k=20 m=8 repeats="<<repeats<<'\n';
    show("established_disjoint_length",[&]{return tpp::tpp_convex_solve_length_binary_search_disjoint(start,target,polygons);});
    show("unchecked_hybrid",[&]{return tpp::tpp_convex_solve_length_hybrid_unchecked(start,target,polygons);});
    show("safe_hybrid",[&]{return tpp::tpp_convex_solve_length_hybrid_safe(start,target,polygons);});
    show("rational_directional_length",[&]{return tpp::detail::length_intersecting_maps(start,target,polygons);});
    show("rational_contacts",[&]{return length(tpp::reconstruct_convex_polyline(start,target,
        tpp::detail::solve_intersecting_map_contacts(start,target,polygons,true)));});
    const auto h=tpp::convex_hybrid_aggregate();
    std::cout<<"hybrid_calls="<<h.total_calls<<" fast_disjoint="<<h.certified_double_disjoint_calls
             <<" rational_disjoint="<<h.rational_disjoint_fallbacks
             <<" dispatch_us="<<1e6*h.dispatch_seconds/h.total_calls
             <<" double_us="<<1e6*h.double_solver_seconds/h.total_calls
             <<" contacts_us="<<1e6*h.contact_materialization_seconds/h.total_calls
             <<" certificate_us="<<1e6*h.certificate_seconds/h.total_calls
             <<" fallback_us="<<1e6*h.rational_fallback_seconds/h.total_calls<<'\n';
    std::cout<<"checksum="<<checksum<<'\n';
}
