#include "tpp/nonconvex/unordered.h"
#include "tpp/convex/cycle.h"
#include "common.h"
#include "solvers/unordered_geometry.h"
#include "solvers/unordered_bounds.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
using Polygon=std::vector<Vector2>;
using Polygons=std::vector<Polygon>;
void require(bool condition,const std::string &message) {if(!condition)throw std::runtime_error(message);}
Polygon box(double x,double y,double w=1,double h=1) {return {{x,y},{x+w,y},{x+w,y+h},{x,y+h}};}
bool covered(const Polygon &q,const Polygons &p) {
    return q.size()>=2&&q.front()==q.back()&&std::all_of(p.begin(),p.end(),[&](const auto &region) {
        return tpp::unordered_detail::contact(q,region,1e-8).distance<=1e-8;
    });
}
std::pair<double,double> enumerate(const Polygons &p) {
    if(p.size()<2)return {0,0};
    std::vector<std::vector<Polygon>> pieces;
    for(auto region:p) {
        double area=0;for(size_t i=0;i<region.size();++i)area+=region[i].cross(region[(i+1)%region.size()]);
        if(area<0)std::reverse(region.begin(),region.end());
        pieces.push_back(tpp::decompose_polygon(region));
    }
    std::vector<size_t> order(p.size());std::iota(order.begin(),order.end(),0);
    double lower=INFINITY,upper=INFINITY;
    do {
        Polygons selected;
        std::function<void(size_t)> visit=[&](size_t i) {
            if(i==p.size()) {
                const auto r=tpp::tpp_convex_solve_cycle(selected);
                require(r.status==tpp::ConvexCycleStatus::Optimal,"Enumeration cycle oracle: "+r.diagnostic);
                lower=std::min(lower,r.certificate.lower_bound);upper=std::min(upper,r.certificate.upper_bound);return;
            }
            for(const auto &piece:pieces[order[i]]){selected.push_back(piece);visit(i+1);selected.pop_back();}
        };
        visit(0);
    } while(std::next_permutation(order.begin()+1,order.end()));
    return {lower,upper};
}
size_t cases=0,interrupted=0,decomposed=0;
void check(const Polygons &p) {
    const auto [lower,upper]=enumerate(p);
    const auto r=tpp::tpp_nonconvex_tspn_solve(p);
    require(covered(r.path,p),"TSPN output is a feasible closed tour");
    require(r.exact&&r.lower_bound<=upper+1e-7&&r.upper_bound>=lower-1e-7&&
            std::abs(r.upper_bound-upper)<=1e-7+1e-9*upper,"TSPN exhaustive order/piece comparison");
    require(r.calls==r.relaxation_calls+r.refinement_calls+r.initial_convex_refinement_calls,"Oracle accounting");
    decomposed+=r.decomposition_branches>0;
    for(size_t cap:{0,1,3,10}) {
        tpp::UnorderedTppSolveOptions options;options.max_calls=cap;
        const auto limited=tpp::tpp_nonconvex_tspn_solve(p,options);
        require(covered(limited.path,p)&&limited.calls<=cap&&limited.lower_bound<=upper+1e-7&&
                limited.upper_bound>=lower-1e-7,"Interrupted cycle frontier bounds");++interrupted;
    }
    tpp::UnorderedTppSolveOptions supplied;supplied.initial_path=r.path;
    supplied.convex_initial_refinement=true;supplied.bidirectional_initial_heuristic=true;
    const auto again=tpp::tpp_nonconvex_tspn_solve(p,supplied);
    require(covered(again.path,p)&&again.exact&&std::abs(again.upper_bound-upper)<1e-6,"Free initial tour may start anywhere");
    auto reversed=p;std::reverse(reversed.begin(),reversed.end());
    for(auto &region:reversed)std::reverse(region.begin(),region.end());
    supplied.initial_path.reset();
    const auto metamorphic=tpp::tpp_nonconvex_tspn_solve(reversed,supplied);
    require(covered(metamorphic.path,reversed)&&metamorphic.exact&&std::abs(metamorphic.upper_bound-upper)<1e-6,
            "Region/winding reversal and cyclic initial refinement");
    ++cases;
}
void insertion_bounds() {
    const Polygons p{box(-4,0),box(0,4),box(4,0)};const auto inserted=box(0,-4);
    std::mt19937 rng(290927);std::uniform_real_distribution<double> pos(-10,10);
    for(size_t n=1;n<=p.size();++n) {
        Polygons regions(p.begin(),p.begin()+n);std::vector<const Polygon*> refs;
        for(const auto &r:regions)refs.push_back(&r);
        std::vector<double> optima;
        for(size_t i=0;i<n;++i) {
            auto child=regions;child.insert(child.begin()+i+1,inserted);
            const auto solved=tpp::tpp_convex_solve_cycle(child);
            require(solved.status==tpp::ConvexCycleStatus::Optimal,"Bound reference optimum");
            optima.push_back(solved.certificate.upper_bound);
        }
        for(size_t trial=0;trial<80;++trial) {
            Polygon hints;for(size_t i=0;i<n;++i)hints.push_back({pos(rng),pos(rng)});
            if(trial%3==0)std::fill(hints.begin(),hints.end(),hints.front());
            hints.push_back(hints.front());
            const auto bounds=tpp::unordered_detail::insertion_lower_bounds(hints,refs,inserted,true);
            for(size_t i=0;i<n;++i)require(bounds[i]<=optima[i],"Rational cyclic dual bound, including closing edge");
        }
    }
}
}
int main() {
    try {
        insertion_bounds();
        check({});check({box(0,0)});check({box(0,0,10,10),box(12,4,1,2)});
        require(std::abs(tpp::tpp_nonconvex_tspn_solve({box(0,0,10,10),box(12,4,1,2)}).upper_bound-4)<1e-8,
                "No artificial fixed endpoint in TSPN");
        check({box(0,0,2,2),box(1,1,2,2),box(.5,.5,1,1)});
        check({box(0,0),box(5,0),box(5,5),box(0,5)});
        check({{{0,0},{6,0},{6,6},{4,6},{4,2},{2,2},{2,6},{0,6}},box(2.5,3,1,1)});
        check({{{0,0},{3,0},{3,1},{1,1},{1,3},{0,3}},box(5,0),box(4,5),box(-2,4)});
        std::mt19937 rng(260927);std::uniform_int_distribution<int> coord(-6,6);
        for(size_t trial=0;trial<12;++trial) {
            Polygons p;for(size_t i=0;i<2+trial%4;++i)p.push_back(box(coord(rng),coord(rng),2,2));
            check(p);
        }
        require(decomposed>0,"Nonconvex decomposition branching exercised");
        tpp::UnorderedTppSolveOptions bad;bad.initial_path=Polygon{{0,0},{1,0}};
        bool rejected=false;try {tpp::tpp_nonconvex_tspn_solve({box(0,0)},bad);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Open supplied tour rejected");
        std::cout<<"TSPN tests passed: "<<cases<<" exhaustive cases, "<<interrupted<<" interrupted searches, "
                 <<decomposed<<" decomposition cases, 240 arbitrary-hint dual checks.\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
