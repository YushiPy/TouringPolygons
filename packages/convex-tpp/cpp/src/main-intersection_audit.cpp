#include "tests.h"
#include "tpp_convex.h"
#include "tpp/convex/certified.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <format>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {
using Polygon = std::vector<Vector2>;
using Polygons = std::vector<Polygon>;
using tpp::TestCase;
size_t failures = 0, checks = 0;

void check(bool condition, const std::string &name) {
    ++checks;
    if (!condition) { ++failures; std::cout << "FAIL " << name << '\n'; }
}
Polygon box(double x0, double y0, double x1, double y1) {
    return {{x0,y0},{x1,y0},{x1,y1},{x0,y1}};
}
double length(const Polygon &path) {
    long double sum = 0;
    for (size_t i=1; i<path.size(); ++i)
        sum += std::hypot((long double)path[i].x-path[i-1].x,
                          (long double)path[i].y-path[i-1].y);
    return double(sum);
}
void normalize(Polygons &polygons) {
    for (auto &p : polygons) {
        double area=0;
        for(size_t j=0;j<p.size();++j) area += (p[j]-p[0]).cross(p[(j+1)%p.size()]-p[0]);
        if(area<0) std::reverse(p.begin(),p.end());
    }
}
// Independent convex support-function dual. For |d_i| <= 1, every feasible
// ordered contact sequence has length >= this value, even with zero steps.
long double lower_bound(const TestCase &c, const Polygon &directions) {
    auto dot=[](Vector2 a,Vector2 b) { return (long double)a.x*b.x+(long double)a.y*b.y; };
    long double bound = -dot(directions.front(),c.start)+dot(directions.back(),c.target);
    for(size_t i=0;i<c.polygons.size();++i) {
        const auto gradient=directions[i]-directions[i+1];
        long double value=std::numeric_limits<long double>::infinity();
        for(auto p:c.polygons[i]) value=std::min(value,dot(gradient,p));
        bound+=value;
    }
    return bound;
}
void certificate(const TestCase &c, Polygon directions, const std::string &name) {
    check(directions.size()==c.polygons.size()+1,name+" dual dimensions");
    // Rounding-safe contraction of nominal unit vectors. This is a test-only
    // numerical check of the algebraic certificates in the accompanying report.
    for(auto &d:directions) {
        check(d.length()<=1+1e-14,name+" dual norm");
        d=d/(std::max(1.0,d.length())*(1+1e-14));
    }
    const auto validation=tpp::validate_ordered_path(c.start,c.target,c.polygons,c.solution);
    const long double bound=lower_bound(c,directions);
    const double tolerance=2e-12*std::max(length(c.solution),1e-30);
    check(validation.valid,name+" oracle ordered feasibility");
    check(std::abs(length(c.solution)-bound)<tolerance,name+" primal-dual equality");
}
void print_path(const Polygon &p) {
    for(auto v:p) std::cout << " (" << v.x << ',' << v.y << ')';
    std::cout << '\n';
}
void solver_check(const TestCase &c,const std::string &name,bool verbose=false) {
    try {
        const auto lazy=tpp::tpp_convex_solve_binary_search_lazy(c.start,c.target,c.polygons);
        const auto eager=tpp::tpp_convex_solve_binary_search_eager(c.start,c.target,c.polygons);
        const double expected=length(c.solution),tol=2e-12*std::max(expected,1e-30);
        for(const auto &[path,label]:std::vector<std::pair<Polygon,std::string>>{{lazy,"lazy"},{eager,"eager"}}) {
            auto v=tpp::validate_ordered_path(c.start,c.target,c.polygons,path);
            check(v.valid,name+" "+label+" ordered visitation");
            check(std::abs(v.length-expected)<=tol,name+" "+label+" optimal length");
        }
        check(std::abs(tpp::tpp_convex_solve_length_binary_search_lazy(c.start,c.target,c.polygons)-length(lazy))<=tol,name+" lazy length API");
        check(std::abs(tpp::tpp_convex_solve_length_binary_search_eager(c.start,c.target,c.polygons)-length(eager))<=tol,name+" eager length API");
        if(verbose) {
            auto v=tpp::validate_ordered_path(c.start,c.target,c.polygons,lazy);
            std::cout<<name<<" actual="<<v.length<<" expected="<<expected<<" visits="<<v.visited_polygons<<'/'<<c.polygons.size()<<" coordinate_tolerance="<<v.coordinate_tolerance<<'\n';
            print_path(lazy);
        }
    } catch(const std::exception &e) {check(false,name+" threw: "+e.what());}
}
Vector2 reflect(Vector2 d,Vector2 edge) {return edge*(2*d.dot(edge)/edge.length_squared())-d;}

void validator_tests() {
    const Vector2 s(-3,0),t(3,0);
    const Polygons p={box(-2,-1,-1,1),box(1,-1,2,1)};
    auto v=[&](const Vector2 &a,const Vector2 &b,const Polygons &q,const Polygon &path) {
        return tpp::validate_ordered_path(a,b,q,path).valid;
    };
    check(v(s,t,p,{s,t}),"straight continuous segment visits two polygons");
    check(!v(s,t,{p[1],p[0]},{s,t}),"unordered contacts rejected");
    check(v(s,t,{p[0],p[1],p[0]},{s,t,s,t}),"repeated visit indices retained");
    check(!v(s,t,{p[0],p[1],p[0]},{s,t}),"missing repeated visit rejected");
    check(v(s,t,{box(-2,-2,2,2),box(-1,-1,1,1),box(-2,-2,2,2)},{s,t}),"containment and simultaneous visits");
    check(v(s,t,{{{-1,1},{0,0},{1,1}}},{s,t}),"vertex-only contact");
    check(v(s,t,{box(-1,0,1,1),box(0,0,2,2)},{s,t}),"collinear shared boundary");
    check(v({0,0},{0,0},{box(-1,-1,1,1)},{{0,0}}),"stationary path");
    check(v(s,t,p,{s,s,{0,0},{0,0},t}),"redundant and zero-length segments");
    check(!v(s,t,p,{s,Vector2::NaN,t}),"nonfinite path rejected");
    check(!v(s,t,p,{s,{1e12,1e12},t}),"distant invalid path cannot inflate tolerance");
    check(!v(s,t,p,{s}),"wrong endpoint rejected");
    // The legacy validator returned early when target was on the first boundary.
    check(!tpp::is_valid_solution(s,{0,0},{box(0,0,1,1),box(2,2,3,3)},{s,{0,0}}),"legacy target early-return guarded");
    auto reverse=p; for(auto &q:reverse)std::reverse(q.begin(),q.end());
    check(v(s,t,reverse,{s,t}),"validator orientation invariant");
}

TestCase transform(TestCase c,double scale,double angle,Vector2 shift,bool reverse=false,bool collinear=false) {
    auto f=[&](Vector2 p) {return Vector2(scale*(std::cos(angle)*p.x-std::sin(angle)*p.y),scale*(std::sin(angle)*p.x+std::cos(angle)*p.y))+shift;};
    c.start=f(c.start);c.target=f(c.target);
    for(auto &p:c.polygons) {
        if(collinear) {Polygon expanded;for(size_t j=0;j<p.size();++j){expanded.push_back(p[j]);expanded.push_back((p[j]+p[(j+1)%p.size()])/2);}p=std::move(expanded);}
        for(auto &v:p)v=f(v);
        if(reverse)std::reverse(p.begin(),p.end());
    }
    for(auto &v:c.solution)v=f(v);
    return c;
}

void continuity_tests(bool proof_only) {
    double previous=0,previous_delta=0;
    for(double delta:{1e-3,1e-6,1e-9,0.,-1e-9,-1e-6,-1e-3}) {
        TestCase c{{-3,-2},{3,-2},{box(-2,0,0,2),box(delta,0,2,2)},{{-3,-2},{0,0},{3,-2}}};
        Polygon directions;
        if(delta>0) {
            c.solution.insert(c.solution.end()-1,{delta,0});
            directions={(Vector2(3,2)).normalized(),{1,0},Vector2(3-delta,-2).normalized()};
        } else {
            const auto u=Vector2(3,2).normalized(),w=Vector2(3,-2).normalized();
            directions={u,(u+w)/2,w};
        }
        const auto name=std::format("edge-edge delta={:.1e}",delta);
        certificate(c,directions,name);
        if(previous!=0)check(std::abs(length(c.solution)-previous)<=2*std::abs(delta-previous_delta)+1e-13,name+" oracle length continuity");
        previous=length(c.solution);previous_delta=delta;
        if(!proof_only)solver_check(c,name);
    }
}

void random_tests(size_t count) {
    std::mt19937 rng(20260913);
    std::uniform_real_distribution<double> center(-2,2),radius(.3,1.7),angle(0,6.283185307179586);
    size_t invalid=0,suboptimal=0,unresolved=0;
    for(size_t trial=0;trial<count;++trial) {
        TestCase c{{-3,-3},{3,-3},{},{}};
        const size_t k=2+rng()%4;
        for(size_t j=0;j<k;++j) {
            Polygon p;const size_t n=3+rng()%6;const double r=radius(rng),a=angle(rng);const Vector2 o(center(rng),center(rng));
            for(size_t v=0;v<n;++v)p.push_back(o+Vector2(std::cos(a+6.283185307179586*v/n),std::sin(a+6.283185307179586*v/n))*r);
            c.polygons.push_back(p);
        }
        try {
            auto path=tpp::tpp_convex_solve_binary_search_lazy(c.start,c.target,c.polygons);
            const auto validation=tpp::validate_ordered_path(c.start,c.target,c.polygons,path);
            if(!validation.valid)++invalid;
            check(validation.valid,"random "+std::to_string(trial)+" ordered visitation");
            // Existing support-dual/interior-point API is used only as a test
            // oracle; no production dispatch is changed by this audit.
            tpp::DynamicConvexTppWorkspace workspace;
            auto oracle=tpp::tpp_convex_solve_certified(c.start,c.target,c.polygons,workspace,1e-10,std::numeric_limits<double>::infinity(),2.0);
            const bool resolved=tpp::validate_ordered_path(c.start,c.target,c.polygons,oracle.path).valid
                && std::isfinite(oracle.lower_bound) && oracle.upper_bound-oracle.lower_bound<2e-8;
            if(!resolved) {++unresolved;continue;}
            if(validation.valid && validation.length>oracle.upper_bound+2e-8)++suboptimal;
            check(!validation.valid || (validation.length>=oracle.lower_bound-2e-8 && validation.length<=oracle.upper_bound+2e-8),"random "+std::to_string(trial)+" oracle length");
        } catch(const std::exception &e) {++invalid;check(false,"random "+std::to_string(trial)+" threw: "+e.what());}
    }
    std::cout<<"Random: "<<count<<" cases, "<<invalid<<" invalid/throwing, "<<suboptimal<<" feasible suboptimal, "<<unresolved<<" unresolved oracle gaps\n";
}

void benchmark(size_t repeats) {
    tpp::set_rng_seed(1729);
    auto [s,t,p]=tpp::generate_test(std::vector<size_t>(20,8));
    TestCase disjoint{s,t,p,{}};
    TestCase overlap{{-3,-4},{-4,-3},{box(0,-6,6,6),{{-8,-4},{8,4},{8,9},{-8,9}}},{}};
    for(auto &[c,name]:std::vector<std::pair<TestCase,std::string>>{{disjoint,"disjoint-k20-m8"},{overlap,"intersecting-known-incorrect"}}) {
        double checksum=0;const auto begin=std::chrono::steady_clock::now();
        for(size_t j=0;j<repeats;++j) checksum+=length(tpp::tpp_convex_solve_binary_search_lazy(c.start,c.target,c.polygons));
        const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/repeats;
        std::cout<<"Benchmark "<<name<<": "<<us<<" us/call, repeats="<<repeats<<", checksum="<<checksum<<'\n';
    }
}
}

int main(int argc,char **argv) {
    bool proof_only=false;size_t random_count=0,bench=0;
    std::string fixture="benchmarks/suites/intersection-audit/Wrong1.bin";
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--proof-only")proof_only=true;
        else if(arg=="--random" && i+1<argc)random_count=std::stoul(argv[++i]);
        else if(arg=="--bench" && i+1<argc)bench=std::stoul(argv[++i]);
        else if(arg=="--fixture" && i+1<argc)fixture=argv[++i];
        else throw std::invalid_argument("Unknown/incomplete argument: "+arg);
    }
    std::cout<<std::setprecision(17);
    validator_tests();
    const Polygons polygons={box(0,-6,6,6),{{-8,-4},{8,4},{8,9},{-8,9}}};
    TestCase simultaneous{{-3,-4},{4,-3},polygons,{{-3,-4},{0,0},{4,-3}}};
    TestCase separate{{-3,-4},{-4,-3},polygons,{{-3,-4},{0,-3},{-3.6,-1.8},{-4,-3}}};
    certificate(simultaneous,{{.6,.8},{.1,.8},{.8,-.6}},"simultaneous counterexample");
    certificate(separate,{Vector2(3,1).normalized(),Vector2(-3,1).normalized(),Vector2(-1,-3).normalized()},"two-reflection counterexample");
    const Vector2 u(.6,.8),edge(2,1);
    const auto single1=reflect(u,{0,1}),single2=reflect(u,edge),twice=reflect(single1,edge);
    check((single1-Vector2(-.6,.8)).length()<1e-14,"first paper ray");
    check((single2-Vector2(1,0)).length()<1e-14,"second paper ray");
    check((twice-Vector2(.28,-.96)).length()<1e-14,"missing double-reflection ray");
    auto angle=[](Vector2 v) { double a=std::atan2(v.y,v.x); return a<0?a+2*std::acos(-1.):a; };
    check(angle(simultaneous.target)>angle(single1) && angle(separate.target)>angle(single1)
        && std::abs(angle(single2))<1e-14,"targets occupy same sector between paper rays");
    check(twice.cross(simultaneous.target)*twice.cross(separate.target)<0,"correct limiting ray separates targets");
    for(double epsilon:{1e-2,1e-4,1e-6,1e-8}) {
        const Vector2 q(-2*epsilon,-epsilon),bend(0,-11*epsilon/(3+2*epsilon));
        const auto incoming=(q-bend).normalized();
        check((incoming-single1).length()<epsilon,"one-sided incoming limit");
    }
    TestCase out_of_order{{-4,-4},{4,-4},
        {box(0,2,2,4),box(0,-2,3,2),box(-1,3,1,4)},
        {{-4,-4},{0,2},{.5,3},{4,-4}}};
    const auto u0=Vector2(2,3).normalized(),u2=Vector2(1,2).normalized();
    certificate(out_of_order,{u0,{u2.x,u0.y},u2,Vector2(1,-2).normalized()},"out-of-order counterexample");
    check(!tpp::validate_ordered_path(out_of_order.start,out_of_order.target,
        out_of_order.polygons,{{-4,-4},{0,3},{4,-4}}).valid,"missed repeat after shared contact rejected");
    if(!proof_only)solver_check(out_of_order,"out-of-order rectangles",true);
    auto cases=tpp::load_test_cases(fixture);
    check(cases.size()==1,"Wrong1 fixture count");
    if(cases.size()!=1)return 2;
    auto wrong=cases.front();Polygon directions;
    for(size_t j=1;j<wrong.solution.size();++j)directions.push_back((wrong.solution[j]-wrong.solution[j-1]).normalized());
    certificate(wrong,directions,"Wrong1");
    if(!proof_only) {
        solver_check(wrong,"Wrong1 original winding",true);
        normalize(wrong.polygons);solver_check(wrong,"Wrong1 CCW",true);
        solver_check(simultaneous,"simultaneous",true);solver_check(separate,"two reflections",true);
        for(const auto &[name,c]:std::vector<std::pair<std::string,TestCase>>{
            {"translate",transform(separate,1,0,{7,-11})},
            {"rotate",transform(separate,1,.37,{0,0})},
            {"small scale",transform(separate,1e-5,0,{0,0})},
            {"large scale",transform(separate,1e5,0,{0,0})},
            {"reverse",transform(separate,1,0,{0,0},true)},
            {"collinear",transform(separate,1,0,{0,0},false,true)}}) solver_check(c,name);
    }
    continuity_tests(proof_only);
    if(random_count)random_tests(random_count);
    if(bench)benchmark(bench);
    std::cout<<"Checks="<<checks<<", failures="<<failures<<" (proof_only="<<proof_only<<")\n";
    return failures?1:0;
}
