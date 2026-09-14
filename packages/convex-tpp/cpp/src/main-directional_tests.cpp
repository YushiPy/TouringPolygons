#include "tests.h"
#include "tpp/convex/certified.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "tpp_convex.h"

#include <algorithm>
#include <cmath>
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
size_t checks=0,failures=0,unresolved=0;
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
        {"thin positive area",{{-3,-2},{3,-2},{box(-2,0,0,1e-13),box(-1,0,2,2e-13)},{{-3,-2},{0,0},{3,-2}}}}
    };
    for(auto &[name,c]:cases) {
        verify(c,name);
        auto reversed=c;
        for(auto &p:reversed.polygons)std::reverse(p.begin(),p.end());
        verify(reversed,name+" CW");
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
    deterministic();continuity();random_boxes(random_count);random_convex(convex_count);
    for(const auto &directory:corpora)corpus(directory);
    std::cout<<"Checks="<<checks<<", failures="<<failures<<", unresolved="<<unresolved<<std::endl;
    return failures?1:0;
}
