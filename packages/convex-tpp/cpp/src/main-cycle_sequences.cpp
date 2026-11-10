// Fixed-order convex cycles through random sequences of regions: the
// boundary-anchor method of paper 1 (Algorithm 4.1: refine_contacts=false,
// gap zero, no polish) against Gurobi's SOCP. Section 7.2 of paper 1; run it
// through `benchmarks/tpp.py paper1-cycles`.
//
// usage: tpp-convex-cycle-sequences INPUT.json REPETITIONS SECONDS [CONFIG]
// CONFIG: exact (default; gap zero, as in paper 1); tolerant (Algorithm 4.1
// with a gap of 1e-6 times the perimeter of the cycle through the polygons'
// vertex means, Gurobi's default scale, as the tolerant run of
// tpp-convex-path-benchmark: boundary_search_gap, so Algorithm 4.1 runs in
// binary64 first, on regions normalized in binary64, and returns at the first
// candidate whose bounds, proved by directed rounding, close the gap; the
// exact path only when it ends open); oracle
// (the same gap given to the branch and bound's binary64 stage, whose
// proposals come from the coordinate-descent sweeps, before Algorithm 4.1).
// INPUT.json: {"instances": [{"name": ..., "polygons": [[[x, y], ...], ...]}]}
// One JSON line per instance and repetition. Within a repetition the two
// solvers run in a rotating order. Our calls shorter than 0.05 s are repeated
// back to back: "seconds" is the first call and "mean" the average of the
// block that follows it. SECONDS caps each of our calls (Interrupted, status 6).
// Gurobi: the model with unit normals, default parameters and one thread,
// built at every call, as in tpp-convex-vertex-scaling.
#include "tpp_convex.h"
#include <gurobi_c++.h>
#include <boost/property_tree/json_parser.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {
using Polygons=std::vector<std::vector<Vector2>>;
using Clock=std::chrono::steady_clock;
double since(Clock::time_point began){return std::chrono::duration<double>(Clock::now()-began).count();}
void json(std::ostream &out,double value){if(std::isfinite(value))out<<value;else out<<"null";}

struct GurobiRun{int status=0;double objective=std::numeric_limits<double>::quiet_NaN(),build=0,optimize=0,total=0;};
GurobiRun gurobi(const Polygons &p,GRBEnv &env) {
    GurobiRun run;const auto began=Clock::now();
    GRBModel model(env);model.set(GRB_IntParam_Threads,1);
    const size_t k=p.size();std::vector<std::array<GRBVar,2>> q(k);
    for(size_t i=0;i<k;++i) {
        q[i]={model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS),model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS)};
        for(size_t j=0;j<p[i].size();++j) {
            // Unit normals: Gurobi's feasibility tolerance is then a distance.
            const auto a=p[i][j],d=p[i][(j+1)%p[i].size()]-a;
            const double length=std::sqrt(d.dot(d));
            if(!(length>0))continue;
            const auto e=d*(1/length);
            model.addConstr(e.y*q[i][0]-e.x*q[i][1]<=e.y*a.x-e.x*a.y);
        }
    }
    GRBLinExpr objective=0;
    for(size_t i=0;i<k;++i) {
        GRBVar d[2]={model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS),model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS)};
        for(int j=0;j<2;++j)model.addConstr(d[j]==q[(i+1)%k][j]-q[i][j]);
        const auto length=model.addVar(0,GRB_INFINITY,0,GRB_CONTINUOUS);
        model.addGenConstrNorm(length,d,2,2.0);objective+=length;
    }
    model.setObjective(objective,GRB_MINIMIZE);
    run.build=since(began);
    const auto optimizing=Clock::now();model.optimize();run.optimize=since(optimizing);
    run.status=model.get(GRB_IntAttr_Status);
    if(model.get(GRB_IntAttr_SolCount))run.objective=model.get(GRB_DoubleAttr_ObjVal);
    run.total=since(began);
    return run;
}

struct OursRun{tpp::ConvexCycleDoubleResult result;double seconds=0,mean=0;size_t calls=0;};
OursRun ours(const Polygons &p,double seconds,double gap,bool boundary_search) {
    tpp::ConvexCycleDoubleOptions options;
    options.refine_contacts=false;options.max_gap=gap;options.float_polish=false;options.max_seconds=seconds;
    if(boundary_search){options.boundary_search_gap=true;options.interval_certificate=true;}
    OursRun run;
    auto began=Clock::now();
    run.result=tpp::tpp_convex_solve_cycle_double(p,options);
    run.seconds=since(began);
    if(run.seconds<0.05) {
        began=Clock::now();
        do{tpp::tpp_convex_solve_cycle_double(p,options);++run.calls;}while(since(began)<0.05);
        run.mean=since(began)/double(run.calls);
    } else run.mean=run.seconds;
    return run;
}

void print(const OursRun &run) {
    const auto &r=run.result;
    std::cout<<",\"ours\":{\"seconds\":"<<run.seconds<<",\"mean\":"<<run.mean<<",\"calls\":"<<run.calls
        <<",\"status\":"<<int(r.status)<<",\"step\":"<<r.method_step<<",\"exact\":"<<(r.exact_contacts.empty()?"false":"true")
        <<",\"lower\":";json(std::cout,r.certificate.lower_bound);
    std::cout<<",\"upper\":";json(std::cout,r.certificate.upper_bound);
    std::cout<<",\"exact_lower\":";json(std::cout,r.exact_contacts.empty()?NAN:r.exact_certificate.lower_bound);
    std::cout<<",\"exact_upper\":";json(std::cout,r.exact_contacts.empty()?NAN:r.exact_certificate.upper_bound);
    std::cout<<",\"anchor\":"<<r.anchor_polygon<<",\"anchored\":"<<r.anchored_solves<<",\"probes\":"<<r.bisection_probes
        <<",\"closed_form\":"<<r.closed_form_steps<<",\"rebuilds\":"<<r.descriptor_rebuilds
        <<",\"rational_anchor_recoveries\":"<<r.rational_anchor_recoveries<<",\"rational_cycle_recoveries\":"<<r.rational_cycle_recoveries
        <<",\"rational_anchored\":"<<r.rational_anchored_solves<<",\"rational_probes\":"<<r.rational_bisection_probes
        <<",\"rational_closed_form\":"<<r.rational_closed_form_steps<<",\"certificate_checks\":"<<r.certificate_checks
        <<",\"construction\":"<<r.timings.construction_seconds<<",\"certification\":"<<r.timings.certification_seconds
        <<",\"recovery\":"<<r.timings.rational_recovery_seconds<<"}";
}
}

int main(int argc,char **argv) {
    try {
        if(argc!=4&&argc!=5)throw std::invalid_argument("usage: tpp-convex-cycle-sequences INPUT.json REPETITIONS SECONDS [exact|tolerant|oracle]");
        const std::string config=argc==5?argv[4]:"exact";
        if(config!="exact"&&config!="tolerant"&&config!="oracle")throw std::invalid_argument("unknown configuration "+config);
        const int repeats=std::stoi(argv[2]);if(repeats<1)throw std::invalid_argument("Positive repetition count required");
        const double seconds=std::stod(argv[3]);
        boost::property_tree::ptree input;read_json(argv[1],input);
        GRBEnv env(true);env.set(GRB_IntParam_OutputFlag,0);env.start();
        std::cout<<std::setprecision(17);
        bool warmed=false;
        for(const auto &[key,item]:input.get_child("instances")) {
            const auto name=item.get<std::string>("name");Polygons p;size_t vertices=0,largest=0;
            for(const auto &[key2,polygon]:item.get_child("polygons")) {
                std::vector<Vector2> region;
                for(const auto &[key3,point]:polygon){auto it=point.begin();const double x=it++->second.get_value<double>();region.push_back({x,it->second.get_value<double>()});}
                vertices+=region.size();largest=std::max(largest,region.size());p.push_back(std::move(region));
            }
            // First touch of the code and of Gurobi's environment, once per process.
            if(!warmed){gurobi(p,env);warmed=true;}
            double gap=0;
            if(config!="exact") {
                std::vector<Vector2> means;
                for(const auto &region:p){Vector2 c{0,0};for(auto v:region)c=c+v;means.push_back(c*(1.0/region.size()));}
                double perimeter=0;
                for(size_t i=0;i<means.size();++i){const auto d=means[(i+1)%means.size()]-means[i];perimeter+=std::sqrt(d.dot(d));}
                gap=1e-6*perimeter;
            }
            for(int repeat=0;repeat<repeats;++repeat) {
                OursRun mine;GurobiRun socp;
                for(int turn=0;turn<2;++turn) {
                    if((turn+repeat)%2==0)mine=ours(p,seconds,gap,config=="tolerant");
                    else socp=gurobi(p,env);
                }
                std::cout<<"{\"name\":\""<<name<<"\",\"k\":"<<p.size()<<",\"vertices\":"<<vertices<<",\"largest\":"<<largest
                    <<",\"repeat\":"<<repeat<<",\"config\":\""<<config<<"\",\"gap\":"<<gap;
                print(mine);
                std::cout<<",\"gurobi\":{\"status\":"<<socp.status<<",\"objective\":";json(std::cout,socp.objective);
                std::cout<<",\"build\":"<<socp.build<<",\"optimize\":"<<socp.optimize<<",\"total\":"<<socp.total<<"}}"<<std::endl;
            }
        }
    } catch(const GRBException &e){std::cerr<<"Gurobi: "<<e.getMessage()<<'\n';return 2;}
      catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
