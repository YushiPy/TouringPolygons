// Fixed-order convex cycle: our binary64 stage against Gurobi's SOCP as the
// number of vertices per region grows. Exploration benchmark; see
// benchmarks/workspace/experiments/vertex-scaling (local).
//
// usage: tpp-convex INPUT.json REPETITIONS
// INPUT.json: {"instances": [{"name": ..., "polygons": [[[x, y], ...], ...]}]}
// One JSON line per instance and repetition on stdout.
#include "tpp_convex.h"
#include <gurobi_c++.h>
#include <boost/property_tree/json_parser.hpp>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace {
using Polygons=std::vector<std::vector<Vector2>>;
using Clock=std::chrono::steady_clock;
double since(Clock::time_point began){return std::chrono::duration<double>(Clock::now()-began).count();}

struct GurobiRun{int status=0;double objective=0,build=0,optimize=0,total=0;};
// Gurobi defaults, one thread: the setting Fekete et al.'s SOCP oracle uses.
GurobiRun gurobi(const Polygons &p,GRBEnv &env) {
    GurobiRun run;const auto began=Clock::now();
    GRBModel model(env);model.set(GRB_IntParam_Threads,1);
    const size_t k=p.size();std::vector<std::array<GRBVar,2>> q(k);
    for(size_t i=0;i<k;++i) {
        q[i]={model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS),model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS)};
        for(size_t j=0;j<p[i].size();++j) {
            const auto a=p[i][j],e=p[i][(j+1)%p[i].size()]-a;
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

struct OursRun{tpp::ConvexCycleDoubleResult result;double seconds=0;};
// The B&B's request: interval certificates and a gap of a quarter of the node gap.
OursRun ours(const Polygons &p,double gap,tpp::ConvexCycleWorkspace *workspace) {
    tpp::ConvexCycleDoubleOptions options;
    options.max_gap=gap;options.interval_certificate=true;options.workspace=workspace;
    const auto began=Clock::now();
    OursRun run{tpp::tpp_convex_solve_cycle_double(p,options),0};
    run.seconds=since(began);
    return run;
}

void print_ours(const char *label,const OursRun &run) {
    const auto &r=run.result;
    std::cout<<",\""<<label<<"\":{\"seconds\":"<<run.seconds<<",\"status\":"<<int(r.status)
        <<",\"lower\":"<<r.certificate.lower_bound<<",\"upper\":"<<r.certificate.upper_bound
        <<",\"interval_closed\":"<<r.float_interval_closed<<",\"polish_closed\":"<<r.float_polish_closed
        <<",\"candidates\":"<<r.float_candidates<<",\"newton\":"<<r.float_newton_iterations
        <<",\"construction\":"<<r.timings.construction_seconds<<",\"certification\":"<<r.timings.certification_seconds
        <<",\"interval_proof\":"<<r.timings.interval_proof_seconds<<",\"polish\":"<<r.timings.polish_seconds
        <<",\"recovery\":"<<r.timings.rational_recovery_seconds<<"}";
}
}

int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::invalid_argument("usage: tpp-convex INPUT.json REPETITIONS");
        const int repeats=std::stoi(argv[2]);if(repeats<1)throw std::invalid_argument("Positive repetition count required");
        boost::property_tree::ptree input;read_json(argv[1],input);
        GRBEnv env(true);env.set(GRB_IntParam_OutputFlag,0);env.start();
        std::cout<<std::setprecision(17);
        for(const auto &[key,item]:input.get_child("instances")) {
            const auto name=item.get<std::string>("name");Polygons p;size_t vertices=0,largest=0;
            for(const auto &[key2,polygon]:item.get_child("polygons")) {
                std::vector<Vector2> region;
                for(const auto &[key3,point]:polygon){auto it=point.begin();const double x=it++->second.get_value<double>();region.push_back({x,it->second.get_value<double>()});}
                vertices+=region.size();largest=std::max(largest,region.size());p.push_back(std::move(region));
            }
            // The length scale of the gap requests, independent of every solver:
            // the cycle through the regions' vertex means.
            std::vector<Vector2> means;
            for(const auto &region:p){Vector2 c{0,0};for(auto v:region)c=c+v;means.push_back(c*(1.0/region.size()));}
            double scale=0;for(size_t i=0;i<p.size();++i){const auto d=means[(i+1)%p.size()]-means[i];scale+=std::sqrt(d.dot(d));}
            // The B&B's request (a quarter of a 0.1% node gap), and 1e-6 to match
            // Gurobi's default barrier convergence tolerance.
            const double gap=2.5e-4*scale,tight=1e-6*scale;
            tpp::ConvexCycleWorkspace workspace;
            ours(p,gap,nullptr);ours(p,gap,&workspace);ours(p,tight,nullptr);gurobi(p,env);
            for(int repeat=0;repeat<repeats;++repeat) {
                OursRun cold,warm,precise;GurobiRun socp;
                for(int turn=0;turn<4;++turn) {
                    const int which=(turn+repeat)%4;
                    if(which==0)cold=ours(p,gap,nullptr);
                    if(which==1)warm=ours(p,gap,&workspace);
                    if(which==2)precise=ours(p,tight,nullptr);
                    if(which==3)socp=gurobi(p,env);
                }
                std::cout<<"{\"name\":\""<<name<<"\",\"k\":"<<p.size()<<",\"vertices\":"<<vertices<<",\"largest\":"<<largest
                    <<",\"repeat\":"<<repeat<<",\"gap\":"<<gap<<",\"tight_gap\":"<<tight;
                print_ours("cold",cold);print_ours("warm",warm);print_ours("tight",precise);
                std::cout<<",\"gurobi\":{\"status\":"<<socp.status<<",\"objective\":"<<socp.objective<<",\"build\":"<<socp.build
                    <<",\"optimize\":"<<socp.optimize<<",\"total\":"<<socp.total<<"}}"<<std::endl;
            }
        }
    } catch(const GRBException &e){std::cerr<<e.getMessage()<<'\n';return 2;}
      catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
