// Fixed-order convex cycle: our binary64 stage against Gurobi's SOCP as the
// number of vertices per region grows. Exploration benchmark; see
// benchmarks/workspace/experiments/vertex-scaling (local).
//
// usage: tpp-convex INPUT.json REPETITIONS
// INPUT.json: {"instances": [{"name": ..., "polygons": [[[x, y], ...], ...]}]}
// One JSON line per instance and repetition on stdout.
//
// usage: tpp-convex --replay CAPTURE.jsonl
// Each "begin" record of a TSPN B&B oracle capture (--oracle-capture): our
// oracle with the B&B's own request (gap, cutoff, warm start, features, one
// workspace for the whole capture, as one worker keeps it), then Gurobi's SOCP
// on the same regions in the same order. One JSON line per call.
#include "tpp_convex.h"
#include <gurobi_c++.h>
#include <boost/property_tree/json_parser.hpp>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

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
            // Unit normals: Gurobi's absolute feasibility tolerance is then a
            // distance (raw edge vectors made it 1e-6/|e|, and on short edges
            // its objective fell below the optimum by up to 1e-4 relative).
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

struct OursRun{tpp::ConvexCycleDoubleResult result;double seconds=0;};
// The B&B's request: interval certificates and a gap of a quarter of the node gap.
OursRun ours(const Polygons &p,double gap,tpp::ConvexCycleWorkspace *workspace,bool ids=false) {
    tpp::ConvexCycleDoubleOptions options;
    options.max_gap=gap;options.interval_certificate=true;options.workspace=workspace;
    if(ids){options.region_ids.resize(p.size());std::iota(options.region_ids.begin(),options.region_ids.end(),0);}
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

// Captured oracle calls: ours as the B&B asked, then Gurobi on the same cycle.
static int replay(const char *path,GRBEnv &env) {
    std::ifstream in(path);
    if(!in)throw std::invalid_argument("cannot read the capture");
    tpp::ConvexCycleWorkspace workspace;
    std::string line;size_t calls=0;
    std::cout<<std::setprecision(17);
    auto points=[](const boost::property_tree::ptree &tree) {
        std::vector<Vector2> out;
        for(const auto &[key,point]:tree){auto it=point.begin();const double x=it++->second.get_value<double>();out.push_back({x,it->second.get_value<double>()});}
        return out;
    };
    while(std::getline(in,line)) {
        if(line.find("\"event\":\"begin\"")==std::string::npos)continue;
        std::istringstream text(line);boost::property_tree::ptree call;read_json(text,call);
        Polygons p;for(const auto &[key,polygon]:call.get_child("polygons"))p.push_back(points(polygon));
        auto number=[&](const char *key){const auto v=call.get<std::string>(key);return v=="null"?INFINITY:std::stod(v);};
        tpp::ConvexCycleDoubleOptions options;
        options.lower_bound_cutoff=number("lower_bound_cutoff");
        options.initial_contacts=points(call.get_child("initial_contacts"));
        if(call.get<bool>("cache"))options.workspace=&workspace;
        if(call.get<bool>("features"))for(const auto &[key,v]:call.get_child("initial_features"))options.initial_features.push_back(v.get_value<int>());
        options.retain_active_features=call.get<bool>("features");
        options.bound_first=call.get<bool>("bound_first");options.interval_certificate=call.get<bool>("interval");
        options.proposal_only=call.get<bool>("proposal_bound");
        const double tolerance=number("tolerance");
        options.max_gap=call.get<bool>("float")&&tolerance>0?tolerance:0;
        size_t vertices=0;for(const auto &region:p)vertices+=region.size();
        const auto began=Clock::now();
        const auto ours=p.size()<2?tpp::ConvexCycleDoubleResult{}:tpp::tpp_convex_solve_cycle_double(p,options);
        const double seconds=since(began);
        const auto socp=p.size()<2?GurobiRun{}:gurobi(p,env);
        std::cout<<"{\"call\":"<<call.get<long>("id")<<",\"node\":"<<call.get<long>("node")<<",\"k\":"<<p.size()
            <<",\"vertices\":"<<vertices<<",\"gap\":"<<options.max_gap<<",\"cutoff\":";
        if(std::isfinite(options.lower_bound_cutoff))std::cout<<options.lower_bound_cutoff;else std::cout<<"null";
        std::cout<<",\"ours\":{\"seconds\":"<<seconds<<",\"status\":"<<int(ours.status)
            <<",\"lower\":"<<ours.certificate.lower_bound<<",\"upper\":";
        if(std::isfinite(ours.certificate.upper_bound))std::cout<<ours.certificate.upper_bound;else std::cout<<"null";
        std::cout<<",\"interval_closed\":"<<ours.float_interval_closed<<",\"polish_closed\":"<<ours.float_polish_closed
            <<"},\"gurobi\":{\"status\":"<<socp.status<<",\"objective\":"<<socp.objective<<",\"build\":"<<socp.build
            <<",\"optimize\":"<<socp.optimize<<",\"total\":"<<socp.total<<"}}"<<std::endl;
        ++calls;
    }
    std::cerr<<calls<<" calls replayed\n";
    return 0;
}

int main(int argc,char **argv) {
    try {
        if(argc==3&&std::string(argv[1])=="--replay") {
            GRBEnv env(true);env.set(GRB_IntParam_OutputFlag,0);env.start();
            return replay(argv[2],env);
        }
        if(argc!=3)throw std::invalid_argument("usage: tpp-convex INPUT.json REPETITIONS | --replay CAPTURE.jsonl");
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
            tpp::ConvexCycleWorkspace workspace,identified;
            ours(p,gap,nullptr);ours(p,gap,&workspace);ours(p,gap,&identified,true);ours(p,tight,nullptr);gurobi(p,env);
            for(int repeat=0;repeat<repeats;++repeat) {
                OursRun cold,warm,ids,precise;GurobiRun socp;
                for(int turn=0;turn<5;++turn) {
                    const int which=(turn+repeat)%5;
                    if(which==0)cold=ours(p,gap,nullptr);
                    if(which==1)warm=ours(p,gap,&workspace);
                    if(which==2)ids=ours(p,gap,&identified,true);
                    if(which==3)precise=ours(p,tight,nullptr);
                    if(which==4)socp=gurobi(p,env);
                }
                std::cout<<"{\"name\":\""<<name<<"\",\"k\":"<<p.size()<<",\"vertices\":"<<vertices<<",\"largest\":"<<largest
                    <<",\"repeat\":"<<repeat<<",\"gap\":"<<gap<<",\"tight_gap\":"<<tight;
                print_ours("cold",cold);print_ours("warm",warm);print_ours("ids",ids);print_ours("tight",precise);
                std::cout<<",\"gurobi\":{\"status\":"<<socp.status<<",\"objective\":"<<socp.objective<<",\"build\":"<<socp.build
                    <<",\"optimize\":"<<socp.optimize<<",\"total\":"<<socp.total<<"}}"<<std::endl;
            }
        }
    } catch(const GRBException &e){std::cerr<<e.getMessage()<<'\n';return 2;}
      catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
