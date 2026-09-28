#include "tpp_convex.h"
#include <gurobi_c++.h>
#include <boost/property_tree/json_parser.hpp>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
using Polygons=std::vector<std::vector<Vector2>>;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point began){return std::chrono::duration<double>(Clock::now()-began).count();}
struct Reference {int status;double objective,bound,runtime;std::vector<Vector2> contacts;};
Reference gurobi(const Polygons &p,GRBEnv &env) {
    GRBModel model(env);model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Method,2);
    model.set(GRB_IntParam_NumericFocus,2);model.set(GRB_DoubleParam_FeasibilityTol,1e-9);
    model.set(GRB_DoubleParam_OptimalityTol,1e-9);model.set(GRB_DoubleParam_BarConvTol,1e-10);
    model.set(GRB_DoubleParam_TimeLimit,30);
    const size_t k=p.size();std::vector<std::array<GRBVar,2>> q(k);
    for(size_t i=0;i<k;++i) {
        auto lo=p[i][0],hi=lo;
        for(auto v:p[i]){lo.x=std::min(lo.x,v.x);lo.y=std::min(lo.y,v.y);hi.x=std::max(hi.x,v.x);hi.y=std::max(hi.y,v.y);}
        q[i]={model.addVar(lo.x,hi.x,0,GRB_CONTINUOUS),model.addVar(lo.y,hi.y,0,GRB_CONTINUOUS)};
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
    model.setObjective(objective,GRB_MINIMIZE);model.optimize();
    Reference r{model.get(GRB_IntAttr_Status),0,0,model.get(GRB_DoubleAttr_Runtime),{}};
    if(!model.get(GRB_IntAttr_SolCount))return r;
    r.objective=model.get(GRB_DoubleAttr_ObjVal);
    try {r.bound=model.get(GRB_DoubleAttr_ObjBound);}catch(const GRBException &){r.bound=r.objective;}
    for(auto vars:q)r.contacts.push_back({vars[0].get(GRB_DoubleAttr_X),vars[1].get(GRB_DoubleAttr_X)});
    return r;
}
// Exact minimal contraction of numerical reference contacts into their regions.
// This is outside every timed solver call and is not an optimality tolerance.
tpp::ConvexCycleCertificateResult reference_certificate(const Polygons &p,const std::vector<Vector2> &q) {
    using R=tpp::ConvexRational;using P=tpp::ConvexRationalPoint;
    tpp::ConvexRationalPolygons exact; tpp::ConvexRationalPolygon contacts;
    for(size_t i=0;i<p.size();++i) {
        tpp::ConvexRationalPolygon poly;P center;
        for(auto v:p[i]){poly.emplace_back(v);center=center+poly.back();}
        center=center*(R(1)/poly.size());const P original(q[i]),d=center-original;R alpha=0;
        for(size_t j=0;j<poly.size();++j) {
            const P e=poly[(j+1)%poly.size()]-poly[j];const R side=e.cross(original-poly[j]);
            if(side<0)alpha=std::max(alpha,R(-side/e.cross(d)));
        }
        contacts.push_back(original+d*alpha);exact.push_back(std::move(poly));
    }
    return tpp::tpp_convex_verify_cycle_certificate(exact,contacts);
}
}
int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::invalid_argument("usage: cycle benchmark INPUT.json REPETITIONS (via benchmarks/tpp.py cycle-benchmark)");
        const int repeats=std::stoi(argv[2]);if(repeats<1)throw std::invalid_argument("Positive repetition count required");
        boost::property_tree::ptree input;read_json(argv[1],input);
        GRBEnv env(true);env.set(GRB_IntParam_OutputFlag,0);env.start();
        std::cout<<std::setprecision(17);
        for(const auto &[key,item]:input.get_child("instances")) {
            const auto name=item.get<std::string>("name");Polygons p;
            for(const auto &[key2,polygon]:item.get_child("polygons")) {
                std::vector<Vector2> vertices;
                for(const auto &[key3,point]:polygon){auto it=point.begin();double x=it++->second.get_value<double>();vertices.push_back({x,it->second.get_value<double>()});}
                p.push_back(std::move(vertices));
            }
            // One untimed warm-up per backend. No Gurobi environment or library
            // startup is charged to one solver but not the others.
            tpp::tpp_convex_solve_cycle(p);tpp::tpp_convex_solve_cycle_double(p);gurobi(p,env);
            for(int repeat=0;repeat<repeats;++repeat) {
                tpp::ConvexCycleResult exact;tpp::ConvexCycleDoubleResult floating;Reference reference;
                double rt=0,dt=0,gt=0;
                for(int turn=0;turn<3;++turn) {
                    const int backend=(turn+repeat)%3;const auto began=Clock::now();
                    if(backend==0){exact=tpp::tpp_convex_solve_cycle(p);rt=seconds(began);}
                    if(backend==1){floating=tpp::tpp_convex_solve_cycle_double(p);dt=seconds(began);}
                    if(backend==2){reference=gurobi(p,env);gt=seconds(began);}
                }
                tpp::ConvexRationalPolygons rational;
                for(const auto &polygon:p){tpp::ConvexRationalPolygon a;for(auto v:polygon)a.emplace_back(v);rational.push_back(std::move(a));}
                const auto ec=tpp::tpp_convex_verify_cycle_certificate(rational,exact.contacts);
                const auto dc=tpp::tpp_convex_verify_cycle_certificate(p,floating.contacts);
                const auto gc=reference.contacts.empty()?tpp::ConvexCycleCertificateResult{}:reference_certificate(p,reference.contacts);
                std::cout<<"{\"name\":\""<<name<<"\",\"k\":"<<p.size()<<",\"repeat\":"<<repeat
                    <<",\"rational_seconds\":"<<rt<<",\"double_seconds\":"<<dt<<",\"gurobi_total_seconds\":"<<gt
                    <<",\"gurobi_optimize_seconds\":"<<reference.runtime
                    <<",\"rational_status\":"<<int(exact.status)<<",\"double_status\":"<<int(floating.status)<<",\"gurobi_status\":"<<reference.status
                    <<",\"rational_certificate\":"<<int(ec.status)<<",\"double_certificate\":"<<int(dc.status)<<",\"gurobi_certificate\":"<<int(gc.status)
                    <<",\"lower\":"<<ec.lower_bound<<",\"upper\":"<<ec.upper_bound
                    <<",\"double_lower\":"<<dc.lower_bound<<",\"double_upper\":"<<dc.upper_bound
                    <<",\"gurobi_objective\":"<<reference.objective<<",\"gurobi_bound\":"<<reference.bound
                    <<",\"gurobi_certified_lower\":"<<gc.lower_bound<<",\"gurobi_certified_upper\":"<<gc.upper_bound
                    <<",\"rational_candidates\":"<<exact.oracle_calls<<",\"double_candidates\":"<<floating.oracle_calls
                    <<",\"rational_recoveries\":"<<floating.rational_cycle_recoveries+floating.rational_anchor_recoveries+floating.rational_feature_recoveries
                    <<",\"rational_anchor_recoveries\":"<<floating.rational_anchor_recoveries
                    <<",\"rational_feature_recoveries\":"<<floating.rational_feature_recoveries
                    <<",\"rational_cycle_recoveries\":"<<floating.rational_cycle_recoveries
                    <<",\"gurobi_contacts\":[";
                for(size_t i=0;i<reference.contacts.size();++i){if(i)std::cout<<',';std::cout<<'['<<reference.contacts[i].x<<','<<reference.contacts[i].y<<']';}
                std::cout<<"]}"<<std::endl;
            }
        }
    } catch(const GRBException &e){std::cerr<<e.getMessage()<<'\n';return 2;}
      catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
