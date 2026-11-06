// Fixed-order TPP with fixed endpoints (paths): our oracle against Gurobi's
// SOCP. Section 7.1 of paper 1; run it through
// `benchmarks/tpp.py paper1-experiments`.
//
// usage: tpp-convex-path-benchmark INPUT.json REPETITIONS
// INPUT.json: {"instances": [{"name": ..., "start": [x, y], "target": [x, y],
//              "polygons": [[[x, y], ...], ...], "runs": [...]}]}
// One JSON line per instance and repetition. The solvers named in "runs"
// (default: the first five below) run in a rotating order within each
// repetition:
//   certified     tpp_convex_solve_hybrid with max_gap 0: the complete
//                 propose-and-certify flow, optimal when it reports L = U;
//   tolerant      the same with max_gap 1e-6 times the length of the path
//                 through the polygons' vertex means (Gurobi's default scale);
//   lazy, eager   the binary64 construction alone (no certificate), building
//                 cones on demand or all of them first, with the number of
//                 cones built (Solution::build_cone or the directional maps);
//   gurobi        the SOCP with unit normals, default parameters, one thread,
//                 the model built at every call (as Fekete et al.'s oracle);
//   tan_jiang     Tan and Jiang's construction (tan_jiang.cpp): every cone,
//                 the vertices of each polygon located together by monotone
//                 scans. Disjoint polygons only; otherwise the solver falls
//                 back to the lazy search, which shows as fewer cones than
//                 vertices;
//   binary64      the binary64 construction of Step 1 of the flow alone (its
//                 trace: the lazy search for disjoint polygons, the binary64
//                 maps otherwise), without replay or certificate;
//   rational      the complete rational construction alone, as Step 3 of the
//                 flow rebuilds the path (Dror et al.'s recursion for disjoint
//                 polygons, the exact maps otherwise), without the certificate;
//   certified_eager, certified_tan_jiang
//                 the certified flow (gap zero) with the eager or Tan and
//                 Jiang's construction proposing the path at Step 1: closed_by
//                 says whether that proposal passed after the exact replay;
//   targets       many targets on the same maps (disjoint polygons): with
//                 "targets": [[x, y], ...] and "target_counts": [r, ...] in the
//                 instance, the lazy, eager and Tan and Jiang constructions
//                 answer the first r targets with one set of maps, timed per r,
//                 with the cones the lazy one builds.
// Calls shorter than 0.05 s are repeated back to back; "seconds" is the first
// call and "mean" the average of the block that follows it.
//
// usage: tpp-convex-path-benchmark --replay CAPTURE.jsonl
// Fixed-endpoint oracle calls captured from the B&B (tpp-unordered
// --oracle-capture, without --cycle): our oracle with the B&B's own request
// (cutoff and tolerance, one workspace for the whole capture, as one worker
// keeps it), then Gurobi's SOCP on the same path. One JSON line per call.
#include "tpp_convex.h"
#include "tpp/convex/solution.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "tpp/convex/detail/rational_disjoint.h"
#include <gurobi_c++.h>
#include <boost/property_tree/json_parser.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace {
using Polygons=std::vector<std::vector<Vector2>>;
using Clock=std::chrono::steady_clock;
using boost::property_tree::ptree;
double since(Clock::time_point began){return std::chrono::duration<double>(Clock::now()-began).count();}

void json(std::ostream &out,double value){if(std::isfinite(value))out<<value;else out<<"null";}

Vector2 point(const ptree &tree) {
    auto it=tree.begin();
    if(it==tree.end())throw std::runtime_error("empty point");
    const double x=it++->second.get_value<double>();
    if(it==tree.end())throw std::runtime_error("point without y");
    return {x,it->second.get_value<double>()};
}
Polygons polygons(const ptree &tree) {
    Polygons out;
    for(const auto &[key,polygon]:tree) {
        std::vector<Vector2> region;
        for(const auto &[key2,vertex]:polygon)region.push_back(point(vertex));
        out.push_back(std::move(region));
    }
    return out;
}

struct GurobiRun{int status=0;double objective=std::numeric_limits<double>::quiet_NaN(),build=0,optimize=0,total=0;};
// The path s, q_1, ..., q_k, t with q_i in P_i (counterclockwise P_i), one norm
// per segment. Unit normals make Gurobi's feasibility tolerance a distance.
GurobiRun gurobi(Vector2 s,Vector2 t,const Polygons &p,GRBEnv &env) {
    GurobiRun run;const auto began=Clock::now();
    GRBModel model(env);model.set(GRB_IntParam_Threads,1);
    const size_t k=p.size();std::vector<std::array<GRBVar,2>> q(k);
    for(size_t i=0;i<k;++i) {
        q[i]={model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS),model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS)};
        for(size_t j=0;j<p[i].size();++j) {
            const auto a=p[i][j],d=p[i][(j+1)%p[i].size()]-a;
            const double length=std::sqrt(d.dot(d));
            if(!(length>0))continue;
            const auto e=d*(1/length);
            model.addConstr(e.y*q[i][0]-e.x*q[i][1]<=e.y*a.x-e.x*a.y);
        }
    }
    GRBLinExpr objective=0;
    for(size_t i=0;i<=k;++i) {
        GRBVar d[2]={model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS),model.addVar(-GRB_INFINITY,GRB_INFINITY,0,GRB_CONTINUOUS)};
        for(int c=0;c<2;++c) {
            const double from=c==0?s.x:s.y,to=c==0?t.x:t.y;
            if(k==0)model.addConstr(d[c]==to-from);
            else if(i==0)model.addConstr(d[c]==q[0][c]-from);
            else if(i==k)model.addConstr(d[c]==to-q[k-1][c]);
            else model.addConstr(d[c]==q[i][c]-q[i-1][c]);
        }
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

// Which step of the propose-and-certify flow closed the call (paper 1,
// Algorithm 5.3): interval bounds (Step 1), exact replay with the certificate
// (Step 2), the recoveries between Steps 2 and 3, the exact rebuild (Step 3),
// or the dual bound at the path's own directions (Step 4).
const char *closed_by(const tpp::ConvexHybridResult &r) {
    const auto &s=r.stats;
    if(s.interval_bounds_certified)return s.interval_bounds_contracted?"interval_contracted":"interval";
    if(s.touching_disjoint_certified)return "touching_disjoint";
    if(s.filtered_certified)return "filtered";
    if(s.rational_fallback)return s.rational_fallback_unverified?"rational_unverified":"rational";
    if(r.cutoff_pruned)return "dual_cutoff";
    if(s.double_certified)return "double_kkt";
    return "other";
}

struct Timed{double seconds=0,mean=0;size_t calls=0;};
// The first call, then the same call back to back for 0.05 s when it is short.
template<class F> Timed timed(F &&call) {
    Timed run;const auto began=Clock::now();call();run.seconds=since(began);
    if(run.seconds<0.05) {
        const auto block=Clock::now();
        do{call();++run.calls;}while(since(block)<0.05);
        run.mean=since(block)/double(run.calls);
    } else run.mean=run.seconds;
    return run;
}

struct OursRun{tpp::ConvexHybridResult result;Timed time;};
OursRun certified(Vector2 s,Vector2 t,const Polygons &p,double gap,
        tpp::ConvexHybridConstruction construction=tpp::ConvexHybridConstruction::Lazy) {
    tpp::ConvexHybridOptions options;options.max_gap=gap;options.construction=construction;
    OursRun run;
    run.time=timed([&]{run.result=tpp::tpp_convex_solve_hybrid(s,t,p,options);});
    return run;
}
struct Construction{double length=0;size_t cones=0;Timed time;};
template<class F> Construction construction(F &&solve) {
    Construction run;
    tpp::detail::cone_builds()=0;
    run.time=timed([&]{run.length=solve();});
    // The counter covers the first call and the block; divide to one call.
    run.cones=tpp::detail::cone_builds()/(run.time.calls+1);
    return run;
}

// Step 3 of the flow alone: the complete rational construction.
size_t rational_construction(Vector2 s,Vector2 t,const Polygons &p,bool disjoint) {
    tpp::ConvexRationalPolygons exact;
    for(const auto &polygon:p){tpp::ConvexRationalPolygon q;for(auto v:polygon)q.emplace_back(v);exact.push_back(std::move(q));}
    const tpp::ConvexRationalPoint start(s),target(t);
    if(disjoint) {
        try {return tpp::detail::solve_rational_disjoint_exact(start,target,exact).contacts.size();}
        catch(const std::exception &) {return tpp::detail::solve_disjoint_map_contacts_exact(start,target,exact).size();}
    }
    return tpp::detail::solve_intersecting_map_contacts_exact(start,target,exact).size();
}

void print(const char *label,const OursRun &run) {
    const auto &r=run.result;
    std::cout<<",\""<<label<<"\":{\"seconds\":"<<run.time.seconds<<",\"mean\":"<<run.time.mean<<",\"calls\":"<<run.time.calls
        <<",\"closed_by\":\""<<closed_by(r)<<"\",\"disjoint\":"<<(r.stats.disjoint?"true":"false")<<",\"lower\":";
    json(std::cout,r.lower_bound);std::cout<<",\"upper\":";json(std::cout,r.upper_bound);std::cout<<"}";
}
void print(const char *label,const Construction &run) {
    std::cout<<",\""<<label<<"\":{\"seconds\":"<<run.time.seconds<<",\"mean\":"<<run.time.mean<<",\"calls\":"<<run.time.calls
        <<",\"length\":";json(std::cout,run.length);std::cout<<",\"cones\":"<<run.cones<<"}";
}
void print(const GurobiRun &run) {
    std::cout<<",\"gurobi\":{\"status\":"<<run.status<<",\"objective\":";json(std::cout,run.objective);
    std::cout<<",\"build\":"<<run.build<<",\"optimize\":"<<run.optimize<<",\"total\":"<<run.total<<"}";
}

double number(const ptree &tree,const char *key) {
    const auto text=tree.get_optional<std::string>(key);
    if(!text||*text=="null")return std::numeric_limits<double>::infinity();
    return std::stod(*text);
}
}

static int replay(const char *path,GRBEnv &env) {
    std::ifstream in(path);
    if(!in)throw std::invalid_argument("cannot read the capture");
    tpp::DynamicConvexTppWorkspace workspace;
    std::string line;size_t calls=0;
    while(std::getline(in,line)) {
        if(line.find("\"event\":\"begin\"")==std::string::npos)continue;
        ptree record;{std::istringstream text(line);read_json(text,record);}
        if(!record.get_child_optional("start"))throw std::runtime_error("capture lacks start/target");
        const Vector2 s=point(record.get_child("start")),t=point(record.get_child("target"));
        const auto p=polygons(record.get_child("polygons"));
        tpp::ConvexHybridOptions options;options.cutoff=number(record,"lower_bound_cutoff");
        const double tolerance=number(record,"tolerance");options.max_gap=std::isfinite(tolerance)?tolerance:0;
        size_t vertices=0;for(const auto &region:p)vertices+=region.size();
        const auto began=Clock::now();
        const auto ours=tpp::tpp_convex_solve_hybrid(s,t,p,options,workspace);
        const double seconds=since(began);
        const auto socp=gurobi(s,t,p,env);
        std::cout<<"{\"call\":"<<record.get<long>("id")<<",\"node\":"<<record.get<long>("node",-1)<<",\"k\":"<<p.size()
            <<",\"vertices\":"<<vertices<<",\"gap\":"<<options.max_gap<<",\"cutoff\":";json(std::cout,options.cutoff);
        std::cout<<",\"ours\":{\"seconds\":"<<seconds<<",\"closed_by\":\""<<closed_by(ours)<<"\",\"disjoint\":"
            <<(ours.stats.disjoint?"true":"false")<<",\"lower\":";json(std::cout,ours.lower_bound);
        std::cout<<",\"upper\":";json(std::cout,ours.upper_bound);std::cout<<"}";
        print(socp);std::cout<<"}"<<std::endl;
        ++calls;
    }
    std::cerr<<calls<<" calls replayed\n";
    return 0;
}

int main(int argc,char **argv) {
    try {
        std::cout<<std::setprecision(17);
        if(argc==3&&std::string(argv[1])=="--replay") {
            GRBEnv env(true);env.set(GRB_IntParam_OutputFlag,0);env.start();
            return replay(argv[2],env);
        }
        if(argc!=3)throw std::invalid_argument("usage: tpp-convex-path-benchmark INPUT.json REPETITIONS | --replay CAPTURE.jsonl");
        const int repeats=std::stoi(argv[2]);if(repeats<1)throw std::invalid_argument("Positive repetition count required");
        ptree input;read_json(argv[1],input);
        GRBEnv env(true);env.set(GRB_IntParam_OutputFlag,0);env.start();
        for(const auto &[key,item]:input.get_child("instances")) {
            const auto name=item.get<std::string>("name");
            const Vector2 s=point(item.get_child("start")),t=point(item.get_child("target"));
            const auto p=polygons(item.get_child("polygons"));
            size_t vertices=0,largest=0;for(const auto &region:p){vertices+=region.size();largest=std::max(largest,region.size());}
            // The length scale of the tolerant request, independent of every
            // solver: the path through the polygons' vertex means.
            double scale=0;Vector2 previous=s;
            for(const auto &region:p){Vector2 c{0,0};for(auto v:region)c=c+v;c=c*(1.0/region.size());const auto d=c-previous;scale+=std::sqrt(d.dot(d));previous=c;}
            {const auto d=t-previous;scale+=std::sqrt(d.dot(d));}
            const double tolerance=1e-6*scale;
            std::vector<std::string> runs;
            if(const auto listed=item.get_child_optional("runs"))
                for(const auto &[key2,run]:*listed)runs.push_back(run.get_value<std::string>());
            if(runs.empty())runs={"certified","tolerant","lazy","eager","gurobi"};
            for(const auto &run:runs)
                if(run!="certified"&&run!="tolerant"&&run!="lazy"&&run!="eager"&&run!="tan_jiang"&&run!="gurobi"&&
                   run!="rational"&&run!="binary64"&&run!="certified_eager"&&run!="certified_tan_jiang"&&run!="targets")
                    throw std::invalid_argument("unknown run "+run);
            // Warm-up: first touch of the code and of Gurobi's environment.
            const bool disjoint=certified(s,t,p,0).result.stats.disjoint;
            std::vector<Vector2> targets;std::vector<size_t> counts;
            if(const auto listed=item.get_child_optional("targets"))for(const auto &[key2,v]:*listed)targets.push_back(point(v));
            if(const auto listed=item.get_child_optional("target_counts"))
                for(const auto &[key2,v]:*listed)counts.push_back(std::min(targets.size(),size_t(v.get_value<long>())));
            for(int repeat=0;repeat<repeats;++repeat) {
                OursRun exact,tolerant,exact_eager,exact_tan_jiang;Construction lazy,eager,batch,rational,binary;GurobiRun socp;
                // Many targets: per count r, the lazy, eager and Tan and Jiang times.
                std::vector<std::array<Construction,3>> many(counts.size());double many_deviation=0;
                for(size_t turn=0;turn<runs.size();++turn) {
                    const auto &which=runs[(turn+size_t(repeat))%runs.size()];
                    if(which=="certified")exact=certified(s,t,p,0);
                    if(which=="tolerant")tolerant=certified(s,t,p,tolerance);
                    if(which=="lazy")lazy=construction([&]{return tpp::tpp_convex_solve_length_binary_search_lazy(s,t,p);});
                    if(which=="eager")eager=construction([&]{return tpp::tpp_convex_solve_length_binary_search_eager(s,t,p);});
                    if(which=="tan_jiang")batch=construction([&]{return tpp::tpp_convex_solve_length_tan_jiang(s,t,p);});
                    if(which=="gurobi")socp=gurobi(s,t,p,env);
                    if(which=="binary64")binary=construction([&]{return double((disjoint?
                        tpp::detail::solve_binary_search_disjoint_trace_unchecked(s,t,p):
                        tpp::detail::solve_intersecting_map_trace_unchecked_double(s,t,p)).size());});
                    if(which=="rational")rational=construction([&]{return double(rational_construction(s,t,p,disjoint));});
                    if(which=="certified_eager")exact_eager=certified(s,t,p,0,tpp::ConvexHybridConstruction::Eager);
                    if(which=="certified_tan_jiang")exact_tan_jiang=certified(s,t,p,0,tpp::ConvexHybridConstruction::TanJiang);
                    if(which=="targets")for(size_t c=0;c<counts.size();++c) {
                        const std::vector<Vector2> first(targets.begin(),targets.begin()+long(counts[c]));
                        std::array<std::vector<double>,3> lengths;
                        for(size_t m=0;m<3;++m) {
                            const size_t which_method=(m+size_t(repeat)+c)%3;
                            many[c][which_method]=construction([&] {
                                lengths[which_method]=which_method==0?tpp::tpp_convex_solve_lengths_binary_search_lazy(s,first,p):
                                    which_method==1?tpp::tpp_convex_solve_lengths_binary_search_eager(s,first,p):
                                    tpp::tpp_convex_solve_lengths_tan_jiang(s,first,p);
                                return lengths[which_method].empty()?0.0:lengths[which_method].back();
                            });
                        }
                        for(size_t r=0;r<first.size();++r)for(size_t m=1;m<3;++m)
                            many_deviation=std::max(many_deviation,std::abs(lengths[m][r]-lengths[0][r])/std::max(1.0,lengths[0][r]));
                    }
                }
                std::cout<<"{\"name\":\""<<name<<"\",\"k\":"<<p.size()<<",\"vertices\":"<<vertices<<",\"largest\":"<<largest
                    <<",\"repeat\":"<<repeat<<",\"tolerance\":"<<tolerance;
                for(const auto &run:runs) {
                    if(run=="certified")print("certified",exact);
                    if(run=="tolerant")print("tolerant",tolerant);
                    if(run=="lazy")print("lazy",lazy);
                    if(run=="eager")print("eager",eager);
                    if(run=="tan_jiang")print("tan_jiang",batch);
                    if(run=="gurobi")print(socp);
                    if(run=="rational")print("rational",rational);
                    if(run=="binary64")print("binary64",binary);
                    if(run=="certified_eager")print("certified_eager",exact_eager);
                    if(run=="certified_tan_jiang")print("certified_tan_jiang",exact_tan_jiang);
                    if(run=="targets") {
                        std::cout<<",\"targets\":{\"max_relative_deviation\":"<<many_deviation<<",\"counts\":[";
                        for(size_t c=0;c<counts.size();++c) {
                            std::cout<<(c?",":"")<<"{\"r\":"<<counts[c];
                            const char *names[3]={"lazy","eager","tan_jiang"};
                            for(size_t m=0;m<3;++m)
                                std::cout<<",\""<<names[m]<<"\":{\"seconds\":"<<many[c][m].time.seconds<<",\"mean\":"<<many[c][m].time.mean
                                    <<",\"calls\":"<<many[c][m].time.calls<<",\"cones\":"<<many[c][m].cones<<"}";
                            std::cout<<"}";
                        }
                        std::cout<<"]}";
                    }
                }
                std::cout<<"}"<<std::endl;
            }
        }
    } catch(const GRBException &e){std::cerr<<"Gurobi: "<<e.getMessage()<<'\n';return 2;}
      catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
