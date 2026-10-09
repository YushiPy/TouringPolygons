// Replays captured fixed-endpoint oracle calls (tpp-unordered --oracle-capture)
// through the certified hybrid oracle and the experimental float oracle, and
// prints one JSON line per call plus a final summary line.
//
// usage: tpp-convex-path-oracle-replay CAPTURE.jsonl [--repeat N] [--no-polish]
//            [--recovery] [--warm-mu-ratio R] [--warm-interior-fraction F]
// --recovery replays the --float-recovery pipeline instead of the float-first
// oracle: hybrid interval proof, then the polish from its seed, then (if still
// open) the complete hybrid oracle.
#include "tpp/convex/float_oracle.h"
#include "tpp/convex/hybrid.h"
#include "tpp/convex/workspace.h"

#include <boost/property_tree/json_parser.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace {
using boost::property_tree::ptree;
using Polygon=std::vector<Vector2>;
using Clock=std::chrono::steady_clock;

Vector2 point(const ptree &tree) {
    auto coordinate=tree.begin();
    if(coordinate==tree.end())throw std::runtime_error("empty point");
    const double x=coordinate++->second.get_value<double>();
    if(coordinate==tree.end())throw std::runtime_error("point without y");
    return {x,coordinate->second.get_value<double>()};
}

double number(const ptree &tree,const char *key) {
    const auto text=tree.get_optional<std::string>(key);
    if(!text||*text=="null")return std::numeric_limits<double>::infinity();
    return std::stod(*text);
}

const char *hybrid_path(const tpp::ConvexHybridResult &r) {
    const auto &s=r.stats;
    if(s.interval_bounds_certified)return s.interval_bounds_contracted?"interval_contracted":"interval";
    if(s.touching_disjoint_certified)return "touching_disjoint";
    if(s.filtered_certified)return "filtered";
    if(s.rational_fallback)return s.rational_fallback_unverified?"rational_unverified":"rational";
    if(r.cutoff_pruned)return "dual_cutoff";
    if(s.double_certified)return "double_kkt";
    return "other";
}

void json(std::ostream &out,double value) {
    if(std::isfinite(value))out<<value;else out<<"null";
}
}

int main(int argc,char **argv) {
    try {
        if(argc<2)throw std::invalid_argument("usage: tpp-convex-path-oracle-replay CAPTURE.jsonl [--repeat N] [--no-polish]");
        std::size_t repeat=1;bool polish=true,recovery=false;double warm_mu_ratio=1e4,warm_fraction=0x1p-7;
        for(int i=2;i<argc;++i) {
            const std::string flag=argv[i];
            if(flag=="--repeat"&&i+1<argc)repeat=std::max<std::size_t>(1,std::stoul(argv[++i]));
            else if(flag=="--no-polish")polish=false;
            else if(flag=="--recovery")recovery=true;
            else if(flag=="--warm-mu-ratio"&&i+1<argc)warm_mu_ratio=std::stod(argv[++i]);
            else if(flag=="--warm-interior-fraction"&&i+1<argc)warm_fraction=std::stod(argv[++i]);
            else throw std::invalid_argument("Unknown option: "+flag);
        }
        std::ifstream input(argv[1]);
        if(!input)throw std::runtime_error("Cannot open capture");
        std::cout<<std::setprecision(17);
        tpp::DynamicConvexTppWorkspace workspace;
        std::size_t calls=0,closed=0,trace_closed=0,polish_closed=0,unsupported=0,violations=0,base_errors=0;
        double base_total=0,float_closed_total=0,float_open_total=0,base_of_open_total=0;
        std::string line;
        while(std::getline(input,line)) {
            if(line.find("\"event\":\"begin\"")==std::string::npos)continue;
            ptree record;{std::istringstream stream(line);boost::property_tree::read_json(stream,record);}
            if(!record.get_child_optional("start"))throw std::runtime_error("Capture lacks start/target; recapture with this revision");
            const Vector2 start=point(record.get_child("start")),target=point(record.get_child("target"));
            std::vector<Polygon> polygons;
            for(const auto &entry:record.get_child("polygons")) {
                Polygon polygon;
                for(const auto &vertex:entry.second)polygon.push_back(point(vertex.second));
                polygons.push_back(std::move(polygon));
            }
            const double cutoff=number(record,"lower_bound_cutoff"),tolerance=number(record,"tolerance");
            tpp::ConvexHybridOptions base_options;base_options.cutoff=cutoff;base_options.max_gap=std::isfinite(tolerance)?tolerance:0;
            tpp::ConvexFloatOracleOptions float_options;float_options.cutoff=cutoff;float_options.max_gap=base_options.max_gap;float_options.polish=polish;
            float_options.warm_mu_ratio=warm_mu_ratio;float_options.warm_interior_fraction=warm_fraction;
            double base_seconds=INFINITY,float_seconds=INFINITY;
            tpp::ConvexHybridResult base;tpp::ConvexFloatOracleResult candidate;bool base_failed=false;std::string base_error;
            for(std::size_t r=0;r<repeat;++r) {
                try {
                    const auto began=Clock::now();
                    base=tpp::tpp_convex_solve_hybrid(start,target,polygons,base_options,workspace);
                    base_seconds=std::min(base_seconds,std::chrono::duration<double>(Clock::now()-began).count());
                } catch(const std::exception &error){base_failed=true;base_error=error.what();}
                const auto began=Clock::now();
                if(recovery) {
                    auto first=base_options;first.stop_after_interval=true;
                    const auto interval=tpp::tpp_convex_solve_hybrid(start,target,polygons,first,workspace);
                    if(!interval.stopped_after_interval) {
                        candidate={};candidate.status=interval.cutoff_pruned?tpp::ConvexFloatOracleStatus::CutoffReached:tpp::ConvexFloatOracleStatus::GapClosed;
                        candidate.trace_closed=true;candidate.lower_bound=interval.lower_bound;candidate.upper_bound=interval.upper_bound;
                    } else {
                        auto polish_options=float_options;polish_options.certify_trace=false;
                        if(interval.interval_seed.size()==polygons.size())polish_options.initial_contacts=&interval.interval_seed;
                        candidate=tpp::tpp_convex_solve_float_certified(start,target,polygons,polish_options);
                        if(candidate.status!=tpp::ConvexFloatOracleStatus::GapClosed&&candidate.status!=tpp::ConvexFloatOracleStatus::CutoffReached) {
                            // Fallback cost is part of this pipeline.
                            const auto full=tpp::tpp_convex_solve_hybrid(start,target,polygons,base_options,workspace);
                            candidate.lower_bound=std::max(candidate.lower_bound,full.lower_bound);
                            candidate.upper_bound=std::min(candidate.upper_bound,full.upper_bound);
                        }
                    }
                } else candidate=tpp::tpp_convex_solve_float_certified(start,target,polygons,float_options);
                float_seconds=std::min(float_seconds,std::chrono::duration<double>(Clock::now()-began).count());
            }
            const bool ok=candidate.status==tpp::ConvexFloatOracleStatus::GapClosed||candidate.status==tpp::ConvexFloatOracleStatus::CutoffReached;
            // Both pairs are rigorous bounds on the same optimum.
            const bool violation=!base_failed&&(candidate.lower_bound>base.upper_bound||base.lower_bound>candidate.upper_bound);
            ++calls;closed+=ok;trace_closed+=candidate.trace_closed;polish_closed+=candidate.polish_closed;
            unsupported+=candidate.status==tpp::ConvexFloatOracleStatus::Unsupported;violations+=violation;base_errors+=base_failed;
            if(!base_failed)base_total+=base_seconds;
            if(ok)float_closed_total+=float_seconds;
            else {float_open_total+=float_seconds;if(!base_failed)base_of_open_total+=base_seconds;}
            std::cout<<"{\"id\":"<<record.get<std::size_t>("id")<<",\"polygons\":"<<polygons.size()
                <<",\"precise\":"<<record.get<std::string>("precise")<<",\"cutoff\":";json(std::cout,cutoff);
            std::cout<<",\"tolerance\":";json(std::cout,tolerance);
            std::cout<<",\"base\":{\"seconds\":";json(std::cout,base_seconds);
            if(base_failed)std::cout<<",\"error\":\""<<base_error<<"\"";
            else {
                std::cout<<",\"path\":\""<<hybrid_path(base)<<"\",\"disjoint\":"<<(base.stats.disjoint?"true":"false")
                    <<",\"cutoff_pruned\":"<<(base.cutoff_pruned?"true":"false")
                    <<",\"lower_bound\":";json(std::cout,base.lower_bound);
                std::cout<<",\"upper_bound\":";json(std::cout,base.upper_bound);
                std::cout<<",\"filtered_seconds\":"<<base.stats.filtered_seconds<<",\"rational_seconds\":"<<base.stats.rational_fallback_seconds;
            }
            std::cout<<"},\"float\":{\"status\":\""<<tpp::to_string(candidate.status)<<"\",\"seconds\":";json(std::cout,float_seconds);
            std::cout<<",\"trace_failed\":"<<(candidate.trace_failed?"true":"false")
                <<",\"trace_closed\":"<<(candidate.trace_closed?"true":"false")
                <<",\"polish_closed\":"<<(candidate.polish_closed?"true":"false")
                <<",\"newton_iterations\":"<<candidate.newton_iterations<<",\"barrier_levels\":"<<candidate.barrier_levels
                <<",\"trace_seconds\":"<<candidate.trace_seconds<<",\"trace_certificate_seconds\":"<<candidate.trace_certificate_seconds
                <<",\"polish_seconds\":"<<candidate.polish_seconds<<",\"lower_bound\":";json(std::cout,candidate.lower_bound);
            std::cout<<",\"upper_bound\":";json(std::cout,candidate.upper_bound);
            std::cout<<"},\"bound_violation\":"<<(violation?"true":"false")<<"}\n";
        }
        std::cout<<"{\"summary\":true,\"calls\":"<<calls<<",\"float_closed\":"<<closed<<",\"trace_closed\":"<<trace_closed
            <<",\"polish_closed\":"<<polish_closed<<",\"unsupported\":"<<unsupported<<",\"bound_violations\":"<<violations
            <<",\"base_errors\":"<<base_errors<<",\"base_seconds\":"<<base_total
            <<",\"float_seconds_closed\":"<<float_closed_total<<",\"float_seconds_open\":"<<float_open_total
            <<",\"base_seconds_of_open\":"<<base_of_open_total
            <<",\"projected_seconds\":"<<float_closed_total+float_open_total+base_of_open_total<<"}\n";
        return violations?3:0;
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
