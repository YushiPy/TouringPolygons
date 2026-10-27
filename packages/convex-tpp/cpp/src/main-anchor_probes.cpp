// The exact cycle path on captured TSPN B&B oracle calls: how many anchored
// fixed-source solves and bisection probes the boundary-anchor search spends,
// with the construction first (as the exact path runs) and without it (the
// anchor search alone). Exploration tool; see docs/algorithms/convex-cycle.md.
//
// usage: tpp-convex-anchor-probes SAMPLES SECONDS CAPTURE.jsonl...
// Takes SAMPLES "begin" records spread evenly over each capture file and solves
// each to gap zero with a limit of SECONDS per solve. One JSON line per call.
#include "tpp_convex.h"
#include <boost/property_tree/json_parser.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
using Polygons=std::vector<std::vector<Vector2>>;
using Clock=std::chrono::steady_clock;

Polygons read_polygons(const boost::property_tree::ptree &call) {
    Polygons p;
    for(const auto &[key,polygon]:call.get_child("polygons")) {
        std::vector<Vector2> region;
        for(const auto &[key2,point]:polygon){auto it=point.begin();const double x=it++->second.get_value<double>();region.push_back({x,it->second.get_value<double>()});}
        p.push_back(std::move(region));
    }
    return p;
}

void print_exact(const char *label,const Polygons &p,bool refine,double seconds) {
    tpp::ConvexCycleOptions options;
    options.refine_contacts=refine;options.max_seconds=seconds;
    const auto began=Clock::now();
    const auto r=tpp::tpp_convex_solve_cycle(p,options);
    const double elapsed=std::chrono::duration<double>(Clock::now()-began).count();
    std::cout<<",\""<<label<<"\":{\"status\":"<<int(r.status)<<",\"seconds\":"<<elapsed
        <<",\"anchored_solves\":"<<r.anchored_solves<<",\"bisection_probes\":"<<r.bisection_probes
        <<",\"oracle_calls\":"<<r.oracle_calls<<",\"anchor_polygon\":"<<r.anchor_polygon
        <<",\"upper\":"<<r.certificate.upper_bound<<"}";
}
}

int main(int argc,char **argv) {
    if(argc<4){std::cerr<<"usage: tpp-convex-anchor-probes SAMPLES SECONDS CAPTURE.jsonl...\n";return 1;}
    const size_t samples=std::stoul(argv[1]);const double seconds=std::stod(argv[2]);
    std::cout<<std::setprecision(17);
    for(int f=3;f<argc;++f) {
        std::vector<std::string> begins;
        {
            std::ifstream in(argv[f]);std::string line;
            while(std::getline(in,line))if(line.find("\"event\":\"begin\"")!=std::string::npos)begins.push_back(std::move(line));
        }
        if(begins.empty())continue;
        const size_t take=std::min(samples,begins.size());
        for(size_t s=0;s<take;++s) {
            const size_t index=(2*s+1)*begins.size()/(2*take);
            std::istringstream text(begins[index]);boost::property_tree::ptree call;read_json(text,call);
            const auto p=read_polygons(call);
            if(p.size()<2)continue;
            size_t vertices=0,smallest=SIZE_MAX;
            for(const auto &region:p){vertices+=region.size();smallest=std::min(smallest,region.size());}
            std::cout<<"{\"file\":\""<<argv[f]<<"\",\"call\":"<<call.get<long>("id")<<",\"k\":"<<p.size()
                <<",\"vertices\":"<<vertices<<",\"smallest\":"<<smallest;
            print_exact("exact",p,true,seconds);
            print_exact("anchors",p,false,seconds);
            std::cout<<"}"<<std::endl;
        }
    }
    return 0;
}
