// Exploration: call the binary64 cycle stage in a loop for a sampling profiler.
// usage: tpp-convex INPUT.json NAME SECONDS [warm]
#include "tpp_convex.h"
#include <boost/property_tree/json_parser.hpp>
#include <chrono>
#include <iostream>
#include <numeric>
int main(int argc,char **argv) {
    boost::property_tree::ptree input;read_json(argv[1],input);
    const std::string wanted=argv[2];const double budget=std::stod(argv[3]);
    for(const auto &[key,item]:input.get_child("instances")) {
        if(item.get<std::string>("name")!=wanted)continue;
        std::vector<std::vector<Vector2>> p;
        for(const auto &[k2,polygon]:item.get_child("polygons")){std::vector<Vector2> r;for(const auto &[k3,pt]:polygon){auto it=pt.begin();const double x=it++->second.get_value<double>();r.push_back({x,it->second.get_value<double>()});}p.push_back(std::move(r));}
        std::vector<Vector2> means;double scale=0;
        for(const auto &r:p){Vector2 c{0,0};for(auto v:r)c=c+v;means.push_back(c*(1.0/r.size()));}
        for(size_t i=0;i<p.size();++i){const auto d=means[(i+1)%p.size()]-means[i];scale+=std::sqrt(d.dot(d));}
        tpp::ConvexCycleWorkspace workspace;
        tpp::ConvexCycleDoubleOptions options;options.max_gap=2.5e-4*scale;options.interval_certificate=true;
        if(argc>4){options.workspace=&workspace;options.region_ids.resize(p.size());std::iota(options.region_ids.begin(),options.region_ids.end(),0);}
        const auto began=std::chrono::steady_clock::now();size_t calls=0;
        while(std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()<budget){tpp::tpp_convex_solve_cycle_double(p,options);++calls;}
        std::cout<<calls<<" calls\n";
    }
}
