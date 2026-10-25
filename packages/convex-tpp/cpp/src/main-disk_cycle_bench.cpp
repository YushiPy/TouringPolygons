// Exploration: the disk cycle oracle on a Mennell CETSP instance in a fixed
// order (by angle around the centroid of the centers), cold and warm, for
// several absolute gaps and starting barrier parameters.
// usage: tpp-convex-disk-cycle-bench INSTANCE.txt
#include "tpp_convex.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

int main(int argc,char **argv) {
    if(argc<2){std::cerr<<"usage: tpp-convex-disk-cycle-bench INSTANCE.txt\n";return 1;}
    std::ifstream in(argv[1]);std::string line;
    std::vector<tpp::ConvexDisk> disks;Vector2 depot{};bool has_depot=false;
    const std::regex depot_line(R"(//\s*Depot is\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+))");
    while(std::getline(in,line)) {
        std::smatch m;
        if(std::regex_search(line,m,depot_line)){depot={std::stod(m[1]),std::stod(m[2])};has_depot=true;continue;}
        if(line.rfind("//",0)==0)continue;
        std::istringstream s(line);double x,y,z,r;
        if(s>>x>>y>>z>>r)disks.push_back({{x,y},r});
    }
    if(!has_depot){std::cerr<<"no depot\n";return 1;}
    Vector2 c{};for(const auto &d:disks)c+=d.center/double(disks.size());
    std::sort(disks.begin(),disks.end(),[&](const auto &a,const auto &b){
        return std::atan2(a.center.y-c.y,a.center.x-c.x)<std::atan2(b.center.y-c.y,b.center.x-c.x);});
    disks.insert(disks.begin(),tpp::ConvexDisk{depot,0});
    std::cout<<std::setprecision(10)<<disks.size()<<" regions (depot first, by angle)\n";
    using Clock=std::chrono::steady_clock;
    for(double gap:{1e-2,1e-4,1e-6,1e-8}) {
        tpp::DiskCycleOptions o;o.max_gap=gap;
        auto t0=Clock::now();const auto cold=tpp::tpp_convex_solve_cycle_disks(disks,o);
        const double tc=std::chrono::duration<double>(Clock::now()-t0).count();
        std::cout<<"gap "<<gap<<" cold: status "<<int(cold.status)<<" ["<<cold.lower_bound<<", "<<cold.upper_bound<<"] it "<<cold.newton_iterations
            <<" levels "<<cold.levels<<" "<<tc*1e6<<" us\n";
        for(double mu:{0.0,1e-2,1e-3,1e-4,1e-5}) {
            tpp::DiskCycleOptions w=o;w.initial_contacts=cold.contacts;w.mu_start=mu;
            t0=Clock::now();const auto warm=tpp::tpp_convex_solve_cycle_disks(disks,w);
            const double tw=std::chrono::duration<double>(Clock::now()-t0).count();
            std::cout<<"   warm mu_start "<<mu<<": status "<<int(warm.status)<<" gap "<<warm.upper_bound-warm.lower_bound<<" it "<<warm.newton_iterations
                <<" levels "<<warm.levels<<" "<<tw*1e6<<" us\n";
        }
    }
}
