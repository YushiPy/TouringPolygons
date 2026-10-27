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
    // By angle, ties by squared distance, then by file order.
    std::stable_sort(disks.begin(),disks.end(),[&](const auto &a,const auto &b){
        const double ta=std::atan2(a.center.y-c.y,a.center.x-c.x),tb=std::atan2(b.center.y-c.y,b.center.x-c.x);
        if(ta!=tb)return ta<tb;
        const double ax=a.center.x-c.x,ay=a.center.y-c.y,bx=b.center.x-c.x,by=b.center.y-c.y;
        return ax*ax+ay*ay<bx*bx+by*by;});
    disks.insert(disks.begin(),tpp::ConvexDisk{depot,0});
    std::cout<<std::setprecision(10)<<disks.size()<<" regions (depot first, by angle)\n";
    using Clock=std::chrono::steady_clock;
    auto run=[&](const char *label,tpp::DiskCycleOptions o,int repeats) {
        tpp::DiskCycleResult r;double best=INFINITY;
        for(int rep=0;rep<repeats;++rep) {
            const auto t0=Clock::now();r=tpp::tpp_convex_solve_cycle_disks(disks,o);
            best=std::min(best,std::chrono::duration<double>(Clock::now()-t0).count());
        }
        std::cout<<"  "<<label<<": status "<<int(r.status)<<" ["<<r.lower_bound<<", "<<r.upper_bound<<"] gap "<<r.upper_bound-r.lower_bound
            <<" construction "<<(r.construction_closed?"closed":"open")<<" rounds "<<r.construction_rounds<<" closing "<<r.closing_iterations
            <<" barrier "<<r.newton_iterations<<" | "<<best*1e6<<" us\n";
        return r;
    };
    for(double gap:{1e-2,1e-4,1e-6,1e-8}) {
        std::cout<<"gap "<<gap<<"\n";
        tpp::DiskCycleOptions o;o.max_gap=gap;
        const auto geometric=run("geometric cold",o,5);
        tpp::DiskCycleOptions b=o;b.construction=false;run("barrier cold  ",b,3);
        tpp::DiskCycleOptions w=o;w.initial_contacts=geometric.contacts;run("geometric warm",w,5);
    }
}
