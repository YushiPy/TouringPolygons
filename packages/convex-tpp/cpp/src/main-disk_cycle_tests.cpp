// Tests of the fixed-order disk cycle oracle (tpp_convex_solve_cycle_disks):
// known optima, the polygonal sandwich (inscribed and circumscribed m-gons
// solved by the polygon oracle must bracket every disk bound), warm starts and
// cutoffs.
#include "tpp_convex.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

namespace {
using tpp::ConvexDisk;
using Disks=std::vector<ConvexDisk>;
void require(bool condition,const std::string &message) {
    if(!condition){std::cerr<<"FAILED: "<<message<<'\n';std::exit(1);}
}
tpp::DiskCycleResult solve(const Disks &disks,double gap,const std::vector<Vector2> &warm={}) {
    tpp::DiskCycleOptions options;options.max_gap=gap;options.initial_contacts=warm;
    const auto r=tpp::tpp_convex_solve_cycle_disks(disks,options);
    require(r.status!=tpp::DiskCycleStatus::InvalidInput,"valid input");
    require(r.lower_bound<=r.upper_bound,"L <= U");
    for(size_t i=0;i<disks.size();++i) {
        const auto d=r.contacts[i]-disks[i].center;
        require(d.dot(d)<=disks[i].radius*disks[i].radius*(1+1e-12)+1e-300,"contact inside its disk");
    }
    return r;
}
void known_optima() {
    // Two disks: the cycle runs twice along the gap between them. Gaps are
    // absolute; 1e-9 on lengths near 10 is about what binary64 proves.
    auto r=solve({{{0,0},1},{{10,0},2}},1e-9);
    require(r.status==tpp::DiskCycleStatus::GapClosed&&r.lower_bound<=14&&14<=r.upper_bound&&r.upper_bound-14<1e-9,"two disks: 2*(10-1-2)");
    // Points only: the perimeter.
    r=solve({{{0,0},0},{{3,0},0},{{0,4},0}},1e-9);
    require(r.lower_bound<=12&&12<=r.upper_bound&&r.upper_bound-12<1e-9,"three points: perimeter 12");
    // A common point: the zero cycle.
    r=solve({{{1,0},1.5},{{-1,0},1.5},{{0,1},1.5}},1e-9);
    require(r.lower_bound<=0&&r.upper_bound<1e-9,"three disks through the origin: zero cycle");
    // A depot point and a disk: twice the distance to the disk.
    r=solve({{{0,0},0},{{5,0},1}},1e-9);
    require(r.lower_bound<=8&&8<=r.upper_bound&&r.upper_bound-8<1e-9,"depot and disk: 2*(5-1)");
    std::cout<<"Disk cycle: known optima passed."<<std::endl;
}
std::vector<Vector2> regular(const ConvexDisk &d,double rho,int m) {
    std::vector<Vector2> p;
    if(d.radius==0)return {d.center};
    for(int j=0;j<m;++j){const double t=2*M_PI*j/m;p.push_back({d.center.x+rho*std::cos(t),d.center.y+rho*std::sin(t)});}
    return p;
}
void sandwich() {
    std::mt19937 rng(20261025);
    std::uniform_real_distribution<double> unit(0,1);
    // The polygon oracle is only a reference here: tighter gaps send heavily
    // overlapping m-gons to its exact path, where clipping them is slow.
    const int m=64;
    size_t cases=0;double widest=0;
    for(int k:{3,5,8,12,20})for(int sample=0;sample<12;++sample) {
        Disks disks;
        for(int i=0;i<k;++i) {
            const double t=2*M_PI*i/k+0.3*unit(rng);
            const double radius=sample%4==0&&i==0?0:(0.2+2*unit(rng))*(sample%3==2?3:1); // overlaps in one third
            disks.push_back({{10*std::cos(t)+2*unit(rng),10*std::sin(t)+2*unit(rng)},radius});
        }
        const auto disk=solve(disks,1e-8);
        std::vector<std::vector<Vector2>> inner,outer;
        for(const auto &d:disks) {
            inner.push_back(regular(d,d.radius*(1-1e-9),m));
            outer.push_back(regular(d,d.radius/std::cos(M_PI/m)*(1+1e-9),m));
        }
        tpp::ConvexCycleDoubleOptions options;options.max_gap=1e-4;
        const auto in=tpp::tpp_convex_solve_cycle_double(inner,options),out=tpp::tpp_convex_solve_cycle_double(outer,options);
        // OPT(outer) <= OPT(disks) <= OPT(inner): every proved bound must agree.
        require(out.certificate.lower_bound<=disk.upper_bound,"outer polygons lower-bound the disks");
        require(disk.lower_bound<=in.certificate.upper_bound,"inner polygons upper-bound the disks");
        require(disk.upper_bound-disk.lower_bound<=1e-8,"disk gap closed");
        widest=std::max(widest,in.certificate.upper_bound-out.certificate.lower_bound);
        ++cases;
    }
    std::cout<<"Disk cycle: "<<cases<<" random cases inside the "<<m<<"-gon sandwich (widest sandwich "<<widest<<") passed."<<std::endl;
}
// The geometric construction and the barrier alone must certify the same value.
void construction_matches_barrier() {
    std::mt19937 rng(20261026);
    std::uniform_real_distribution<double> unit(0,1);
    size_t cases=0,closed=0;
    for(int k:{3,6,10,20,40,80})for(int sample=0;sample<10;++sample) {
        Disks disks;
        for(int i=0;i<k;++i) {
            const double t=2*M_PI*i/k+0.5*unit(rng)/k;
            const double radius=sample%5==0&&i==0?0:(0.1+unit(rng))*(sample%3==2?4:1)*10/k; // overlaps in one third
            disks.push_back({{10*std::cos(t)+unit(rng),10*std::sin(t)+unit(rng)},radius});
        }
        const auto geometric=solve(disks,1e-8);
        tpp::DiskCycleOptions options;options.max_gap=1e-8;options.construction=false;
        const auto barrier=tpp::tpp_convex_solve_cycle_disks(disks,options);
        require(geometric.status==tpp::DiskCycleStatus::GapClosed&&barrier.status==tpp::DiskCycleStatus::GapClosed,"both close the gap");
        require(geometric.lower_bound<=barrier.upper_bound&&barrier.lower_bound<=geometric.upper_bound,"the bounds agree");
        closed+=geometric.construction_closed;++cases;
    }
    std::cout<<"Disk cycle: construction and barrier agree on "<<cases<<" cases (construction closed "<<closed<<")."<<std::endl;
}
void warm_and_cutoff() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> unit(0,1);
    Disks disks;
    for(int i=0;i<30;++i)disks.push_back({{20*unit(rng),20*unit(rng)},0.3+unit(rng)});
    const auto cold=solve(disks,1e-6);
    const auto warm=solve(disks,1e-6,cold.contacts);
    require(cold.status==tpp::DiskCycleStatus::GapClosed&&warm.status==tpp::DiskCycleStatus::GapClosed,"cold and warm close the gap");
    require(std::abs(warm.upper_bound-cold.upper_bound)<=2e-6,"warm start reaches the same value");
    tpp::DiskCycleOptions options;options.lower_bound_cutoff=cold.lower_bound*0.9;
    const auto cut=tpp::tpp_convex_solve_cycle_disks(disks,options);
    require(cut.status==tpp::DiskCycleStatus::CertifiedBound&&cut.lower_bound>=options.lower_bound_cutoff,"cutoff certified");
    std::cout<<"Disk cycle: warm start ("<<cold.newton_iterations<<" -> "<<warm.newton_iterations
        <<" Newton iterations) and cutoff passed."<<std::endl;
}
}

int main() {
    known_optima();
    sandwich();
    construction_matches_barrier();
    warm_and_cutoff();
    std::cout<<"Disk cycle tests passed.\n";
}
