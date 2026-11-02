// Tests of the fixed-order disk cycle oracle (tpp_convex_solve_cycle_disks):
// known optima, the polygonal sandwich (inscribed and circumscribed m-gons
// solved by the polygon oracle must bracket every disk bound), warm starts and
// cutoffs.
#include "tpp_convex.h"
#include "tpp/convex/rational.h"
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
// Insertion bounds never exceed the child's optimum (checked against its
// certified upper bound) and are mostly close to its lower bound.
void insertion_bounds() {
    std::mt19937 rng(20261027);
    std::uniform_real_distribution<double> unit(0,1);
    size_t checked=0,close=0;double worst=0;
    for(int k:{1,2,3,6,12,25})for(int sample=0;sample<12;++sample) {
        Disks disks;
        for(int i=0;i<k;++i) {
            const double t=2*M_PI*i/k;
            const double radius=i==0?0:(0.2+unit(rng))*(sample%3==2?3:1);
            disks.push_back({{10*std::cos(t)+unit(rng),10*std::sin(t)+unit(rng)},radius});
        }
        const ConvexDisk inserted{{20*unit(rng)-10,20*unit(rng)-10},0.5+unit(rng)};
        const auto parent=solve(disks,1e-9);
        const auto bounds=tpp::tpp_convex_disk_insertion_bounds(disks,parent.contacts,inserted);
        require(bounds.size()==size_t(k),"one bound per position");
        for(int p=1;p<=k;++p) {
            Disks child=disks;child.insert(child.begin()+p,inserted);
            const auto c=solve(child,1e-9);
            require(bounds[size_t(p-1)]<=c.upper_bound,"insertion bound below the child's certified upper bound");
            const double slack=(c.lower_bound-bounds[size_t(p-1)])/std::max(1.0,c.lower_bound);
            worst=std::max(worst,slack);close+=slack<1e-2;++checked;
        }
    }
    std::cout<<"Disk cycle: "<<checked<<" insertion bounds valid ("<<close<<" within 1% of the child's bound, worst "<<worst<<")."<<std::endl;
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

// Coverage of a disk by the polyline is proved only when the excess plus its
// error bound is at most zero. With coordinates up to 1e15 and radii set at
// the computed distance (the borderline), every proof holds in rational
// arithmetic; the review's segment of 2e16 that a fixed margin took as
// meeting a disk 2 away (radius 1) proves nothing.
void coverage_error() {
    using R=tpp::ConvexRational;
    {
        const std::vector<tpp::ConvexDisk> disks{{{-1e16,-1e16},0},{{1e16,1e16},0},{{1e16,1e16+2},1}};
        const auto e=tpp::tpp_convex_disk_excess(disks,{0,1},{disks[0].center,disks[1].center});
        require(!(e.excess[2]+e.error[2]<=0),"a disk 2 away from a long link is not proved met");
    }
    std::mt19937 rng(20261028);std::uniform_real_distribution<double> unit(-1,1);
    size_t proved=0,cases=0;
    for(const double scale:{1.0,1e3,1e8,1e15})for(int trial=0;trial<200;++trial) {
        std::vector<Vector2> cycle;for(int i=0;i<4;++i)cycle.push_back(Vector2{unit(rng),unit(rng)}*scale);
        const Vector2 center=Vector2{unit(rng),unit(rng)}*scale;
        // Exact squared distance from the center to the closed polyline.
        R best=-1;
        for(size_t l=0;l<cycle.size();++l) {
            const Vector2 a=cycle[l],b=cycle[(l+1)%cycle.size()];
            const R ex=R(b.x)-R(a.x),ey=R(b.y)-R(a.y),fx=R(center.x)-R(a.x),fy=R(center.y)-R(a.y);
            const R ee=ex*ex+ey*ey;R t=ee>0?(fx*ex+fy*ey)/ee:R(0);
            if(t<0)t=0;if(t>1)t=1;
            const R gx=fx-ex*t,gy=fy-ey*t,d2=gx*gx+gy*gy;
            if(best<0||d2<best)best=d2;
        }
        std::vector<tpp::ConvexDisk> disks{{cycle[0],0},{center,0}};
        const double computed=tpp::tpp_convex_disk_excess(disks,{0},cycle).excess[1];
        for(const double radius:{computed,std::nextafter(computed,INFINITY),computed*(1+1e-12)}) {
            disks[1].radius=radius;
            const auto e=tpp::tpp_convex_disk_excess(disks,{0},cycle);
            ++cases;
            if(e.excess[1]+e.error[1]<=0) {
                ++proved;
                require(best<=R(radius)*R(radius),"a disk proved met is met in exact arithmetic");
            }
            if(scale<=1e3)require(e.error[1]<=1e-9*std::max(1.0,radius),"the error bound stays below the margin at small scales");
        }
    }
    require(proved>0,"some borderline disks are proved met");
    std::cout<<"Disk coverage: "<<proved<<" of "<<cases<<" borderline disks proved met, all met exactly; the review's long link proves nothing.\n";
}
int main() {
    known_optima();
    sandwich();
    construction_matches_barrier();
    insertion_bounds();
    warm_and_cutoff();
    coverage_error();
    std::cout<<"Disk cycle tests passed.\n";
}
