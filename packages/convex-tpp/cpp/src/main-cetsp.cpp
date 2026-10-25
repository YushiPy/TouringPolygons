// Exploration: a best-first branch-and-bound for the 2D Close-Enough TSP on
// top of the disk cycle oracle (tpp_convex_solve_cycle_disks), to compare
// with Coutinho et al. (2016). A node is a cyclic sequence of disks starting
// at the depot; its certified lower bound is the oracle's L for that
// sequence (visiting a subset can only shorten the tour). If the oracle's
// cycle already meets every disk, it is a feasible tour; otherwise the
// uncovered disk farthest from the cycle is inserted at every position.
//
// usage: tpp-convex-cetsp INSTANCE.txt [SECONDS] [RELATIVE_GAP]
// One JSON line on stdout.
#include "tpp_convex.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <regex>
#include <sstream>

namespace {
using Disks=std::vector<tpp::ConvexDisk>;
using Clock=std::chrono::steady_clock;

struct Instance {Disks disks;}; // disks[0] is the depot, radius 0
Instance read_mennell(const char *path) {
    std::ifstream in(path);std::string line;Instance inst;Vector2 depot{};bool has_depot=false;
    const std::regex depot_line(R"(//\s*Depot is\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+))");
    Disks targets;
    while(std::getline(in,line)) {
        std::smatch m;
        if(std::regex_search(line,m,depot_line)){depot={std::stod(m[1]),std::stod(m[2])};has_depot=true;continue;}
        if(line.rfind("//",0)==0)continue;
        std::istringstream s(line);double x,y,z,r;
        if(s>>x>>y>>z>>r)targets.push_back({{x,y},r});
    }
    if(!has_depot)throw std::runtime_error("no depot line");
    inst.disks.push_back({depot,0});
    inst.disks.insert(inst.disks.end(),targets.begin(),targets.end());
    return inst;
}
double segment_distance(Vector2 p,Vector2 a,Vector2 b) {
    const Vector2 e=b-a;const double ee=e.dot(e);
    const double t=ee>0?std::clamp((p-a).dot(e)/ee,0.0,1.0):0.0;
    const Vector2 d=p-(a+e*t);return std::sqrt(d.dot(d));
}
// How far each disk is from the closed polyline through the contacts, minus
// its radius (<= 0: the tour meets it). Disks of the sequence are met at their
// contacts, which the oracle proved inside them; the others are decided with
// a margin, so a borderline disk is inserted rather than assumed covered.
std::vector<double> excess(const Disks &disks,const std::vector<int> &sequence,const std::vector<Vector2> &cycle) {
    std::vector<double> out(disks.size(),INFINITY);
    std::vector<bool> member(disks.size(),false);for(int i:sequence)member[size_t(i)]=true;
    for(size_t j=0;j<disks.size();++j) {
        if(member[j]){out[j]=-INFINITY;continue;}
        double best=INFINITY;
        for(size_t l=0;l<cycle.size();++l)best=std::min(best,segment_distance(disks[j].center,cycle[l],cycle[(l+1)%cycle.size()]));
        out[j]=best-disks[j].radius;
    }
    return out;
}
constexpr double kCovered=-1e-9; // relative to the radius scale; conservative

struct Node {std::vector<int> sequence;std::vector<Vector2> contacts;double lower=0;};
struct ByBound {bool operator()(const Node &a,const Node &b) const {return a.lower>b.lower;}};

struct Solver {
    const Disks &all;double gap;double deadline;Clock::time_point began;
    double upper=INFINITY;std::vector<int> best_sequence;std::vector<Vector2> best_cycle;
    size_t oracle_calls=0,nodes=0,newton=0;double oracle_seconds=0;
    double elapsed() const {return std::chrono::duration<double>(Clock::now()-began).count();}
    tpp::DiskCycleResult oracle(const std::vector<int> &sequence,const std::vector<Vector2> &warm,double cutoff) {
        Disks d;for(int i:sequence)d.push_back(all[size_t(i)]);
        tpp::DiskCycleOptions o;o.max_gap=std::isfinite(upper)?0.25*gap*upper:1e-9;o.lower_bound_cutoff=cutoff;
        if(warm.size()==d.size())o.initial_contacts=warm;
        const auto t0=Clock::now();const auto r=tpp::tpp_convex_solve_cycle_disks(d,o);
        oracle_seconds+=std::chrono::duration<double>(Clock::now()-t0).count();++oracle_calls;newton+=r.newton_iterations;
        return r;
    }
    // A feasible tour if the cycle meets every disk; returns the uncovered disk farthest from it, or -1.
    int consider(const std::vector<int> &sequence,const tpp::DiskCycleResult &r) {
        const auto ex=excess(all,sequence,r.contacts);
        int far=-1;double worst=0;
        for(size_t j=0;j<all.size();++j)if(ex[j]>kCovered*std::max(1.0,all[j].radius)&&ex[j]>worst-1e300){if(far<0||ex[j]>worst){far=int(j);worst=ex[j];}}
        if(far<0&&r.upper_bound<upper){upper=r.upper_bound;best_sequence=sequence;best_cycle=r.contacts;}
        return far;
    }
    // Warm contacts for a child: the parent's, and for the inserted disk the
    // point of it closest to the parent's link it splits.
    static std::vector<Vector2> child_warm(const std::vector<Vector2> &parent,size_t position,const tpp::ConvexDisk &disk) {
        std::vector<Vector2> w=parent;
        const Vector2 a=parent[position-1],b=parent[position%parent.size()];
        const Vector2 e=b-a;const double ee=e.dot(e);
        const double t=ee>0?std::clamp((disk.center-a).dot(e)/ee,0.0,1.0):0.0;
        Vector2 p=a+e*t,d=p-disk.center;const double n=std::sqrt(d.dot(d));
        if(n>disk.radius&&n>0)p=disk.center+d*(disk.radius/n);
        w.insert(w.begin()+long(position),p);
        return w;
    }
    // Insertion heuristic: repeatedly insert the farthest uncovered disk where the cycle grows least.
    void heuristic() {
        std::vector<int> seq{0};std::vector<Vector2> cyc{all[0].center};
        for(;;) {
            const auto ex=excess(all,seq,cyc);
            int far=-1;for(size_t j=1;j<all.size();++j)if(ex[j]>kCovered*std::max(1.0,all[j].radius)&&(far<0||ex[j]>ex[size_t(far)]))far=int(j);
            if(far<0)break;
            double bestU=INFINITY;std::vector<int> bestS;std::vector<Vector2> bestC;
            for(size_t p=1;p<=seq.size();++p) {
                auto s=seq;s.insert(s.begin()+long(p),far);
                const auto r=oracle(s,child_warm(cyc,p,all[size_t(far)]),INFINITY);
                if(r.upper_bound<bestU){bestU=r.upper_bound;bestS=s;bestC=r.contacts;}
            }
            seq=bestS;cyc=bestC;
            if(elapsed()>deadline)return;
        }
        // The final cycle covers everything: a feasible tour.
        double length=0;for(size_t l=0;l<cyc.size();++l){const auto d=cyc[(l+1)%cyc.size()]-cyc[l];length+=std::sqrt(d.dot(d));}
        upper=length*(1+1e-15);best_sequence=seq;best_cycle=cyc;
    }
    double run() {
        heuristic();
        std::priority_queue<Node,std::vector<Node>,ByBound> queue;
        Node root;root.sequence={0};root.contacts={all[0].center};
        {
            const auto r=oracle(root.sequence,root.contacts,INFINITY);
            root.contacts=r.contacts;root.lower=r.lower_bound;
        }
        queue.push(root);
        double global=0;
        while(!queue.empty()) {
            if(elapsed()>deadline){global=queue.top().lower;break;}
            Node node=queue.top();queue.pop();
            const double cutoff=upper*(1-gap);
            if(node.lower>=cutoff)continue;
            ++nodes;
            const auto ex=excess(all,node.sequence,node.contacts);
            int far=-1;for(size_t j=1;j<all.size();++j)if(ex[j]>kCovered*std::max(1.0,all[j].radius)&&(far<0||ex[j]>ex[size_t(far)]))far=int(j);
            if(far<0)continue; // feasible; its tour was considered when it was created
            const size_t k=node.sequence.size();
            // With the depot first, [0 a ... ] and its reverse are the same cycle:
            // inserting the second disk keeps one orientation only.
            const size_t last=k==2?1:k;
            for(size_t p=1;p<=last;++p) {
                Node child;child.sequence=node.sequence;child.sequence.insert(child.sequence.begin()+long(p),far);
                const auto r=oracle(child.sequence,child_warm(node.contacts,p,all[size_t(far)]),upper*(1-gap));
                if(r.lower_bound>=upper*(1-gap))continue;
                consider(child.sequence,r);
                child.contacts=r.contacts;child.lower=std::max(node.lower,r.lower_bound);
                if(child.lower<upper*(1-gap))queue.push(std::move(child));
            }
        }
        if(queue.empty())global=upper*(1-gap);
        return std::min(global,upper);
    }
};
}

int main(int argc,char **argv) {
    if(argc<2){std::cerr<<"usage: tpp-convex-cetsp INSTANCE.txt [SECONDS] [RELATIVE_GAP]\n";return 1;}
    const auto inst=read_mennell(argv[1]);
    Solver s{inst.disks,argc>3?std::stod(argv[3]):1e-6,argc>2?std::stod(argv[2]):60.0,Clock::now()};
    const double lower=s.run();
    std::cout<<std::setprecision(12)<<"{\"instance\":\""<<argv[1]<<"\",\"disks\":"<<inst.disks.size()-1
        <<",\"lower_bound\":"<<lower<<",\"upper_bound\":"<<s.upper<<",\"gap\":"<<(s.upper-lower)/s.upper
        <<",\"closed\":"<<(s.upper-lower<=s.gap*s.upper*1.0000001?"true":"false")
        <<",\"seconds\":"<<s.elapsed()<<",\"nodes\":"<<s.nodes<<",\"oracle_calls\":"<<s.oracle_calls
        <<",\"oracle_seconds\":"<<s.oracle_seconds<<",\"newton_iterations\":"<<s.newton
        <<",\"tour_regions\":"<<s.best_sequence.size()<<"}"<<std::endl;
}
