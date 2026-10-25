// Exploration: a best-first branch-and-bound for the 2D Close-Enough TSP on
// top of the disk cycle oracle (tpp_convex_solve_cycle_disks), to compare
// with Coutinho et al. (2016). A node is a cyclic sequence of disks starting
// at the depot; its certified lower bound is the oracle's L for that
// sequence (visiting a subset can only shorten the tour). If the oracle's
// cycle already meets every disk, it is a feasible tour; otherwise the
// uncovered disk farthest from the cycle is inserted at every position.
//
// The initial tour comes from cheapest insertion, improved by local search
// (drop, relocate, segment reversal) and iterated local search: every move is
// evaluated by the oracle, whose U bounds a feasible tour when its cycle meets
// every disk.
//
// usage: tpp-convex-cetsp INSTANCE.txt [SECONDS] [RELATIVE_GAP] [TOUR.json|-] [STRONG] [HEURISTIC_SECONDS]
// One JSON line on stdout; TOUR.json, if given, receives the best tour
// (sequence and contacts) for an independent check. STRONG (default 1) is
// the number of branching candidates.
#include "tpp_convex.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
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

// The search tree keeps 8 bytes a node: its parent and the insertion that
// made it. A node's sequence is rebuilt from the root, and its cycle solved
// again, when it is expanded; open nodes wait in the queue with their bound.
struct Entry {std::uint32_t parent;std::uint16_t position,disk;};
struct Open {double lower;std::uint32_t id;};
struct ByBound {bool operator()(const Open &a,const Open &b) const {return a.lower>b.lower;}};

struct Solver {
    const Disks &all;double gap;double deadline;Clock::time_point began;size_t strong=1;
    double upper=INFINITY;std::vector<int> best_sequence;std::vector<Vector2> best_cycle;
    size_t oracle_calls=0,nodes=0,newton=0,constructed=0,improvements=0;double oracle_seconds=0,heuristic_seconds=0;
    double elapsed() const {return std::chrono::duration<double>(Clock::now()-began).count();}
    tpp::DiskCycleResult oracle(const std::vector<int> &sequence,const std::vector<Vector2> &warm,double cutoff) {
        Disks d;for(int i:sequence)d.push_back(all[size_t(i)]);
        tpp::DiskCycleOptions o;o.max_gap=std::isfinite(upper)?0.25*gap*upper:1e-9;o.lower_bound_cutoff=cutoff;
        if(warm.size()==d.size())o.initial_contacts=warm;
        const auto t0=Clock::now();const auto r=tpp::tpp_convex_solve_cycle_disks(d,o);
        oracle_seconds+=std::chrono::duration<double>(Clock::now()-t0).count();++oracle_calls;newton+=r.newton_iterations;constructed+=r.construction_closed;
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
    // A tour: a sequence whose oracle cycle meets every disk, and its U.
    struct Tour {std::vector<int> sequence;std::vector<Vector2> cycle;double length=INFINITY;};
    bool covers(const std::vector<int> &sequence,const std::vector<Vector2> &cycle) const {
        const auto ex=excess(all,sequence,cycle);
        for(size_t j=1;j<all.size();++j)if(ex[j]>kCovered*std::max(1.0,all[j].radius))return false;
        return true;
    }
    // The sequence's cycle (warm from `warm` if it fits), as a tour if it covers every disk.
    Tour evaluate(const std::vector<int> &sequence,const std::vector<Vector2> &warm) {
        const auto r=oracle(sequence,warm,INFINITY);
        Tour t{sequence,r.contacts,INFINITY};
        if(covers(sequence,r.contacts))t.length=r.upper_bound;
        return t;
    }
    void offer(const Tour &t) {
        if(t.length<upper){upper=t.length;best_sequence=t.sequence;best_cycle=t.cycle;++improvements;}
    }
    // Cheapest insertion: repeatedly the farthest uncovered disk, where the cycle grows least.
    Tour repair(std::vector<int> seq,std::vector<Vector2> cyc,double until=INFINITY) {
        for(;;) {
            if(elapsed()>until)return {};
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
        }
        return evaluate(seq,cyc);
    }
    // Each disk's 10 nearest disks, by center distance.
    std::vector<std::vector<int>> nearest;
    void prepare_neighbours() {
        const size_t n=all.size();nearest.assign(n,{});
        std::vector<int> order(n);
        for(size_t i=0;i<n;++i) {
            for(size_t j=0;j<n;++j)order[j]=int(j);
            const size_t keep=std::min<size_t>(10,n-1);
            std::partial_sort(order.begin(),order.begin()+long(keep)+1,order.end(),[&](int a,int b){
                const auto da=all[size_t(a)].center-all[i].center,db=all[size_t(b)].center-all[i].center;return da.dot(da)<db.dot(db);});
            for(size_t q=0;q<=keep;++q)if(size_t(order[q])!=i&&nearest[i].size()<keep)nearest[i].push_back(order[q]);
        }
    }
    // First-improvement local search: drop a disk the rest of the tour meets,
    // move one next to one of its nearest disks, or reverse a segment so that
    // a disk follows one of its nearest.
    Tour local_search(Tour t,double until) {
        if(nearest.empty())prepare_neighbours();
        for(bool improved=true;improved&&elapsed()<until;) {
            improved=false;
            const size_t n=t.sequence.size();
            std::vector<long> at(all.size(),-1);
            for(size_t i=0;i<n;++i)at[size_t(t.sequence[i])]=long(i);
            auto attempt=[&](std::vector<int> s,std::vector<Vector2> w) {
                const Tour c=evaluate(s,w);
                if(c.length<t.length*(1-1e-12)){t=c;improved=true;}
                return improved;
            };
            for(size_t i=1;i<n&&!improved;++i) {
                auto s=t.sequence;auto w=t.cycle;s.erase(s.begin()+long(i));w.erase(w.begin()+long(i));
                attempt(s,w);
            }
            for(size_t i=1;i<n&&!improved&&elapsed()<until;++i)for(const int nb:nearest[size_t(t.sequence[i])]) {
                if(at[size_t(nb)]<0||improved)continue;
                for(const long side:{0L,1L}) {
                    // Remove disk i, then insert it before (side 0) or after (side 1) its neighbour.
                    auto s=t.sequence;auto w=t.cycle;
                    const int d=s[i];const Vector2 p=w[i];
                    s.erase(s.begin()+long(i));w.erase(w.begin()+long(i));
                    long j=at[size_t(nb)];if(j>long(i))--j;
                    j+=side;
                    if(j<1||j>long(s.size())||(j==long(i)))continue;
                    s.insert(s.begin()+j,d);w.insert(w.begin()+j,p);
                    if(attempt(s,w))break;
                }
            }
            for(size_t i=1;i+1<n&&!improved&&elapsed()<until;++i)for(const int nb:nearest[size_t(t.sequence[i-1])]) {
                const long j=at[size_t(nb)];
                if(j<=long(i)||improved)continue;
                // Reverse i..j: disk i-1 is then followed by its neighbour.
                auto s=t.sequence;auto w=t.cycle;
                std::reverse(s.begin()+long(i),s.begin()+j+1);std::reverse(w.begin()+long(i),w.begin()+j+1);
                attempt(s,w);
            }
        }
        return t;
    }
    // ---- Full-order tours ----------------------------------------------
    // A CETSP tour is also a tour visiting every disk in some order, one
    // contact each: a disk the cycle crosses anyway is a straight passage at
    // no cost. A sequence's tour becomes one by putting each other disk where
    // the cycle meets it, in order along the cycle (same length).
    Tour expand(const Tour &t) const {
        const size_t k=t.sequence.size();
        struct Visit {size_t link;double along;int disk;Vector2 point;};
        std::vector<Visit> visits;
        std::vector<bool> member(all.size(),false);for(int i:t.sequence)member[size_t(i)]=true;
        for(size_t l=0;l<k;++l)visits.push_back({l,-1.0,t.sequence[l],t.cycle[l]});
        for(size_t j=1;j<all.size();++j) {
            if(member[j])continue;
            size_t best=0;double dist=INFINITY,along=0;Vector2 point{};
            for(size_t l=0;l<k;++l) {
                const Vector2 a=t.cycle[l],e=t.cycle[(l+1)%k]-a;const double ee=e.dot(e);
                const double u=ee>0?std::clamp((all[j].center-a).dot(e)/ee,0.0,1.0):0.0;
                const Vector2 q=a+e*u,d=q-all[j].center;
                if(d.dot(d)<dist){dist=d.dot(d);best=l;along=u;point=q;}
            }
            visits.push_back({best,along,int(j),point});
        }
        std::stable_sort(visits.begin(),visits.end(),[](const Visit &x,const Visit &y){return x.link<y.link||(x.link==y.link&&x.along<y.along);});
        Tour full;full.length=t.length;
        for(const auto &v:visits){full.sequence.push_back(v.disk);full.cycle.push_back(v.point);}
        return full;
    }
    // Local search on a full order: 2-opt and Or-opt (blocks of 1 to 3, either
    // way round) toward each disk's nearest disks, screened with the contacts
    // fixed; a move that shortens the fixed-contact cycle is applied and its
    // contacts re-optimized by the oracle (warm), which can only shorten it.
    // With `exact` false the contacts stay fixed (a tour of fixed points).
    Tour full_search(Tour t,double until,bool exact=true) {
        if(nearest.empty())prepare_neighbours();
        const size_t n=t.sequence.size();
        if(n<5)return t;
        std::vector<long> at(all.size());
        auto d=[&](size_t i,size_t j){const Vector2 v=t.cycle[i%n]-t.cycle[j%n];return std::sqrt(v.dot(v));};
        auto fixed=[&](const std::vector<Vector2> &w){double s=0;for(size_t i=0;i<n;++i){const Vector2 v=w[(i+1)%n]-w[i];s+=std::sqrt(v.dot(v));}return s;};
        auto apply=[&](std::vector<int> s,std::vector<Vector2> w) {
            const Tour c=exact?evaluate(s,w):Tour{s,w,fixed(w)};
            if(c.length<t.length*(1-1e-12)){t=c;return true;}
            return false;
        };
        if(!exact)t.length=fixed(t.cycle);
        for(bool improved=true;improved&&elapsed()<until;) {
            improved=false;
            for(size_t i=0;i<n;++i)at[size_t(t.sequence[i])]=long(i);
            const double eps=1e-12*t.length;
            // 2-opt: links (i,i+1) and (j,j+1) become (i,j) and (i+1,j+1).
            for(size_t i=0;i+1<n&&!improved;++i)for(const int nb:nearest[size_t(t.sequence[i])]) {
                size_t j=size_t(at[size_t(nb)]);
                size_t lo=std::min(i,j),hi=std::max(i,j);
                if(hi<=lo+1)continue;
                const double delta=d(lo,hi)+d(lo+1,hi+1)-d(lo,lo+1)-d(hi,hi+1);
                if(!(delta<-eps))continue;
                auto s=t.sequence;auto w=t.cycle;
                std::reverse(s.begin()+long(lo)+1,s.begin()+long(hi)+1);std::reverse(w.begin()+long(lo)+1,w.begin()+long(hi)+1);
                if(apply(s,w)){improved=true;break;}
            }
            // Or-opt: the block [b, e] (not the depot) between j and j+1.
            for(size_t len=1;len<=3&&!improved;++len)for(size_t b=1;b+len-1<n&&!improved&&elapsed()<until;++b) {
                const size_t e=b+len-1;
                const double removal=d(b-1,b)+d(e,e+1)-d(b-1,e+1);
                for(const int nb:nearest[size_t(t.sequence[b])]) {
                    const size_t j=size_t(at[size_t(nb)]);
                    for(const size_t at_j:{j,(j+n-1)%n}) { // between at_j and at_j+1
                        if(at_j+1>=b&&at_j<=e)continue;
                        const double plain=d(at_j,b)+d(e,at_j+1)-d(at_j,at_j+1);
                        const double flipped=d(at_j,e)+d(b,at_j+1)-d(at_j,at_j+1);
                        const bool flip=flipped<plain;
                        if(!(std::min(plain,flipped)-removal<-eps))continue;
                        std::vector<int> block(t.sequence.begin()+long(b),t.sequence.begin()+long(e)+1);
                        std::vector<Vector2> wb(t.cycle.begin()+long(b),t.cycle.begin()+long(e)+1);
                        if(flip){std::reverse(block.begin(),block.end());std::reverse(wb.begin(),wb.end());}
                        auto s=t.sequence;auto w=t.cycle;
                        s.erase(s.begin()+long(b),s.begin()+long(e)+1);w.erase(w.begin()+long(b),w.begin()+long(e)+1);
                        const size_t to=at_j<b?at_j+1:at_j+1-len;
                        s.insert(s.begin()+long(to),block.begin(),block.end());w.insert(w.begin()+long(to),wb.begin(),wb.end());
                        if(apply(s,w)){improved=true;break;}
                    }
                    if(improved)break;
                }
            }
        }
        return t;
    }
    // Iterated local search on full orders: a random double bridge, then the
    // search; a tour within eta of the best is accepted, eta falling from 1%
    // by 5% every 10 iterations without a new best and reset below 1e-4.
    // It stops after max(4 per disk, 1000) iterations without a new best.
    Tour full_ils(Tour best,double until) {
        std::mt19937 rng(20261026);
        Tour current=best=full_search(best,until);offer(best);
        double eta=0.01;size_t stale=0;
        const size_t n=best.sequence.size();
        while(n>=8&&elapsed()<until) {
            size_t cut[3];
            for(auto &c:cut)c=1+rng()%(n-1);
            std::sort(cut,cut+3);
            if(cut[0]==cut[1]||cut[1]==cut[2])continue;
            auto s=current.sequence;auto w=current.cycle;
            std::rotate(s.begin()+long(cut[0]),s.begin()+long(cut[1]),s.begin()+long(cut[2]));
            std::rotate(w.begin()+long(cut[0]),w.begin()+long(cut[1]),w.begin()+long(cut[2]));
            Tour c=evaluate(s,w);
            if(!std::isfinite(c.length))continue;
            c=full_search(c,until);
            if(c.length<best.length*(1-1e-12)){best=c;offer(best);stale=0;}
            else if(++stale%10==0)eta*=0.95;
            if(stale>=std::max<size_t>(4*n,1000))break;
            if(c.length<best.length*(1+eta))current=c;
            if(eta<1e-4){eta=0.01;current=best;}
        }
        return best;
    }
    // A tour of the centers: nearest neighbour from the depot, then the full
    // order's local search with the centers as fixed points, and the oracle
    // for its contacts.
    Tour center_tour(double until) {
        const size_t n=all.size();
        std::vector<int> order{0};std::vector<bool> used(n,false);used[0]=true;
        for(size_t step=1;step<n;++step) {
            const Vector2 here=all[size_t(order.back())].center;
            int next=-1;double best=INFINITY;
            for(size_t j=0;j<n;++j)if(!used[j]){const Vector2 v=all[j].center-here;if(v.dot(v)<best){best=v.dot(v);next=int(j);}}
            used[size_t(next)]=true;order.push_back(next);
        }
        std::vector<Vector2> centers;for(int i:order)centers.push_back(all[size_t(i)].center);
        const Tour t=full_search(Tour{order,centers,INFINITY},until,false);
        return evaluate(t.sequence,t.cycle);
    }
    // The initial tour: the centers' tour, and (unless time runs out) cheapest
    // insertion with local search on sequences, expanded to a full order;
    // then iterated local search on full orders from the shorter.
    void heuristic(double budget) {
        const double until=std::min(deadline,elapsed()+budget);
        Tour start=center_tour(until);offer(start);
        Tour t=repair({0},{all[0].center},until);
        if(std::isfinite(t.length)) {
            t=local_search(t,until);offer(t);
            if(t.length<start.length)start=expand(t);
        }
        if(std::isfinite(start.length))full_ils(start,until);
    }
    std::vector<Entry> tree;
    std::vector<int> sequence_of(std::uint32_t id) const {
        std::vector<Entry> path;
        for(std::uint32_t at=id;at!=0;at=tree[at].parent)path.push_back(tree[at]);
        std::vector<int> sequence{0};
        for(auto it=path.rbegin();it!=path.rend();++it)sequence.insert(sequence.begin()+it->position,int(it->disk));
        return sequence;
    }
    double run(double budget) {
        heuristic(budget);heuristic_seconds=elapsed();
        tree.assign(1,Entry{0,0,0});
        std::priority_queue<Open,std::vector<Open>,ByBound> queue;
        queue.push({0.0,0});
        double global=0,feasible_floor=INFINITY;
        struct Child {double lower;std::uint16_t position;bool feasible;};
        std::vector<Child> chosen,children;
        while(!queue.empty()) {
            if(elapsed()>deadline){global=std::min(queue.top().lower,upper*(1-gap));break;}
            const Open open=queue.top();queue.pop();
            if(open.lower>=upper*(1-gap))continue;
            ++nodes;
            const auto sequence=sequence_of(open.id);
            const auto r=oracle(sequence,{},INFINITY);
            const double lower=std::max(open.lower,r.lower_bound);
            if(lower>=upper*(1-gap))continue;
            const auto ex=excess(all,sequence,r.contacts);
            std::vector<int> uncovered;
            for(size_t j=1;j<all.size();++j)if(ex[j]>kCovered*std::max(1.0,all[j].radius))uncovered.push_back(int(j));
            // Feasible: no refinement beats OPT(sequence) >= its L. If the
            // oracle left a gap there, that L still bounds the optimum.
            if(uncovered.empty()){consider(sequence,r);feasible_floor=std::min(feasible_floor,lower);continue;}
            // Branch on the farthest uncovered disk, or (strong branching) on
            // the one of the `strong` farthest whose weakest child is
            // strongest; every candidate's children bound the node.
            const size_t candidates=std::min(strong,uncovered.size());
            std::partial_sort(uncovered.begin(),uncovered.begin()+long(candidates),uncovered.end(),[&](int a,int b){return ex[size_t(a)]>ex[size_t(b)];});
            const size_t k=sequence.size();
            // With the depot first, [0 a ... ] and its reverse are the same cycle:
            // inserting the second disk keeps one orientation only.
            const size_t last=k==2?1:k;
            double strongest=-INFINITY;int branched=-1;chosen.clear();
            for(size_t q=0;q<candidates;++q) {
                const int far=uncovered[q];
                children.clear();double weakest=INFINITY;
                for(size_t p=1;p<=last;++p) {
                    auto child=sequence;child.insert(child.begin()+long(p),far);
                    const auto c=oracle(child,child_warm(r.contacts,p,all[size_t(far)]),upper*(1-gap));
                    if(c.lower_bound>=upper*(1-gap))continue;
                    const bool feasible=consider(child,c)<0;
                    const double bound=std::max(lower,c.lower_bound);
                    if(bound>=upper*(1-gap))continue;
                    weakest=std::min(weakest,bound);
                    children.push_back({bound,std::uint16_t(p),feasible});
                }
                if(weakest>strongest){strongest=weakest;chosen.swap(children);branched=far;}
                if(chosen.empty())break; // every child pruned: so is the node
            }
            // A feasible child needs no node: its subtree is its own sequence.
            for(const auto &child:chosen) {
                if(child.lower>=upper*(1-gap))continue;
                if(child.feasible){feasible_floor=std::min(feasible_floor,child.lower);continue;}
                tree.push_back({open.id,child.position,std::uint16_t(branched)});
                queue.push({child.lower,std::uint32_t(tree.size()-1)});
            }
        }
        if(queue.empty())global=upper*(1-gap);
        return std::min({global,feasible_floor,upper});
    }
};
}

int main(int argc,char **argv) {
    if(argc<2){std::cerr<<"usage: tpp-convex-cetsp INSTANCE.txt [SECONDS] [RELATIVE_GAP] [TOUR.json|-] [STRONG] [HEURISTIC_SECONDS]\n";return 1;}
    const auto inst=read_mennell(argv[1]);
    const double seconds=argc>2?std::stod(argv[2]):60.0;
    Solver s{inst.disks,argc>3?std::stod(argv[3]):1e-6,seconds,Clock::now()};
    if(argc>5)s.strong=std::max<size_t>(1,std::stoul(argv[5]));
    // The initial tour: 5% of the time unless given (its local searches
    // stop on their own when they stagnate).
    const double lower=s.run(argc>6?std::stod(argv[6]):0.05*seconds);
    std::cout<<std::setprecision(12)<<"{\"instance\":\""<<argv[1]<<"\",\"disks\":"<<inst.disks.size()-1
        <<",\"lower_bound\":"<<lower<<",\"upper_bound\":"<<s.upper<<",\"gap\":"<<(s.upper-lower)/s.upper
        <<",\"closed\":"<<(s.upper-lower<=s.gap*s.upper*1.0000001?"true":"false")
        <<",\"seconds\":"<<s.elapsed()<<",\"heuristic_seconds\":"<<s.heuristic_seconds<<",\"heuristic_tours\":"<<s.improvements
        <<",\"nodes\":"<<s.nodes<<",\"oracle_calls\":"<<s.oracle_calls
        <<",\"oracle_seconds\":"<<s.oracle_seconds<<",\"newton_iterations\":"<<s.newton<<",\"construction_closed\":"<<s.constructed
        <<",\"tree_nodes\":"<<s.tree.size()<<",\"strong\":"<<s.strong<<",\"tour_regions\":"<<s.best_sequence.size()<<"}"<<std::endl;
    if(argc>4&&std::string(argv[4])!="-") {
        std::ofstream out(argv[4]);out<<std::setprecision(17)<<"{\"instance\":\""<<argv[1]<<"\",\"upper_bound\":"<<s.upper<<",\"sequence\":[";
        for(size_t i=0;i<s.best_sequence.size();++i)out<<(i?",":"")<<s.best_sequence[i];
        out<<"],\"contacts\":[";
        for(size_t i=0;i<s.best_cycle.size();++i)out<<(i?",":"")<<"["<<s.best_cycle[i].x<<","<<s.best_cycle[i].y<<"]";
        out<<"]}\n";
    }
}
