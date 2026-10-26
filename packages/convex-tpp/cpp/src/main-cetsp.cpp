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
// usage: tpp-convex-cetsp INSTANCE.txt [SECONDS] [RELATIVE_GAP] [TOUR.json|-]
//        [--strong K] [--heuristic SECONDS] [--mode eager|screen|lazy]
// One JSON line on stdout; TOUR.json, if given, receives the best tour
// (sequence and contacts) for an independent check. --strong (default 1) is
// the number of branching candidates; --heuristic the initial tour's budget
// (default 5% of the time); --mode how children are bounded (default lazy);
// --seed the iterated local search's; --threads the workers (independent
// iterated local searches, then nodes expanded in parallel); --radius every
// disk's, for Behdani and Smith's files (points only, the depot first);
// --max-memory a ceiling (GB) on the peak resident size: the B&B stops there
// with the bound it has, as at the time limit.
#include "tpp_convex.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <csignal>
#include <barrier>
#include <deque>
#include <memory>
#include <sys/mman.h>
#include <type_traits>
#include <thread>
#ifdef __APPLE__
#include <malloc/malloc.h>
#include <mach/mach.h>
#else
#include <unistd.h>
#endif
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
// Mennell's format ("x y z radius" lines, "// Depot is x, y"), or, given a
// radius, Behdani and Smith's ("x y" lines, the depot first).
Instance read_instance(const char *path,double radius) {
    std::ifstream in(path);std::string line;Instance inst;Vector2 depot{};bool has_depot=false;
    if(!in)throw std::runtime_error("cannot read the instance");
    // "//Depot is x, y, z" (geometric files) or "//Depot: x, y, z" (TSPLIB, team).
    const std::regex depot_line(R"(//\s*Depot(?:\s+is|:)\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+))");
    Disks targets;std::vector<Vector2> points;
    while(std::getline(in,line)) {
        std::smatch m;
        if(std::regex_search(line,m,depot_line)){depot={std::stod(m[1]),std::stod(m[2])};has_depot=true;continue;}
        if(line.rfind("//",0)==0)continue;
        std::istringstream s(line);double x,y,z,r;
        if(!(s>>x>>y))continue;
        points.push_back({x,y});
        if(s>>z>>r)targets.push_back({{x,y},r});
    }
    if(has_depot) {
        inst.disks.push_back({depot,0});
        inst.disks.insert(inst.disks.end(),targets.begin(),targets.end());
    } else {
        if(!(radius>0)||points.empty())throw std::runtime_error("no depot line (Behdani's format needs --radius)");
        inst.disks.push_back({points[0],0});
        for(size_t i=1;i<points.size();++i)inst.disks.push_back({points[i],radius});
    }
    return inst;
}
// The point of the disk nearest the segment ab (its detour point when the
// segment meets it): a warm start, cheaper than Alhazen's point.
Vector2 nearest_to_link(Vector2 a,Vector2 b,const tpp::ConvexDisk &d) {
    const Vector2 e=b-a;const double ee=e.dot(e);
    const double t=ee>0?std::clamp((d.center-a).dot(e)/ee,0.0,1.0):0.0;
    const Vector2 foot=a+e*t,r=foot-d.center;const double rr=r.dot(r);
    if(rr<=d.radius*d.radius)return foot;
    return d.center+r*(d.radius/std::sqrt(rr));
}
// How far each disk is from the closed polyline through the contacts, minus
// its radius (<= 0: the tour meets it). Disks of the sequence are met at their
// contacts, which the oracle proved inside them; the others are decided with
// a margin, so a borderline disk is inserted rather than assumed covered.
std::vector<double> excess(const Disks &disks,const std::vector<int> &sequence,const std::vector<Vector2> &cycle) {
    std::vector<double> out(disks.size(),INFINITY);
    std::vector<bool> member(disks.size(),false);for(int i:sequence)member[size_t(i)]=true;
    // Squared distances to each link, one square root per disk.
    const size_t k=cycle.size();
    std::vector<Vector2> edge(k);std::vector<double> inverse(k);
    for(size_t l=0;l<k;++l){edge[l]=cycle[(l+1)%k]-cycle[l];const double ee=edge[l].dot(edge[l]);inverse[l]=ee>0?1/ee:0;}
    for(size_t j=0;j<disks.size();++j) {
        if(member[j]){out[j]=-INFINITY;continue;}
        double best=INFINITY;
        for(size_t l=0;l<k;++l) {
            const Vector2 f=disks[j].center-cycle[l];
            const double t=std::clamp(f.dot(edge[l])*inverse[l],0.0,1.0);
            const Vector2 g=f-edge[l]*t;
            best=std::min(best,g.dot(g));
        }
        out[j]=std::sqrt(best)-disks[j].radius;
    }
    return out;
}
constexpr double kCovered=-1e-9; // relative to the radius scale; conservative

// The search keeps a family per expanded node: the disk its children insert,
// and per child its position and bound, sorted by bound, with the next one
// to expand. A child costs 6 bytes: its bound in single precision, rounded
// down (still a lower bound), and its position. The queue holds families,
// not children: a family while it has children left, keyed by the next
// one's bound. A family records its own node's insertion, so a node's
// sequence is rebuilt from the root through families alone, each disk
// entering at its point nearest the link between its neighbours there; its
// cycle is then solved again, warm from those points.
// The process's resident size now, in bytes (0 if unknown).
double resident_bytes() {
#ifdef __APPLE__
    mach_task_basic_info_data_t info{};mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
    if(task_info(mach_task_self(),MACH_TASK_BASIC_INFO,task_info_t(&info),&count)!=KERN_SUCCESS)return 0;
    return double(info.resident_size);
#else
    std::ifstream statm("/proc/self/statm");long pages=0,resident=0;
    if(!(statm>>pages>>resident))return 0;
    return double(resident)*double(sysconf(_SC_PAGESIZE));
#endif
}
// Set by SIGINT, SIGTERM or SIGHUP: the run stops as at its time limit and
// still reports what it has (marked "interrupted").
std::atomic<bool> interrupted{false};
extern "C" void on_stop_signal(int){interrupted.store(true);}
constexpr std::uint32_t kNone=0xffffffffu;
struct Family {std::uint32_t parent,first;std::uint16_t own_position,own_disk,disk,count,next;};
struct Open {float lower;std::uint32_t family;};
float rounded_down(double x) {
    float f=float(x);
    if(double(f)>x)f=std::nextafter(f,-INFINITY);
    return f;
}
struct ByBound {bool operator()(const Open &a,const Open &b) const {return a.lower>b.lower;}};
// Room for `extra` more elements, growing by a quarter rather than doubling,
// so that a large vector does not briefly need three times its size.
template<class T> void make_room(std::vector<T> &v,size_t extra) {
    if(v.size()+extra>v.capacity())v.reserve(std::max(v.size()+extra,v.capacity()+v.capacity()/4));
}
// An array of plain values in fixed chunks of 2^20 elements, each mapped
// from the system and unmapped when freed: it grows without reallocating,
// and freed chunks leave the resident size at once (the allocator would keep
// large freed blocks resident in its cache).
template<class T> class Chunked {
    static_assert(std::is_trivially_copyable_v<T>);
    struct Unmap {void operator()(T *p) const {munmap(p,kChunk*sizeof(T));}};
    std::vector<std::unique_ptr<T,Unmap>> chunks_;size_t size_=0;
public:
    static constexpr size_t kBits=20,kChunk=size_t(1)<<kBits;
    T &operator[](size_t i){return chunks_[i>>kBits].get()[i&(kChunk-1)];}
    const T &operator[](size_t i) const {return chunks_[i>>kBits].get()[i&(kChunk-1)];}
    void push_back(const T &x) {
        if(size_==chunks_.size()*kChunk) {
            void *m=mmap(nullptr,kChunk*sizeof(T),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
            if(m==MAP_FAILED)throw std::bad_alloc();
            chunks_.emplace_back(static_cast<T*>(m));
        }
        (*this)[size_++]=x;
    }
    size_t size() const {return size_;}
    size_t bytes() const {return chunks_.size()*kChunk*sizeof(T);}
    void clear(){chunks_.clear();size_=0;}
    void swap(Chunked &o){chunks_.swap(o.chunks_);std::swap(size_,o.size_);}
};

struct Solver {
    const Disks &all;double gap;double deadline;Clock::time_point began;size_t strong=1;
    // Children: all solved when created (eager), screened by insertion bounds
    // first (screen), or queued with those bounds and solved when expanded (lazy).
    enum class Mode {Eager,Screen,Lazy} mode=Mode::Lazy;
    size_t ils_iterations=0;
    unsigned seed=20261026; // of the iterated local search
    size_t threads=1; // expanding nodes in parallel
    double max_memory=0; // bytes; 0: no ceiling
    std::string stopped="closed"; // or "time", "memory"
    double upper=INFINITY;std::vector<int> best_sequence;std::vector<Vector2> best_cycle;
    struct Counters {size_t oracle_calls=0,newton=0,constructed=0,screened=0;double oracle_seconds=0;};
    Counters counted; // the main thread's; workers keep their own and add them after each batch
    size_t nodes=0,improvements=0;double heuristic_seconds=0;
    // Seconds since the start; past every deadline once interrupted, so that
    // all searches stop as at their time limits.
    double elapsed() const {return wall()+(interrupted.load(std::memory_order_relaxed)?1e9:0);}
    double wall() const {return std::chrono::duration<double>(Clock::now()-began).count();}
    tpp::DiskCycleResult oracle(const std::vector<int> &sequence,const std::vector<Vector2> &warm,double cutoff,Counters &c,double bound) const {
        Disks d;for(int i:sequence)d.push_back(all[size_t(i)]);
        tpp::DiskCycleOptions o;o.max_gap=std::isfinite(bound)?0.25*gap*bound:1e-9;o.lower_bound_cutoff=cutoff;
        if(warm.size()==d.size())o.initial_contacts=warm;
        const auto t0=Clock::now();const auto r=tpp::tpp_convex_solve_cycle_disks(d,o);
        c.oracle_seconds+=std::chrono::duration<double>(Clock::now()-t0).count();++c.oracle_calls;c.newton+=r.newton_iterations;c.constructed+=r.construction_closed;
        return r;
    }
    tpp::DiskCycleResult oracle(const std::vector<int> &sequence,const std::vector<Vector2> &warm,double cutoff) {
        return oracle(sequence,warm,cutoff,counted,upper);
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
    Tour evaluate(const std::vector<int> &sequence,const std::vector<Vector2> &warm,Counters *c=nullptr) {
        const auto r=c?oracle(sequence,warm,INFINITY,*c,upper):oracle(sequence,warm,INFINITY);
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
    // Ruin and recreate on sequences, for tours that turn at few disks: drop
    // a random run of up to 10 disks, or up to 10 random disks, insert the
    // uncovered ones again (cheapest insertion), and search locally. A tour
    // not 0.1% longer than the current one replaces it; after 50 tries
    // without a new best the search restarts from the best. Until `until`.
    Tour subset_ils(Tour best,double until) {
        std::mt19937 rng(seed);
        Tour current=best;size_t stale=0;
        while(elapsed()<until&&current.sequence.size()>=4) {
            auto s=current.sequence;auto w=current.cycle;
            const size_t m=1+rng()%std::min<size_t>(10,(s.size()-1)/2);
            if(rng()%2) {
                const size_t first=1+rng()%(s.size()-m);
                s.erase(s.begin()+long(first),s.begin()+long(first+m));w.erase(w.begin()+long(first),w.begin()+long(first+m));
            } else for(size_t q=0;q<m;++q) {
                const size_t i=1+rng()%(s.size()-1);
                s.erase(s.begin()+long(i));w.erase(w.begin()+long(i));
            }
            Tour c=repair(s,w,until);
            if(!std::isfinite(c.length))break;
            c=local_search(c,until);
            if(c.length<best.length*(1-1e-12)){best=c;stale=0;}
            else if(++stale%50==0)current=best;
            if(c.length<=current.length*1.001)current=c;
        }
        return best;
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
    // way round) toward each disk's nearest disks, with the other contacts
    // fixed. A moved block's contacts are placed again by one Alhazen pass
    // between its new neighbours, so a disk moved to where the cycle already
    // crosses it costs nothing. A move is applied when it shortens the cycle
    // through the contacts, which stays feasible. Only disks near a change
    // are examined again (a work queue, like don't-look bits in TSP local
    // search); `seeds` start it, all disks if empty. At a local optimum the
    // contacts are swept (detour points), then re-optimized and certified by
    // the oracle; the search resumes while either shortens the cycle. With
    // `exact` false the contacts stay fixed (a tour of fixed points).
    Tour full_search(Tour t,double until,bool exact=true,const std::vector<int> &seeds={},Counters *counters=nullptr) {
        if(nearest.empty())prepare_neighbours();
        const size_t n=t.sequence.size();
        if(n<5)return exact?evaluate(t.sequence,t.cycle,counters):t;
        std::vector<long> at(all.size(),-1);
        auto index=[&]{for(size_t i=0;i<n;++i)at[size_t(t.sequence[i])]=long(i);};
        auto d=[&](size_t i,size_t j){const Vector2 v=t.cycle[i%n]-t.cycle[j%n];return std::sqrt(v.dot(v));};
        auto fixed=[&](const std::vector<Vector2> &w){double s=0;for(size_t i=0;i<n;++i){const Vector2 v=w[(i+1)%n]-w[i];s+=std::sqrt(v.dot(v));}return s;};
        std::vector<char> queued(all.size(),0);std::deque<int> work;
        auto wake=[&](int disk){if(!queued[size_t(disk)]){queued[size_t(disk)]=1;work.push_back(disk);}};
        auto wake_at=[&](long position){wake(t.sequence[size_t((position%long(n)+long(n))%long(n))]);};
        auto wake_all=[&]{for(size_t i=0;i<n;++i)wake(t.sequence[i]);};
        // One Gauss-Seidel pass of detour points (the depot stays).
        auto sweep=[&] {
            for(size_t i=1;i<n;++i)t.cycle[i]=tpp::tpp_convex_disk_detour_point(t.cycle[i-1],t.cycle[(i+1)%n],all[size_t(t.sequence[i])]);
            t.length=fixed(t.cycle);
        };
        std::vector<Vector2> placed;
        // The block's contacts between positions `before` and `after`, and the
        // cost of putting it there instead of the link between them.
        auto place=[&](size_t b,size_t e,size_t before,size_t after,bool flip) {
            placed.clear();
            Vector2 from=t.cycle[before%n];
            for(size_t q=0;q<=e-b;++q) {
                const size_t i=flip?e-q:b+q;
                from=tpp::tpp_convex_disk_detour_point(from,t.cycle[after%n],all[size_t(t.sequence[i])]);
                placed.push_back(from);
            }
            double cost=0;Vector2 prev=t.cycle[before%n];
            for(const auto &q:placed){const Vector2 v=q-prev;cost+=std::sqrt(v.dot(v));prev=q;}
            const Vector2 v=t.cycle[after%n]-prev;cost+=std::sqrt(v.dot(v));
            return cost-d(before,after);
        };
        // A lower bound on place() without Alhazen: each step of the path is
        // at least the distance between centers less the radii, and the path
        // at least the link it replaces.
        auto reach=[&](Vector2 p,const tpp::ConvexDisk &disk){const Vector2 v=disk.center-p;return std::max(0.0,std::sqrt(v.dot(v))-disk.radius);};
        auto place_bound=[&](size_t b,size_t e,size_t before,size_t after,bool flip) {
            double path=reach(t.cycle[before%n],all[size_t(t.sequence[flip?e:b])])+reach(t.cycle[after%n],all[size_t(t.sequence[flip?b:e])]);
            for(size_t q=b;q<e;++q) {
                const auto &x=all[size_t(t.sequence[q])],&y=all[size_t(t.sequence[q+1])];const Vector2 v=y.center-x.center;
                path+=std::max(0.0,std::sqrt(v.dot(v))-x.radius-y.radius);
            }
            return std::max(path,d(before,after))-d(before,after);
        };
        // 2-opt with links (p, p+1) and (j, j+1), j a position of a nearest disk.
        auto two_opt=[&](size_t p,double eps) {
            for(const int nb:nearest[size_t(t.sequence[p])]) {
                const size_t j=size_t(at[size_t(nb)]);
                const size_t lo=std::min(p,j),hi=std::max(p,j);
                if(hi<=lo+1||hi+1>n)continue;
                const double delta=d(lo,hi)+d(lo+1,hi+1)-d(lo,lo+1)-d(hi,hi+1);
                if(!(delta<-eps))continue;
                std::reverse(t.sequence.begin()+long(lo)+1,t.sequence.begin()+long(hi)+1);
                std::reverse(t.cycle.begin()+long(lo)+1,t.cycle.begin()+long(hi)+1);
                t.length+=delta;index();
                for(const long q:{long(lo),long(lo)+1,long(hi),long(hi)+1})wake_at(q);
                return true;
            }
            return false;
        };
        // Or-opt of the block [b, e] (not the depot) next to a nearest disk.
        auto or_opt=[&](size_t b,size_t e,double eps) {
            if(b<1||e>=n||e<b)return false;
            const size_t len=e-b+1;
            double removal=d(b-1,b)+d(e,e+1)-d(b-1,e+1);
            for(size_t i=b;i<e;++i)removal+=d(i,i+1);
            for(const int nb:nearest[size_t(t.sequence[b])]) {
                const size_t j=size_t(at[size_t(nb)]);
                for(const size_t at_j:{j,(j+n-1)%n}) { // between at_j and at_j+1
                    if(at_j+1>=b&&at_j<=e)continue;
                    for(const bool flip:{false,true}) {
                        if(!(place_bound(b,e,at_j,at_j+1,flip)-removal<-eps))continue;
                        const double delta=place(b,e,at_j,at_j+1,flip)-removal;
                        if(!(delta<-eps))continue;
                        std::vector<int> block;
                        for(size_t q=0;q<len;++q)block.push_back(t.sequence[flip?e-q:b+q]);
                        const int left=t.sequence[b-1],right=t.sequence[(e+1)%n];
                        t.sequence.erase(t.sequence.begin()+long(b),t.sequence.begin()+long(e)+1);
                        t.cycle.erase(t.cycle.begin()+long(b),t.cycle.begin()+long(e)+1);
                        const size_t to=at_j<b?at_j+1:at_j+1-len;
                        t.sequence.insert(t.sequence.begin()+long(to),block.begin(),block.end());
                        t.cycle.insert(t.cycle.begin()+long(to),placed.begin(),placed.end());
                        t.length+=delta;index();
                        wake(left);wake(right);
                        for(const long q:{long(to)-1,long(to),long(to+len)-1,long(to+len)})wake_at(q);
                        return true;
                    }
                }
            }
            return false;
        };
        t.length=fixed(t.cycle);index();
        if(seeds.empty())wake_all();else for(const int disk:seeds)if(at[size_t(disk)]>=0)wake(disk);
        for(;;) {
            while(!work.empty()&&elapsed()<until) {
                const int disk=work.front();work.pop_front();queued[size_t(disk)]=0;
                const size_t i=size_t(at[size_t(disk)]);
                const double eps=1e-12*t.length;
                bool moved=two_opt(i,eps)||(i>0&&two_opt(i-1,eps));
                for(size_t len=1;len<=3&&!moved;++len)moved=or_opt(i,i+len-1,eps)||(i+1>=len&&or_opt(i+1-len,i,eps));
                if(moved)wake(disk);
            }
            if(!exact)return t;
            // No move left with these contacts: sweep them, and search again
            // if that shortened the cycle noticeably (sweeps converge slowly);
            // else the oracle's contacts, certified.
            const double before=t.length;
            sweep();
            if(t.length<before*(1-1e-7)&&elapsed()<until){wake_all();continue;}
            const Tour c=evaluate(t.sequence,t.cycle,counters);
            const bool shorter=c.length<t.length*(1-1e-9);
            t=c;
            if(!shorter||elapsed()>=until)return t;
            t.length=fixed(t.cycle);wake_all();
        }
    }
    // Iterated local search on full orders: swap two adjacent segments of
    // up to 50 disks each (a local double bridge), search again from the
    // disks around the cuts; a tour within eta of the best is accepted, eta
    // falling from 1% by 5% every 10 iterations without a new best and reset
    // below 1e-4. It stops after max(4 per disk, 200) iterations without a
    // new best.
    // Thread-safe given its own counters: it offers nothing; the caller does.
    Tour full_ils(Tour best,double until,unsigned seed,Counters &counters,size_t &iterations) {
        std::mt19937 rng(seed);
        Tour current=best=full_search(best,until,true,{},&counters);
        double eta=0.01;size_t stale=0;
        const size_t n=best.sequence.size();
        while(n>=8&&elapsed()<until) {
            const size_t first=1+rng()%(n-1),one=1+rng()%std::min<size_t>(50,n/3),two=1+rng()%std::min<size_t>(50,n/3);
            if(first+one+two>n)continue;
            auto s=current.sequence;auto w=current.cycle;
            std::rotate(s.begin()+long(first),s.begin()+long(first+one),s.begin()+long(first+one+two));
            std::rotate(w.begin()+long(first),w.begin()+long(first+one),w.begin()+long(first+one+two));
            std::vector<int> seeds;
            for(const size_t q:{first-1,first,first+two-1,first+two,first+one+two-1,first+one+two})seeds.push_back(s[q%n]);
            Tour c=full_search(Tour{s,w,INFINITY},until,true,seeds,&counters);
            ++iterations;
            if(!std::isfinite(c.length))continue;
            if(c.length<best.length*(1-1e-12)){best=c;stale=0;}
            else if(++stale%10==0)eta*=0.95;
            if(stale>=std::max<size_t>(4*n,200))break;
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
    // The initial tour: the centers' tour, and (within 30% of the budget)
    // cheapest insertion with local search on sequences, then ruin and
    // recreate if the sequence has at most a quarter of the disks, expanded
    // to a full order; then iterated local search on full orders from the
    // shorter, in every thread.
    void heuristic(double budget) {
        const double until=std::min(deadline,elapsed()+budget);
        Tour start=center_tour(until);offer(start);
        const double subset=std::min(until,elapsed()+0.3*budget); // sequences: at most 30%
        Tour t=repair({0},{all[0].center},subset);
        if(std::isfinite(t.length)) {
            t=local_search(t,subset);offer(t);
            // When the tour turns at few disks, sequences are the better
            // search space; otherwise the time goes to full orders.
            if(4*t.sequence.size()<=all.size()){t=subset_ils(t,subset);offer(t);}
            if(t.length<start.length)start=expand(t);
        }
        if(!std::isfinite(start.length))return;
        // Independent iterated local searches, one per thread (seeds
        // seed, seed+1, ...), from the same start; the shortest is kept.
        if(nearest.empty())prepare_neighbours();
        std::vector<Tour> found(threads);std::vector<Counters> work(threads);std::vector<size_t> iterations(threads,0);
        std::vector<std::thread> pool;
        for(size_t t=1;t<threads;++t)pool.emplace_back([&,t]{found[t]=full_ils(start,until,seed+unsigned(t),work[t],iterations[t]);});
        found[0]=full_ils(start,until,seed,work[0],iterations[0]);
        for(auto &t:pool)t.join();
        for(size_t t=0;t<threads;++t) {
            offer(found[t]);ils_iterations+=iterations[t];
            counted.oracle_calls+=work[t].oracle_calls;counted.newton+=work[t].newton;
            counted.constructed+=work[t].constructed;counted.oracle_seconds+=work[t].oracle_seconds;
        }
    }
    Chunked<Family> families;Chunked<float> bounds;Chunked<std::uint16_t> positions;
    std::vector<Open> heap;
    size_t children_created=0,compactions=0,dropped=0;
    double dropped_floor=INFINITY; // children dropped for memory had bounds at least this
    double resident_start=0;       // bytes, when the B&B starts
    double probe_seconds=0;        // the first, short B&B with the centers' tour
    size_t bytes() const {
        return families.bytes()+bounds.bytes()+positions.bytes()+heap.capacity()*sizeof(Open);
    }
    // Child `child` of `family` (the root itself if `family` is kNone).
    std::vector<int> sequence_of(std::uint32_t family,std::uint16_t child,std::vector<Vector2> &warm) const {
        std::vector<std::pair<std::uint16_t,std::uint16_t>> path; // (disk, position), leaf first
        if(family!=kNone) {
            path.push_back({families[family].disk,positions[families[family].first+child]});
            for(std::uint32_t f=family;families[f].parent!=kNone;f=families[f].parent)path.push_back({families[f].own_disk,families[f].own_position});
        }
        std::vector<int> sequence{0};warm.assign(1,all[0].center);
        for(auto it=path.rbegin();it!=path.rend();++it) {
            const size_t p=it->second,k=warm.size();
            const Vector2 point=nearest_to_link(warm[p-1],warm[p%k],all[it->first]);
            sequence.insert(sequence.begin()+long(p),int(it->first));
            warm.insert(warm.begin()+long(p),point);
        }
        return sequence;
    }
    // Keep only what the queue still needs: families with children left (and
    // only those children), and their ancestors (for their insertions). A
    // family's parent is always older, so one pass in order renumbers them.
    void compact() {
        ++compactions;
        std::vector<std::uint32_t> renamed(families.size(),kNone);
        std::vector<char> keep(families.size(),0),open(families.size(),0);
        for(const auto &o:heap)open[o.family]=keep[o.family]=1;
        for(size_t f=families.size();f-->0;)if(keep[f]&&families[f].parent!=kNone)keep[families[f].parent]=1;
        Chunked<Family> kept;Chunked<float> kept_bounds;Chunked<std::uint16_t> kept_positions;
        for(size_t f=0;f<families.size();++f) {
            if(!keep[f])continue;
            Family g=families[f];
            if(g.parent!=kNone)g.parent=renamed[g.parent];
            const std::uint32_t first=std::uint32_t(kept_bounds.size());
            if(open[f])for(size_t c=g.next;c<g.count;++c){kept_bounds.push_back(bounds[g.first+c]);kept_positions.push_back(positions[g.first+c]);}
            g.count=open[f]?std::uint16_t(g.count-g.next):0;g.next=0;g.first=first;
            renamed[f]=std::uint32_t(kept.size());kept.push_back(g);
        }
        for(auto &o:heap)o.family=renamed[o.family];
        families.swap(kept);bounds.swap(kept_bounds);positions.swap(kept_positions);
        std::vector<Open>(heap).swap(heap); // its capacity back to its size
        kept.clear();kept_bounds.clear();kept_positions.clear();
#ifdef __APPLE__
        malloc_zone_pressure_relief(nullptr,0); // freed chunks back to the system now
#endif
    }
    // Over budget even when compact: drop about a third of the children left,
    // those with the largest bounds. Every dropped child's bound is at least
    // the threshold, which therefore stays a lower bound for their subtrees:
    // the search can no longer prove more than it (dropped_floor).
    void drop() {
        double low=INFINITY,high=-INFINITY;size_t total=0;
        for(const auto &o:heap){const auto &f=families[o.family];low=std::min(low,double(bounds[f.first+f.next]));high=std::max(high,double(bounds[f.first+f.count-1]));total+=f.count-f.next;}
        if(!(high>low))return;
        std::vector<size_t> histogram(1024,0);
        auto bin=[&](double b){return std::min<size_t>(1023,size_t((b-low)/(high-low)*1024));};
        for(const auto &o:heap){const auto &f=families[o.family];for(size_t c=f.next;c<f.count;++c)++histogram[bin(bounds[f.first+c])];}
        size_t above=0,cut=1023;
        for(;cut>0&&above+histogram[cut]<total/3;--cut)above+=histogram[cut];
        const float threshold=rounded_down(low+(high-low)*double(cut)/1024);
        std::vector<Open> left;
        for(const auto &o:heap) {
            auto &f=families[o.family];
            size_t end=f.next;
            while(end<f.count&&bounds[f.first+end]<threshold)++end;
            dropped+=f.count-end;f.count=std::uint16_t(end);
            if(f.next<f.count)left.push_back(o);
        }
        dropped_floor=std::min(dropped_floor,double(threshold));
        heap.swap(left);std::make_heap(heap.begin(),heap.end(),ByBound{});
        compact();
    }
    // One node's expansion, independent of the others in its batch: its
    // cycle solved again, then its children (bounds by insertion, or solved)
    // and the feasible tours met on the way. `cutoff` is the batch's.
    struct Child {double lower;std::uint16_t position;};
    struct Expansion {
        std::uint32_t family=kNone;std::uint16_t child=0;double bound=0;
        bool pruned=false;double floor=INFINITY;int branched=-1;
        std::vector<Child> children;
        std::vector<Tour> tours; // feasible, to offer
        Counters counted;
    };
    void expand(Expansion &x,double cutoff,double incumbent) const {
        std::vector<Vector2> warm;
        const auto sequence=sequence_of(x.family,x.child,warm);
        const auto r=oracle(sequence,warm,INFINITY,x.counted,incumbent);
        const double lower=std::max(x.bound,r.lower_bound);
        if(lower>=cutoff){x.pruned=true;return;}
        const auto ex=excess(all,sequence,r.contacts);
        std::vector<int> uncovered;
        for(size_t j=1;j<all.size();++j)if(ex[j]>kCovered*std::max(1.0,all[j].radius))uncovered.push_back(int(j));
        // Feasible: no refinement beats OPT(sequence) >= its L. If the oracle
        // left a gap there, that L still bounds the optimum.
        if(uncovered.empty()){x.tours.push_back({sequence,r.contacts,r.upper_bound});x.floor=lower;return;}
        // Branch on the farthest uncovered disk, or (strong branching) on
        // the one of the `strong` farthest whose weakest child is strongest;
        // every candidate's children bound the node.
        const size_t candidates=std::min(strong,uncovered.size());
        std::partial_sort(uncovered.begin(),uncovered.begin()+long(candidates),uncovered.end(),[&](int a,int b){return ex[size_t(a)]>ex[size_t(b)];});
        const size_t k=sequence.size();
        // With the depot first, [0 a ... ] and its reverse are the same cycle:
        // inserting the second disk keeps one orientation only.
        const size_t last=k==2?1:k;
        double strongest=-INFINITY;
        std::vector<Child> children;std::vector<Tour> tours;double floor=INFINITY;
        Disks here;for(int i:sequence)here.push_back(all[size_t(i)]);
        for(size_t q=0;q<candidates;++q) {
            const int far=uncovered[q];
            children.clear();tours.clear();floor=INFINITY;double weakest=INFINITY;
            // Every position's insertion bound first (O(k) for all): a child
            // it prunes costs no oracle call; lazily, the others wait in the
            // queue with it and are solved when expanded.
            std::vector<double> inserted;
            if(mode!=Mode::Eager)inserted=tpp::tpp_convex_disk_insertion_bounds(here,r.contacts,all[size_t(far)]);
            for(size_t p=1;p<=last;++p) {
                double bound=std::max(lower,inserted.empty()?0.0:inserted[p-1]);
                if(bound>=cutoff){++x.counted.screened;continue;}
                if(mode==Mode::Lazy){weakest=std::min(weakest,bound);children.push_back({bound,std::uint16_t(p)});continue;}
                auto child=sequence;child.insert(child.begin()+long(p),far);
                const auto c=oracle(child,child_warm(r.contacts,p,all[size_t(far)]),cutoff,x.counted,incumbent);
                bound=std::max(bound,c.lower_bound);
                if(bound>=cutoff)continue;
                weakest=std::min(weakest,bound);
                // A feasible child needs no node: its subtree is its own sequence.
                if(covers(child,c.contacts)){tours.push_back({child,c.contacts,c.upper_bound});floor=std::min(floor,bound);continue;}
                children.push_back({bound,std::uint16_t(p)});
            }
            if(weakest>strongest){strongest=weakest;x.children.swap(children);x.tours.swap(tours);x.floor=floor;x.branched=far;}
            if(x.children.empty()&&x.tours.empty())break; // every child pruned: so is the node
        }
    }
    // A small instance closes before an elaborate tour pays off: with only
    // the centers' tour as incumbent, the B&B gets a moment (a second, at
    // most 2% of the time). If that does not close it, the initial tour gets
    // its budget and the B&B starts over.
    double run(double budget) {
        const double t0=wall();
        if(nearest.empty())prepare_neighbours();
        offer(center_tour(deadline));
        const double probe_until=std::min(deadline,elapsed()+std::min(1.0,0.02*deadline));
        double lower=search(probe_until);
        probe_seconds=wall()-t0;
        if(upper-lower<=gap*upper*1.0000001||elapsed()>=deadline)return lower;
        const double t1=wall();
        heuristic(budget);heuristic_seconds=wall()-t1;
        return search(deadline);
    }
    double search(double until) {
        families.clear();bounds.clear();positions.clear();heap.clear();
        dropped_floor=INFINITY;stopped="closed";
        double global=0,feasible_floor=INFINITY;
        // The search structures' share of the memory ceiling: 40% of what
        // the process does not hold already, since a compaction briefly holds
        // the old and the new structures at once.
        resident_start=resident_bytes();
        const double room=max_memory>0?std::max(0.1*max_memory,0.4*(max_memory-resident_start)):INFINITY;
        // A batch of the best open nodes (4 per worker, so that uneven
        // expansions balance) is expanded by `threads` workers; its results
        // are merged in order by this thread. The root comes first, alone.
        std::vector<Expansion> batch(1);batch[0].family=kNone;
        std::atomic<size_t> next{0};
        double cutoff=0,incumbent=0;
        auto work=[&]{for(size_t i;(i=next.fetch_add(1))<batch.size();)expand(batch[i],cutoff,incumbent);};
        std::barrier<> start{std::ptrdiff_t(threads)},done{std::ptrdiff_t(threads)};
        std::atomic<bool> stop{false};
        std::vector<std::thread> pool;
        for(size_t t=1;t<threads;++t)pool.emplace_back([&]{for(;;){start.arrive_and_wait();if(stop)return;work();done.arrive_and_wait();}});
        for(size_t round=0;;++round) {
            cutoff=std::min(upper*(1-gap),dropped_floor);incumbent=upper;
            if(round>0) {
                if(heap.empty())break;
                if(elapsed()>until){global=std::min(double(heap.front().lower),cutoff);stopped=interrupted?"interrupted":"time";break;}
                if(bytes()>room) {
                    compact();
                    if(bytes()>0.6*room)drop();
                    cutoff=std::min(upper*(1-gap),dropped_floor);
                }
                // A hard ceiling on the resident size, should the estimate
                // above miss something.
                if(max_memory>0&&round%4096==0&&resident_bytes()>1.5*max_memory&&!heap.empty()) {
                    global=std::min(double(heap.front().lower),cutoff);stopped="memory";break;
                }
                batch.clear();
                while(!heap.empty()&&batch.size()<(threads>1?4*threads:1)) {
                    std::pop_heap(heap.begin(),heap.end(),ByBound{});
                    const Open o=heap.back();heap.pop_back();
                    if(o.lower>=cutoff)continue; // and so are its family's later children
                    auto &f=families[o.family];
                    const std::uint16_t c=f.next++;
                    if(f.next<f.count&&bounds[f.first+f.next]<cutoff){heap.push_back({bounds[f.first+f.next],o.family});std::push_heap(heap.begin(),heap.end(),ByBound{});}
                    Expansion x;x.family=o.family;x.child=c;x.bound=bounds[f.first+c];batch.push_back(std::move(x));
                }
                if(batch.empty())continue;
            }
            next=0;
            if(threads>1&&batch.size()>1){start.arrive_and_wait();work();done.arrive_and_wait();}
            else work();
            for(auto &x:batch) {
                ++nodes;
                counted.oracle_calls+=x.counted.oracle_calls;counted.newton+=x.counted.newton;counted.constructed+=x.counted.constructed;
                counted.screened+=x.counted.screened;counted.oracle_seconds+=x.counted.oracle_seconds;
                for(const auto &t:x.tours)if(t.length<upper){upper=t.length;best_sequence=t.sequence;best_cycle=t.cycle;}
                feasible_floor=std::min(feasible_floor,x.floor);
                if(x.pruned)continue;
                const double limit=std::min(upper*(1-gap),dropped_floor);
                std::sort(x.children.begin(),x.children.end(),[](const Child &a,const Child &b){return a.lower<b.lower;});
                size_t n=0;while(n<x.children.size()&&x.children[n].lower<limit)++n;
                if(!n)continue;
                Family g{kNone,std::uint32_t(bounds.size()),0,0,std::uint16_t(x.branched),std::uint16_t(n),0};
                if(x.family!=kNone){g.parent=x.family;g.own_position=positions[families[x.family].first+x.child];g.own_disk=families[x.family].disk;}
                make_room(heap,1);
                for(size_t i=0;i<n;++i){bounds.push_back(rounded_down(x.children[i].lower));positions.push_back(x.children[i].position);}
                children_created+=n;
                families.push_back(g);
                heap.push_back({bounds[g.first],std::uint32_t(families.size()-1)});std::push_heap(heap.begin(),heap.end(),ByBound{});
            }
            if(round==0&&heap.empty())break;
        }
        if(threads>1){stop=true;start.arrive_and_wait();for(auto &t:pool)t.join();}
        if(heap.empty()){global=std::min(upper*(1-gap),dropped_floor);if(stopped=="closed"&&dropped_floor<upper*(1-gap))stopped="dropped";}
        return std::min({global,feasible_floor,upper});
    }
};
}

int main(int argc,char **argv) {
    const char *usage="usage: tpp-convex-cetsp INSTANCE.txt [SECONDS] [RELATIVE_GAP] [TOUR.json|-] [--strong K] [--heuristic SECONDS] [--mode eager|screen|lazy] [--seed S] [--threads T] [--radius R] [--max-memory GB]\n";
    std::vector<std::string> positional;double heuristic=-1,radius=0,max_memory=0;size_t strong=1,threads=1;unsigned seed=20261026;std::string mode="lazy";
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];
        if((a=="--strong"||a=="--heuristic"||a=="--mode"||a=="--seed"||a=="--threads"||a=="--radius"||a=="--max-memory")&&i+1<argc) {
            const std::string v=argv[++i];
            if(a=="--max-memory")max_memory=std::stod(v)*1e9;
            else if(a=="--radius")radius=std::stod(v);
            else if(a=="--seed")seed=unsigned(std::stoul(v));
            else if(a=="--threads")threads=std::max<size_t>(1,std::stoul(v));
            else if(a=="--strong")strong=std::max<size_t>(1,std::stoul(v));
            else if(a=="--heuristic")heuristic=std::stod(v);
            else mode=v;
        } else if(a.rfind("--",0)==0){std::cerr<<usage;return 1;}
        else positional.push_back(a);
    }
    if(positional.empty()||(mode!="eager"&&mode!="screen"&&mode!="lazy")){std::cerr<<usage;return 1;}
    for(const int sig:{SIGINT,SIGTERM,SIGHUP})std::signal(sig,on_stop_signal);
    const auto inst=read_instance(positional[0].c_str(),radius);
    const double seconds=positional.size()>1?std::stod(positional[1]):60.0;
    Solver s{inst.disks,positional.size()>2?std::stod(positional[2]):1e-6,seconds,Clock::now()};
    s.strong=strong;s.seed=seed;s.threads=threads;s.max_memory=max_memory;
    s.mode=mode=="eager"?Solver::Mode::Eager:mode=="screen"?Solver::Mode::Screen:Solver::Mode::Lazy;
    const std::string tour=positional.size()>3?positional[3]:"-";
    // The initial tour: 5% of the time unless given (its local searches
    // stop on their own when they stagnate).
    const double lower=s.run(heuristic>=0?heuristic:0.05*seconds);
    if(interrupted)s.stopped="interrupted";
    std::cout<<std::setprecision(12)<<"{\"instance\":\""<<positional[0]<<"\",\"disks\":"<<inst.disks.size()-1
        <<",\"lower_bound\":"<<lower<<",\"upper_bound\":"<<s.upper<<",\"gap\":"<<(s.upper-lower)/s.upper
        <<",\"closed\":"<<(s.upper-lower<=s.gap*s.upper*1.0000001?"true":"false")<<",\"stopped\":\""<<s.stopped<<"\""
        <<",\"seconds\":"<<s.wall()<<",\"heuristic_seconds\":"<<s.heuristic_seconds<<",\"probe_seconds\":"<<s.probe_seconds<<",\"heuristic_tours\":"<<s.improvements<<",\"ils_iterations\":"<<s.ils_iterations
        <<",\"nodes\":"<<s.nodes<<",\"oracle_calls\":"<<s.counted.oracle_calls
        <<",\"oracle_seconds\":"<<s.counted.oracle_seconds<<",\"newton_iterations\":"<<s.counted.newton<<",\"construction_closed\":"<<s.counted.constructed
        <<",\"children\":"<<s.children_created<<",\"families\":"<<s.families.size()<<",\"compactions\":"<<s.compactions
        <<",\"dropped\":"<<s.dropped<<",\"resident_start\":"<<s.resident_start<<",\"resident_end\":"<<resident_bytes()<<",\"search_bytes\":"<<s.bytes()<<",\"dropped_floor\":"<<(std::isfinite(s.dropped_floor)?s.dropped_floor:-1)<<",\"screened\":"<<s.counted.screened<<",\"threads\":"<<s.threads<<",\"mode\":\""<<mode<<"\",\"strong\":"<<s.strong<<",\"tour_regions\":"<<s.best_sequence.size()<<"}"<<std::endl;
    if(tour!="-") {
        std::ofstream out(tour);out<<std::setprecision(17)<<"{\"instance\":\""<<positional[0]<<"\",\"upper_bound\":"<<s.upper<<",\"sequence\":[";
        for(size_t i=0;i<s.best_sequence.size();++i)out<<(i?",":"")<<s.best_sequence[i];
        // Contacts pulled toward their centers by 2^-40 of the radius: inside
        // the decimal disks too, which binary64 reading moves by about 1e-14.
        out<<"],\"contacts\":[";
        for(size_t i=0;i<s.best_cycle.size();++i) {
            const auto &d=inst.disks[size_t(s.best_sequence[i])];
            const Vector2 c=d.radius>0?d.center+(s.best_cycle[i]-d.center)*(1-0x1p-40):s.best_cycle[i];
            out<<(i?",":"")<<"["<<c.x<<","<<c.y<<"]";
        }
        out<<"]}\n";
    }
}
