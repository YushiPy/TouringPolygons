#include "tpp/nonconvex/unordered.h"
#include "tpp/convex/cycle.h"
#include "common.h"
#include "solvers/unordered_geometry.h"
#include "solvers/unordered_bounds.h"
#include "solvers/unordered_portfolio.h"
#include <atomic>
#include <thread>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
using Polygon=std::vector<Vector2>;
using Polygons=std::vector<Polygon>;
void require(bool condition,const std::string &message) {if(!condition)throw std::runtime_error(message);}
Polygon box(double x,double y,double w=1,double h=1) {return {{x,y},{x+w,y},{x+w,y+h},{x,y+h}};}
void check_oracle_profile(const tpp::UnorderedTppSolveResult &r) {
    const auto calls=std::accumulate(r.oracle_call_histogram.begin(),r.oracle_call_histogram.end(),size_t{0});
    const auto seconds=std::accumulate(r.oracle_seconds_histogram.begin(),r.oracle_seconds_histogram.end(),0.0);
    const double tolerance=1e-9*std::max(1.0,r.convex_oracle_seconds);
    require(calls==r.oracle_profiled_calls,"Oracle timing histogram accounts for every profiled call");
    require(r.oracle_profiled_calls<=r.calls,"Profiled oracle requests are included in total calls");
    require(std::isfinite(seconds)&&std::abs(seconds-r.convex_oracle_seconds)<=tolerance,
        "Oracle timing histogram sums to convex oracle seconds");
    require(r.oracle_max_call_seconds<=r.convex_oracle_seconds+tolerance,
        "Maximum oracle call duration is bounded by accumulated time");
    require(r.oracle_fallback_call_seconds<=r.convex_oracle_seconds+tolerance,
        "Fallback-attributed complete-call time is bounded by oracle total");
}
bool covered(const Polygon &q,const Polygons &p) {
    return q.size()>=2&&q.front()==q.back()&&std::all_of(p.begin(),p.end(),[&](const auto &region) {
        return tpp::unordered_detail::contact(q,region,1e-8).distance<=1e-8;
    });
}
std::pair<double,double> enumerate(const Polygons &p) {
    if(p.size()<2)return {0,0};
    std::vector<std::vector<Polygon>> pieces;
    for(auto region:p) {
        double area=0;for(size_t i=0;i<region.size();++i)area+=region[i].cross(region[(i+1)%region.size()]);
        if(area<0)std::reverse(region.begin(),region.end());
        pieces.push_back(tpp::decompose_polygon(region));
    }
    std::vector<size_t> order(p.size());std::iota(order.begin(),order.end(),0);
    double lower=INFINITY,upper=INFINITY;
    do {
        Polygons selected;
        std::function<void(size_t)> visit=[&](size_t i) {
            if(i==p.size()) {
                const auto r=tpp::tpp_convex_solve_cycle(selected);
                require(r.status==tpp::ConvexCycleStatus::Optimal,"Enumeration cycle oracle: "+r.diagnostic);
                lower=std::min(lower,r.certificate.lower_bound);upper=std::min(upper,r.certificate.upper_bound);return;
            }
            for(const auto &piece:pieces[order[i]]){selected.push_back(piece);visit(i+1);selected.pop_back();}
        };
        visit(0);
    } while(std::next_permutation(order.begin()+1,order.end()));
    return {lower,upper};
}
size_t cases=0,interrupted=0,decomposed=0,parallel_batches=0,portfolio_cases=0,portfolio_limited=0;
void check(const Polygons &p) {
    const auto [lower,upper]=enumerate(p);
    const auto r=tpp::tpp_nonconvex_tspn_solve(p);
    check_oracle_profile(r);
    require(covered(r.path,p),"TSPN output is a feasible closed tour");
    require(r.exact&&r.lower_bound<=upper+1e-7&&r.upper_bound>=lower-1e-7&&
            std::abs(r.upper_bound-upper)<=1e-7+1e-9*upper,"TSPN exhaustive order/piece comparison");
    require(r.calls==r.relaxation_calls+r.refinement_calls+r.initial_convex_refinement_calls,"Oracle accounting");
    decomposed+=r.decomposition_branches>0;
    tpp::UnorderedTppSolveOptions parallel;parallel.threads=2;
    const auto concurrent=tpp::tpp_nonconvex_tspn_solve(p,parallel);
    check_oracle_profile(concurrent);
    require(covered(concurrent.path,p)&&concurrent.exact&&concurrent.threads==2&&
            concurrent.lower_bound<=upper+1e-7&&concurrent.upper_bound>=lower-1e-7&&
            std::abs(concurrent.upper_bound-upper)<=1e-7+1e-9*upper,"Parallel TSPN exhaustive comparison");
    parallel_batches+=concurrent.parallel_oracle_batches;
    for(size_t threads:{1,2})for(size_t cap:{0,1,3,10}) {
        tpp::UnorderedTppSolveOptions options;options.max_calls=cap;options.threads=threads;
        const auto limited=tpp::tpp_nonconvex_tspn_solve(p,options);
        check_oracle_profile(limited);
        require(covered(limited.path,p)&&limited.calls<=cap&&limited.lower_bound<=upper+1e-7&&
                limited.upper_bound>=lower-1e-7,"Interrupted cycle frontier bounds");++interrupted;
    }
    tpp::UnorderedTppSolveOptions dfs;
    dfs.search_strategy=tpp::UnorderedSearchStrategy::DfsBfs;
    const auto alternative=tpp::tpp_nonconvex_tspn_solve(p,dfs);
    check_oracle_profile(alternative);
    require(covered(alternative.path,p)&&alternative.exact&&alternative.lower_bound<=upper+1e-7&&
        std::abs(alternative.upper_bound-upper)<=1e-7+1e-9*upper,"DFS/BFS root and frontier exhaustive comparison");
    for(bool sharing:{false,true}) {
        tpp::UnorderedTppSolveOptions options;options.portfolio=true;options.portfolio_share_incumbents=sharing;
        const auto cooperative=tpp::tpp_nonconvex_tspn_solve(p,options);
        check_oracle_profile(cooperative);
        require(covered(cooperative.path,p)&&cooperative.exact&&cooperative.threads==2&&
            cooperative.portfolio_workers==2&&cooperative.portfolio_runs.size()==2&&
            cooperative.lower_bound<=upper+1e-7&&std::abs(cooperative.upper_bound-upper)<=1e-7+1e-9*upper,
            "Two-search portfolio exhaustive comparison");
        require(cooperative.calls==cooperative.relaxation_calls+cooperative.refinement_calls+
            cooperative.initial_convex_refinement_calls,"Portfolio sums both workers' oracle calls");
        require(cooperative.portfolio_winner<2&&
            cooperative.portfolio_runs[cooperative.portfolio_winner].termination==tpp::UnorderedTppTermination::Optimal,
            "Only a completed proof cancels the peer");
        require(cooperative.seconds>=cooperative.portfolio_proof_seconds&&cooperative.portfolio_join_seconds>=0,
            "Timing includes cooperative shutdown");
        for(const auto &run:cooperative.portfolio_runs)require(run.error.empty(),"Portfolio worker must not fail");
        if(!sharing)require(cooperative.portfolio_incumbent_publications==0&&cooperative.portfolio_incumbent_imports==0,
            "Independent race does not exchange incumbents");
        ++portfolio_cases;
        for(size_t cap:{0,1,3,10}) {
            options.max_calls=cap;
            const auto partial=tpp::tpp_nonconvex_tspn_solve(p,options);
            check_oracle_profile(partial);
            require(covered(partial.path,p)&&partial.calls<=cap&&partial.lower_bound<=upper+1e-7&&
                partial.upper_bound>=lower-1e-7,"Shared call cap and interrupted portfolio frontier");
            require(partial.calls==partial.portfolio_runs[0].calls+partial.portfolio_runs[1].calls,
                "No hidden per-worker call budget");
            if(!partial.exact)require(partial.termination==tpp::UnorderedTppTermination::CallLimit&&
                partial.portfolio_winner==std::numeric_limits<size_t>::max(),"Call limit is not a proof");
            ++portfolio_limited;
        }
    }
    for(int mode=0;mode<14;++mode) {
        tpp::UnorderedTppSolveOptions optimized;
        optimized.cycle_cache=mode==0||mode==13;
        optimized.cycle_dual_reuse=mode==1||mode==13;
        optimized.cycle_active_features=mode==2||mode==13;
        optimized.cycle_lazy=mode==3||mode==13;
        optimized.cycle_separated_root=mode==4||mode==13;
        optimized.cycle_strong_branching=mode==5||mode==13;
        optimized.cycle_one_tree=mode==6||mode==13;
        optimized.cycle_learned_branching=mode==7||mode==13;
        optimized.cycle_memo=mode==8||mode==13;
        optimized.cycle_bound_first=mode==9||mode==13;
        optimized.cycle_dual_screen=mode==10||mode==13;
        optimized.cycle_interval_certificate=mode==11||mode==13;
        optimized.cycle_share_bounds=mode==12||mode==13;
        if(mode==12)optimized.portfolio=true;
        for(size_t cap:{size_t(1),size_t(3),std::numeric_limits<size_t>::max()}) {
            optimized.max_calls=cap;
            const auto run=tpp::tpp_nonconvex_tspn_solve(p,optimized);
            require(covered(run.path,p)&&run.calls<=cap&&run.lower_bound<=upper+1e-7&&run.upper_bound>=lower-1e-7,
                "Optimization frontier certificate mode="+std::to_string(mode));
            if(cap==std::numeric_limits<size_t>::max())require(run.exact&&std::abs(run.upper_bound-upper)<=1e-7+1e-9*upper,
                "Optimization exhaustive objective mode="+std::to_string(mode));
        }
        if(mode==13) {
            optimized.max_calls=3;optimized.portfolio=true;
            const auto partial=tpp::tpp_nonconvex_tspn_solve(p,optimized);
            require(covered(partial.path,p)&&partial.calls<=3&&partial.lower_bound<=upper+1e-7&&partial.upper_bound>=lower-1e-7,
                "Combined optimizations preserve portfolio interruption bounds");
            optimized.portfolio=false;optimized.threads=2;optimized.cycle_lazy=false;
            const auto parallel=tpp::tpp_nonconvex_tspn_solve(p,optimized);
            require(covered(parallel.path,p)&&parallel.calls<=3&&parallel.lower_bound<=upper+1e-7&&parallel.upper_bound>=lower-1e-7,
                "Prepared cache is private to each sibling worker");
        }
    }
    tpp::UnorderedTppSolveOptions supplied;supplied.initial_path=r.path;
    supplied.convex_initial_refinement=true;supplied.bidirectional_initial_heuristic=true;
    const auto again=tpp::tpp_nonconvex_tspn_solve(p,supplied);
    require(covered(again.path,p)&&again.exact&&std::abs(again.upper_bound-upper)<1e-6,"Free initial tour may start anywhere");
    auto reversed=p;std::reverse(reversed.begin(),reversed.end());
    for(auto &region:reversed)std::reverse(region.begin(),region.end());
    supplied.initial_path.reset();
    const auto metamorphic=tpp::tpp_nonconvex_tspn_solve(reversed,supplied);
    require(covered(metamorphic.path,reversed)&&metamorphic.exact&&std::abs(metamorphic.upper_bound-upper)<1e-6,
            "Region/winding reversal and cyclic initial refinement");
    ++cases;
}
void portfolio_protocol() {
    using tpp::unordered_detail::PortfolioControl;
    PortfolioControl budget(97,INFINITY,true);
    std::atomic<size_t> accepted{0};
    auto consume=[&]{while(budget.reserve_call())accepted.fetch_add(1);};
    {std::jthread a(consume),b(consume),c(consume),d(consume);}
    require(accepted==97&&budget.calls==97,"Atomic global call reservations never overrun");
    require(!budget.proved(),"Exhausting the budget does not cancel as proven");
    PortfolioControl incumbent(100,INFINITY,true);
    auto publish=[&](int parity){for(int i=20-parity;i>=1;i-=2) {
        Polygon path{{0,0},{double(i),0},{0,0}};incumbent.publish(path,2*i);
    }};
    {std::jthread a([&]{publish(0);}),b([&]{publish(1);});}
    Polygon received;
    require(incumbent.receive(INFINITY,received)&&tpp::unordered_detail::path_length(received)==2,
        "Concurrent incumbent snapshot carries the corresponding best path");
    require(!incumbent.receive(2,received),"Unchanged incumbents require no transfer");
    {
        const Polygons regions{box(0,0),box(3,0)};
        PortfolioControl::CycleKey key;
        for(size_t i=0;i<regions.size();++i) {
            std::vector<std::pair<double,double>> coordinates;
            for(auto q:regions[i])coordinates.emplace_back(q.x,q.y);
            key.emplace_back(i,std::move(coordinates));
        }
        auto entry=[&](Polygon contacts) {
            const auto certificate=tpp::tpp_convex_verify_cycle_certificate(regions,contacts);
            tpp::unordered_detail::CycleMemo::Entry value;
            value.contacts=std::move(contacts);value.lower_bound=certificate.lower_bound;value.upper_bound=certificate.upper_bound;
            return value;
        };
        const auto longer=entry({{0,0},{4,0}}),shorter=entry({{1,0},{3,0}});
        {std::jthread a([&]{for(int i=0;i<32;++i)incumbent.store_cycle(key,longer);}),
            b([&]{for(int i=0;i<32;++i)incumbent.store_cycle(key,shorter);});}
        const auto cached=incumbent.find_cycle(key);
        require(cached&&cached->lower_bound==4&&cached->upper_bound==4&&
            tpp::tpp_convex_verify_cycle_certificate(regions,cached->contacts).status==tpp::ConvexCycleCertificateStatus::Optimal,
            "Shared relaxation cache retains a consistent best certified contact snapshot");
        auto changed=key;changed[1].second.front().first+=1;
        require(!incumbent.find_cycle(changed),"Equal local labels cannot alias different shared geometry");
        PortfolioControl isolated(100,INFINITY,false);isolated.store_cycle(key,shorter);
        require(!isolated.find_cycle(key),"No-sharing also disables relaxation-result exchange");
        auto extended=key;extended.insert(extended.begin()+1,{7,{{1,4},{2,4},{2,5},{1,5}}});
        for(int reverse=0;reverse<2;++reverse)for(int rotation=0;rotation<3;++rotation) {
            size_t queries=0,hits=0;
            require(incumbent.compatible_cycle_bound(extended,INFINITY,queries,hits)==4&&hits==1,
                "Certified subcycle bound survives insertion, reversal and rotation");
            std::rotate(extended.begin(),extended.begin()+1,extended.end());
            if(rotation==2)std::reverse(extended.begin(),extended.end());
        }
        size_t queries=0,hits=0;
        require(isolated.compatible_cycle_bound(extended,0,queries,hits)==0&&queries==0,
            "No-sharing disables compatible-bound queries");
        require(incumbent.compatible_cycle_bound(changed,INFINITY,queries,hits)==0,
            "Different geometry cannot import a subset bound");
        std::atomic<bool> bounds_consistent{true};
        {std::jthread publisher([&]{for(int i=0;i<32;++i)incumbent.store_cycle(key,i%2?shorter:longer);}),
            reader([&]{for(int i=0;i<32;++i) {
                size_t q=0,h=0;
                if(incumbent.compatible_cycle_bound(extended,INFINITY,q,h)!=4)bounds_consistent=false;
            }});}
        require(bounds_consistent,"Concurrent interned-bound publication preserves exact geometry IDs and the strongest lower bound");
        PortfolioControl orders(100,INFINITY,true);
        auto ordered=extended;ordered.push_back({9,{{6,7}}});
        Polygons order_regions;Polygon order_contacts;
        for(const auto &item:ordered) {
            Polygon region;for(auto [x,y]:item.second)region.push_back({x,y});
            order_contacts.push_back(region.front());order_regions.push_back(std::move(region));
        }
        const auto order_certificate=tpp::tpp_convex_verify_cycle_certificate(order_regions,order_contacts);
        auto certified_entry=shorter;certified_entry.contacts=order_contacts;
        certified_entry.lower_bound=order_certificate.lower_bound;certified_entry.upper_bound=order_certificate.upper_bound;
        orders.store_cycle(PortfolioControl::canonical_key(ordered),certified_entry);
        auto different=ordered;std::swap(different[1],different[2]);
        require(orders.compatible_cycle_bound(different,INFINITY,queries,hits)==0,
            "A different cyclic order cannot import the stronger cycle's bound");
        require(orders.compatible_cycle_bound(key,INFINITY,queries,hits)==0,
            "A bound on a larger constraint set cannot flow to a smaller one");
    }
    incumbent.finish_proof(1);incumbent.finish_proof(0);
    require(incumbent.winner==1&&!incumbent.reserve_call(),"First proof stops new oracle calls");
    const Polygons p{box(0,0),box(5,0),box(5,5)};
    tpp::UnorderedTppSolveOptions zero;zero.portfolio=true;zero.max_seconds=0;
    const auto timed=tpp::tpp_nonconvex_tspn_solve(p,zero);
    require(covered(timed.path,p)&&timed.calls==0&&!timed.exact&&
        timed.termination==tpp::UnorderedTppTermination::TimeLimit&&
        timed.portfolio_winner==std::numeric_limits<size_t>::max(),"Timeout keeps a feasible unproven tour");
    for(int invalid=0;invalid<2;++invalid) {
        tpp::UnorderedTppSolveOptions bad;bad.portfolio=true;
        if(invalid==0)bad.threads=2;else bad.max_seconds=std::numeric_limits<double>::quiet_NaN();
        bool rejected=false;try{tpp::tpp_nonconvex_tspn_solve(p,bad);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Invalid portfolio options rejected");
    }
}
void insertion_bounds() {
    const Polygons p{box(-4,0),box(0,4),box(4,0)};const auto inserted=box(0,-4);
    std::mt19937 rng(290927);std::uniform_real_distribution<double> pos(-10,10);
    for(size_t n=1;n<=p.size();++n) {
        Polygons regions(p.begin(),p.begin()+n);std::vector<const Polygon*> refs;
        for(const auto &r:regions)refs.push_back(&r);
        std::vector<double> optima;
        for(size_t i=0;i<n;++i) {
            auto child=regions;child.insert(child.begin()+i+1,inserted);
            const auto solved=tpp::tpp_convex_solve_cycle(child);
            require(solved.status==tpp::ConvexCycleStatus::Optimal,"Bound reference optimum");
            optima.push_back(solved.certificate.upper_bound);
        }
        for(size_t trial=0;trial<80;++trial) {
            Polygon hints;for(size_t i=0;i<n;++i)hints.push_back({pos(rng),pos(rng)});
            if(trial%3==0)std::fill(hints.begin(),hints.end(),hints.front());
            hints.push_back(hints.front());
            const auto bounds=tpp::unordered_detail::insertion_lower_bounds(hints,refs,inserted,true);
            for(size_t i=0;i<n;++i)require(bounds[i]<=optima[i],"Rational cyclic dual bound, including closing edge");
            tpp::ConvexRationalPolygon dual;
            for(size_t i=0;i<n;++i)dual.emplace_back(tpp::ConvexRational(int(rng()%3)-1)/2,tpp::ConvexRational(int(rng()%3)-1)/2);
            const auto reused=tpp::unordered_detail::insertion_lower_bounds(hints,refs,inserted,true,dual);
            for(size_t i=0;i<n;++i)require(reused[i]>=bounds[i]&&reused[i]<=optima[i],"Inherited subunit dual remains a valid stronger bound");
        }
    }
}
void replacement_bounds() {
    using namespace tpp;
    const Polygons p{box(-4,0,3,3),box(0,4,3,3),box(4,0,3,3)};
    std::mt19937 rng(260930);std::uniform_int_distribution<int> coord(-8,8);
    for(size_t n=1;n<=p.size();++n) {
        Polygons regions(p.begin(),p.begin()+n);std::vector<const Polygon *> refs;
        for(const auto &r:regions)refs.push_back(&r);
        for(size_t position=0;position<n;++position) {
            const auto origin=regions[position].front();
            const Polygons pieces{box(origin.x,origin.y,1,1),box(origin.x+2,origin.y+2,1,1)};
            std::vector<double> optima(2,0);
            for(size_t j=0;j<2&&n>1;++j) {
                auto child=regions;child[position]=pieces[j];
                const auto solved=tpp_convex_solve_cycle(child);
                require(solved.status==ConvexCycleStatus::Optimal,"Replacement exact reference optimum");
                optima[j]=solved.certificate.upper_bound;
            }
            for(size_t trial=0;trial<12;++trial) {
                Polygon contacts;for(size_t i=0;i<n;++i)contacts.push_back({double(coord(rng)),double(coord(rng))});
                if(trial%3==0)std::fill(contacts.begin(),contacts.end(),contacts.front());
                contacts.push_back(contacts.front());
                ConvexRationalPolygon inherited(n,ConvexRationalPoint{ConvexRational(1)/2,ConvexRational(1)/2});
                const auto bounds=unordered_detail::cycle_replacement_lower_bounds(contacts,refs,pieces,position,inherited);
                for(size_t j=0;j<2;++j)require(bounds[j]<=optima[j],"Replacement dual valid for arbitrary and coincident parent contacts");
            }
        }
    }
}
}

void one_tree_bounds() {
    using R=tpp::ConvexRational;
    using namespace tpp::unordered_detail;
    std::mt19937 random(20260930);bool strengthened=false;
    for(size_t n=2;n<=6;++n)for(size_t sample=0;sample<8;++sample) {
        std::vector<std::vector<R>> costs(n,std::vector<R>(n));
        for(size_t i=0;i<n;++i)for(size_t j=0;j<i;++j)
            costs[i][j]=costs[j][i]=R(1+random()%20)+R(random()%4)/R(tpp::ConvexInteger(1)<<70);
        std::vector<size_t> order(n);std::iota(order.begin(),order.end(),0);R optimum=0;bool first=true;
        do {
            R value=0;for(size_t i=0;i<n;++i)value+=costs[order[i]][order[(i+1)%n]];
            if(first||value<optimum){optimum=value;first=false;}
        } while(std::next_permutation(order.begin()+1,order.end()));
        const double upper=std::nextafter(optimum.convert_to<double>(),INFINITY);
        const auto base=held_karp_bound(costs,upper,1),bound=held_karp_bound(costs,upper);
        require(R(bound.lower_bound)<=optimum,"Exact one-tree lower bound versus exhaustive graph cycles");
        require(bound.lower_bound>=base.lower_bound,"Ascent retains the best certified bound");
        strengthened|=bound.lower_bound>base.lower_bound;
        const auto arbitrary_target=held_karp_bound(costs,upper/2);
        require(R(arbitrary_target.lower_bound)<=optimum,"Incumbent below node optimum cannot invalidate the dual");
        if(n>=3)require(held_karp_bound(costs,upper,32,[]{return true;}).lower_bound==0,"Stopped ascent is conservative");
    }
    require(strengthened,"Held-Karp ascent improves an unpenalized one-tree");
    Polygons p{box(0,0),box(4,5)};
    CycleOneTreeWorkspace workspace;
    std::vector<const Polygon *> regions{&p[0],&p[1]};
    const auto first=workspace.bound(regions,20),cached=workspace.bound(regions,20);
    require(first.lower_bound==10&&first.distance_queries==1&&cached.cached&&cached.lower_bound==10,
        "Exact support distance on the 3-4-5 gap and cached two-region cycle");
    const Polygon contained=box(0.2,0.2,.2,.2);regions[1]=&contained;
    require(workspace.bound(regions,20).lower_bound==0,"Intersecting regions admit zero pair distance");
    const Polygon strip{{-.5,.4},{1.5,.4},{1.5,.6},{-.5,.6}};regions[1]=&strip;
    require(workspace.bound(regions,20).lower_bound==0,"Crossing edges with no interior vertices cannot yield positive separation");
    std::cout<<"One-tree graph enumeration, exact rounding and geometric support tests passed.\n";
}
void memo_cycle_keys() {
    using Key=std::vector<std::pair<size_t,size_t>>;
    auto canonical=[](const Key &input) {
        Key key;for(auto i:tpp::unordered_detail::canonical_cycle_indices(input))key.push_back(input[i]);return key;
    };
    const Key input{{4,3},{1,2},{7,0},{2,5}};const auto reference=canonical(input);
    auto rotated=input;
    for(size_t i=0;i<input.size();++i) {
        require(canonical(rotated)==reference,"Cyclic memo key transports rotations");
        auto reversed=rotated;std::reverse(reversed.begin(),reversed.end());
        require(canonical(reversed)==reference,"Cyclic memo key transports reversal");
        std::rotate(rotated.begin(),rotated.begin()+1,rotated.end());
    }
    auto changed=input;changed[0].second++;
    require(canonical(changed)!=reference,"Memo never aliases different decomposition pieces");
    changed=input;std::swap(changed[0],changed[1]);
    require(canonical(changed)!=reference,"Memo never aliases a different visit order");
}
int main() {
    try {
        memo_cycle_keys();
        one_tree_bounds();
        portfolio_protocol();
        insertion_bounds();
        replacement_bounds();
        check({});check({box(0,0)});check({box(0,0,10,10),box(12,4,1,2)});
        require(std::abs(tpp::tpp_nonconvex_tspn_solve({box(0,0,10,10),box(12,4,1,2)}).upper_bound-4)<1e-8,
                "No artificial fixed endpoint in TSPN");
        check({box(0,0,2,2),box(1,1,2,2),box(.5,.5,1,1)});
        check({box(0,0),box(5,0),box(5,5),box(0,5)});
        check({{{0,0},{6,0},{6,6},{4,6},{4,2},{2,2},{2,6},{0,6}},box(2.5,3,1,1)});
        check({{{0,0},{3,0},{3,1},{1,1},{1,3},{0,3}},box(5,0),box(4,5),box(-2,4)});
        std::mt19937 rng(260927);std::uniform_int_distribution<int> coord(-6,6);
        for(size_t trial=0;trial<12;++trial) {
            Polygons p;for(size_t i=0;i<2+trial%4;++i)p.push_back(box(coord(rng),coord(rng),2,2));
            check(p);
        }
        require(decomposed>0,"Nonconvex decomposition branching exercised");
        require(parallel_batches>0,"Concurrent cyclic oracle batches exercised");
        tpp::UnorderedTppSolveOptions bad;bad.initial_path=Polygon{{0,0},{1,0}};
        bool rejected=false;try {tpp::tpp_nonconvex_tspn_solve({box(0,0)},bad);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Open supplied tour rejected");
        std::cout<<"TSPN tests passed: "<<cases<<" exhaustive cases with 1 and 2 threads, "<<interrupted<<" interrupted searches, "
                 <<portfolio_cases<<" portfolios, "<<portfolio_limited<<" shared-budget searches, "<<decomposed<<" decomposition cases, "<<parallel_batches<<" concurrent oracle batches, 240 arbitrary-hint plus 240 inherited-dual checks; 798 optimization/call-cap comparisons and 38 combined concurrency checks.\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
