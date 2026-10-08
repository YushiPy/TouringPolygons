#include "tpp/nonconvex/unordered.h"
#include "tpp/convex/certified.h"
#include "tpp/convex/float_oracle.h"
#include "tpp/convex/hybrid.h"
#include "tpp/nonconvex/decomposition.h"
#include "solvers/unordered_geometry.h"
#include "solvers/unordered_bounds.h"
#include "tpp/convex/dual.h"
#include "solvers/unordered_sequence.h"
#include <functional>
#include <mutex>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace tpp;
using Polygon = std::vector<Vector2>;

void same_search(const UnorderedTppSolveResult &a, const UnorderedTppSolveResult &b) {
    if (a.lower_bound!=b.lower_bound || a.upper_bound!=b.upper_bound || a.exact!=b.exact
        || a.termination!=b.termination || a.calls!=b.calls || a.nodes!=b.nodes
        || a.children_generated!=b.children_generated || a.children_queued!=b.children_queued
        || a.insertion_branches!=b.insertion_branches || a.decomposition_branches!=b.decomposition_branches
        || a.peak_queue!=b.peak_queue || a.order!=b.order || a.path.size()!=b.path.size()
        || a.trace.size()!=b.trace.size()) throw std::runtime_error("Sequence storage changed the search.");
    auto same_path=[](const auto &x,const auto &y) {
        if(x.size()!=y.size())return false;
        for(size_t i=0;i<x.size();++i)if(x[i].x!=y[i].x||x[i].y!=y[i].y)return false;
        return true;
    };
    if(!same_path(a.path,b.path))throw std::runtime_error("Sequence storage changed contacts.");
    for(size_t i=0;i<a.trace.size();++i) {
        const auto &x=a.trace[i],&y=b.trace[i];
        if(x.kind!=y.kind||x.node!=y.node||x.parent!=y.parent||x.polygon!=y.polygon||x.piece!=y.piece
            ||x.position!=y.position||x.pass!=y.pass||x.sequence!=y.sequence||x.order!=y.order
            ||!same_path(x.path,y.path)||x.lower_bound!=y.lower_bound||x.upper_bound!=y.upper_bound
            ||x.length!=y.length||x.pruned!=y.pruned||x.source!=y.source||x.reason!=y.reason)
            throw std::runtime_error("Sequence storage changed the trace.");
    }
}

void check_prepared_contacts() {
    using namespace tpp::unordered_detail;
    std::mt19937 random(20261003);
    std::uniform_real_distribution<double> coordinate(-8,8);
    size_t checks=0;
    for(size_t trial=0;trial<2000;++trial) {
        Polygon polygon=trial%2?Polygon{{0,0},{3,0},{3,1},{1,1},{1,3},{0,3}}:
            Polygon{{-1,-1},{2,-1},{2,2},{-1,2}};
        const Vector2 shift{coordinate(random),coordinate(random)};
        for(auto &v:polygon)v+=shift;
        if(trial%3==0)std::reverse(polygon.begin(),polygon.end());
        if(trial%5==0)polygon.push_back(polygon.front());
        const PreparedContactPolygon prepared_polygon(polygon);
        SegmentContactCache cache;
        PreparedContactPath prepared_path;
        for(size_t count:{size_t(0),size_t(1),size_t(2),size_t(7)}) {
            Polygon path;
            for(size_t i=0;i<count;++i)path.push_back({coordinate(random),coordinate(random)});
            if(count>1&&trial%7==0)path.back()=path.front();
            if(count>1&&trial%11==0)path.front()=polygon.front();
            prepared_path.prepare(path);
            for(double tolerance:{0.0,1e-8,0.01}) {
                const auto raw=contact(path,polygon,tolerance),prepared=contact(prepared_path,prepared_polygon,tolerance);
                if(raw.distance!=prepared.distance||raw.position!=prepared.position)
                    throw std::runtime_error("Prepared contact changed distance or visit position.");
                // Each cache has one immutable polygon and tolerance. Repeat
                // paths, shared segments and eviction are exercised separately.
                SegmentContactCache local;
                for(size_t repeat=0;repeat<2;++repeat) {
                    const auto cached=local.query(prepared_path,prepared_polygon,0,1,tolerance);
                    if(raw.distance!=cached.distance||raw.position!=cached.position)
                        throw std::runtime_error("Segment cache changed a visit or its tie order");
                }
                ++checks;
            }
        }
        // Changed paths share an unchanged first segment with earlier calls.
        Polygon changing{{coordinate(random),coordinate(random)},polygon.front(),{coordinate(random),coordinate(random)}};
        for(size_t repeat=0;repeat<4;++repeat) {
            changing.back()={coordinate(random),coordinate(random)};
            prepared_path.prepare(changing);
            const auto raw=contact(changing,polygon,1e-8),cached=cache.query(prepared_path,prepared_polygon,0,1,1e-8);
            if(raw.distance!=cached.distance||raw.position!=cached.position)throw std::runtime_error("Segment reuse changed a visit");
        }
    }
    const Polygon stable{{-1,-1},{2,-1},{2,2},{-1,2}};
    const PreparedContactPolygon prepared_stable(stable);
    SegmentContactCache evicting;
    for(size_t i=0;i<2400;++i) {
        const Polygon path{{-3,double(i)/100},{3,double(i)/100},{4,4}};
        const PreparedContactPath prepared(path);
        const auto raw=contact(path,stable,1e-8),cached=evicting.query(prepared,prepared_stable,0,1,1e-8);
        if(raw.distance!=cached.distance||raw.position!=cached.position)throw std::runtime_error("Segment eviction changed a contact");
        ++checks;
    }
    // Reusing the scratch object across tolerances, polygon identities and
    // polygon counts must never reuse a contact from different geometry.
    const Polygon shifted{{9,9},{12,9},{12,12},{9,12}};
    const PreparedContactPolygon prepared_shifted(shifted);
    const Polygon near_path{{-3,-1.0005},{3,-1.0005}};
    const PreparedContactPath prepared_near(near_path);
    for(double tolerance:{0.0,0.01,0.0})for(size_t count:{size_t(1),size_t(2)}) {
        for(const auto *polygon:{&prepared_stable,&prepared_shifted,&prepared_stable}) {
            const auto raw=contact(prepared_near,*polygon,tolerance);
            const auto cached=evicting.query(prepared_near,*polygon,0,count,tolerance);
            if(raw.distance!=cached.distance||raw.position!=cached.position)
                throw std::runtime_error("Segment cache reused incompatible geometry or tolerance");
            ++checks;
        }
    }
    std::cout<<"Prepared visits passed "<<checks<<" distance/position comparisons.\n";
}

void check_sequence_storage() {
    using namespace tpp::unordered_detail;
    if(sequence_index_bytes(255,257)!=1||sequence_index_bytes(256,3)!=2
        ||sequence_index_bytes(1,258)!=2||sequence_index_bytes(65535,3)!=2
        ||sequence_index_bytes(65536,3)!=4||sequence_index_bytes(UINT32_MAX,3)!=4
        ||sequence_index_bytes(size_t(UINT32_MAX)+1,3)!=8)
        throw std::runtime_error("Incorrect sequence width boundary.");
    bool overflow=false;
    try {encode_sequence_index<uint8_t>(255);}catch(const std::overflow_error &){overflow=true;}
    if(!overflow||decode_sequence_index(encode_sequence_index<uint8_t>(no_sequence_index))!=no_sequence_index)
        throw std::runtime_error("Sequence sentinel or overflow failure.");
    std::mt19937 random(20261002);
    size_t comparisons=0;
    for(size_t bytes:{1,2,4,8})for(size_t n:{1,3,60,70})for(size_t root_size:{size_t(0),std::min(n,size_t(3))}) {
        SequenceHistory arena(bytes,n);
        std::vector<SequenceElement> root;
        for(size_t i=0;i<root_size;++i)root.push_back({i});
        auto root_reference=arena.snapshot(root);
        using State=std::pair<SequenceReference,std::vector<SequenceElement>>;
        std::vector<State> states{{root_reference,root}};
        for(size_t step=0;step<1200;++step) {
            auto &parent=states[random()%states.size()];
            auto expected=parent.second;
            std::vector<size_t> missing;
            for(size_t i=0;i<n;++i)
                if(std::none_of(expected.begin(),expected.end(),[&](auto e){return e.polygon==i;}))missing.push_back(i);
            size_t polygon,piece=no_sequence_index,position;
            if(!missing.empty()&&(expected.empty()||random()%3)) {
                polygon=missing[random()%missing.size()];position=random()%(expected.size()+1);
                expected.insert(expected.begin()+position,{polygon});
            } else {
                position=random()%expected.size();polygon=expected[position].polygon;
                piece=bytes==1?random()%255:bytes==2?size_t(300+random()%100):bytes==4?size_t(70000+random()%100):size_t(UINT32_MAX)+1+random()%100;
                expected[position].piece=piece;
            }
            auto child=arena.child(parent.first,polygon,piece,position);
            if(child.expand()!=expected)throw std::runtime_error("Delta reconstruction differs from vector edits.");
            ++comparisons;
            NodeSequence packed(expected);packed.pack(bytes);
            packed.restore(bytes);
            if(packed.elements()!=expected)throw std::runtime_error("Packed indices differ from vector edits.");
            // Copies, moves, shared ancestors, arbitrary deletion and recycled IDs.
            states.emplace_back(std::move(child),std::move(expected));
            if(states.size()>80)states.erase(states.begin()+1+random()%(states.size()-1));
            if(step%17==0)for(const auto &state:states) {
                if(state.first.expand()!=state.second)throw std::runtime_error("Recycling damaged a retained ancestor.");
                ++comparisons;
            }
        }
        states.clear();root_reference={};
        if(arena.live_records()!=0)throw std::runtime_error("Sequence ancestors leaked after the last owner.");
        // Force a long history so the >64-position Fenwick reconstruction runs.
        // A separate arena has its own root and can be destroyed independently.
        SequenceHistory long_arena(bytes,n);
        auto tail=long_arena.snapshot({});std::vector<SequenceElement> expected;
        for(size_t i=0;i<n;++i) {
            const size_t position=random()%(expected.size()+1);
            expected.insert(expected.begin()+position,{i});tail=long_arena.child(tail,i,no_sequence_index,position);
        }
        if(tail.expand()!=expected)throw std::runtime_error("Long delta history reconstruction failed.");
        tail={};if(long_arena.live_records()!=0)throw std::runtime_error("Long delta history leaked.");
    }
    const std::vector<Polygon> polygons={
        {{0,0},{3,0},{3,1},{1,1},{1,3},{0,3}},
        {{5,0},{6,0},{6,1},{5,1}},{{4,5},{5,5},{5,6},{4,6}},{{-2,4},{-1,4},{-1,5},{-2,5}}};
    for(bool cycle:{false,true})for(auto strategy:{UnorderedSearchStrategy::BestBoundDive,UnorderedSearchStrategy::DfsBfs})
        for(size_t cap:{0,1,3,10,1000}) {
            UnorderedTppSolveOptions options;options.trace=true;options.max_calls=cap;options.search_strategy=strategy;
            options.sequence_storage=UnorderedSequenceStorage::Native;
            auto solve=[&]{return cycle?tpp_nonconvex_tspn_solve(polygons,options):tpp_nonconvex_unordered_solve({-3,-2},{8,8},polygons,options);};
            const auto baseline=solve();
            options.prepared_visit_queries=false;same_search(baseline,solve());
            options.prepared_visit_queries=true;
            options.segment_visit_cache=true;same_search(baseline,solve());
            options.segment_visit_cache=false;
            options.oracle_borrow_geometry=true;same_search(baseline,solve());
            options.oracle_borrow_geometry=false;
            for(auto caches:{std::pair{false,true},std::pair{true,false},std::pair{false,false}}) {
                options.oracle_dispatch_cache=caches.first;
                options.oracle_interval_geometry_cache=caches.second;
                same_search(baseline,solve());
            }
            options.oracle_dispatch_cache=true;
            options.oracle_interval_geometry_cache=true;
            for(auto storage:{UnorderedSequenceStorage::Packed,UnorderedSequenceStorage::Deltas}) {
                options.sequence_storage=storage;const auto result=solve();same_search(baseline,result);
                if(result.node_index_bits!=8)throw std::runtime_error("Small nodes should use byte indices.");
            }
        }
    std::cout<<"Sequence storage passed "<<comparisons<<" reconstruction/recycling checks, 40 traced storage comparisons and 60 oracle-cache comparisons.\n";
}

double length(const Polygon &p) {
	double value = 0;
	for (size_t i = 1; i < p.size(); ++i) value += p[i - 1].distance_to(p[i]);
	return value;
}

void check(Vector2 s, Vector2 t, const std::vector<Polygon> &polygons) {
	const auto result = tpp::tpp_nonconvex_unordered_solve(s, t, polygons);
    UnorderedTppSolveOptions uncached;uncached.oracle_dispatch_cache=false;
    uncached.oracle_interval_geometry_cache=false;
    uncached.prepared_visit_queries=false;
    same_search(result,tpp_nonconvex_unordered_solve(s,t,polygons,uncached));
    for(auto storage:{UnorderedSequenceStorage::Native,UnorderedSequenceStorage::Deltas}) {
        UnorderedTppSolveOptions options;options.sequence_storage=storage;
        same_search(result,tpp_nonconvex_unordered_solve(s,t,polygons,options));
    }
	if (result.fallback_calls != result.fallback_geometric_path_invalid_calls + result.fallback_certificate_gap_calls
		|| result.extended_precision_calls > result.fallback_calls
		|| result.repaired_geometric_path_calls > result.calls
		|| result.oracle_dispatch_pair_cache_hits > result.oracle_dispatch_pair_queries
		|| result.oracle_dispatch_pair_exact_checks > result.oracle_dispatch_pair_queries
		|| result.convex_dispatch_seconds > result.convex_oracle_seconds
		|| result.calls != result.relaxation_calls + result.refinement_calls + result.initial_convex_refinement_calls
		|| result.branch_events != result.insertion_branches + result.decomposition_branches
		|| result.partial_states_created < 1
		|| result.nodes > result.partial_states_created
		|| result.children_queued > result.children_generated
		|| result.sibling_bound_prunes > result.children_generated
		|| result.insertion_positions_pruned > result.insertion_positions_considered
		|| result.pruned_nodes > result.pruned_states
		|| result.pruned_states != result.bound_prunes + result.incumbent_prunes
		|| result.best_updates > result.incumbent_updates
		|| (result.best_updates > 0 && !std::isfinite(result.first_best_update_length))
		|| (result.best_updates == 0 && std::isfinite(result.first_best_update_length))
		|| (result.best_updates > 0 && result.first_best_update_length > result.incumbent_length)
		|| result.final_length != result.upper_bound
		|| result.initial_length != result.initial_upper_bound
		|| result.incumbent_length != result.initial_upper_bound
		|| !std::isfinite(result.order_space_log2)
		|| result.convex_oracle_wall_seconds + result.decomposition_seconds + result.search_visit_check_seconds
			+ result.search_maintenance_seconds > result.search_seconds + 1e-9
		|| result.heuristic_visit_check_seconds + result.search_visit_check_seconds
			+ result.finalization_visit_check_seconds > result.visit_check_seconds + 1e-9)
		{
		throw std::runtime_error("Inconsistent unordered profiling metrics.");
		}
	std::vector<size_t> order(polygons.size());
	std::iota(order.begin(), order.end(), 0);
	std::vector<std::vector<Polygon>> pieces;
	for (auto p : polygons) {
		double area = 0;
		for (size_t i = 0; i < p.size(); ++i) area += p[i].cross(p[(i + 1) % p.size()]);
		if (area < 0) std::reverse(p.begin(), p.end());
		pieces.push_back(tpp::decompose_polygon(p));
	}
	double best = std::numeric_limits<double>::infinity();
	tpp::DynamicConvexTppWorkspace workspace;
	do {
		std::vector<Polygon> ordered;
		std::function<void(size_t)> enumerate = [&](size_t i) {
			if (i == order.size()) {
				const auto fixed = tpp::tpp_convex_solve_certified(s, t, ordered, workspace, 1e-7);
				if (fixed.upper_bound - fixed.lower_bound > 1e-6) {
					std::cerr << "Oracle " << fixed.lower_bound << ' ' << fixed.upper_bound << " polygons " << ordered << " endpoints " << s << ' ' << t << "\n";
					throw std::runtime_error("Enumeration oracle gap.");
				}
				best = std::min(best, fixed.upper_bound);
				return;
			}
			for (const auto &piece : pieces[order[i]]) {
				ordered.push_back(piece);
				enumerate(i + 1);
				ordered.pop_back();
			}
		};
		enumerate(0);
	} while (std::next_permutation(order.begin(), order.end()));
	if (!result.exact || result.lower_bound > best + 1e-6 || result.lower_bound > result.upper_bound + 1e-8
		|| std::abs(best - result.upper_bound) > 1e-6 * (1 + best)) {
		std::cerr << "Expected " << best << ", got " << result.upper_bound << '\n';
		throw std::runtime_error("Permutation enumeration mismatch.");
	}
    for(unsigned variant=0;variant<15;++variant) {
        UnorderedTppSolveOptions candidate;
        // Parallel node rounds (and with the LNS) keep the same guarantees.
        candidate.parallel_nodes=variant>=11;
        candidate.threads=variant>=11?3:1;
        // Exact window LNS (smallest windows, to exercise splicing) and the
        // insertion lookahead must preserve bounds, budget and optimality.
        candidate.window_lns=variant==8||variant==10||variant==12;
        candidate.window_lns_size=2;
        candidate.window_lns_max_size=4;
        candidate.insertion_lookahead=variant>=9?8:0;
        candidate.oracle_borrow_geometry=variant==0||variant==6;
        candidate.lazy_oracles=variant==1||variant==6;
        candidate.oracle_bound_first=variant==2||variant==6;
        candidate.segment_visit_cache=variant==3||variant==6;
        candidate.path_dual_reuse=variant==4||variant==6;
        candidate.path_strong_branching=variant==5||variant==6;
        candidate.path_certificate_dual=variant==7;
        // Multi-insertion node bounds, alone and with lazy oracles, no dives
        // and the lookahead.
        candidate.multi_insertion_bound=variant>=13;
        if(variant==14){candidate.lazy_oracles=true;candidate.dive_interval=0;candidate.insertion_lookahead=8;}
        for(size_t cap:{size_t(0),size_t(1),size_t(3),size_t(10),size_t(1000000)}) {
            candidate.max_calls=cap;
            const auto changed=tpp_nonconvex_unordered_solve(s,t,polygons,candidate);
            if(changed.calls>cap||changed.lower_bound>best+1e-6||changed.upper_bound<best-1e-6
                ||(cap==1000000&&(!changed.exact||std::abs(changed.upper_bound-best)>1e-6*(1+best))))
                throw std::runtime_error("Path optimization violated exhaustive bounds or budget");
            for(const auto &p:polygons)if(unordered_detail::contact(changed.path,p,1e-8).distance>1e-8)
                throw std::runtime_error("Path optimization returned an infeasible route");
        }
    }
    // Anchor upper bounds only skip exact contacts; the search must not change,
    // with and without the lookahead's K-candidate threshold.
    for(size_t lookahead:{size_t(0),size_t(3)}) {
        UnorderedTppSolveOptions bounded,unbounded;
        bounded.insertion_lookahead=unbounded.insertion_lookahead=lookahead;
        unbounded.visit_upper_bounds=false;
        same_search(tpp_nonconvex_unordered_solve(s,t,polygons,bounded),tpp_nonconvex_unordered_solve(s,t,polygons,unbounded));
    }
	tpp::UnorderedTppSolveOptions root_options;
	root_options.detour_root = true;
	const auto detour_result = tpp::tpp_nonconvex_unordered_solve(s, t, polygons, root_options);
	if (!detour_result.exact || detour_result.lower_bound > best + 1e-6
		|| detour_result.lower_bound > detour_result.upper_bound + 1e-8
		|| std::abs(best - detour_result.upper_bound) > 1e-6 * (1 + best))
		throw std::runtime_error("Detour-root permutation enumeration mismatch.");
	for (size_t cap : {0, 1, 3, 10}) {
		tpp::UnorderedTppSolveOptions options;
		options.max_calls = cap;
		if (cap == 10) options.oracle_relative_gap = .01;
		const auto limited = tpp::tpp_nonconvex_unordered_solve(s, t, polygons, options);
        for(auto storage:{UnorderedSequenceStorage::Native,UnorderedSequenceStorage::Deltas}) {
            options.sequence_storage=storage;
            same_search(limited,tpp_nonconvex_unordered_solve(s,t,polygons,options));
        }
		if (limited.calls > cap || limited.lower_bound > best + 1e-6 || limited.upper_bound < best - 1e-6)
			throw std::runtime_error("Invalid interrupted search bounds.");
	}
}

void check_oracle_certificates() {
	const Vector2 s{0, 0}, t{10, 0};
	const std::vector<Polygon> polygons = {
		{{6, 1}, {8, 1}, {8, 3}, {6, 3}},
		{{2, 2}, {7, 2}, {7, 4}, {2, 4}}
	};
	const double optimum = std::sqrt(40.) + std::sqrt(20.);
	DynamicConvexTppWorkspace workspace;
	for (double cutoff : {0., 10.5, std::numeric_limits<double>::infinity()}) {
		const auto r = tpp_convex_solve_certified(s, t, polygons, workspace, 1e-7, cutoff);
		if (r.lower_bound > optimum + 1e-8 || r.upper_bound < optimum - 1e-8
			|| (r.lower_bound < cutoff && r.upper_bound - r.lower_bound > 1e-7))
			throw std::runtime_error("Invalid convex cutoff certificate.");
        workspace.retain_binary_dual=true;
        const auto retained=tpp_convex_solve_certified(s,t,polygons,workspace,1e-7,cutoff);
        workspace.retain_binary_dual=false;
        if(retained.path!=r.path||retained.lower_bound!=r.lower_bound||retained.upper_bound!=r.upper_bound)
            throw std::runtime_error("Retaining a binary dual changed oracle bounds or contacts");
        if(!retained.binary_dual.empty()&&retained.binary_dual.size()!=polygons.size()+1)
            throw std::runtime_error("Invalid retained dual dimension");
        for(auto u:retained.binary_dual) {
            const ConvexRationalPoint exact(u);
            if(exact.dot(exact)>1)throw std::runtime_error("Retained binary dual leaves the unit disk");
        }
		if (cutoff == 10.5) {
			// Complete zero-block certification now proves this candidate
			// directly; the cutoff must still be met without rational recovery.
			if (r.used_fallback || r.lower_bound < cutoff
				|| r.path.size() != polygons.size() + 2 || length(r.path) > r.upper_bound + 1e-7)
				throw std::runtime_error("Expected a certified coincident-contact cutoff.");
			for (size_t i = 0; i < polygons.size(); ++i)
				if (tpp::unordered_detail::contact({r.path[i + 1], r.path[i + 1]}, polygons[i], 1e-8).distance > 1e-8)
					throw std::runtime_error("Dual cutoff path missed an ordered polygon.");
		}
	}
	// This proposal is feasible but suboptimal. The early dual cutoff must
	// precede filtered construction as well as complete rational recovery.
	const std::vector<Polygon> suboptimal = {
		{{-2,0},{0,0},{0,2},{-2,2}},
		{{-1e-6,1},{1.999999,0},{1.999999,2}}
	};
	const auto cut = tpp_convex_solve_certified({-3,-2},{3,-2},suboptimal,workspace,1e-7,0.0);
	if (!cut.dual_cutoff_pruned || cut.used_fallback || cut.lower_bound < 0
		|| cut.path.size() != suboptimal.size()+2 || length(cut.path) > cut.upper_bound+1e-7)
		throw std::runtime_error("Expected a feasible early dual cutoff for a rejected candidate.");
	const auto interrupted = tpp_convex_solve_certified(
		s, t, polygons, workspace, 0.0, std::numeric_limits<double>::infinity(), 0.0
	);
	if (!interrupted.time_limited || interrupted.lower_bound > optimum + 1e-8
		|| interrupted.upper_bound < optimum - 1e-8)
		throw std::runtime_error("Invalid deadline-interrupted convex bounds.");
	std::mt19937 rng(9162026);
	std::uniform_real_distribution<double> coordinate(-100, 100);
	const Polygon inserted{{4, -2}, {5, -2}, {5, -1}, {4, -1}};
	std::vector<double> optima;
	for (size_t j = 0; j <= polygons.size(); ++j) {
		auto sequence = polygons;
		sequence.insert(sequence.begin() + j, inserted);
		optima.push_back(tpp_convex_solve_certified(s, t, sequence, workspace, 1e-7).upper_bound);
	}
	for (size_t trial = 0; trial < 200; ++trial) {
		// Dual screening must remain valid even for completely infeasible hints.
		Polygon q{s, {coordinate(rng), coordinate(rng)}, {coordinate(rng), coordinate(rng)}, t};
		if (trial % 3 == 0) q[2] = q[1];
		const auto bounds = unordered_detail::insertion_lower_bounds(q, {&polygons[0], &polygons[1]}, inserted);
		for (size_t j = 0; j < bounds.size(); ++j) if (bounds[j] > optima[j] + 1e-8)
			throw std::runtime_error("Invalid incremental insertion bound.");
        Polygon dual,proposals;
        for(size_t j=0;j<3;++j) {
            dual.push_back({coordinate(rng)/200,coordinate(rng)/200});
            proposals.push_back(trial%3==0?q[j]:trial%3==1?Vector2{coordinate(rng),coordinate(rng)}:
                unordered_detail::best_contact(q[j],q[j+1],inserted,inserted.front()));
        }
        const auto certified=tpp_convex_binary_dual_insertion_bounds(s,t,q,{&polygons[0],&polygons[1]},inserted,proposals,dual);
        if(certified.size()!=3)throw std::runtime_error("Binary dual screening unexpectedly declined");
        for(size_t j=0;j<3;++j)if(certified[j]>optima[j]+1e-8)
            throw std::runtime_error("Binary dual insertion exceeded exact optimum");
        dual[trial%3]={2,0};
        if(!tpp_convex_binary_dual_insertion_bounds(s,t,q,{&polygons[0],&polygons[1]},inserted,proposals,dual).empty())
            throw std::runtime_error("Invalid dual hint authorized screening");
	}
    const Polygon hint{s,{6,2},{4,2},t},proposals{{5,-1},{5,-1},{5,-1}};
    Polygon dual(3,{.25,.25});
    // Even a barely exterior binary vector must decline; pruning never uses
    // a tolerance for membership of the dual unit disk.
    for(auto invalid:{Vector2{std::nextafter(1.0,INFINITY),0},
            Vector2{1,std::numeric_limits<double>::denorm_min()},Vector2{NAN,0}}) {
        dual[1]=invalid;
        if(!tpp_convex_binary_dual_insertion_bounds(s,t,hint,{&polygons[0],&polygons[1]},inserted,proposals,dual).empty())
            throw std::runtime_error("Exterior or nonfinite dual authorized screening");
    }
    dual[1]={.25,.25};
    const int rounding=std::fegetround();
    if(std::fesetround(FE_UPWARD)==0) {
        const bool declined=tpp_convex_binary_dual_insertion_bounds(s,t,hint,
            {&polygons[0],&polygons[1]},inserted,proposals,dual).empty();
        std::fesetround(rounding);
        if(!declined)throw std::runtime_error("Unsupported rounding authorized screening");
    }
	const Polygon rectangle{{4, 2}, {6, 2}, {6, 4}, {4, 4}};
	const auto point = unordered_detail::best_contact(s, t, rectangle, rectangle.front());
	if (point.distance_to({5, 2}) > 1e-12)
		throw std::runtime_error("Analytic edge contact missed reflection point.");
	const auto crossing = unordered_detail::best_contact({0, 3}, {10, 3}, rectangle, rectangle.front());
	if (std::abs(Vector2{0, 3}.distance_to(crossing) + crossing.distance_to({10, 3}) - 10) > 1e-12)
		throw std::runtime_error("Analytic edge contact missed pass-through.");
}

void check_coordinate_normalization() {
	const Vector2 start{-3, -2}, target{13, 8};
	const std::vector<Polygon> polygons = {
		{{0, 0}, {2, 0}, {2, .6}, {.6, .6}, {.6, 2}, {0, 2}},
		{{5, 3}, {7, 3}, {7, 5}, {5, 5}},
	};
	const auto base = tpp_nonconvex_unordered_solve(start, target, polygons);
	for (double factor : {1e-9, 1e9}) {
		const Vector2 offset{700 * factor, -400 * factor};
		auto transformed = polygons;
		for (auto &polygon : transformed) for (auto &point : polygon) point = point * factor + offset;
		const auto scaled = tpp_nonconvex_unordered_solve(
			start * factor + offset, target * factor + offset, transformed
		);
		if (!scaled.exact || !std::isfinite(scaled.upper_bound)
			|| std::abs(scaled.upper_bound - factor * base.upper_bound) > 1e-7 * std::max(1.0, factor * base.upper_bound)
			|| std::any_of(transformed.begin(), transformed.end(), [&](const auto &polygon) {
				return unordered_detail::contact(scaled.path, polygon, 1e-8).distance > 1e-8;
			}))
			throw std::runtime_error("Coordinate normalization regression.");
	}
}

void check_provided_initial_path() {
	const Vector2 start{0, 0}, target{10, 0};
	const std::vector<Polygon> polygons = {{{4, 2}, {6, 2}, {6, 4}, {4, 4}}};
	const Polygon hint{start, {4, 2}, {6, 2}, target};
	UnorderedTppSolveOptions options;
	options.initial_path = hint;
	options.max_calls = 0;
	const auto interrupted = tpp_nonconvex_unordered_solve(start, target, polygons, options);
	if (interrupted.exact || interrupted.termination != UnorderedTppTermination::CallLimit
		|| interrupted.calls != 0 || interrupted.nodes != 0
		|| std::abs(interrupted.initial_upper_bound - length(hint)) > 1e-10
		|| std::abs(interrupted.upper_bound - length(hint)) > 1e-10
		|| interrupted.lower_bound > 10 + 1e-10)
		throw std::runtime_error("Provided path was treated as a certificate or was not used.");
	options.max_calls = 1000;
	const auto solved = tpp_nonconvex_unordered_solve(start, target, polygons, options);
	if (!solved.exact || solved.upper_bound > interrupted.upper_bound + 1e-9
		|| solved.lower_bound > solved.upper_bound + 1e-9)
		throw std::runtime_error("Search failed with a provided initial path.");
	for (const Polygon invalid : {Polygon{start, target}, Polygon{{1, 0}, {4, 2}, target}}) {
		options.initial_path = invalid;
		try {
			(void)tpp_nonconvex_unordered_solve(start, target, polygons, options);
			throw std::runtime_error("Invalid initial path was accepted.");
		} catch (const std::invalid_argument &) {}
	}
}

void check_cooperative_interruption() {
	const Vector2 start{0, 0}, target{12, 0};
	const std::vector<Polygon> polygons = {{{5, 3}, {7, 3}, {7, 5}, {5, 5}}};
	UnorderedTppSolveOptions options;
	options.stop_requested = [] { return true; };
	const auto partial = tpp_nonconvex_unordered_solve(start, target, polygons, options);
	if (partial.exact || partial.termination != UnorderedTppTermination::Interrupted
		|| partial.path.size() < 2
		|| partial.path.front().distance_to(start) > options.feasibility_tolerance
		|| partial.path.back().distance_to(target) > options.feasibility_tolerance
		|| !std::isfinite(partial.upper_bound) || partial.lower_bound > partial.upper_bound + 1e-9
		|| unordered_detail::contact(partial.path, polygons.front(), options.feasibility_tolerance).distance
			> options.feasibility_tolerance)
		throw std::runtime_error("Cooperative interruption did not preserve a feasible incumbent and bounds: exact="
			+ std::to_string(partial.exact) + ", termination="
			+ std::to_string(static_cast<int>(partial.termination)) + ", path="
			+ std::to_string(partial.path.size()) + ", LB=" + std::to_string(partial.lower_bound)
			+ ", UB=" + std::to_string(partial.upper_bound));
}

void check_progress_reports() {
	// Progress reporting only observes the search: results are identical with and
	// without it, in original units even when the solver normalizes coordinates.
	const std::vector<Polygon> polygons = {
		{{0, 0}, {2, 0}, {2, .6}, {.6, .6}, {.6, 2}, {0, 2}},
		{{5, 3}, {7, 3}, {7, 5}, {5, 5}},
		{{9, 0}, {11, 0}, {11, 2}, {9, 2}},
		{{3, 7}, {5, 7}, {5, 9}, {3, 9}},
	};
	for (const double factor : {1.0, 1e6}) {
		auto scaled = polygons;
		for (auto &polygon : scaled) for (auto &point : polygon) point = point * factor;
		for (const bool cycle : {false, true}) {
			UnorderedTppSolveOptions options;
			auto solve = [&](const UnorderedTppSolveOptions &used) {
				return cycle ? tpp_nonconvex_tspn_solve(scaled, used)
					: tpp_nonconvex_unordered_solve({-3 * factor, -2 * factor}, {13 * factor, 8 * factor}, scaled, used);
			};
			const auto silent = solve(options);
			std::vector<UnorderedTppProgress> reports;
			options.progress_interval_seconds = 1e-9;  // report at every search iteration
			options.progress = [&](const UnorderedTppProgress &report) { reports.push_back(report); };
			const auto observed = solve(options);
			same_search(silent, observed);
			if (reports.empty()) throw std::runtime_error("Progress was never reported.");
			size_t previous_calls = 0;
			double previous_time = 0;
			for (const auto &report : reports) {
				if (report.worker != 0 || report.region_count != scaled.size()
					|| report.calls < previous_calls || report.elapsed_seconds < previous_time
					|| report.lower_bound > report.upper_bound * (1 + 1e-9) + 1e-9
					// Reported bracket must contain the final answer, in original units.
					|| report.upper_bound < observed.upper_bound * (1 - 1e-9)
					|| report.lower_bound > observed.upper_bound * (1 + 1e-9))
					throw std::runtime_error("Progress report is inconsistent with the final result.");
				previous_calls = report.calls;
				previous_time = report.elapsed_seconds;
			}
			// Disabled by a zero interval or a missing callback.
			options.progress_interval_seconds = 0;
			size_t calls = 0;
			options.progress = [&](const UnorderedTppProgress &) { ++calls; };
			(void)solve(options);
			options.progress_interval_seconds = 1;
			options.progress = nullptr;
			(void)solve(options);
			if (calls) throw std::runtime_error("Progress was reported although disabled.");
		}
	}
	// Both portfolio searches report from their own threads.
	UnorderedTppSolveOptions options;
	options.portfolio = true;
	options.progress_interval_seconds = 1e-9;
	std::mutex lock;
	std::vector<size_t> workers;
	options.progress = [&](const UnorderedTppProgress &report) {
		const std::scoped_lock guard(lock);
		workers.push_back(report.worker);
	};
	(void)tpp_nonconvex_tspn_solve(polygons, options);
	if (workers.empty() || std::any_of(workers.begin(), workers.end(), [](size_t worker) { return worker > 1; }))
		throw std::runtime_error("Portfolio progress reports are inconsistent.");
}

void check_initial_heuristic_strategies() {
	const Vector2 start{0, 0}, target{20, 0};
	const std::vector<Polygon> polygons = {
		{{2, 2}, {4, 2}, {4, 4}, {2, 4}},
		{{8, -4}, {10, -4}, {10, -2}, {8, -2}},
		{{14, 1}, {16, 1}, {16, 3}, {14, 3}},
	};
	UnorderedTppSolveOptions baseline_options;
	baseline_options.max_calls = 0;
	baseline_options.max_seconds = 10;
	const auto baseline = tpp_nonconvex_unordered_solve(start, target, polygons, baseline_options);
	if (baseline.initial_upper_bound <= 0 || baseline.initial_convex_refinement_calls != 0)
		throw std::runtime_error("Initial heuristic baseline was not constructed.");

	for (int mask = 1; mask < 16; ++mask) {
		UnorderedTppSolveOptions options = baseline_options;
		options.sampled_perimeter_initial_heuristic = mask & 1;
		options.convex_initial_refinement = mask & 2;
		options.bidirectional_initial_heuristic = mask & 4;
		options.relocate_initial_heuristic = mask & 8;
		if (options.convex_initial_refinement) options.max_calls = 1;
		const auto result = tpp_nonconvex_unordered_solve(start, target, polygons, options);
		if (result.initial_upper_bound > baseline.initial_upper_bound + 1e-9
			|| result.calls > options.max_calls
			|| result.calls != result.relaxation_calls + result.refinement_calls + result.initial_convex_refinement_calls
			|| result.initial_convex_refinement_calls > size_t(options.convex_initial_refinement)
			|| (options.sampled_perimeter_initial_heuristic && result.initial_sampling_work_budget <= 0)
			|| (options.convex_initial_refinement && result.initial_convex_refinement_error.empty()
				&& result.initial_convex_refinement_calls != 1))
			throw std::runtime_error("Initial heuristic strategy violated its budget or non-worsening contract.");
		for (const auto &polygon : polygons)
			if (unordered_detail::contact(result.path, polygon, 1e-8).distance > 1e-8)
				throw std::runtime_error("An initial heuristic strategy returned an infeasible path.");
	}
}

void check_relocation_and_zero_dual() {
    std::mt19937 random(20261003);
    std::uniform_real_distribution<double> coordinate(-5,5);
    for(size_t trial=0;trial<24;++trial) {
        std::vector<Polygon> polygons;
        for(size_t i=0;i<3+trial%4;++i) {
            const double x=coordinate(random),y=coordinate(random);
            polygons.push_back({{x,y},{x+1,y},{x+1,y+1},{x,y+1}});
        }
        for(bool cycle:{false,true}) {
            UnorderedTppSolveOptions options;options.max_calls=0;
            auto solve=[&]{return cycle?tpp_nonconvex_tspn_solve(polygons,options):tpp_nonconvex_unordered_solve({-7,0},{7,0},polygons,options);};
            const auto base=solve();options.relocate_initial_heuristic=true;
            const auto moved=solve();
            if(moved.calls!=0||moved.upper_bound>base.upper_bound+1e-10||moved.order.size()!=polygons.size())
                throw std::runtime_error("Relocation worsened the initial route or violated its call budget.");
            for(const auto &p:polygons)if(unordered_detail::contact(moved.path,p,1e-8).distance>1e-8)
                throw std::runtime_error("Relocation produced an infeasible route.");
            if((cycle&&moved.path.front()!=moved.path.back())||(!cycle&&(moved.path.front()!=base.path.front()||moved.path.back()!=base.path.back())))
                throw std::runtime_error("Relocation changed the route endpoints.");
        }
    }
    const std::vector<Polygon> touching={{{0,0},{1,0},{1,1},{0,1}},{{1,0},{2,0},{2,1},{1,1}},{{1,1},{2,1},{2,2},{1,2}}};
    const auto reference=tpp_nonconvex_unordered_solve({-1,-1},{3,3},touching);
    for(size_t cap:{0,1,5,10000}) {
        UnorderedTppSolveOptions options;options.max_calls=cap;options.interpolated_zero_dual=true;
        const auto result=tpp_nonconvex_unordered_solve({-1,-1},{3,3},touching,options);
        if(result.calls>cap||result.lower_bound>reference.upper_bound+1e-10||result.upper_bound<reference.lower_bound-1e-10)
            throw std::runtime_error("Interpolated zero dual violated bounds or its call cap.");
    }
}

// The float oracle must bracket the exact optimum with proved bounds, and both
// experimental modes must close the global gap with routes the hybrid accepts.
void check_float_oracle() {
    std::mt19937 rng(20261008);
    std::uniform_real_distribution<double> offset(-1, 1);
    std::vector<std::vector<Polygon>> instances = {
        {{{0,0},{1,0},{1,1},{0,1}},{{1,0},{2,0},{2,1},{1,1}},{{1,1},{2,1},{2,2},{1,2}}},
        {{{0,0},{2,0},{2,2},{0,2}},{{2,2},{4,2},{4,4},{2,4}},{{2,0},{4,0},{4,2},{2,2}}},
    };
    for (size_t trial = 0; trial < 40; ++trial) {
        std::vector<Polygon> polygons;
        for (size_t i = 0; i < 3 + trial % 4; ++i) {
            Polygon p = trial % 2 ? Polygon{{0,0},{2,0},{2,2},{0,2}} : Polygon{{0,0},{2,0},{1,1.5}};
            const Vector2 shift{1.7 * double(i % 3) + offset(rng), 1.7 * double(i / 3) + offset(rng)};
            for (auto &v : p) v += shift;
            if (trial % 3 == 0) std::reverse(p.begin(), p.end());
            polygons.push_back(std::move(p));
        }
        instances.push_back(std::move(polygons));
    }
    for (size_t index = 0; index < instances.size(); ++index) {
        const auto &polygons = instances[index];
        const Vector2 start{-2, -1}, target = index % 3 ? Vector2{7, 6} : Vector2{-2, -1};
        const auto exact = tpp_convex_solve_hybrid(start, target, polygons);
        for (double gap : {1e-3, 1e-7}) {
            ConvexFloatOracleOptions options;options.max_gap = gap;
            const auto bounds = tpp_convex_solve_float_certified(start, target, polygons, options);
            // Bounds must always bracket the optimum; closing is required at the
            // search's tolerance scale. A much tighter gap may stay open (the
            // search then falls back to the hybrid oracle).
            const bool closed = bounds.status == ConvexFloatOracleStatus::GapClosed;
            if ((gap >= 1e-3 && !closed) || bounds.contacts.size() != polygons.size()
                || bounds.lower_bound > exact.upper_bound || bounds.upper_bound < exact.lower_bound
                || (closed && bounds.upper_bound - bounds.lower_bound > gap))
                { std::ostringstream message; message.precision(17);
                  message << "Float oracle bounds do not bracket the optimum (instance " << index << ", gap " << gap << "): status "
                      << to_string(bounds.status) << " L " << bounds.lower_bound << " U " << bounds.upper_bound
                      << " exact [" << exact.lower_bound << ", " << exact.upper_bound << "] contacts " << bounds.contacts.size()
                      << " newton " << bounds.newton_iterations << " trace_failed " << bounds.trace_failed;
                  throw std::runtime_error(message.str()); }
            for (size_t i = 0; i < polygons.size(); ++i)
                if (unordered_detail::contact(Polygon{bounds.contacts[i], bounds.contacts[i]}, polygons[i], 0).distance > 0)
                    throw std::runtime_error("Float oracle contact outside its polygon.");
        }
        UnorderedTppSolveOptions reference_options;reference_options.relative_gap = 1e-3;reference_options.absolute_gap = 0;
        reference_options.float_recovery = false;
        const auto reference = tpp_nonconvex_unordered_solve(start, target, polygons, reference_options);
        {
            auto options = reference_options;options.float_recovery = true;
            const auto result = tpp_nonconvex_unordered_solve(start, target, polygons, options);
            if (!result.exact || result.lower_bound > reference.upper_bound + 1e-12 || reference.lower_bound > result.upper_bound + 1e-12)
                throw std::runtime_error("Float oracle search disagrees with the hybrid search (instance " + std::to_string(index) + ").");
            for (const auto &polygon : polygons)
                if (unordered_detail::contact(result.path, polygon, options.feasibility_tolerance).distance > options.feasibility_tolerance)
                    throw std::runtime_error("Float oracle search returned an infeasible path.");
        }
    }
}

void check_intra_instance_threads() {
	const Vector2 start{0, 0}, target{28, 0};
	std::vector<Polygon> polygons;
	for (size_t i = 0; i < 5; ++i) {
		Polygon l = {{0, 0}, {2, 0}, {2, .6}, {.6, .6}, {.6, 2}, {0, 2}};
		for (auto &point : l) point += Vector2{3.0 + 5.0 * i, i % 2 ? -12.0 : 12.0};
		polygons.push_back(std::move(l));
	}
	UnorderedTppSolveOptions options;
	options.threads = 4;
	options.relative_gap = 1e-3;
	options.absolute_gap = 0;
	options.max_calls = 100000;
	options.max_seconds = 20;
	options.sampled_perimeter_initial_heuristic = true;
	options.convex_initial_refinement = true;
	options.bidirectional_initial_heuristic = true;
	const auto result = tpp_nonconvex_unordered_solve(start, target, polygons, options);
	if (!result.exact || result.threads != options.threads || result.parallel_oracle_calls < 2
		|| result.parallel_oracle_batches == 0
		|| result.calls != result.relaxation_calls + result.refinement_calls + result.initial_convex_refinement_calls
		|| result.upper_bound - result.lower_bound > 1e-3 * std::abs(result.upper_bound) + 1e-10)
		throw std::runtime_error("Multi-threaded search failed: exact=" + std::to_string(result.exact)
			+ ", calls=" + std::to_string(result.calls) + ", parallel=" + std::to_string(result.parallel_oracle_calls)
			+ ", batches=" + std::to_string(result.parallel_oracle_batches)
			+ ", gap=" + std::to_string(result.upper_bound - result.lower_bound)
			+ ", termination=" + std::to_string(static_cast<int>(result.termination)));
	for (const auto &polygon : polygons)
		if (unordered_detail::contact(result.path, polygon, options.feasibility_tolerance).distance > options.feasibility_tolerance)
			throw std::runtime_error("Multi-threaded search returned an infeasible path.");
}

void check_endpoint_portfolio() {
	const Vector2 start{0, 0}, target{20, 0};
	const std::vector<Polygon> polygons = {
		{{2, 2}, {4, 2}, {4, 4}, {2, 4}},
		{{8, -4}, {10, -4}, {10, -2}, {8, -2}},
		{{14, 1}, {16, 1}, {16, 3}, {14, 3}},
	};
	UnorderedTppSolveOptions reference_options;
	reference_options.max_calls = 100000;
	reference_options.max_seconds = 20;
	const auto reference = tpp_nonconvex_unordered_solve(start, target, polygons, reference_options);
	if (!reference.exact) throw std::runtime_error("Endpoint portfolio reference did not close.");
	for (bool share : {true, false}) {
		UnorderedTppSolveOptions options = reference_options;
		options.portfolio = true;
		options.portfolio_share_incumbents = share;
		const auto result = tpp_nonconvex_unordered_solve(start, target, polygons, options);
		if (!result.exact || result.portfolio_workers != 2 || result.portfolio_runs.size() != 2
			|| result.portfolio_winner >= 2 || result.lower_bound > reference.upper_bound + 1e-7
			|| result.upper_bound + 1e-7 < reference.lower_bound
			|| result.calls > options.max_calls)
			throw std::runtime_error("Endpoint portfolio disagreed with the isolated reference.");
		for (const auto &polygon : polygons)
			if (unordered_detail::contact(result.path, polygon, options.feasibility_tolerance).distance
				> options.feasibility_tolerance)
				throw std::runtime_error("Endpoint portfolio returned an infeasible path.");
		if (result.path.size() < 2 || result.path.front() != start || result.path.back() != target)
			throw std::runtime_error("Endpoint portfolio lost its fixed endpoints.");
		for (size_t cap : {size_t{0}, size_t{1}, size_t{3}}) {
			options.max_calls = cap;
			const auto limited = tpp_nonconvex_unordered_solve(start, target, polygons, options);
			if (limited.calls > cap || limited.lower_bound > reference.upper_bound + 1e-7
				|| limited.upper_bound + 1e-7 < reference.lower_bound
				|| limited.path.size() < 2 || limited.path.front() != start || limited.path.back() != target)
				throw std::runtime_error("Endpoint portfolio violated its shared call cap or bounds.");
			for (const auto &polygon : polygons)
				if (unordered_detail::contact(limited.path, polygon, options.feasibility_tolerance).distance
					> options.feasibility_tolerance)
					throw std::runtime_error("Endpoint portfolio cap returned an infeasible path.");
		}
	}
}

void check_lower_dimensional_regions() {
    struct Case { Vector2 start,target; std::vector<Polygon> regions; double optimum; };
    const std::vector<Case> cases{
        {{0,0},{0,0},{{{2,1}}},2*std::sqrt(5.)},
        {{0,0},{0,0},{{{2,-1},{2,1}}},4},
        {{0,0},{0,0},{{{-2,0},{2,0}},{{0,-2},{0,2}}},0},
        {{0,0},{0,0},{{{1,0},{3,0}},{{2,0},{4,0}}},4},
        {{-1,1},{2,1},{{{0,0}},{{1,0},{2,0}}},1+2*std::sqrt(2.)},
        {{0,0},{0,0},{{{2,1}},{{2,1}},{{2,-1}}},2+2*std::sqrt(5.)},
        {{0,0},{0,0},{{{4,1}},{{2,-1},{2,2}},{{1,-1},{3,-1},{2,1}}},2*std::sqrt(17.)},
    };
    for(const auto &c:cases) {
        check(c.start,c.target,c.regions);
        for(size_t threads:{size_t(1),size_t(2)}) {
            UnorderedTppSolveOptions options; options.threads=threads;
            const auto result=tpp_nonconvex_unordered_solve(c.start,c.target,c.regions,options);
            if(!result.exact || std::abs(result.upper_bound-c.optimum)>1e-7
                || result.lower_bound>c.optimum+1e-7)
                throw std::runtime_error("Point/segment analytic optimum mismatch.");
            options.initial_path=result.path;
            const auto supplied=tpp_nonconvex_unordered_solve(c.start,c.target,c.regions,options);
            if(!supplied.exact || std::abs(supplied.upper_bound-c.optimum)>1e-7)
                throw std::runtime_error("Point/segment initial path rejected or changed optimum.");
        }
        UnorderedTppSolveOptions heuristic;
        heuristic.sampled_perimeter_initial_heuristic=true;
        heuristic.convex_initial_refinement=true;
        heuristic.bidirectional_initial_heuristic=true;
        heuristic.relocate_initial_heuristic=true;
        heuristic.trace=true;
        const auto polished=tpp_nonconvex_unordered_solve(c.start,c.target,c.regions,heuristic);
        if(!polished.exact || std::abs(polished.upper_bound-c.optimum)>1e-7
            || !polished.initial_convex_refinement_error.empty())
            throw std::runtime_error("Point/segment heuristic options failed.");
    }
    // Check fixed-order contact materialization when a point lies beyond the
    // supporting line's finite segment, and when the prefix is collinear.
    DynamicConvexTppWorkspace workspace;
    const auto fixed=tpp_convex_solve_certified({-1,1},{2,1},{{{0,0}},{{1,0},{2,0}}},workspace,0);
    if(fixed.path.size()!=4 || fixed.path[2]!=Vector2{1,0}
        || std::abs(fixed.upper_bound-(1+2*std::sqrt(2.)))>1e-12)
        throw std::runtime_error("Segment endpoint directional derivative regression.");
    bool rejected=false;
    try {tpp_nonconvex_unordered_solve({0,0},{1,0},{{{0,1},{1,1},{2,1}}});}
    catch(const std::invalid_argument &){rejected=true;}
    if(!rejected)throw std::runtime_error("Collinear polygon must use the segment representation.");
    std::cout<<"Passed seven analytic point/segment cases, initial paths and two-thread checks.\n";
}

int main() {
	try {
        check_prepared_contacts();check_relocation_and_zero_dual();
        check_sequence_storage();
		check_lower_dimensional_regions();
		check_oracle_certificates();
		check_coordinate_normalization();
		check_provided_initial_path();
		check_cooperative_interruption();
		check_progress_reports();
		check_initial_heuristic_strategies();
		check_intra_instance_threads();
		check_float_oracle();
		check_endpoint_portfolio();
		check({0, 0}, {10, 0}, {});
		check({0, 0}, {10, 0}, {{{2, -1}, {3, -1}, {3, 1}, {2, 1}}});
		check({0, 0}, {0, 0}, {{{2, -1}, {3, -1}, {3, 1}, {2, 1}}});
		check({0, 0}, {0, 0}, {{{-2, -2}, {2, -2}, {2, 2}, {-2, 2}}});
		check({1, 1.2}, {1.8, 1.3}, {{{0, 0}, {2, 0}, {2, .6}, {.6, .6}, {.6, 2}, {0, 2}}});
		check({0, 0}, {0, 1}, {{{-2, -2}, {2, -2}, {2, 2}, {1, 2}, {1, -1}, {-1, -1}, {-1, 2}, {-2, 2}}});
		std::mt19937 rng(342026);
		std::uniform_real_distribution<double> offset(-1, 1);
		for (size_t trial = 0; trial < 80; ++trial) {
			std::vector<Polygon> polygons;
			const size_t count = 2 + trial % 4;
			for (size_t i = 0; i < count; ++i) {
				const double x = (trial % 2 ? 1.5 : 5) * double(i % 3) + offset(rng);
				const double y = 5 * double(i / 3) + offset(rng);
				Polygon p = {{0, 0}, {2, 0}, {2, .6}, {.6, .6}, {.6, 2}, {0, 2}};
				if (trial % 3 == 0) p = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
				for (auto &v : p) v += Vector2{x, y};
				if (trial % 4 == 0) std::reverse(p.begin(), p.end());
				polygons.push_back(p);
			}
			
			check({-3, -2}, trial % 5 ? Vector2{13, 8} : Vector2{-3, -2}, polygons);
		}
		std::cout << "Passed oracle certificate/contact regressions, 86 exhaustive-order cases, 344 interrupted-search checks, and multi-threaded child evaluation.\n";
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
