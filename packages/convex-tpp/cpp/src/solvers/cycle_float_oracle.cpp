#include "tpp/convex/float_oracle.h"
#include "tpp/convex/cycle_certificate.h"
#include "cycle_execution.h"
#include "cycle_refinement.h"
#include "float_chain.h"
#include "float_proof.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace tpp {
namespace {
using Clock=std::chrono::steady_clock;
using Region=detail::FloatRegion;
using Interval=detail::CycleInterval;
using Kernel=detail::CycleRefinement<double>;

double since(Clock::time_point began) {return std::chrono::duration<double>(Clock::now()-began).count();}

// Best proved interval over all proposals of one call.
struct CycleBounds {
    const std::vector<Region> &regions;
    Vector2 reference;
    double max_gap,cutoff;
    double lower=0,upper=INFINITY;
    std::vector<Vector2> contacts;
    // Both closures need a cycle of proved contacts, as CertifiedBound does.
    bool closed() const {
        if(!std::isfinite(upper))return false;
        if(lower>=cutoff)return true;
        return max_gap>0&&(Interval(upper)-Interval(lower)).hi<=max_gap;
    }
    // Algorithm ProvaLimites for a cycle: proved membership and enclosed
    // length for U, each dual proposal through D(u) for L.
    bool prove(const std::vector<Vector2> &q,const std::vector<std::vector<Vector2>> &duals) {
        detail::CyclePhaseScope phase(detail::CyclePhase::IntervalProof);
        for(const auto &u:duals)lower=std::max(lower,detail::cycle_dual_lower(regions,u,reference));
        std::vector<detail::IntervalPoint> boxes;std::vector<Vector2> points;
        boxes.reserve(q.size());points.reserve(q.size());
        for(size_t i=0;i<q.size();++i) {
            const auto proved=detail::prove_contact(q[i],regions[i]);
            if(!proved)return closed();
            boxes.push_back(proved->box);points.push_back(proved->point);
        }
        const double length=detail::enclosed_length_upper(boxes,true);
        if(length<upper){upper=length;contacts=std::move(points);}
        return closed();
    }
    double short_link(const std::vector<Vector2> &q) const {
        double scale=0;
        for(const auto &v:q)scale=std::max({scale,std::abs(v.x),std::abs(v.y)});
        return std::max(32*std::numeric_limits<double>::epsilon()*scale,
            max_gap>0?max_gap/(16*double(q.size())):0);
    }
    bool prove_proposal(const std::vector<Vector2> &q) {
        return prove(q,detail::cycle_dual_proposals(q,short_link(q)));
    }
};

} // namespace

std::size_t ConvexCycleWorkspace::BitsHash::operator()(const std::vector<Vector2> &p) const {
    std::size_t h=p.size();
    for(const auto &v:p)for(double c:{v.x,v.y})
        h^=std::hash<std::uint64_t>{}(std::bit_cast<std::uint64_t>(c))+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
    return h;
}
bool ConvexCycleWorkspace::BitsEqual::operator()(const std::vector<Vector2> &a,const std::vector<Vector2> &b) const {
    return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin(),[](Vector2 p,Vector2 q) {
        return std::bit_cast<std::uint64_t>(p.x)==std::bit_cast<std::uint64_t>(q.x)&&
               std::bit_cast<std::uint64_t>(p.y)==std::bit_cast<std::uint64_t>(q.y);});
}
const std::optional<std::vector<Vector2>> &ConvexCycleWorkspace::float_region(const std::vector<Vector2> &p) {
    if(auto found=float_regions_.find(p);found!=float_regions_.end())return found->second;
    if(float_regions_.size()>=(std::size_t(1)<<16))float_regions_.clear();
    return float_regions_.emplace(p,detail::binary_region(p)).first->second;
}

ConvexCycleFloatResult tpp_convex_solve_cycle_float_certified(const std::vector<std::vector<Vector2>> &input,
        const ConvexCycleFloatOptions &options) {
    const auto began=Clock::now();
    ConvexCycleFloatResult result;
    auto finish=[&](ConvexFloatOracleStatus status){result.status=status;result.total_seconds=since(began);return result;};
    const size_t k=input.size();
    if(!detail::cycle_interval_environment()||k<2||std::isnan(options.cutoff)||
       !(options.max_gap>0||std::isfinite(options.cutoff)))return finish(ConvexFloatOracleStatus::Unsupported);
    std::vector<Region> regions;regions.reserve(k);
    for(const auto &p:input) {
        if(options.workspace) {
            const auto &normalized=options.workspace->float_region(p);
            if(!normalized)return finish(ConvexFloatOracleStatus::Unsupported);
            regions.push_back(*normalized);
            continue;
        }
        auto normalized=detail::binary_region(p);
        if(!normalized)return finish(ConvexFloatOracleStatus::Unsupported);
        regions.push_back(std::move(*normalized));
    }
    Vector2 reference{};
    for(const auto &p:regions)reference+=detail::vertex_mean(p)/double(k);
    if(!reference.is_finite())return finish(ConvexFloatOracleStatus::Unsupported);
    CycleBounds bounds{regions,reference,options.max_gap,options.cutoff};
    auto export_bounds=[&]{result.contacts=bounds.contacts;result.lower_bound=bounds.lower;result.upper_bound=bounds.upper;};
    auto closed_status=[&] {
        export_bounds();
        return bounds.lower>=options.cutoff?ConvexFloatOracleStatus::CutoffReached:ConvexFloatOracleStatus::GapClosed;
    };
    const std::vector<Vector2> *initial=options.initial_contacts&&options.initial_contacts->size()==k&&
        std::all_of(options.initial_contacts->begin(),options.initial_contacts->end(),[](Vector2 q){return q.is_finite();})
        ?options.initial_contacts:nullptr;
    // Seed for the polish: the construction candidate with the best proved
    // upper bound, else the last one, else the inherited contacts.
    std::optional<std::vector<Vector2>> seed,last;
    try {
        if(options.construct) {
            Kernel::Polygons p;
            for(const auto &region:regions) {
                Kernel::Polygon q;for(const auto &v:region)q.emplace_back(v);p.push_back(std::move(q));
            }
            // A common point gives the zero cycle.
            auto joint=p.front();
            for(size_t i=1;i<k&&!joint.empty();++i)joint=Kernel::intersect(std::move(joint),p[i]);
            if(!joint.empty()) {
                ++result.candidates;
                const std::vector<Vector2> q(k,joint.front().external());
                if(bounds.prove_proposal(q)){result.construction_closed=true;return finish(closed_status());}
            }
            Kernel::Polygon start;
            if(initial)for(const auto &v:*initial)start.emplace_back(v);
            static const std::vector<int> none;
            const auto &features=options.initial_features?*options.initial_features:none;
            const bool accepted=Kernel::run(p,[&](const Kernel::Polygon &candidate,const std::vector<int> &active,bool) {
                ++result.candidates;
                std::vector<Vector2> q;q.reserve(k);
                for(const auto &v:candidate)q.push_back(v.external());
                const double before=bounds.upper;
                const bool done=bounds.prove_proposal(q);
                if(bounds.upper<before||!seed)seed=bounds.upper<before?bounds.contacts:q;
                last=std::move(q);
                if(done)result.active_features=active;
                return done;
            },start,features);
            if(accepted){result.construction_closed=true;return finish(closed_status());}
        }
    } catch(const detail::CycleInterrupted &) {
        result.interrupted=true;export_bounds();return finish(ConvexFloatOracleStatus::Open);
    } catch(const std::exception &) {} // a failed construction only loses its proposal
    if(!seed&&last)seed=last;
    if(!seed&&initial)seed=*initial;
    if(options.polish) {
        detail::CyclePhaseScope phase(detail::CyclePhase::Polish);
        result.polish_attempted=true;
        double scale=0;size_t faces=0;
        for(const auto &p:regions)for(const auto &v:p)scale=std::max({scale,std::abs(v.x-reference.x),std::abs(v.y-reference.y)});
        if(scale>0&&std::isfinite(scale)) {
            std::vector<detail::ChainNode> nodes;nodes.reserve(k);
            for(size_t i=0;i<k;++i) {
                nodes.push_back(detail::region_node(regions[i],reference,scale,seed?&(*seed)[i]:nullptr,options.warm_interior_fraction));
                faces+=nodes.back().faces.size();
            }
            const double target_gap=options.max_gap>0?options.max_gap:1e-9*std::max(1.0,std::abs(options.cutoff));
            // Proposition (cycle): at a barrier minimizer D(u) >= length - mu*(k+F)
            // in scaled coordinates, so this level closes target_gap/2.
            const double mu_end=std::max(1e-15,.5*target_gap/(scale*double(k+faces)));
            const double mu=seed?std::max(mu_end,std::min(1e-3,mu_end*options.warm_mu_ratio)):1e-1;
            const auto stats=detail::chain_interior_point(nodes,true,mu,mu_end,options.max_newton_iterations,
                [&](const detail::ChainLevel &level) {
                    std::vector<Vector2> q;q.reserve(k);
                    for(size_t i=0;i<k;++i)q.push_back(detail::region_contact(regions[i],level,i,reference,scale));
                    return bounds.prove(q,{level.duals});
                });
            if(stats){result.newton_iterations=stats->iterations;result.barrier_levels=stats->levels;}
        }
        if(bounds.closed()){result.polish_closed=true;return finish(closed_status());}
    }
    export_bounds();
    return finish(ConvexFloatOracleStatus::Open);
}
} // namespace tpp
