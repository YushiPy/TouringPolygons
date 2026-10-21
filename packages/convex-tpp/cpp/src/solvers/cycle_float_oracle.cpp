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
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>

namespace tpp {
struct PreparedFloatRegion {
    std::optional<std::vector<Vector2>> region;
    std::vector<ConvexArithmeticPoint<double>> kernel;
    // The construction's starting point, computed as CycleRefinement::run does.
    ConvexArithmeticPoint<double> centroid{};
    Vector2 mean{};
    // The input it was prepared from, for a cheap check of identifier reuse.
    std::size_t size=0;Vector2 front{},back{};
    // Common regions with other prepared regions (move_zero_blocks), each
    // valid while that region is alive. One workspace serves one thread.
    mutable std::map<const PreparedFloatRegion*,std::pair<std::weak_ptr<const PreparedFloatRegion>,
        std::vector<ConvexArithmeticPoint<double>>>> common;
};
namespace {
using Clock=std::chrono::steady_clock;
using Region=detail::FloatRegion;
using Interval=detail::CycleInterval;
using Kernel=detail::CycleRefinement<double>;

double since(Clock::time_point began) {return std::chrono::duration<double>(Clock::now()-began).count();}

// Best proved interval over all proposals of one call.
struct CycleBounds {
    const std::vector<const Region*> &regions;
    const std::vector<Vector2> &means;
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
            const auto proved=detail::prove_contact(q[i],*regions[i],&means[i]);
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
        const double link=short_link(q);
        auto duals=detail::cycle_dual_proposals(q,link);
        if(duals.size()>1)duals.push_back(detail::cycle_dual_by_runs(regions,q,link,reference));
        return prove(q,duals);
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
std::shared_ptr<const PreparedFloatRegion> ConvexCycleWorkspace::prepared_region(std::uint64_t id,const std::vector<Vector2> &p) {
    auto same=[](Vector2 a,Vector2 b) {
        return std::bit_cast<std::uint64_t>(a.x)==std::bit_cast<std::uint64_t>(b.x)&&
               std::bit_cast<std::uint64_t>(a.y)==std::bit_cast<std::uint64_t>(b.y);
    };
    if(auto found=prepared_regions_.find(id);found!=prepared_regions_.end()) {
        const auto &prepared=*found->second;
        if(prepared.size==p.size()&&(p.empty()||(same(prepared.front,p.front())&&same(prepared.back,p.back()))))
            return found->second;
    }
    auto prepared=std::make_shared<PreparedFloatRegion>();
    prepared->size=p.size();
    if(!p.empty()){prepared->front=p.front();prepared->back=p.back();}
    prepared->region=detail::binary_region(p);
    if(prepared->region) {
        for(const auto &v:*prepared->region)prepared->kernel.emplace_back(v);
        for(const auto &v:prepared->kernel)prepared->centroid=prepared->centroid+v;
        prepared->centroid=prepared->centroid*(1.0/double(prepared->kernel.size()));
        prepared->mean=detail::vertex_mean(*prepared->region);
    }
    return prepared_regions_[id]=std::move(prepared);
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
    // Regions by reference: prepared once per caller identifier, or else
    // normalized (from the workspace's content-keyed cache) into owned copies.
    std::vector<const Region*> regions;regions.reserve(k);
    std::vector<Kernel::View> views;views.reserve(k);
    std::vector<Region> owned;owned.reserve(k);
    std::vector<Kernel::Polygon> owned_kernel;owned_kernel.reserve(k);
    std::vector<std::shared_ptr<const PreparedFloatRegion>> held;
    std::vector<Vector2> means;means.reserve(k);
    Vector2 reference{};
    const bool by_id=options.workspace&&options.region_ids&&options.region_ids->size()==k;
    for(size_t i=0;i<k;++i) {
        if(by_id) {
            auto prepared=options.workspace->prepared_region((*options.region_ids)[i],input[i]);
            if(!prepared->region)return finish(ConvexFloatOracleStatus::Unsupported);
            regions.push_back(&*prepared->region);views.emplace_back(prepared->kernel);
            means.push_back(prepared->mean);reference+=means.back()/double(k);
            held.push_back(std::move(prepared));
            continue;
        }
        if(options.workspace) {
            const auto &normalized=options.workspace->float_region(input[i]);
            if(!normalized)return finish(ConvexFloatOracleStatus::Unsupported);
            owned.push_back(*normalized);
        } else {
            auto normalized=detail::binary_region(input[i]);
            if(!normalized)return finish(ConvexFloatOracleStatus::Unsupported);
            owned.push_back(std::move(*normalized));
        }
        regions.push_back(&owned.back());
        means.push_back(detail::vertex_mean(owned.back()));reference+=means.back()/double(k);
    }
    if(!reference.is_finite())return finish(ConvexFloatOracleStatus::Unsupported);
    CycleBounds bounds{regions,means,reference,options.max_gap,options.cutoff};
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
            if(!by_id)for(const auto *region:regions) {
                Kernel::Polygon q;for(const auto &v:*region)q.emplace_back(v);
                owned_kernel.push_back(std::move(q));views.emplace_back(owned_kernel.back());
            }
            const auto &p=views;
            // Prepared regions keep the common regions of overlapping pairs.
            const Kernel::PairRegion common=[&](size_t i,size_t j)->const Kernel::Polygon* {
                if(!by_id||views[i].size()<Kernel::kLogarithmicVertices||views[j].size()<Kernel::kLogarithmicVertices)return nullptr;
                auto &cache=held[i]->common;const auto *other=held[j].get();
                if(const auto found=cache.find(other);found!=cache.end()&&found->second.first.lock().get()==other)
                    return &found->second.second;
                if(cache.size()>=16)cache.clear();
                auto region=Kernel::intersect(Kernel::Polygon(views[i].begin(),views[i].end()),views[j]);
                auto &slot=cache[other];slot={held[j],std::move(region)};
                return &slot.second;
            };
            // A common point gives the zero cycle. Regions 0 and 1 separated by
            // an edge of one facing the other have none: that is the empty
            // intersection the clip would reach after copying region 0.
            const bool apart=Kernel::separated_toward(p[0],p[1],means[1]-means[0])||
                             Kernel::separated_toward(p[1],p[0],means[0]-means[1]);
            Kernel::Polygon joint;
            size_t joined=1;
            if(!apart) {
                if(const auto *known=common(0,1)){joint=*known;joined=2;}
                else joint.assign(p.front().begin(),p.front().end());
            }
            for(size_t i=joined;i<k&&!joint.empty();++i)joint=Kernel::intersect(std::move(joint),p[i]);
            if(!joint.empty()) {
                ++result.candidates;
                const std::vector<Vector2> q(k,joint.front().external());
                if(bounds.prove_proposal(q)){result.construction_closed=true;return finish(closed_status());}
            }
            Kernel::Polygon start;
            if(initial)for(const auto &v:*initial)start.emplace_back(v);
            else if(by_id)for(const auto &prepared:held)start.push_back(prepared->centroid);
            static const std::vector<int> none;
            const auto &features=options.initial_features?*options.initial_features:none;
            const Kernel::PairScope scope(&common);
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
        for(const auto *p:regions)scale=std::max(scale,detail::coordinate_reach(*p,reference));
        if(scale>0&&std::isfinite(scale)) {
            // The seed's descent direction at each contact: the sum of the unit
            // vectors toward its neighbors.
            auto outward=[&](size_t i) {
                Vector2 g{};
                if(seed)for(const size_t j:{(i+k-1)%k,(i+1)%k}) {
                    const auto d=(*seed)[j]-(*seed)[i];const double length=d.length();
                    if(length>0)g+=d/length;
                }
                return g;
            };
            std::vector<detail::ChainNode> nodes;nodes.reserve(k);bool windowed=false;
            for(size_t i=0;i<k;++i) {
                nodes.push_back(detail::region_node(*regions[i],reference,scale,seed?&(*seed)[i]:nullptr,options.warm_interior_fraction,
                    outward(i)));
                faces+=nodes.back().faces.size();windowed|=bool(nodes.back().refine);
            }
            const double target_gap=options.max_gap>0?options.max_gap:1e-9*std::max(1.0,std::abs(options.cutoff));
            // Proposition (cycle): at a barrier minimizer D(u) >= length - mu*(k+F)
            // in scaled coordinates, so this level closes target_gap/2.
            const double mu_end=std::max(1e-15,.5*target_gap/(scale*double(k+faces)));
            const double mu=seed?std::max(mu_end,std::min(1e-3,mu_end*options.warm_mu_ratio)):1e-1;
            const auto stats=detail::chain_interior_point(nodes,true,mu,mu_end,options.max_newton_iterations,
                [&](const detail::ChainLevel &level) {
                    std::vector<Vector2> q;q.reserve(k);
                    for(size_t i=0;i<k;++i)q.push_back(detail::region_contact(*regions[i],level,i,reference,scale));
                    return bounds.prove(q,{level.duals});
                },windowed?.5*target_gap/scale:0);
            if(stats){result.newton_iterations=stats->iterations;result.barrier_levels=stats->levels;}
        }
        if(bounds.closed()){result.polish_closed=true;return finish(closed_status());}
    }
    export_bounds();
    return finish(ConvexFloatOracleStatus::Open);
}
} // namespace tpp
