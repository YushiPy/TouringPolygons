#include "tpp/convex/float_oracle.h"
#include "binary_dual.h"
#include "float_chain.h"
#include "float_proof.h"
#include "cycle_refinement.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>

namespace tpp {
namespace detail {
// Defined in hybrid.cpp, next to the trace replay it reuses.
std::vector<Vector2> double_candidate_chain(const Vector2 &start,const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);
}

namespace {
using Clock=std::chrono::steady_clock;
using Polygon=std::vector<Vector2>;
using Interval=detail::CycleInterval;
using IntervalPoint=detail::IntervalPoint;

double since(Clock::time_point began) {return std::chrono::duration<double>(Clock::now()-began).count();}

// Same convex set: consecutive duplicates removed, counter-clockwise order.
std::optional<Polygon> counter_clockwise(const Polygon &input) {
    Polygon p;
    for(const auto &v:input)if(p.empty()||v!=p.back())p.push_back(v);
    while(p.size()>1&&p.front()==p.back())p.pop_back();
    if(p.size()<3)return {};
    double area=0;
    for(size_t i=0;i<p.size();++i)area+=p[i].cross(p[(i+1)%p.size()]);
    if(!(std::abs(area)>0))return {};
    if(area<0)std::reverse(p.begin(),p.end());
    return p;
}

using detail::proved_inside;
using detail::vertex_mean;

// Moves an unproved contact a tiny fraction toward the vertex mean. The
// length bound is computed afterwards, so the move only costs bound quality.
bool make_proved(Vector2 &q,const Polygon &p) {
    if(proved_inside(q,p))return true;
    const auto center=vertex_mean(p);
    for(double fraction:{0x1p-45,0x1p-40,0x1p-30,0x1p-20,0x1p-10}) {
        const auto candidate=q+(center-q)*fraction;
        if(proved_inside(candidate,p)){q=candidate;return true;}
    }
    return false;
}

double length_upper(const Polygon &chain) {
    Interval length;
    for(size_t i=1;i<chain.size();++i) {
        if(chain[i]==chain[i-1])continue;
        const auto d=IntervalPoint(chain[i])-IntervalPoint(chain[i-1]);
        length=length+(d.x.square()+d.y.square()).sqrt();
    }
    return length.finite()?length.hi:INFINITY;
}

// D(u) = (t-s).u_n + sum_i min_{v in P_i} (v-s).(u_i-u_{i+1}); valid for any u
// in the unit disk. Each proposal is replaced by a binary vector proved to lie
// in the disk (or zero), and the sum is enclosed with directed rounding.
double dual_lower(Vector2 start,Vector2 target,const std::vector<Polygon> &polygons,const std::vector<Vector2> &proposal) {
    if(proposal.size()!=polygons.size()+1)return -INFINITY;
    std::vector<IntervalPoint> u;u.reserve(proposal.size());
    for(const auto &v:proposal) {
        if(!v.is_finite()){u.emplace_back();continue;}
        u.emplace_back(detail::binary_dual_vector(IntervalPoint(v)));
    }
    const IntervalPoint origin(start);
    Interval dual=(IntervalPoint(target)-origin).dot(u.back());
    for(size_t i=0;i<polygons.size();++i) {
        const auto normal=u[i]-u[i+1];
        Interval support(INFINITY);
        for(const auto &v:polygons[i]) {
            const auto term=normal.dot(IntervalPoint(v)-origin);
            support.lo=std::min(support.lo,term.lo);
            support.hi=std::min(support.hi,term.hi);
        }
        dual=dual+support;
    }
    return dual.finite()?dual.lo:-INFINITY;
}

double direct_lower(Vector2 start,Vector2 target) {
    const auto d=IntervalPoint(target)-IntervalPoint(start);
    const auto norm=(d.x.square()+d.y.square()).sqrt();
    return norm.finite()?norm.lo:0;
}

struct Bounds {
    double lower=-INFINITY,upper=INFINITY;
    Polygon contacts;
    void offer_lower(double value) {lower=std::max(lower,value);}
    void offer_upper(Vector2 start,Vector2 target,Polygon interior) {
        Polygon chain{start};chain.insert(chain.end(),interior.begin(),interior.end());chain.push_back(target);
        const double value=length_upper(chain);
        if(value<upper){upper=value;contacts=std::move(interior);}
    }
    bool closed(const ConvexFloatOracleOptions &options) const {
        if(lower>=options.cutoff)return true;
        if(!(options.max_gap>0)||!std::isfinite(upper)||!std::isfinite(lower))return false;
        return (Interval(upper)-Interval(lower)).hi<=options.max_gap;
    }
};

// Link directions of a chain. Links no longer than short_link (zero ones
// included) borrow a neighbour's direction, or the start-target direction;
// as in the hybrid interval certificate, length only selects a proposal.
std::vector<std::vector<Vector2>> chain_duals(const Polygon &chain,double short_link) {
    std::vector<Vector2> base;std::vector<bool> zero;
    for(size_t i=1;i<chain.size();++i) {
        const auto d=IntervalPoint(chain[i])-IntervalPoint(chain[i-1]);
        const auto norm=(d.x.square()+d.y.square()).sqrt();
        base.push_back(detail::binary_dual_direction(chain[i-1],chain[i]));
        zero.push_back(base.back()==Vector2{}||!norm.finite()||norm.hi<=short_link);
    }
    if(std::none_of(zero.begin(),zero.end(),[](bool z){return z;}))return {base};
    const auto direct=detail::binary_dual_direction(chain.front(),chain.back());
    std::vector<std::vector<Vector2>> policies(3,base);
    for(int policy=0;policy<3;++policy)for(size_t i=0;i<base.size();++i)if(zero[i]) {
        if(policy==2){policies[policy][i]=direct;continue;}
        std::optional<size_t> left,right;
        for(size_t j=i;j>0;)if(!zero[--j]){left=j;break;}
        for(size_t j=i+1;j<base.size();++j)if(!zero[j]){right=j;break;}
        const auto chosen=policy==0?(left?left:right):(right?right:left);
        policies[policy][i]=chosen?base[*chosen]:direct;
    }
    return policies;
}

// Log-barrier Newton method on the smoothed lengths (float_chain.h), in
// coordinates relative to start divided by scale, with start and target as
// fixed nodes of an open chain. At a barrier minimizer the smoothed directions
// satisfy u_i - u_{i+1} = sum (mu/slack) n_f, so D(u) >= length - mu*(links+faces)
// (scaled); the stopping level follows from the requested gap.
struct Polish {
    std::vector<Vector2> contacts,duals;
    std::size_t iterations=0,levels=0;
};

std::optional<Polish> interior_point(Vector2 start,Vector2 target,const std::vector<Polygon> &polygons,
        const Polygon *warm,double gap,double cutoff,const ConvexFloatOracleOptions &options,
        const std::function<bool(const Polish&)> &done) {
    const size_t n=polygons.size();
    double scale=std::max(std::abs(target.x-start.x),std::abs(target.y-start.y));
    for(const auto &p:polygons)for(const auto &v:p)scale=std::max({scale,std::abs(v.x-start.x),std::abs(v.y-start.y)});
    if(!(scale>0)||!std::isfinite(scale))return {};
    auto local=[&](Vector2 v){return Vector2{(v.x-start.x)/scale,(v.y-start.y)/scale};};
    using Kind=detail::ChainNode::Kind;
    std::vector<detail::ChainNode> nodes(n+2);
    nodes.front()={.kind=Kind::Fixed,.offset={0,0}};
    nodes.back()={.kind=Kind::Fixed,.offset=local(target)};
    size_t face_count=0;
    for(size_t i=0;i<n;++i) {
        const auto &p=polygons[i];
        auto &node=nodes[i+1];
        for(size_t j=0;j<p.size();++j) {
            const auto a=local(p[j]),b=local(p[(j+1)%p.size()]);
            Vector2 normal{a.y-b.y,b.x-a.x};
            const double length=normal.length();
            if(!(length>0))continue;
            normal=normal/length;
            node.faces.push_back({normal,normal.dot(a)});
        }
        face_count+=node.faces.size();
        const auto center=local(vertex_mean(p));
        node.y=center;
        if(warm&&warm->size()==n)node.y=center+(local((*warm)[i])-center)*(1-options.warm_interior_fraction);
        auto feasible=[&]{return std::all_of(node.faces.begin(),node.faces.end(),[&](const auto &f){return f.normal.dot(node.y)-f.offset>0;});};
        if(!feasible()) {
            node.y=center;
            if(!feasible())return {};
        }
    }
    const double constraints=double(n+1+face_count);
    const double target_gap=gap>0?gap:1e-9*std::max(1.0,std::abs(cutoff));
    const double mu_end=std::max(1e-15,.5*target_gap/(scale*constraints));
    const double mu=warm?std::max(mu_end,std::min(1e-3,mu_end*options.warm_mu_ratio)):1e-1;
    Polish polish;
    const auto stats=detail::chain_interior_point(nodes,false,mu,mu_end,options.max_newton_iterations,
        [&](const detail::ChainLevel &level) {
            polish.contacts.clear();
            for(size_t i=1;i<=n;++i)polish.contacts.push_back({start.x+scale*level.points[i].x,start.y+scale*level.points[i].y});
            polish.duals=level.duals;
            return done(polish);
        });
    if(!stats)return {};
    polish.iterations=stats->iterations;polish.levels=stats->levels;
    return polish;
}
}

namespace {
// Points and segments (fewer than three distinct vertices), with positive-area
// polygons alongside. The directional trace needs areas, so proposals come
// from the shared cycle construction on {s}, P_1, ..., P_m, {t}: its closing
// link is constant, so its cycle optima are the path optima. Proofs are the
// same as below, with segment contacts enclosed (float_proof.h).
ConvexFloatOracleResult solve_degenerate_path(Vector2 start,Vector2 target,const std::vector<Polygon> &input,
        const ConvexFloatOracleOptions &options,ConvexFloatOracleResult result) {
    using Kernel=detail::CycleRefinement<double>;
    const size_t m=input.size();
    std::vector<Polygon> regions;regions.reserve(m);
    for(const auto &p:input) {
        auto normalized=detail::binary_region(p);
        if(!normalized){result.status=ConvexFloatOracleStatus::Unsupported;return result;}
        regions.push_back(std::move(*normalized));
    }
    double lower=direct_lower(start,target),upper=INFINITY;
    Polygon best;
    auto closed=[&] {
        if(!std::isfinite(upper))return false;
        if(lower>=options.cutoff)return true;
        return options.max_gap>0&&(Interval(upper)-Interval(lower)).hi<=options.max_gap;
    };
    auto prove=[&](const Polygon &contacts,const std::vector<std::vector<Vector2>> &duals) {
        for(const auto &u:duals)lower=std::max(lower,dual_lower(start,target,regions,u));
        std::vector<IntervalPoint> boxes{IntervalPoint(start)};Polygon points;
        for(size_t i=0;i<m;++i) {
            const auto proved=detail::prove_contact(contacts[i],regions[i]);
            if(!proved)return closed();
            boxes.push_back(proved->box);points.push_back(proved->point);
        }
        boxes.emplace_back(target);
        const double length=detail::enclosed_length_upper(boxes,false);
        if(length<upper){upper=length;best=std::move(points);}
        return closed();
    };
    auto prove_chain=[&](const Polygon &contacts) {
        Polygon chain{start};chain.insert(chain.end(),contacts.begin(),contacts.end());chain.push_back(target);
        double scale=0;
        for(const auto &v:chain)scale=std::max({scale,std::abs(v.x),std::abs(v.y)});
        const double short_link=std::max(32*std::numeric_limits<double>::epsilon()*scale,
            options.max_gap>0?options.max_gap/(16*double(chain.size())):0);
        return prove(contacts,chain_duals(chain,short_link));
    };
    auto finish=[&](ConvexFloatOracleStatus status) {
        result.contacts=best;result.lower_bound=lower;result.upper_bound=upper;result.status=status;return result;
    };
    auto done=[&]{return finish(lower>=options.cutoff?ConvexFloatOracleStatus::CutoffReached:ConvexFloatOracleStatus::GapClosed);};
    std::optional<Polygon> seed;
    const auto trace_began=Clock::now();
    result.trace_attempted=true;
    if(options.initial_contacts&&options.initial_contacts->size()==m)seed=*options.initial_contacts;
    else try {
        Kernel::Polygons cycle;cycle.push_back({Kernel::P(start)});
        for(const auto &region:regions) {
            Kernel::Polygon q;for(const auto &v:region)q.emplace_back(v);cycle.push_back(std::move(q));
        }
        cycle.push_back({Kernel::P(target)});
        Polygon last;
        const bool accepted=Kernel::run(cycle,[&](const Kernel::Polygon &candidate) {
            Polygon contacts;
            for(size_t i=1;i<=m;++i)contacts.push_back(candidate[i].external());
            const double before=upper;
            const bool finished=prove_chain(contacts);
            if(upper<before)seed=best;
            last=std::move(contacts);
            return finished;
        });
        if(!seed&&!last.empty())seed=last;
        result.trace_seconds=since(trace_began);
        if(accepted){result.trace_closed=true;return done();}
    } catch(const std::exception &) {result.trace_seconds=since(trace_began);}
    result.trace_failed=!seed;
    if(options.initial_contacts&&seed&&options.certify_trace&&prove_chain(*seed)){result.trace_closed=true;return done();}
    if(options.polish) {
        result.polish_attempted=true;
        const auto polish_began=Clock::now();
        double scale=std::max(std::abs(target.x-start.x),std::abs(target.y-start.y));
        for(const auto &p:regions)for(const auto &v:p)scale=std::max({scale,std::abs(v.x-start.x),std::abs(v.y-start.y)});
        if(scale>0&&std::isfinite(scale)) {
            using Kind=detail::ChainNode::Kind;
            std::vector<detail::ChainNode> nodes{{.kind=Kind::Fixed,.offset={0,0}}};
            size_t faces=0;
            for(size_t i=0;i<m;++i) {
                nodes.push_back(detail::region_node(regions[i],start,scale,seed?&(*seed)[i]:nullptr,options.warm_interior_fraction));
                faces+=nodes.back().faces.size();
            }
            nodes.push_back({.kind=Kind::Fixed,.offset={(target.x-start.x)/scale,(target.y-start.y)/scale}});
            const double target_gap=options.max_gap>0?options.max_gap:1e-9*std::max(1.0,std::abs(options.cutoff));
            const double mu_end=std::max(1e-15,.5*target_gap/(scale*double(m+1+faces)));
            const double mu=seed?std::max(mu_end,std::min(1e-3,mu_end*options.warm_mu_ratio)):1e-1;
            const auto stats=detail::chain_interior_point(nodes,false,mu,mu_end,options.max_newton_iterations,
                [&](const detail::ChainLevel &level) {
                    Polygon contacts;
                    for(size_t i=0;i<m;++i)contacts.push_back(detail::region_contact(regions[i],level,i+1,start,scale));
                    return prove(contacts,{level.duals});
                });
            if(stats){result.newton_iterations=stats->iterations;result.barrier_levels=stats->levels;}
        }
        result.polish_seconds=since(polish_began);
        if(closed()){result.polish_closed=true;return done();}
    }
    return finish(ConvexFloatOracleStatus::Open);
}
}

const char *to_string(ConvexFloatOracleStatus status) {
    switch(status) {
        case ConvexFloatOracleStatus::GapClosed:return "gap_closed";
        case ConvexFloatOracleStatus::CutoffReached:return "cutoff_reached";
        case ConvexFloatOracleStatus::Open:return "open";
        case ConvexFloatOracleStatus::Unsupported:return "unsupported";
    }
    return "unknown";
}

bool tpp_convex_solve_double_trusted(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &input,std::vector<Vector2> &contacts,double &length) {
    std::vector<Polygon> polygons;polygons.reserve(input.size());
    for(const auto &p:input) {
        auto normalized=counter_clockwise(p);
        if(!normalized)return false;
        polygons.push_back(std::move(*normalized));
    }
    try {
        const auto chain=detail::double_candidate_chain(start,target,polygons);
        if(chain.size()!=polygons.size()+2)return false;
        contacts.assign(chain.begin()+1,chain.end()-1);
        length=0;
        for(size_t i=1;i<chain.size();++i)length+=(chain[i]-chain[i-1]).length();
        return std::isfinite(length);
    } catch(const std::exception &) {return false;}
}

ConvexFloatOracleResult tpp_convex_solve_float_certified(const Vector2 &start,const Vector2 &target,
        const std::vector<std::vector<Vector2>> &input,const ConvexFloatOracleOptions &options) {
    const auto began=Clock::now();
    ConvexFloatOracleResult result;
    auto finish=[&](ConvexFloatOracleStatus status){result.status=status;result.total_seconds=since(began);return result;};
    if(!detail::cycle_interval_environment()||!start.is_finite()||!target.is_finite())
        return finish(ConvexFloatOracleStatus::Unsupported);
    std::vector<Polygon> polygons;polygons.reserve(input.size());
    for(const auto &p:input) {
        auto normalized=counter_clockwise(p);
        if(!normalized) {
            // Fewer than three distinct vertices (or no area): points and
            // segments take their own proposals; other inputs are unsupported.
            auto degenerate=solve_degenerate_path(start,target,input,options,result);
            degenerate.total_seconds=since(began);
            return degenerate;
        }
        polygons.push_back(std::move(*normalized));
    }
    Bounds bounds;
    bounds.offer_lower(direct_lower(start,target));
    if(polygons.empty()) {
        bounds.offer_upper(start,target,{});
        result.lower_bound=bounds.lower;result.upper_bound=bounds.upper;
        return finish(ConvexFloatOracleStatus::GapClosed);
    }
    auto status=[&] {
        result.contacts=bounds.contacts;result.lower_bound=bounds.lower;result.upper_bound=bounds.upper;
        return bounds.lower>=options.cutoff?ConvexFloatOracleStatus::CutoffReached:ConvexFloatOracleStatus::GapClosed;
    };
    // 1. The binary64 directional trace (no exact predicates, no certification).
    std::optional<Polygon> candidate;
    if(options.initial_contacts&&options.initial_contacts->size()==polygons.size()
            &&std::all_of(options.initial_contacts->begin(),options.initial_contacts->end(),[](Vector2 q){return q.is_finite();}))
        candidate=*options.initial_contacts;
    else {
        result.trace_attempted=true;
        const auto trace_began=Clock::now();
        try {
            const auto chain=detail::double_candidate_chain(start,target,polygons);
            if(chain.size()==polygons.size()+2)candidate=Polygon(chain.begin()+1,chain.end()-1);
        } catch(const std::exception &) {}
        result.trace_seconds=since(trace_began);
    }
    result.trace_failed=!candidate;
    if(candidate&&options.certify_trace) {
        const auto certificate_began=Clock::now();
        Polygon chain{start};chain.insert(chain.end(),candidate->begin(),candidate->end());chain.push_back(target);
        double scale=0;
        for(const auto &v:chain)scale=std::max({scale,std::abs(v.x),std::abs(v.y)});
        const double short_link=std::max(32*std::numeric_limits<double>::epsilon()*scale,
            options.max_gap>0?options.max_gap/(16*double(chain.size())):0);
        for(const auto &proposal:chain_duals(chain,short_link))bounds.offer_lower(dual_lower(start,target,polygons,proposal));
        Polygon proved=*candidate;
        bool feasible=true;
        for(size_t i=0;i<proved.size()&&feasible;++i)feasible=make_proved(proved[i],polygons[i]);
        if(feasible)bounds.offer_upper(start,target,std::move(proved));
        result.trace_certificate_seconds=since(certificate_began);
        if(bounds.closed(options)) {
            result.trace_closed=true;
            return finish(status());
        }
    }
    // 2. Interior-point polish; every barrier level offers its own bounds.
    if(options.polish) {
        result.polish_attempted=true;
        const auto polish_began=Clock::now();
        const double gap=options.max_gap>0?options.max_gap:0;
        const auto polished=interior_point(start,target,polygons,candidate?&*candidate:nullptr,gap,options.cutoff,
            options,[&](const Polish &level) {
                bounds.offer_lower(dual_lower(start,target,polygons,level.duals));
                Polygon proved=level.contacts;
                bool feasible=true;
                for(size_t i=0;i<proved.size()&&feasible;++i)feasible=make_proved(proved[i],polygons[i]);
                if(feasible)bounds.offer_upper(start,target,std::move(proved));
                return bounds.closed(options);
            });
        if(polished){result.newton_iterations=polished->iterations;result.barrier_levels=polished->levels;}
        result.polish_seconds=since(polish_began);
        if(bounds.closed(options)) {
            result.polish_closed=true;
            return finish(status());
        }
    }
    result.contacts=bounds.contacts;result.lower_bound=bounds.lower;result.upper_bound=bounds.upper;
    return finish(ConvexFloatOracleStatus::Open);
}

}
