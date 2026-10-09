#include "tpp/convex/float_oracle.h"
#include "binary_dual.h"

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

// Proof that q lies in the closed CCW polygon: an interval sign first, the
// exact integer determinant when the interval straddles zero. Unknown is "no".
bool proved_inside(Vector2 q,const Polygon &p) {
    if(!q.is_finite())return false;
    for(size_t i=0;i<p.size();++i) {
        const auto &a=p[i],&b=p[(i+1)%p.size()];
        const auto side=(IntervalPoint(b)-IntervalPoint(a)).cross(IntervalPoint(q)-IntervalPoint(a));
        if(side.hi<0)return false;
        if(side.lo>=0)continue;
        const auto sign=detail::dyadic_orientation(a,b,q);
        if(!sign||*sign<0)return false;
    }
    return true;
}

Vector2 vertex_mean(const Polygon &p) {
    Vector2 c{};
    for(const auto &v:p)c+=v/double(p.size());
    return c;
}

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

struct Face {Vector2 normal;double offset;};

// Log-barrier Newton method on the smoothed lengths, in coordinates relative
// to start divided by scale. At a barrier minimizer the smoothed directions
// satisfy u_i - u_{i+1} = sum (mu/slack) n_f, so D(u) >= length - mu*(links+faces)
// (scaled); the stopping level follows from the requested gap.
struct Polish {
    std::vector<Vector2> contacts,duals;
    std::size_t iterations=0,levels=0;
};

std::optional<Polish> interior_point(Vector2 start,Vector2 target,const std::vector<Polygon> &polygons,
        const Polygon *warm,double gap,double cutoff,const ConvexFloatOracleOptions &options,
        const std::function<bool(const Polish&)> &done) {
    const std::size_t max_iterations=options.max_newton_iterations;
    const size_t n=polygons.size();
    double scale=std::max(std::abs(target.x-start.x),std::abs(target.y-start.y));
    for(const auto &p:polygons)for(const auto &v:p)scale=std::max({scale,std::abs(v.x-start.x),std::abs(v.y-start.y)});
    if(!(scale>0)||!std::isfinite(scale))return {};
    auto local=[&](Vector2 v){return Vector2{(v.x-start.x)/scale,(v.y-start.y)/scale};};
    std::vector<std::vector<Face>> faces(n);size_t face_count=0;
    std::vector<Vector2> x(n+2);
    x.front()={0,0};x.back()=local(target);
    for(size_t i=0;i<n;++i) {
        const auto &p=polygons[i];
        for(size_t j=0;j<p.size();++j) {
            const auto a=local(p[j]),b=local(p[(j+1)%p.size()]);
            Vector2 normal{a.y-b.y,b.x-a.x};
            const double length=normal.length();
            if(!(length>0))continue;
            normal=normal/length;
            faces[i].push_back({normal,normal.dot(a)});
        }
        face_count+=faces[i].size();
        const auto center=local(vertex_mean(p));
        x[i+1]=center;
        if(warm&&warm->size()==n)x[i+1]=center+(local((*warm)[i])-center)*(1-options.warm_interior_fraction);
    }
    auto slack=[&](size_t i,const Face &f,const std::vector<Vector2> &z){return f.normal.dot(z[i+1])-f.offset;};
    for(size_t i=0;i<n;++i)for(const auto &f:faces[i])if(!(slack(i,f,x)>0)) {
        x[i+1]=local(vertex_mean(polygons[i]));
        for(const auto &g:faces[i])if(!(slack(i,g,x)>0))return {};
        break;
    }
    const double constraints=double(n+1+face_count);
    const double target_gap=gap>0?gap:1e-9*std::max(1.0,std::abs(cutoff));
    const double mu_end=std::max(1e-15,.5*target_gap/(scale*constraints));
    auto objective=[&](const std::vector<Vector2> &z,double mu)->double {
        double value=0;
        for(size_t i=1;i<z.size();++i) {
            const auto d=z[i]-z[i-1];value+=std::sqrt(d.dot(d)+mu*mu);
        }
        for(size_t i=0;i<n;++i)for(const auto &f:faces[i]) {
            const double s=slack(i,f,z);
            if(!(s>0))return INFINITY;
            value-=mu*std::log(s);
        }
        return value;
    };
    struct M2 {double a=0,b=0,c=0,d=0;}; // [[a b] [c d]]
    auto add=[](M2 &m,const M2 &h){m.a+=h.a;m.b+=h.b;m.c+=h.c;m.d+=h.d;};
    auto mul=[](const M2 &m,const M2 &h){return M2{m.a*h.a+m.b*h.c,m.a*h.b+m.b*h.d,m.c*h.a+m.d*h.c,m.c*h.b+m.d*h.d};};
    auto apply=[](const M2 &m,Vector2 v){return Vector2{m.a*v.x+m.b*v.y,m.c*v.x+m.d*v.y};};
    auto transpose=[](const M2 &m){return M2{m.a,m.c,m.b,m.d};};
    auto inverse=[](const M2 &m)->std::optional<M2> {
        const double det=m.a*m.d-m.b*m.c;
        if(!(std::abs(det)>0)||!std::isfinite(det))return {};
        return M2{m.d/det,-m.b/det,-m.c/det,m.a/det};
    };
    Polish polish;
    double mu=warm?std::max(mu_end,std::min(1e-3,mu_end*options.warm_mu_ratio)):1e-1;
    std::vector<M2> diagonal(n),off(n),factor(n),inverses(n);
    std::vector<Vector2> gradient(n),rhs(n),step(n),trial;
    for(;;mu=std::max(mu_end,mu*.1)) {
        ++polish.levels;
        for(;polish.iterations<max_iterations;++polish.iterations) {
            std::fill(diagonal.begin(),diagonal.end(),M2{});std::fill(off.begin(),off.end(),M2{});
            std::fill(gradient.begin(),gradient.end(),Vector2{});
            for(size_t i=0;i<=n;++i) {
                const auto d=x[i+1]-x[i];
                const double norm=std::sqrt(d.dot(d)+mu*mu);
                const Vector2 g=d/norm;
                const M2 h{(1-g.x*g.x)/norm,-g.x*g.y/norm,-g.x*g.y/norm,(1-g.y*g.y)/norm};
                if(i>0){gradient[i-1]-=g;add(diagonal[i-1],h);}
                if(i<n){gradient[i]+=g;add(diagonal[i],h);}
                if(i>0&&i<n)off[i]=M2{-h.a,-h.b,-h.c,-h.d};
            }
            for(size_t i=0;i<n;++i) {
                for(const auto &f:faces[i]) {
                    const double s=slack(i,f,x),w=mu/(s*s);
                    gradient[i]-=f.normal*(mu/s);
                    add(diagonal[i],M2{w*f.normal.x*f.normal.x,w*f.normal.x*f.normal.y,w*f.normal.x*f.normal.y,w*f.normal.y*f.normal.y});
                }
                diagonal[i].a+=1e-14;diagonal[i].d+=1e-14;
            }
            bool singular=false;
            for(size_t i=0;i<n;++i) {
                rhs[i]=gradient[i]*-1.0;
                if(i) {
                    factor[i]=mul(off[i],inverses[i-1]);
                    const auto correction=mul(factor[i],transpose(off[i]));
                    diagonal[i].a-=correction.a;diagonal[i].b-=correction.b;diagonal[i].c-=correction.c;diagonal[i].d-=correction.d;
                    rhs[i]-=apply(factor[i],rhs[i-1]);
                }
                const auto inv=inverse(diagonal[i]);
                if(!inv){singular=true;break;}
                inverses[i]=*inv;
            }
            if(singular)break;
            for(size_t i=n;i-->0;)
                step[i]=apply(inverses[i],rhs[i]-(i+1<n?apply(transpose(off[i+1]),step[i+1]):Vector2{}));
            double slope=0;
            for(size_t i=0;i<n;++i)slope+=gradient[i].dot(step[i]);
            if(!std::isfinite(slope)||slope>=0||-slope<1e-10*mu)break;
            const double before=objective(x,mu);
            // Below the objective's resolution Armijo cannot see progress;
            // near convergence the feasible Newton step is then taken as is.
            const bool resolved=-slope>64*std::numeric_limits<double>::epsilon()*std::abs(before);
            double alpha=1;trial=x;
            for(;alpha>1e-12;alpha*=.5) {
                for(size_t i=0;i<n;++i)trial[i+1]=x[i+1]+step[i]*alpha;
                const double after=objective(trial,mu);
                if(resolved?after<=before+.01*alpha*slope:std::isfinite(after))break;
            }
            if(alpha<=1e-12)break;
            x.swap(trial);
        }
        polish.contacts.clear();polish.duals.clear();
        for(size_t i=1;i<=n;++i)polish.contacts.push_back({start.x+scale*x[i].x,start.y+scale*x[i].y});
        for(size_t i=0;i<=n;++i) {
            const auto d=x[i+1]-x[i];
            polish.duals.push_back(d/std::sqrt(d.dot(d)+mu*mu));
        }
        if(done(polish)||mu<=mu_end||polish.iterations>=max_iterations)return polish;
    }
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
        if(!normalized)return finish(ConvexFloatOracleStatus::Unsupported);
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
