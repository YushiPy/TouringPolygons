#include "tpp/convex/dual.h"
#include "binary_dual.h"
#include <algorithm>
#include <stdexcept>

namespace tpp {
namespace {
using I=detail::CycleInterval;using P=detail::IntervalPoint;
using Region=std::vector<Vector2>;
using detail::Support;using detail::support_bounds;using detail::point_bounds;using detail::unit_direction;
double down(double x) {return detail::below(x);}
double up(double x) {return detail::above(x);}
// Zero links take the previous nonzero vector in the scan order (both
// directions, so every zero link of a chain with one nonzero link is filled).
std::vector<Vector2> filled(std::vector<Vector2> u,bool forward,bool cyclic) {
    const size_t n=u.size();
    for(int pass=0;pass<2;++pass) {
        Vector2 previous;
        for(size_t step=0;step<(cyclic?2*n:n);++step) {
            const size_t k=step%n,i=(forward!=bool(pass))?k:n-1-k;
            if(u[i]==Vector2{})u[i]=previous;
            else previous=u[i];
        }
        if(cyclic)break;
    }
    return u;
}
ConvexBinaryChainDual chain_dual(const std::vector<Vector2> &contacts,
        const std::vector<const Region *> &regions,bool cyclic) {
    ConvexBinaryChainDual best;best.cyclic=cyclic;
    if(!detail::cycle_interval_environment())return best;
    const size_t n=regions.size(),links=cyclic?n:n+1;
    const auto origin=contacts.front();
    std::vector<Vector2> raw(links);
    for(size_t i=0;i<links;++i)raw[i]=unit_direction(contacts[i],contacts[(i+1)%contacts.size()]);
    const bool zero=std::any_of(raw.begin(),raw.end(),[](Vector2 v){return v==Vector2{};});
    for(int fill=0;fill<(zero?3:1);++fill) {
        ConvexBinaryChainDual candidate;candidate.cyclic=cyclic;
        candidate.directions=fill?filled(raw,fill==1,cyclic):raw;
        const auto &u=candidate.directions;
        double lower=0,upper=0;
        if(!cyclic) {
            const auto end=point_bounds(contacts.back(),origin,u.back(),{});
            lower=end.lower;upper=end.upper;
        }
        for(size_t i=0;i<n;++i) {
            if(regions[i]->empty())throw std::invalid_argument("Empty binary dual region");
            const auto term=cyclic?support_bounds<true>(*regions[i],origin,u[(i+n-1)%n],u[i])
                :support_bounds<true>(*regions[i],origin,u[i],u[i+1]);
            candidate.support_lower.push_back(term.lower);candidate.support_upper.push_back(term.upper);
            candidate.width_upper.push_back(term.width);
            lower=down(lower+term.lower);upper=up(upper+term.upper);
        }
        if(!std::isfinite(lower)||!std::isfinite(upper))continue;
        candidate.lower=lower;candidate.upper=upper;
        if(!best.valid()||candidate.lower>best.lower)best=std::move(candidate);
    }
    return best;
}
}

ConvexBinaryChainDual tpp_convex_binary_path_dual(const std::vector<Vector2> &contacts,
        const std::vector<const std::vector<Vector2> *> &regions) {
    if(contacts.size()!=regions.size()+2)throw std::invalid_argument("Invalid binary path dual dimensions");
    return chain_dual(contacts,regions,false);
}
ConvexBinaryChainDual tpp_convex_binary_cycle_dual(const std::vector<Vector2> &contacts,
        const std::vector<const std::vector<Vector2> *> &regions) {
    if(regions.empty()||contacts.size()!=regions.size())throw std::invalid_argument("Invalid binary cycle dual dimensions");
    return chain_dual(contacts,regions,true);
}
std::pair<double,double> tpp_convex_binary_insertion_gain(const ConvexBinaryChainDual &dual,
        const std::vector<Vector2> &contacts,const std::vector<const std::vector<Vector2> *> &regions,
        const std::vector<Vector2> &inserted,std::size_t j,Vector2 proposal) {
    const size_t n=regions.size(),links=dual.cyclic?n:n+1;
    if(!dual.valid()||dual.directions.size()!=links||contacts.size()!=(dual.cyclic?n:n+2)||j>=links||inserted.empty())
        throw std::invalid_argument("Invalid binary insertion gain");
    const auto &u=dual.directions;
    const auto origin=contacts.front();
    const auto a=contacts[j],b=contacts[(j+1)%contacts.size()];
    const auto left=unit_direction(a,proposal),right=unit_direction(proposal,b);
    auto gain=[&](Vector2 left,Vector2 right) {
        double lower=0,upper=0;
        auto add=[&](const Support &term) {lower=down(lower+term.lower);upper=up(upper+term.upper);};
        // The new term of region i minus its stored one.
        auto replace=[&](size_t i,Vector2 x,Vector2 y) {
            const auto term=support_bounds(*regions[i],origin,x,y);
            add({down(term.lower-dual.support_upper[i]),up(term.upper-dual.support_lower[i])});
        };
        add(support_bounds(inserted,origin,left,right));
        if(dual.cyclic) {
            const size_t next=(j+1)%n;
            if(n==1)replace(0,right,left);
            else {replace(j,u[(j+n-1)%n],left);replace(next,right,u[next]);}
        } else {
            if(j)replace(j-1,u[j-1],left);
            if(j<n)replace(j,right,u[j+1]);
            else add(point_bounds(contacts.back(),origin,right,u.back()));
        }
        if(!std::isfinite(lower)||!std::isfinite(upper))return std::pair<double,double>{-INFINITY,INFINITY};
        return std::pair<double,double>{lower,upper};
    };
    auto best=gain(left,right);
    // A zero new link admits any unit-disk vector; link j's own is the other
    // natural choice, and neither dominates.
    if(left==Vector2{}||right==Vector2{}) {
        const auto filled=gain(left==Vector2{}?u[j]:left,right==Vector2{}?u[j]:right);
        if(filled.first>best.first)best=filled;
    }
    return best;
}
double tpp_convex_distance_lower(Vector2 a,Vector2 b) {
    if(!detail::cycle_interval_environment())return 0;
    const auto d=P(b)-P(a);
    const auto length=(d.x.square()+d.y.square()).sqrt();
    return std::isfinite(length.lo)?std::max(0.0,length.lo):0;
}

std::vector<double> tpp_convex_binary_dual_insertion_bounds(
    Vector2 start,Vector2 target,const std::vector<Vector2> &contacts,
    const std::vector<const std::vector<Vector2> *> &regions,
    const std::vector<Vector2> &inserted,const std::vector<Vector2> &proposals,
    const std::vector<Vector2> &dual) {
    using I=detail::CycleInterval;using P=detail::IntervalPoint;
    const size_t n=regions.size();
    if(contacts.size()!=n+2||dual.size()!=n+1||proposals.size()!=n+1||inserted.empty())
        throw std::invalid_argument("Invalid binary dual insertion dimensions");
    if(!detail::cycle_interval_environment())return {};
    for(const auto &u:dual)if(!detail::binary_dual_feasible(u))return {};
    const P origin(start),destination(target);
    auto support=[&](const std::vector<Vector2> &p,const P &normal) {
        if(p.empty())throw std::invalid_argument("Empty binary dual region");
        I value(INFINITY);
        for(const auto &v:p) {
            const auto term=(P(v)-origin).dot(normal);
            value.lo=std::min(value.lo,term.lo);value.hi=std::min(value.hi,term.hi);
        }
        return value;
    };
    // Reuse unchanged supports for all siblings. Bounds remain valid even
    // when the new optimum moves every contact of the reference path.
    std::vector<I> terms;terms.reserve(n);
    I total=(destination-origin).dot(P(dual.back()));
    for(size_t i=0;i<n;++i)total=total+terms.emplace_back(support(*regions[i],P(dual[i])-P(dual[i+1])));
    std::vector<double> bounds(n+1,0);
    for(size_t j=0;j<=n;++j) {
        auto left=detail::binary_dual_direction(contacts[j],proposals[j]);
        auto right=detail::binary_dual_direction(proposals[j],contacts[j+1]);
        if(left==Vector2{})left=dual[j];
        if(right==Vector2{})right=dual[j];
        I value=total+support(inserted,P(left)-P(right));
        if(j)value=value+support(*regions[j-1],P(dual[j-1])-P(left))-terms[j-1];
        if(j<n)value=value+support(*regions[j],P(right)-P(dual[j+1]))-terms[j];
        else value=value+(destination-origin).dot(P(right)-P(dual.back()));
        if(value.finite())bounds[j]=std::max(0.0,value.lo);
    }
    return bounds;
}
std::vector<double> tpp_convex_binary_cycle_insertion_bounds(
    const std::vector<Vector2> &contacts,const std::vector<const std::vector<Vector2> *> &regions,
    const std::vector<Vector2> &inserted,const std::vector<Vector2> &proposals) {
    const size_t n=regions.size();
    if(!n||contacts.size()!=n||proposals.size()!=n||inserted.empty())
        throw std::invalid_argument("Invalid binary cycle insertion dimensions");
    if(!detail::cycle_interval_environment())return {};
    const auto origin=contacts.front();
    // Link i enters region i+1. Inserting at gap i replaces link i by left
    // and right, and changes only the terms of regions i and i+1 and its own.
    std::vector<Vector2> u(n),left(n),right(n);
    for(size_t i=0;i<n;++i) {
        const auto next=contacts[(i+1)%n];
        u[i]=unit_direction(contacts[i],next);
        left[i]=unit_direction(contacts[i],proposals[i]);
        right[i]=unit_direction(proposals[i],next);
    }
    std::vector<Support> terms;terms.reserve(n);
    double parent_lower=0;
    for(size_t i=0;i<n;++i) {
        terms.push_back(support_bounds(*regions[i],origin,u[(i+n-1)%n],u[i]));
        parent_lower=down(parent_lower+terms.back().lower);
    }
    std::vector<double> bounds(n,0);
    for(size_t i=0;i<n;++i) {
        const size_t next=(i+1)%n;
        double value=support_bounds(inserted,origin,left[i],right[i]).lower;
        if(n==1)value=down(value+support_bounds(*regions[0],origin,right[0],left[0]).lower);
        else {
            value=down(value+down(down(parent_lower-terms[i].upper)-terms[next].upper));
            value=down(value+support_bounds(*regions[i],origin,u[(i+n-1)%n],left[i]).lower);
            value=down(value+support_bounds(*regions[next],origin,right[i],u[next]).lower);
        }
        if(std::isfinite(value))bounds[i]=std::max(0.0,value);
    }
    return bounds;
}
}
