#include "tpp/convex/dual.h"
#include "binary_dual.h"
#include <stdexcept>

namespace tpp {
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
}
