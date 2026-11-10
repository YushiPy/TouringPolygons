#pragma once
// The proved interval of binary64 cycle proposals, shared by the binary64
// cycle stage (cycle_float_oracle.cpp) and Algorithm 4.1 with a tolerance
// (ConvexCycleDoubleOptions::boundary_search_gap in cycle.cpp).
#include "tpp/geometry/common.h"
#include "cycle_execution.h"
#include "cycle_interval.h"
#include "float_proof.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tpp::detail {
// Best proved interval over all proposals of one call.
struct CycleBounds {
    const std::vector<const FloatRegion*> &regions;
    const std::vector<Vector2> &means;
    Vector2 reference;
    double max_gap,cutoff;
    double lower=0,upper=INFINITY;
    std::vector<Vector2> contacts;
    // Both closures need a cycle of proved contacts, as CertifiedBound does.
    bool closed() const {
        if(!std::isfinite(upper))return false;
        if(lower>=cutoff)return true;
        return max_gap>0&&(CycleInterval(upper)-CycleInterval(lower)).hi<=max_gap;
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

} // namespace tpp::detail
