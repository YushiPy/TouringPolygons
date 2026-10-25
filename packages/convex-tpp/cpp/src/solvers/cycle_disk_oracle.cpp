#include "tpp/convex/disk_cycle.h"
#include "cycle_interval.h"
#include "float_chain.h"
#include <algorithm>
#include <cmath>

namespace tpp {
namespace {
using detail::CycleInterval;

// r^2 - |x-c|^2 >= 0 with directed rounding; a point region holds only its center.
bool proved_inside(Vector2 x,const ConvexDisk &d) {
    if(d.radius==0)return x.x==d.center.x&&x.y==d.center.y;
    const auto dx=CycleInterval(x.x)-CycleInterval(d.center.x),dy=CycleInterval(x.y)-CycleInterval(d.center.y);
    return (CycleInterval(d.radius).square()-(dx.square()+dy.square())).lo>=0;
}
// The proposal, moved toward the center until it is proved inside; the center always is.
Vector2 inside_contact(Vector2 x,const ConvexDisk &d) {
    if(d.radius==0||!x.is_finite())return d.center;
    if(proved_inside(x,d))return x;
    for(const double f:{0x1p-45,0x1p-40,0x1p-30,0x1p-20,0x1p-10,0x1p-6,0x1p-3,0.5}) {
        const Vector2 y=d.center+(x-d.center)*(1-f);
        if(proved_inside(y,d))return y;
    }
    return d.center;
}
CycleInterval link_length(Vector2 a,Vector2 b) {
    const auto dx=CycleInterval(b.x)-CycleInterval(a.x),dy=CycleInterval(b.y)-CycleInterval(a.y);
    return (dx.square()+dy.square()).sqrt();
}
// u itself if proved in the unit disk, else shrunk until it is.
Vector2 unit_disk(Vector2 u) {
    if(!u.is_finite())return {};
    for(int attempt=0;attempt<8;++attempt) {
        if((CycleInterval(u.x).square()+CycleInterval(u.y).square()).hi<=1)return u;
        u=u*(1-0x1p-40);
        if(attempt==3)u=u/std::sqrt(u.dot(u))*(1-0x1p-30);
    }
    return {};
}
}

DiskCycleResult tpp_convex_solve_cycle_disks(const std::vector<ConvexDisk> &disks,const DiskCycleOptions &options) {
    DiskCycleResult result;
    const size_t k=disks.size();
    if(!k||std::isnan(options.lower_bound_cutoff))return result;
    for(const auto &d:disks)if(!d.center.is_finite()||!std::isfinite(d.radius)||d.radius<0)return result;
    if(!options.initial_contacts.empty()&&options.initial_contacts.size()!=k)return result;
    if(!detail::cycle_interval_environment())return result;

    // Both bounds are proved on the input; each level of the polish only proposes.
    auto certify=[&](const std::vector<Vector2> &points,const std::vector<Vector2> &duals) {
        std::vector<Vector2> contacts(k);
        for(size_t i=0;i<k;++i)contacts[i]=inside_contact(points[i],disks[i]);
        CycleInterval upper{};
        if(k>1)for(size_t i=0;i<k;++i)upper=upper+link_length(contacts[i],contacts[(i+1)%k]);
        if(upper.hi<result.upper_bound){result.upper_bound=upper.hi;result.contacts=std::move(contacts);}
        if(duals.size()==k) {
            Vector2 r{};for(const auto &d:disks)r+=d.center/double(k);
            std::vector<Vector2> u(k);
            for(size_t l=0;l<k;++l)u[l]=unit_disk(duals[l]);
            CycleInterval lower{};
            for(size_t i=0;i<k;++i) {
                const auto &in=u[(i+k-1)%k],&out=u[i];
                const auto wx=CycleInterval(in.x)-CycleInterval(out.x),wy=CycleInterval(in.y)-CycleInterval(out.y);
                const auto support=(CycleInterval(disks[i].center.x)-CycleInterval(r.x))*wx+
                                   (CycleInterval(disks[i].center.y)-CycleInterval(r.y))*wy-
                                   CycleInterval(disks[i].radius)*(wx.square()+wy.square()).sqrt();
                lower=lower+support;
            }
            if(std::isfinite(lower.lo))result.lower_bound=std::max(result.lower_bound,lower.lo);
        }
        return result.upper_bound-result.lower_bound<=options.max_gap||result.lower_bound>=options.lower_bound_cutoff;
    };
    if(k==1) {
        certify({disks[0].center},{});
        result.status=DiskCycleStatus::GapClosed;
        return result;
    }

    // Scaled coordinates x = ref + scale*y.
    Vector2 ref{};for(const auto &d:disks)ref+=d.center/double(k);
    double scale=0;
    for(const auto &d:disks)scale=std::max({scale,std::abs(d.center.x-ref.x)+d.radius,std::abs(d.center.y-ref.y)+d.radius});
    if(!(scale>0)||!std::isfinite(scale))scale=1;
    auto local=[&](Vector2 v){return (v-ref)/scale;};
    std::vector<detail::ChainNode> nodes(k);
    size_t barriers=0;
    for(size_t i=0;i<k;++i) {
        auto &node=nodes[i];const auto &d=disks[i];
        const Vector2 c=local(d.center);
        if(d.radius==0){node.kind=detail::ChainNode::Kind::Fixed;node.offset=c;continue;}
        const double radius=d.radius/scale;
        node.kind=detail::ChainNode::Kind::Free;
        node.disk=detail::ChainNode::Disk{c,radius*radius};
        node.y=c;
        if(!options.initial_contacts.empty()&&options.initial_contacts[i].is_finite()) {
            const Vector2 y=c+(local(options.initial_contacts[i])-c)*(1-0x1p-7);
            if((y-c).dot(y-c)<radius*radius*(1-0x1p-20))node.y=y;
        }
        ++barriers;
    }
    const bool warm=!options.initial_contacts.empty();
    const double mu_end=std::max(1e-15,options.max_gap/(2*scale*double(k+barriers)));
    // A warm start near the boundary needs a moderate first level: starting
    // far below the central path, Newton crawls (1e-5 to 1e-3 all work).
    const double mu_start=std::max(mu_end,options.mu_start>0?options.mu_start:warm?1e-4:0.1);
    const auto stats=detail::chain_interior_point(nodes,true,mu_start,mu_end,options.max_iterations,
        [&](const detail::ChainLevel &level) {
            std::vector<Vector2> points(k);
            for(size_t i=0;i<k;++i)points[i]=ref+level.points[i]*scale;
            return certify(points,level.duals);
        });
    if(stats){result.newton_iterations=stats->iterations;result.levels=stats->levels;}
    if(result.contacts.empty()) {
        std::vector<Vector2> centers(k);for(size_t i=0;i<k;++i)centers[i]=disks[i].center;
        certify(centers,{});
    }
    result.lower_bound=std::max(result.lower_bound,0.0);
    result.status=result.upper_bound-result.lower_bound<=options.max_gap?DiskCycleStatus::GapClosed:
        result.lower_bound>=options.lower_bound_cutoff?DiskCycleStatus::CertifiedBound:DiskCycleStatus::Limit;
    return result;
}
} // namespace tpp
