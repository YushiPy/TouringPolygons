#include <random>
#include "tpp/convex/cycle.h"
#include "tpp/convex/float_oracle.h"
#include "tpp/convex/detail/rational_disjoint.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "cycle_internal.h"
#include "cycle_execution.h"
#include "disjoint_arithmetic.h"
#include "cycle_refinement.h"
#include "cycle_float_bounds.h"
#include "float_proof.h"
#include "binary_certificate.h"
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <optional>

#include <algorithm>
#include <stdexcept>

namespace tpp {
namespace detail {
// Defined in hybrid.cpp: the binary64 construction of the TPP flow (trace,
// replay and contact repair), a proposal only.
std::vector<Vector2> double_candidate_chain(const Vector2 &start,const Vector2 &target,
    const std::vector<std::vector<Vector2>> &polygons);
}
namespace {
using R = ConvexRational;
using Point = ConvexRationalPoint;
using Integer = tpp::ConvexInteger;

// A verified exact dual bound lower-bounds every feasible rounded contact set.
// Keep that proof when exporting to binary64; its contact-derived dual can be
// much weaker after a zero link becomes a tiny nonzero link during rounding.
void retain_exact_bound(ConvexCycleDoubleResult &result,
        const ConvexCycleCertificateResult &exact,double cutoff) {
    if(result.contacts.empty()||(exact.status!=ConvexCycleCertificateStatus::Optimal&&
       exact.status!=ConvexCycleCertificateStatus::Feasible))return;
    result.certificate.lower_bound=std::max(result.certificate.lower_bound,exact.lower_bound);
    if(result.status!=ConvexCycleStatus::Optimal)
        result.status=result.certificate.lower_bound>=cutoff?ConvexCycleStatus::CertifiedBound:ConvexCycleStatus::FloatingPointLimit;
}

// The edge searches make at most this many closed-form probes in a row before a
// round of bisection (a midpoint and a simplest rational). Each round halves the
// bracket, so a piece takes O(1+log Q) probes, Q the smallest denominator of a
// rational minimizer (paper 1, Theorem 4.1), whatever the number of descriptors
// the closed-form step meets. On the 1638 cycles of paper 1 no search made more
// than two in a row, so the cap changes none of those runs.
constexpr std::size_t kClosedFormRun=4;

// Simplest rational in the closed interval. Continued fractions reconstruct a
// rational root exactly; they do not round it to a prescribed grid or tolerance.
template<class S> S simplest_rational(S left, S right) {
    using I=std::conditional_t<std::is_same_v<S,double>,double,Integer>;
    const S original_mid=(left+right)/2;
    std::vector<I> prefix;
    S result;
    for (;;) {
        detail::cycle_checkpoint();
        const auto floor_value=[](const S &x)->I {
            if constexpr(std::is_same_v<S,double>)return std::floor(x);
            else return numerator(x)/denominator(x);
        };
        const I a=floor_value(left),b=floor_value(right);
        if (left == a) { result = left; break; }
        if (a < b) { result = S(a + 1); break; }
        prefix.push_back(a);
        const S next_left = S(1) / (right - a);
        right = S(1) / (left - a);
        if constexpr(std::is_same_v<S,double>)
            if(!std::isfinite(next_left)||!std::isfinite(right)||next_left>=right)return original_mid;
        left = next_left;
    }
    for (auto it = prefix.rbegin(); it != prefix.rend(); ++it) result = S(*it) + S(1) / result;
    return result;
}

template<class S> int direction_support_sign(const ConvexArithmeticPoint<S> &incoming,
        const ConvexArithmeticPoint<S> &outgoing,const ConvexArithmeticPoint<S> &direction) {
    if(incoming.zero()||outgoing.zero())throw std::runtime_error("Missing nonzero incident cycle direction");
    if constexpr(std::is_same_v<S,double>) {
        const double ni=std::hypot(incoming.x,incoming.y),no=std::hypot(outgoing.x,outgoing.y);
        const double value=(incoming.x/ni-outgoing.x/no)*direction.x+
                           (incoming.y/ni-outgoing.y/no)*direction.y;
        if(!std::isfinite(value))throw std::runtime_error("Nonfinite double support direction");
        return value>0?1:value<0?-1:0;
    } else return detail::cycle_normalized_difference_sign(incoming.dot(direction),incoming.dot(incoming),
                                                          outgoing.dot(direction),outgoing.dot(outgoing));
}
template<class S> int support_sign(const std::vector<ConvexArithmeticPoint<S>> &q,size_t i,
        const ConvexArithmeticPoint<S> &direction) {
    return direction_support_sign(q[i]-q[(i+q.size()-1)%q.size()],q[(i+1)%q.size()]-q[i],direction);
}
// The combinatorial type of a cycle through the polygons: vertex 2j, edge
// 2j+1, or -1 where the cycle goes straight through the contact (strictly
// between its neighbours on one line, even on the region's boundary).
template<class S> std::vector<int> contact_features(const std::vector<std::vector<ConvexArithmeticPoint<S>>> &p,
        const std::vector<ConvexArithmeticPoint<S>> &q) {
    std::vector<int> feature(p.size(),-1);
    const std::size_t k=p.size();
    for(std::size_t i=0;i<k;++i) {
        const auto in=q[i]-q[(i+k-1)%k],out=q[(i+1)%k]-q[i];
        if(!in.zero()&&!out.zero()&&in.cross(out)==0&&in.dot(out)>0)continue;
        const std::size_t m=p[i].size();
        for(std::size_t j=0;j<m&&feature[i]<0;++j)if(q[i]==p[i][j])feature[i]=int(2*j);
        if(feature[i]>=0||m<2)continue;
        for(std::size_t j=0;j<m;++j)if((p[i][(j+1)%m]-p[i][j]).cross(q[i]-p[i][j])==0){feature[i]=int(2*j+1);break;}
    }
    return feature;
}
// Identical floating-anchor reduction for both arithmetic backends. A Step,
// when given, solves an anchored cycle's own type in closed form on an anchor
// edge and proposes the next probe; see search_intersecting_boundaries.
template<class S,class Evaluate,class Solved,class Step=std::nullptr_t>
void search_cycle(const std::vector<std::vector<ConvexArithmeticPoint<S>>> &polygons,
                  std::size_t anchor_index,Evaluate evaluate,Solved solved, std::string *floating_failure=nullptr,
                  std::size_t *probes=nullptr,Step step=nullptr) {
    const auto &anchor=polygons[anchor_index];const auto m=anchor.size();
    std::vector<int> forward(m),backward(m);
    std::vector<bool> available(m,false);
    std::vector<std::vector<ConvexArithmeticPoint<S>>> at_vertex(m);
    auto try_step=[&](std::size_t j,const std::vector<ConvexArithmeticPoint<S>> &q,std::optional<S> &next) {
        if constexpr(std::is_same_v<Step,std::nullptr_t>)return false;
        else return !q.empty()&&step(j,q,next);
    };
    // Subgradient cuts (paper 1, Lemma 4.3): the anchored route at x has
    // nonzero segments at x, because the polygons are disjoint, and
    // g = u_in - u_out is a subgradient of F_i at x, so every minimizer y has
    // g.(y-x) <= 0. An edge whose two ends lie strictly beyond is discarded,
    // and a vertex is evaluated only while an edge at it remains. Each
    // evaluation takes the middle vertex of the longest run of vertices still
    // needed, so a cut removes at least half of that run.
    std::vector<char> edge_alive(m,1),tried(m,0);
    auto needed=[&](std::size_t j){return !tried[j]&&(edge_alive[j]||edge_alive[(j+m-1)%m]);};
    for(;;) {
        detail::cycle_checkpoint();
        std::size_t best_start=0,best_length=0;
        {
            std::size_t first=m;
            for(std::size_t j=0;j<m;++j)if(!needed(j)){first=j;break;}
            if(first==m){best_start=0;best_length=m;}
            else for(std::size_t step=1,run=0,start=0;step<=m;++step) {
                const std::size_t j=(first+step)%m;
                if(needed(j)){if(run==0)start=j;++run;if(run>best_length){best_length=run;best_start=start;}}
                else run=0;
            }
        }
        if(best_length==0)break;
        const std::size_t j=(best_start+best_length/2)%m;
        tried[j]=1;
        try {
            const auto q=evaluate(anchor[j]);if(solved())return;
            at_vertex[j]=q;
            forward[j]=support_sign(q,anchor_index,anchor[(j+1)%m]-anchor[j]);
            backward[j]=support_sign(q,anchor_index,anchor[j]-anchor[(j+m-1)%m]);
            available[j]=true;
            std::vector<int> side(m);
            for(std::size_t v=0;v<m;++v)side[v]=v==j?0:support_sign(q,anchor_index,anchor[v]-anchor[j]);
            for(std::size_t e=0;e<m;++e)if(edge_alive[e]&&side[e]>0&&side[(e+1)%m]>0)edge_alive[e]=0;
        } catch(const std::exception &error) {
            if(!floating_failure)throw;
            *floating_failure=error.what();
        }
    }
    for(std::size_t j=0;j<m;++j) {
        if(!edge_alive[j]||!available[j]||!available[(j+1)%m]||forward[j]>=0||backward[(j+1)%m]<=0)continue;
        const auto a=anchor[j],edge=anchor[(j+1)%m]-a;
        S left=0,right=1;std::optional<S> next;
        auto inside=[&]{return next&&*next>left&&*next<right;};
        // The route at the right end of the bracket, for end_steps.
        auto right_route=at_vertex[(j+1)%m];bool right_moved=true;
        auto probe=[&](const S &t) {
            if(probes)++*probes;
            const auto q=evaluate(a+edge*t);if(solved())return true;
            const int derivative=support_sign(q,anchor_index,edge);
            if(derivative<0)left=t;else if(derivative>0){right=t;right_route=q;right_moved=true;}
            return derivative==0||try_step(j,q,next);
        };
        // Exact arithmetic: every closed-form proposal of the route at the
        // right end, probed when inside the bracket. The right end tends to
        // the largest minimizer from above and enters the open interval of
        // one type that starts there; the closed form of that type is the
        // minimizer, which ends the piece for any real input (Theorem 4.1).
        auto end_steps=[&] {
            if constexpr(std::is_same_v<S,double>||std::is_same_v<Step,std::nullptr_t>)return false;
            else {
                if(!right_moved||right_route.empty())return false;
                right_moved=false;
                std::vector<S> proposals;std::optional<S> unused;
                if(step(j,right_route,unused,&proposals))return true;
                for(const S &t:proposals)if(t>left&&t<right&&probe(t))return true;
                return false;
            }
        };
        try {
            // A step at an end returns true when it solved the cycle or when it
            // reached a fixed point of the closed form; the latter ends only
            // this piece, as a fixed point at a probe does below.
            if(try_step(j,at_vertex[j],next)){if(solved())return;continue;}
            if(!inside()&&try_step(j,at_vertex[(j+1)%m],next)){if(solved())return;continue;}
            std::size_t closed_form_run=0;
            for(;;) {
                detail::cycle_checkpoint();
                if(inside()&&closed_form_run<kClosedFormRun) {
                    ++closed_form_run;const S t=*next;next.reset();if(probe(t))break;continue;
                }
                closed_form_run=0;
                next.reset();
                if(end_steps())break;
                const S mid=(left+right)/2;
                if(mid==left||mid==right)break; // Only reachable in finite arithmetic.
                if(probe(mid))break;
                const S candidate=simplest_rational(left,right);
                if(candidate>left&&candidate<right&&probe(candidate))break;
            }
        } catch(const std::exception &error) {
            if(!floating_failure)throw;
            *floating_failure=error.what();
        }
        if(solved())return;
    }
}

// refine_contacts=false: the boundary-anchor method alone (paper 1, Algorithm
// 4.1). The descriptor of a binary64 cycle records, at each contact, the
// feature that the local optimality conditions select between its two
// neighbours (a vertex, an edge or a straight passage), so that a reflection
// rounded off its edge line stays a reflection; the anchor gets the given
// feature. It only proposes a cycle, which the certificate decides.
using DPolygon=std::vector<ConvexArithmeticPoint<double>>;
// kLocal gives the anchor its own local feature too.
constexpr int kLocal=std::numeric_limits<int>::min();
std::vector<int> local_descriptor(const std::vector<DPolygon> &p,const DPolygon &q,
        std::size_t anchor,int anchor_feature) {
    const std::size_t k=p.size();std::vector<int> feature(k,-1);
    for(std::size_t i=0;i<k;++i) {
        if(i==anchor&&anchor_feature!=kLocal){feature[i]=anchor_feature;continue;}
        try {feature[i]=detail::CycleRefinement<double>::coordinate(q[(i+k-1)%k],q[(i+1)%k],q[i],p[i]).second;}
        catch(const std::runtime_error &){feature[i]=-1;}
    }
    return feature;
}
std::optional<std::size_t> vertex_at(const DPolygon &p,const ConvexArithmeticPoint<double> &x) {
    for(std::size_t j=0;j<p.size();++j)if(p[j]==x)return j;
    return std::nullopt;
}
// close() pins a vertex feature at the contact's current point: put a
// binary64 contact that only approximates its vertex on it first.
template<class S>
void snap_vertices(const std::vector<std::vector<ConvexArithmeticPoint<S>>> &polygons,const std::vector<int> &feature,
        std::vector<ConvexArithmeticPoint<S>> &q) {
    for(std::size_t i=0;i<q.size();++i)
        if(feature[i]>=0&&feature[i]%2==0)q[i]=polygons[i][std::size_t(feature[i])/2];
}
// close() with pivots: a binary64 contact next to a vertex can carry the
// feature of an incident edge, and the exact (or binary64) reflection across
// that edge then falls beyond the vertex, so close() fails. With pivots, the
// contact is pinned at that blocking vertex and the descriptor closed again,
// as the feature sweeps do; each pivot turns an edge into a vertex, so at
// most k occur. The result is still only a proposal for the certificate.
// *pivoted, when given, says whether a pivot was needed.
template<class S,class Regions>
bool close_descriptor(const Regions &polygons,std::vector<int> feature,std::vector<ConvexArithmeticPoint<S>> &q,bool pivots,
        bool *pivoted=nullptr) {
    if(pivoted)*pivoted=false;
    if(!pivots)return detail::CycleRefinement<S>::close(polygons,feature,q);
    for(std::size_t pivot=0;pivot<=q.size();++pivot) {
        auto blocking=feature;
        if(detail::CycleRefinement<S>::close(polygons,feature,q,&blocking))return true;
        if(blocking==feature)return false;
        feature=std::move(blocking);
        if(pivoted)*pivoted=true;
    }
    return false;
}
// The anchor's other closed form on edge j: the cycle passes straight through
// the edge (its neighbours on opposite sides of the edge line) instead of
// reflecting on it. The descriptor is closed with the anchor straight, and
// the anchor is placed where the segment between its neighbours crosses edge
// j; *t receives that parameter. This is where an eligible polygon's
// boundary optimum lies when the optimal cycle enters the polygon (Lemma 4.1).
template<class S,class Regions>
bool close_straight(const Regions &polygons,std::vector<int> feature,std::vector<ConvexArithmeticPoint<S>> &q,
        std::size_t anchor,std::size_t j,bool pivots,S *t=nullptr) {
    const std::size_t k=q.size();
    if(k<3)return false;
    feature[anchor]=-1;
    snap_vertices(polygons,feature,q);
    try {if(!close_descriptor<S>(polygons,feature,q,pivots))return false;}
    catch(const std::runtime_error &){return false;}
    const std::size_t before=(anchor+k-1)%k;std::size_t after=(anchor+1)%k;
    while(after!=before&&q[after]==q[before])after=(after+1)%k;
    if(after==before)return false;
    const auto &region=polygons[anchor];const std::size_t m=region.size();
    const auto a=region[j],e=region[(j+1)%m]-a,o=q[before],d=q[after]-o;
    const S determinant=d.cross(e);
    if(determinant==0)return false;
    const S s=(a-o).cross(e)/determinant,u=(a-o).cross(d)/determinant;
    if(s<0||s>1||u<0||u>1)return false;
    q[anchor]=a+e*u;
    if(t)*t=u;
    return true;
}
// A block of coincident visit points need not follow its polygons' own
// features: it may be pinned at a corner of their common region, such as a
// crossing of two edges, which no polygon's descriptor records. The block
// descriptor replaces each block's features by the face of that region that
// holds its local optimum between its neighbours (move_zero_blocks: a vertex
// there pins the block). True when q has a block and this changed something.
template<class S>
bool block_descriptor(const std::vector<std::vector<ConvexArithmeticPoint<S>>> &polygons,
        std::vector<ConvexArithmeticPoint<S>> &q,std::vector<int> &feature) {
    const std::size_t k=q.size();bool block=false;
    for(std::size_t i=0;i<k&&!block;++i)block=q[i]==q[(i+1)%k];
    if(!block)return false;
    const auto before=feature;
    try {detail::CycleRefinement<S>::move_zero_blocks(polygons,q,feature);}
    catch(const std::runtime_error &){return false;}
    return feature!=before;
}
// Step 2 of Algorithm 4.1: a binary64 candidate, read as binary rationals, is
// rebuilt from its descriptor by unfolding (the closed-form step of the
// rational search) and certified exactly, each descriptor once per anchor.
// The anchor's feature is the given one (a reflection on the probed edge, or
// its vertex); with coincident visit points the block descriptor is tried
// next, and last the anchor's own local feature, which differs when an
// optimal cycle passes straight through the anchor polygon.
struct DescriptorRebuilds {
    std::set<std::vector<long long>> tried;
    bool pivots=false; // ConvexCycleDoubleOptions::descriptor_pivots
    bool straight=true; // ConvexCycleDoubleOptions::straight_anchor
    template<class Verify>
    bool operator()(const ConvexRationalPolygons &polygons,std::size_t anchor,const std::vector<int> &feature,
            int local_anchor,const DPolygon &q,Verify verify,std::size_t &count,ConvexRationalPolygon &exact,
            ConvexCycleCertificateResult &certificate) {
        ConvexRationalPolygon lifted;lifted.reserve(q.size());
        for(const auto &v:q)lifted.emplace_back(v.external());
        auto attempt=[&](ConvexRationalPolygon candidate,const std::vector<int> &features,int kind) {
            // A block pinned at a corner (kFixed) is identified by its point.
            std::vector<long long> key(features.begin(),features.end());key.push_back(int(anchor));key.push_back(kind);
            for(std::size_t i=0;i<features.size();++i)if(features[i]==detail::CycleRefinement<R>::kFixed) {
                const auto point=candidate[i].external();
                key.push_back(std::bit_cast<long long>(point.x));key.push_back(std::bit_cast<long long>(point.y));
            }
            if(!tried.insert(std::move(key)).second)return false;
            ++count;
            snap_vertices(polygons,features,candidate);
            try {if(!close_descriptor<R>(polygons,features,candidate,pivots))return false;}
            catch(const std::runtime_error &){return false;}
            auto c=verify(candidate);
            if(c.status!=ConvexCycleCertificateStatus::Optimal)return false;
            exact=std::move(candidate);certificate=c;return true;
        };
        if(attempt(lifted,feature,0))return true;
        auto blocks=feature;auto moved=lifted;
        const bool has_blocks=block_descriptor(polygons,moved,blocks);
        if(has_blocks&&attempt(moved,blocks,1))return true;
        // The anchor straight through its edge, with the polygon descriptor
        // and with the block one.
        if(straight&&feature[anchor]>=0&&feature[anchor]%2==1) {
            const std::size_t j=std::size_t(feature[anchor])/2;
            auto straight=[&](ConvexRationalPolygon candidate,const std::vector<int> &features,int kind) {
                std::vector<long long> key(features.begin(),features.end());key.push_back(int(anchor));key.push_back(kind);
                for(std::size_t i=0;i<features.size();++i)if(features[i]==detail::CycleRefinement<R>::kFixed) {
                    const auto point=candidate[i].external();
                    key.push_back(std::bit_cast<long long>(point.x));key.push_back(std::bit_cast<long long>(point.y));
                }
                if(!tried.insert(std::move(key)).second)return false;
                ++count;
                if(!close_straight<R>(polygons,features,candidate,anchor,j,pivots))return false;
                auto c=verify(candidate);
                if(c.status!=ConvexCycleCertificateStatus::Optimal)return false;
                exact=std::move(candidate);certificate=c;return true;
            };
            if(straight(lifted,feature,3))return true;
            if(has_blocks&&straight(moved,blocks,4))return true;
        }
        if(local_anchor==feature[anchor])return false;
        auto own=feature;own[anchor]=local_anchor;
        return attempt(std::move(lifted),own,2);
    }
};
// Binary64 bisection never meets an exact zero of the derivative, so a piece
// whose minimum is not global would be bisected down to representability.
// Step 2 instead ends a piece at a fixed point of the closed-form step: the
// probe its descriptor proposed returns that same descriptor and proposal.
// That descriptor's exact rebuild has then already failed the certificate.
// This only ends a binary64 search early; Step 3 remains complete.
struct ClosedFormFixedPoint {
    std::map<std::pair<std::size_t,std::size_t>,std::pair<std::vector<int>,double>> last;
    bool operator()(std::size_t anchor,std::size_t edge,const std::vector<int> &feature,double next) {
        auto &entry=last[{anchor,edge}];
        const bool fixed=entry.first==feature&&entry.second==next;
        entry={feature,next};
        return fixed;
    }
};
// The double contacts themselves passed the exact certificate.
// Algorithm 4.1 with a tolerance (ConvexCycleDoubleOptions::boundary_search_gap):
// U - L <= gap, with the sum L + gap rounded down so that rounding never
// accepts a larger gap.
bool certificate_closes_gap(const ConvexCycleCertificateResult &c,double gap) {
    if(c.status!=ConvexCycleCertificateStatus::Feasible&&c.status!=ConvexCycleCertificateStatus::Optimal)return false;
    if(!std::isfinite(c.upper_bound)||!std::isfinite(c.lower_bound))return false;
    return c.upper_bound<=std::nextafter(c.lower_bound+gap,-INFINITY);
}

void record_exact_contacts(ConvexCycleDoubleResult &result) {
    if(result.status!=ConvexCycleStatus::Optimal||!result.exact_contacts.empty())return;
    for(const auto &v:result.contacts)result.exact_contacts.emplace_back(v);
    result.exact_certificate=result.certificate;
}

// The rational search for disjoint polygons, after the feature sweeps when
// refine is true (the public solver) and without them when false (Step 3 of
// Algorithm 4.1).
ConvexCycleResult solve_cycle_disjoint_rational(const ConvexRationalPolygons &input,bool refine) {
    ConvexCycleResult result;
    if (input.size() < 2) return result;
    ConvexRationalPolygons polygons;
    try {
        if (!detail::prepare_cycle_polygons(input, polygons)) {
            result.status = ConvexCycleStatus::UnsupportedIntersection;
            return result;
        }
    } catch (const std::invalid_argument &) { return result; }

    const ConvexCycleCertificateGeometry prepared(polygons);
    if(refine)try {
        if(detail::CycleRefinement<R>::run(polygons,[&](const ConvexRationalPolygon &q) {
            ++result.oracle_calls;++result.certificate_checks;
            const auto c=detail::cycle_optimality_certificate(prepared,q);
            if(c.status!=ConvexCycleCertificateStatus::Optimal)return false;
            result.status=ConvexCycleStatus::Optimal;result.contacts=q;result.certificate=c;
            for(size_t i=0;i<q.size();++i){const auto d=q[(i+1)%q.size()]-q[i];result.squared_link_lengths.push_back(d.dot(d));}
            return true;
        }))return result;
    } catch(const std::runtime_error &) {} // Checked proposals never affect fallback correctness.

    const auto anchor = std::min_element(polygons.begin(), polygons.end(),
        [](const auto &a, const auto &b) { return a.size() < b.size(); });
    result.anchor_polygon = std::size_t(anchor - polygons.begin());
    const std::size_t k = polygons.size(), m = anchor->size();
    ConvexRationalPolygons remaining;
    for (std::size_t j = 1; j < k; ++j)
        remaining.push_back(polygons[(result.anchor_polygon + j) % k]);

    auto evaluate = [&](const Point &point) {
        detail::cycle_checkpoint();
        ++result.oracle_calls;++result.anchored_solves;
        const auto path = detail::solve_rational_disjoint_exact(point, point, remaining);
        if (path.contacts.size() != k - 1) throw std::runtime_error("Incomplete anchored cycle");
        ConvexRationalPolygon contacts(k);
        contacts[result.anchor_polygon] = point;
        for (std::size_t j = 1; j < k; ++j)
            contacts[(result.anchor_polygon + j) % k] = path.contacts[j - 1];
        const auto certificate = tpp_convex_verify_cycle_certificate(prepared, contacts);
        ++result.certificate_checks;
        if (certificate.status != ConvexCycleCertificateStatus::Optimal &&
            certificate.status != ConvexCycleCertificateStatus::Feasible)
            throw std::runtime_error("Infeasible rational anchored cycle");
        // A search direction is usable only when the existing fixed-source
        // oracle satisfies every free-contact support condition exactly.
        for (std::size_t i = 0; i < k; ++i) if (i != result.anchor_polygon)
            for (const auto &v : polygons[i])
                if (support_sign(contacts, i, v - contacts[i]) < 0)
                    throw std::runtime_error("Rational anchored path failed its support certificate");
        result.contacts = contacts;
        result.certificate = certificate;
        if (certificate.status == ConvexCycleCertificateStatus::Optimal) {
            result.status = ConvexCycleStatus::Optimal;
            for (std::size_t i = 0; i < k; ++i) {
                const auto d = contacts[(i + 1) % k] - contacts[i];
                result.squared_link_lengths.push_back(d.dot(d));
            }
        }
        return contacts;
    };
    auto solved = [&] { return result.status == ConvexCycleStatus::Optimal; };
    try {
        // The closed form of the anchored cycle's type, with the anchor
        // reflecting on edge j: its anchor is the next probe, and the cycle
        // itself is accepted only by the global certificate.
        // proposals (the right end of a bracket): also the anchor straight
        // through edge j, the other closed form of the type.
        auto step=[&](std::size_t j,const ConvexRationalPolygon &q,std::optional<R> &next,std::vector<R> *proposals=nullptr) {
            ++result.closed_form_steps;
            auto feature=contact_features<R>(polygons,q);feature[result.anchor_polygon]=int(2*j+1);
            const auto a=(*anchor)[j],edge=(*anchor)[(j+1)%m]-a;
            if(proposals) {
                auto crossing=q;R u;
                if(close_straight<R>(polygons,feature,crossing,result.anchor_polygon,j,false,&u))proposals->push_back(u);
            }
            auto closed=q;
            try {
                if(!detail::CycleRefinement<R>::close(polygons,feature,closed))return false;
            } catch(const std::runtime_error &) {return false;}
            next=(closed[result.anchor_polygon]-a).dot(edge)/edge.dot(edge);
            if(proposals)proposals->push_back(*next);
            ++result.certificate_checks;
            const auto certificate=detail::cycle_optimality_certificate(prepared,closed);
            if(certificate.status!=ConvexCycleCertificateStatus::Optimal)return false;
            result.status=ConvexCycleStatus::Optimal;result.contacts=closed;result.certificate=certificate;
            result.squared_link_lengths.clear();
            for(std::size_t i=0;i<k;++i){const auto d=closed[(i+1)%k]-closed[i];result.squared_link_lengths.push_back(d.dot(d));}
            return true;
        };
        search_cycle<R>(polygons,result.anchor_polygon,evaluate,solved,nullptr,&result.bisection_probes,step);
        if(solved())return result;
    } catch (const std::runtime_error &error) {
        result.diagnostic=error.what();
        result.status = ConvexCycleStatus::OracleFailure;
        return result;
    }
    result.status = ConvexCycleStatus::OracleFailure;
    return result;
}
} // namespace

ConvexCycleResult tpp_convex_solve_cycle_disjoint(const ConvexRationalPolygons &input) {
    return solve_cycle_disjoint_rational(input,true);
}

ConvexCycleResult tpp_convex_solve_cycle_disjoint(const std::vector<std::vector<Vector2>> &input) {
    ConvexRationalPolygons exact;
    for (const auto &polygon : input) {
        ConvexRationalPolygon p;
        for (auto point : polygon) {
            if (!point.is_finite()) return {};
            p.emplace_back(point);
        }
        exact.push_back(std::move(p));
    }
    return tpp_convex_solve_cycle_disjoint(exact);
}

namespace {
bool rational_inside(const Point &q,const ConvexRationalPolygon &p) {
    // Points and segments have no area: their edge halfplanes alone accept
    // the whole plane or a whole line.
    if(p.size()<=2)return detail::CycleRefinement<R>::inside(q,p);
    for(std::size_t j=0;j<p.size();++j)
        if((p[(j+1)%p.size()]-p[j]).cross(q-p[j])<0)return false;
    return true;
}

// Repair only an outward-rounded exported contact. The initial displacement
// comes from nextafter (one representable step), never from a chosen epsilon.
// This is output rounding, not an optimization or rational-solver fallback.
void round_contacts_inward(const ConvexRationalPolygons &polygons,std::vector<Vector2> &q) {
    for(std::size_t i=0;i<q.size();++i) {
        if(!q[i].is_finite())throw std::runtime_error("Nonfinite double contact");
        const Point original(q[i]);const auto &p=polygons[i];
        if(rational_inside(original,p))continue;
        Point center;for(const auto &v:p)center=center+v;
        center=center*(R(1)/p.size());const auto toward=center-original;
        R fraction=0;
        for(std::size_t j=0;j<p.size();++j) {
            const auto edge=p[(j+1)%p.size()]-p[j];const R side=edge.cross(original-p[j]);
            if(side<0)fraction=std::max(fraction,R(-side/edge.cross(toward)));
        }
        for(int coordinate=0;coordinate<2;++coordinate) {
            const R d=coordinate==0?toward.x:toward.y;if(d==0)continue;
            const double value=coordinate==0?q[i].x:q[i].y;
            const double adjacent=std::nextafter(value,d>0?std::numeric_limits<double>::infinity():
                                                           -std::numeric_limits<double>::infinity());
            if(!std::isfinite(adjacent))continue;
            fraction=std::max(fraction,R((R(adjacent)-R(value))/d));
        }
        fraction=std::min(fraction,R(1));
        for(;;) {
            const auto candidate=(original+toward*fraction).external();
            if(candidate.is_finite()&&rational_inside(Point(candidate),p)){q[i]=candidate;break;}
            if(fraction==1){q[i]=p.front().external();break;}
            fraction=std::min(R(1),R(2*fraction));
        }
    }
}
} // namespace

namespace {
// The binary64 search for pairwise disjoint polygons, which
// prepare_cycle_polygons has normalized and checked.
ConvexCycleDoubleResult solve_cycle_disjoint_double(ConvexCycleDoubleResult &result,
        const std::vector<std::vector<Vector2>> &input,const ConvexRationalPolygons &normalized,
        const ConvexCycleDoubleOptions &options) {
    using DPoint=ConvexArithmeticPoint<double>;
    using DPolygon=std::vector<DPoint>;
    std::vector<DPolygon> polygons;
    for(const auto &p:normalized){DPolygon q;for(const auto &v:p)q.emplace_back(v.external());polygons.push_back(q);}
    DPolygon initial;for(const auto &v:options.initial_contacts)initial.emplace_back(v);
    result.anchor_polygon=std::size_t(std::min_element(polygons.begin(),polygons.end(),
        [](const auto &a,const auto &b){return a.size()<b.size();})-polygons.begin());
    const auto k=polygons.size();std::vector<DPolygon> remaining;std::vector<std::vector<Vector2>> remaining_input;
    for(std::size_t j=1;j<k;++j) {
        remaining.push_back(polygons[(result.anchor_polygon+j)%k]);
        remaining_input.emplace_back();for(const auto &v:remaining.back())remaining_input.back().push_back(v.external());
    }
    // refine_contacts=false with rational recovery: Algorithm 4.1 of paper 1,
    // with one prepared geometry for its exact certificates.
    const bool method=!options.refine_contacts&&options.recover_arithmetic_failures;
    std::optional<ConvexCycleCertificateGeometry> prepared;
    auto geometry=[&]()->const ConvexCycleCertificateGeometry & {
        if(!prepared)prepared.emplace(input,options.interval_certificate);
        return *prepared;
    };
    auto verify_double=[&](const std::vector<Vector2> &contacts,double cut) {
        return method?tpp_convex_verify_cycle_certificate(geometry(),contacts,cut,options.interval_certificate):
                      tpp_convex_verify_cycle_certificate(input,contacts,cut,options.interval_certificate);
    };
    auto consider=[&](std::vector<Vector2> contacts) {
        detail::cycle_checkpoint();
        const double cut=(options.bound_first||options.interval_certificate)?options.lower_bound_cutoff:INFINITY;
        auto certificate=verify_double(contacts,cut);++result.certificate_checks;
        if(certificate.status==ConvexCycleCertificateStatus::InvalidCandidate) {
            round_contacts_inward(normalized,contacts);
            certificate=verify_double(contacts,cut);++result.certificate_checks;
        }
        result.certificate_cutoff_skips+=certificate.optimality_check_skipped;
        result.certificate_interval_uses+=certificate.interval_bounds_used;
        if(certificate.status!=ConvexCycleCertificateStatus::Feasible&&
           certificate.status!=ConvexCycleCertificateStatus::Optimal)
            throw std::runtime_error("Double candidate failed exact feasibility after rounding");
        if(result.contacts.empty()||certificate.upper_bound<result.certificate.upper_bound||
           (certificate.upper_bound==result.certificate.upper_bound&&certificate.lower_bound>result.certificate.lower_bound)||
           certificate.status==ConvexCycleCertificateStatus::Optimal||certificate.lower_bound>=options.lower_bound_cutoff) {
            result.contacts=std::move(contacts);result.certificate=certificate;
        }
        if(certificate.status==ConvexCycleCertificateStatus::Optimal)result.status=ConvexCycleStatus::Optimal;
        else if(certificate.lower_bound>=options.lower_bound_cutoff)result.status=ConvexCycleStatus::CertifiedBound;
    };
    auto solved=[&]{return result.status==ConvexCycleStatus::Optimal||result.status==ConvexCycleStatus::CertifiedBound||
                           result.status==ConvexCycleStatus::GapClosed||!result.exact_contacts.empty();};
    const bool gap_search=method&&options.boundary_search_gap&&options.max_gap>0;
    // Algorithm 4.1 with a tolerance: accept a binary64 candidate whose
    // certificate closes the gap, before any exact rebuild.
    auto gap_closes=[&](const DPolygon &q) {
        if(!gap_search)return false;
        std::vector<Vector2> contacts;for(const auto &v:q)contacts.push_back(v.external());
        auto certificate=verify_double(contacts,INFINITY);++result.certificate_checks;
        if(certificate.status==ConvexCycleCertificateStatus::InvalidCandidate) {
            round_contacts_inward(normalized,contacts);
            certificate=verify_double(contacts,INFINITY);++result.certificate_checks;
        }
        if(!certificate_closes_gap(certificate,options.max_gap))return false;
        result.contacts=std::move(contacts);result.certificate=certificate;
        result.status=certificate.status==ConvexCycleCertificateStatus::Optimal?ConvexCycleStatus::Optimal:ConvexCycleStatus::GapClosed;
        return true;
    };
    auto recover_cycle=[&] {
        detail::cycle_checkpoint();
        detail::CyclePhaseScope recovery(detail::CyclePhase::RationalRecovery);
        ++result.rational_cycle_recoveries;
        // Step 3 of Algorithm 4.1 runs the rational search without sweeps.
        const auto recovered=solve_cycle_disjoint_rational(normalized,options.refine_contacts);
        if(method) {
            result.rational_anchored_solves+=recovered.anchored_solves;
            result.rational_bisection_probes+=recovered.bisection_probes;
            result.rational_closed_form_steps+=recovered.closed_form_steps;
        }
        if(recovered.status!=ConvexCycleStatus::Optimal){result.diagnostic=recovered.diagnostic;return false;}
        if(method) {
            result.method_step=3;result.exact_contacts=recovered.contacts;result.exact_certificate=recovered.certificate;
        }
        std::vector<Vector2> q;for(const auto &v:recovered.contacts)q.push_back(v.external());
        ++result.oracle_calls;consider(q);
        retain_exact_bound(result,recovered.certificate,options.lower_bound_cutoff);
        return !result.contacts.empty();
    };
    try {
        if(options.bound_first&&detail::cycle_cutoff_promising(normalized,options.initial_contacts,options.lower_bound_cutoff)) {
            ++result.initial_contact_checks;consider(options.initial_contacts);
            if(solved()){++result.initial_contact_accepts;return result;}
        }
        if(options.refine_contacts)detail::CycleRefinement<double>::run(polygons,
                [&](const DPolygon &q,const std::vector<int> &features,bool closed) {
            ++result.oracle_calls;
            std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());
            consider(candidate);if(solved())return true;
            if(closed&&options.recover_arithmetic_failures&&!options.proposal_only) {
                detail::CyclePhaseScope recovery(detail::CyclePhase::RationalRecovery);
                ++result.rational_feature_recoveries;
                ConvexRationalPolygon lifted;for(auto v:candidate)lifted.emplace_back(v);
                if(detail::CycleRefinement<R>::close(normalized,features,lifted)) {
                    ++result.oracle_calls;++result.certificate_checks;
                    const auto exact_certificate=tpp_convex_verify_cycle_certificate(normalized,lifted);
                    if(exact_certificate.status==ConvexCycleCertificateStatus::Optimal) {
                        candidate.clear();for(const auto &v:lifted)candidate.push_back(v.external());
                        ++result.oracle_calls;consider(candidate);
                        retain_exact_bound(result,exact_certificate,options.lower_bound_cutoff);
                        return true;
                    }
                }
            }
            return false;
        },initial);
        if(solved()||result.status==ConvexCycleStatus::FloatingPointLimit)return result;
        // Exhausting the finite proposal does not prove convergence. With
        // recovery enabled, use the complete exact reduction now instead of
        // repeating rational repairs at many double anchor probes.
    } catch(const std::exception &error){result.diagnostic=error.what();}
    if(options.proposal_only) {
        result.status=ConvexCycleStatus::ProposalLimit;
        if(result.contacts.empty())result.certificate.upper_bound=INFINITY;
        return result;
    }
    if(options.refine_contacts&&options.recover_arithmetic_failures)try {
        if(recover_cycle())return result;
    } catch(const std::exception &error){result.diagnostic=error.what();}
    try {
        DescriptorRebuilds rebuilds;ClosedFormFixedPoint fixed_point;rebuilds.pivots=options.descriptor_pivots;rebuilds.straight=options.straight_anchor;
        auto rebuild=[&](std::size_t anchor,const std::vector<int> &feature,const DPolygon &q) {
            ConvexRationalPolygon exact;ConvexCycleCertificateResult certificate;
            const int local=local_descriptor(polygons,q,anchor,kLocal)[anchor];
            if(!rebuilds(normalized,anchor,feature,local,q,[&](const ConvexRationalPolygon &c) {
                    ++result.oracle_calls;++result.certificate_checks;
                    // Kept only when Optimal: no reporting bounds otherwise.
                    return detail::cycle_optimality_certificate(geometry(),c);
                },result.descriptor_rebuilds,exact,certificate))return false;
            std::vector<Vector2> rounded;for(const auto &v:exact)rounded.push_back(v.external());
            result.exact_contacts=std::move(exact);result.exact_certificate=certificate;
            ++result.oracle_calls;
            try {consider(std::move(rounded));} catch(const std::runtime_error &) {}
            retain_exact_bound(result,certificate,options.lower_bound_cutoff);
            return true;
        };
        // The closed form of the anchored cycle's descriptor, with the anchor
        // reflecting on edge j: its anchor proposes the next probe, and its
        // exact rebuild goes to the certificate. Binary64 contacts are not
        // certified themselves: they rarely are exact, and the rebuild of
        // their descriptor is what can pass.
        auto step=[&](std::size_t j,const DPolygon &q,std::optional<double> &next) {
            ++result.closed_form_steps;
            const std::size_t anchor=result.anchor_polygon,m=polygons[anchor].size();
            const auto feature=local_descriptor(polygons,q,anchor,int(2*j+1));
            auto closed=q;bool built=false;snap_vertices(polygons,feature,closed);
            bool pivoted=false;
            try {built=close_descriptor<double>(polygons,feature,closed,options.descriptor_pivots,&pivoted);}
            catch(const std::runtime_error &) {}
            if(built) {
                const auto a=polygons[anchor][j],edge=polygons[anchor][(j+1)%m]-a;
                next=(closed[anchor]-a).dot(edge)/edge.dot(edge);
                if(gap_closes(closed))return true;
            }
            if(rebuild(anchor,feature,q))return true;
            // A pivoted closed form is not the descriptor's own: its fixed
            // point says nothing about the piece, which goes on (bisection).
            return built&&!pivoted&&fixed_point(anchor,j,feature,*next);
        };
        auto evaluate=[&](const DPoint &point) {
            detail::cycle_checkpoint();
            ++result.oracle_calls;
            if(method)++result.anchored_solves;
            const auto path=[&] {
                try {
                    if(!method)return detail::DisjointArithmetic<double>::Solver(point,point,remaining).solve(true);
                    // Algorithm 4.1: the binary64 construction of the TPP flow.
                    const auto chain=detail::double_candidate_chain(point.external(),point.external(),remaining_input);
                    if(chain.size()!=k+1)throw std::runtime_error("Incomplete binary64 anchored chain");
                    detail::DisjointArithmetic<double>::Result converted;
                    for(std::size_t j=1;j<k;++j)converted.contacts.emplace_back(chain[j]);
                    return converted;
                }
                catch(const std::exception &) {
                    if(!options.recover_arithmetic_failures)throw;
                    detail::CyclePhaseScope recovery(detail::CyclePhase::RationalRecovery);
                    ++result.rational_anchor_recoveries;
                    ConvexRationalPolygons exact_remaining;
                    for(std::size_t j=1;j<k;++j)exact_remaining.push_back(normalized[(result.anchor_polygon+j)%k]);
                    const auto recovered=detail::solve_rational_disjoint_exact(Point(point.external()),Point(point.external()),exact_remaining);
                    detail::DisjointArithmetic<double>::Result converted;
                    for(const auto &q:recovered.contacts)converted.contacts.emplace_back(q.external());
                    return converted;
                }
            }();
            if(path.contacts.size()!=k-1)throw std::runtime_error("Incomplete double anchored cycle");
            DPolygon q(k);q[result.anchor_polygon]=point;
            for(std::size_t j=1;j<k;++j)q[(result.anchor_polygon+j)%k]=path.contacts[j-1];
            if(!method) {
                std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());
                consider(std::move(candidate));
                return q;
            }
            // At a vertex of the anchor polygon where F does not decrease
            // along either edge, the anchored cycle with that vertex pinned
            // is rebuilt exactly.
            if(const auto j=vertex_at(polygons[result.anchor_polygon],point)) {
                const auto &a=polygons[result.anchor_polygon];const std::size_t m=a.size();
                int forward=0,backward=0;
                try {
                    forward=support_sign(q,result.anchor_polygon,a[(*j+1)%m]-a[*j]);
                    backward=support_sign(q,result.anchor_polygon,a[*j]-a[(*j+m-1)%m]);
                } catch(const std::runtime_error &) {forward=backward=0;}
                if(forward>=0&&backward<=0&&!gap_closes(q))
                    rebuild(result.anchor_polygon,local_descriptor(polygons,q,result.anchor_polygon,int(2**j)),q);
            }
            return q;
        };
        if(method) {
            search_cycle<double>(polygons,result.anchor_polygon,evaluate,solved,&result.diagnostic,&result.bisection_probes,step);
            if(solved()){result.method_step=2;record_exact_contacts(result);}
        } else search_cycle<double>(polygons,result.anchor_polygon,evaluate,solved,&result.diagnostic);
        if(!solved())result.status=result.contacts.empty()?ConvexCycleStatus::OracleFailure:ConvexCycleStatus::FloatingPointLimit;
    } catch(const std::runtime_error &error) {
        result.status=ConvexCycleStatus::OracleFailure;result.diagnostic=error.what();
    }
    if(!solved()&&options.recover_arithmetic_failures)try {
        if(!recover_cycle())result.status=ConvexCycleStatus::OracleFailure;
    } catch(const std::exception &error){result.status=ConvexCycleStatus::OracleFailure;result.diagnostic=error.what();}
    return result;
}
} // namespace

ConvexCycleDoubleResult tpp_convex_solve_cycle_disjoint_double(const std::vector<std::vector<Vector2>> &input,
        const ConvexCycleDoubleOptions &options) {
    ConvexCycleDoubleResult result;
    return detail::run_cycle_execution(result,options,[&]() -> ConvexCycleDoubleResult {
    if(input.size()<2)return result;
    if(std::isnan(options.lower_bound_cutoff)||(!options.initial_contacts.empty()&&options.initial_contacts.size()!=input.size())||
       !std::all_of(options.initial_contacts.begin(),options.initial_contacts.end(),[](auto q){return q.is_finite();}))return result;
    ConvexRationalPolygons normalized;
    for(const auto &polygon:input)for(auto v:polygon)if(!v.is_finite())return result;
    try {
        if(!detail::prepare_cycle_polygons(input,normalized)) {
            result.status=ConvexCycleStatus::UnsupportedIntersection;return result;
        }
    } catch(const std::invalid_argument &){return result;}
    return solve_cycle_disjoint_double(result,input,normalized,options);
    });
}

namespace {
// Exact orientation of binary64 points: the int128 determinant, or rationals.
int exact_orientation(Vector2 a,Vector2 b,Vector2 q) {
    if(const auto sign=detail::dyadic_orientation(a,b,q))return *sign;
    const R d=(Point(b)-Point(a)).cross(Point(q)-Point(a));
    return d>0?1:d<0?-1:0;
}
// Whether q meets int(p), exactly. p is convex, counterclockwise, without
// repeated vertices; q is a convex polygon (counterclockwise), a segment or a
// point. Two convex sets with disjoint interiors are weakly separated by the
// line of an edge of one of them (for a segment, its own line), so q misses
// int(p) exactly when such a line leaves them on opposite closed sides.
// The vertex of the convex polygon r (counterclockwise) farthest to the left of
// the directed line from a to b, decided exactly. The search starts at the
// binary64 guess `start` and climbs to a strictly better neighbor, crossing
// plateaus of equal value, so the guess only saves time: along a convex
// polygon the signed distance to a line is unimodal.
size_t farthest_left(Vector2 a,Vector2 b,const std::vector<Vector2> &r,size_t start) {
    const size_t n=r.size();
    // The sign of (b-a) x (r[w]-r[v]): binary64 when its value exceeds an a
    // priori bound of its rounding error (two rounded differences, two
    // products and a subtraction: under 4u of the sum of the products' sizes,
    // taken as 8u here), and in rational arithmetic otherwise.
    const double dx=b.x-a.x,dy=b.y-a.y;
    auto compare=[&](size_t w,size_t v) {
        const double ex=r[w].x-r[v].x,ey=r[w].y-r[v].y,t1=dx*ey,t2=dy*ex,size=std::abs(t1)+std::abs(t2);
        const double c=t1-t2;
        if(size>0x1p-900&&std::abs(c)>0x1p-50*size)return c>0?1:-1;
        const R exact=(Point(b)-Point(a)).cross(Point(r[w])-Point(r[v]));return exact>0?1:exact<0?-1:0;
    };
    size_t v=start%n;
    for(size_t moves=0;moves<2*n;++moves) {
        const size_t f=(v+1)%n,g=(v+n-1)%n;
        if(compare(f,v)>0){v=f;continue;}
        if(compare(g,v)>0){v=g;continue;}
        size_t w=f,count=0;
        while(count<n&&compare(w,v)==0){w=(w+1)%n;++count;}
        if(count<n&&compare(w,v)>0){v=w;continue;}
        break;
    }
    return v;
}
// Whether no vertex of r lies strictly to the left (side 1) or strictly to the
// right (side -1) of the line from a to b; guess is updated for the next call.
bool closed_outside(Vector2 a,Vector2 b,const std::vector<Vector2> &r,int side,size_t &guess) {
    const Vector2 from=side>0?a:b,to=side>0?b:a;
    const Vector2 d=to-from;
    // Binary64 rotating pointer: advance while the next vertex is farther.
    const size_t n=r.size();
    for(size_t steps=0;steps<n;++steps) {
        const size_t f=(guess+1)%n;
        if(d.cross(r[f]-r[guess])>0)guess=f;else break;
    }
    if(exact_orientation(from,to,r[guess])>0)return false; // a vertex strictly left
    guess=farthest_left(from,to,r,guess);
    return exact_orientation(from,to,r[guess])<=0;
}
bool meets_interior(const std::vector<Vector2> &p,const std::vector<Vector2> &q) {
    if(p.size()<3)return false; // no interior
    // Rotating calipers: as the edges of p turn counterclockwise, the vertex
    // of q farthest to their left moves counterclockwise, so the guesses
    // advance about |p|+|q| steps in all, and every decision is exact.
    size_t guess=0;
    {
        const Vector2 d=p[1]-p[0];double best=-INFINITY;
        for(size_t v=0;v<q.size();++v){const double c=d.cross(q[v]-p[0]);if(c>best){best=c;guess=v;}}
    }
    for(size_t j=0;j<p.size();++j)if(closed_outside(p[j],p[(j+1)%p.size()],q,1,guess))return false;
    if(q.size()==2) {
        size_t g1=0,g2=0;
        return !closed_outside(q[0],q[1],p,1,g1)&&!closed_outside(q[0],q[1],p,-1,g2);
    }
    if(q.size()>=3) {
        size_t back=0;
        const Vector2 d=q[1]-q[0];double best=-INFINITY;
        for(size_t v=0;v<p.size();++v){const double c=d.cross(p[v]-q[0]);if(c>best){best=c;back=v;}}
        for(size_t j=0;j<q.size();++j)if(closed_outside(q[j],q[(j+1)%q.size()],p,1,back))return false;
    }
    return true;
}
} // namespace

namespace detail {
// The one boundary that Algorithm 4.1 searches for polygons that are not
// pairwise disjoint (ConvexCycleAnchorChoice), or none: every boundary.
// Eligibility is exact; the order among eligible polygons is a heuristic.
std::optional<size_t> eligible_anchor(const std::vector<std::vector<Vector2>> &p,ConvexCycleAnchorChoice choice) {
    const size_t k=p.size();
    if(choice==ConvexCycleAnchorChoice::EveryBoundary||k<2)return std::nullopt;
    std::optional<size_t> best;
    for(size_t i=0;i<k;++i) {
        if(p[i].size()<2)continue; // a point region is searched alone already
        // Eligible (Lemma 4.1): one cyclic neighbour misses int(P_i). A link
        // from that neighbour's visit point to an interior visit point of P_i
        // crosses the boundary, and visiting P_i there is no longer.
        if(meets_interior(p[i],p[(i+k-1)%k])&&meets_interior(p[i],p[(i+1)%k]))continue;
        if(!best||p[i].size()<p[*best].size())best=i;
    }
    return best;
}
// Rational regions: their binary64 vertices, when every vertex is one, and
// otherwise every boundary (the rational path keeps its earlier behavior).
std::optional<size_t> eligible_anchor(const ConvexRationalPolygons &p,ConvexCycleAnchorChoice choice) {
    if(choice==ConvexCycleAnchorChoice::EveryBoundary)return std::nullopt;
    std::vector<std::vector<Vector2>> binary;binary.reserve(p.size());
    for(const auto &polygon:p) {
        auto &q=binary.emplace_back();q.reserve(polygon.size());
        for(const auto &v:polygon){const Vector2 w=v.external();if(!w.is_finite()||Point(w)!=v)return std::nullopt;q.push_back(w);}
    }
    return eligible_anchor(binary,choice);
}
} // namespace detail

namespace {
// A point common to all regions, or nothing, decided exactly: Seidel's
// randomized incremental linear program in the plane, over the halfplanes of
// every edge (and the lines and ends of segments, and the coordinates of
// points), with the lexicographic maximum of (x, y) as the unique optimum. Its
// expected time is linear in the number of halfplanes, where clipping the
// regions one by another took quadratic time on regions with many vertices.
// The order is random but fixed, so the result is reproducible.
// With `leading`, the program first runs on the first `leading` regions alone:
// when they have no common point, neither do all of them. Callers pass the
// regions after which the binary64 intersection became empty, which on
// small polygons finds the empty set sooner than the program over every
// halfplane. The answer is the same.
ConvexRationalPolygon common_region_of(const ConvexRationalPolygons &p) {
    struct Half{Point g;R h;}; // g.x >= h
    std::vector<Half> halves;
    R lo_x=0,lo_y=0,hi_x=0,hi_y=0;bool first=true;
    for(const auto &region:p)for(const auto &v:region) {
        if(first){lo_x=hi_x=v.x;lo_y=hi_y=v.y;first=false;}
        lo_x=std::min(lo_x,v.x);hi_x=std::max(hi_x,v.x);lo_y=std::min(lo_y,v.y);hi_y=std::max(hi_y,v.y);
    }
    if(first)return {};
    lo_x-=1;lo_y-=1;hi_x+=1;hi_y+=1;
    for(const auto &region:p) {
        const size_t m=region.size();
        if(m==0)return {};
        if(m==1) {
            const Point c=region[0];
            halves.push_back({Point{R(1),R(0)},c.x});halves.push_back({Point{R(-1),R(0)},-c.x});
            halves.push_back({Point{R(0),R(1)},c.y});halves.push_back({Point{R(0),R(-1)},-c.y});
        } else if(m==2) {
            const Point a=region[0],b=region[1],e=b-a,g{-e.y,e.x};
            halves.push_back({g,g.dot(a)});halves.push_back({Point{-g.x,-g.y},-g.dot(a)});
            halves.push_back({e,e.dot(a)});halves.push_back({Point{-e.x,-e.y},-e.dot(b)});
        } else for(size_t j=0;j<m;++j) {
            const Point a=region[j],e=region[(j+1)%m]-a,g{-e.y,e.x};
            halves.push_back({g,g.dot(a)});
        }
    }
    std::mt19937 shuffle_rng(20261127);
    std::shuffle(halves.begin(),halves.end(),shuffle_rng);
    const std::array<Half,4> box{Half{Point{R(1),R(0)},lo_x},Half{Point{R(-1),R(0)},-hi_x},Half{Point{R(0),R(1)},lo_y},Half{Point{R(0),R(-1)},-hi_y}};
    Point v{hi_x,hi_y};
    for(size_t i=0;i<halves.size();++i) {
        if(i%256==0)detail::cycle_checkpoint();
        const auto &[g,h]=halves[i];
        if(g.dot(v)>=h)continue;
        // The new optimum lies on the line g.x = h: a linear program in t.
        const Point u{-g.y,g.x};
        const Point origin=g.x!=0?Point{h/g.x,R(0)}:Point{R(0),h/g.y};
        bool has_lo=false,has_hi=false;R t_lo=0,t_hi=0;
        auto bound=[&](const Half &c)->bool {
            const R slope=c.g.dot(u),rhs=c.h-c.g.dot(origin);
            if(slope>0){const R t=rhs/slope;if(!has_lo||t>t_lo){t_lo=t;has_lo=true;}}
            else if(slope<0){const R t=rhs/slope;if(!has_hi||t<t_hi){t_hi=t;has_hi=true;}}
            else if(rhs>0)return false;
            return true;
        };
        for(const auto &c:box)if(!bound(c))return {};
        for(size_t j=0;j<i;++j)if(!bound(halves[j]))return {};
        if(!has_lo||!has_hi||t_lo>t_hi)return {};
        const bool up=u.x!=0?u.x>0:u.y>0;
        const R t=up?t_hi:t_lo;
        v=origin+u*t;
    }
    return {v};
}
ConvexRationalPolygon common_region(const ConvexRationalPolygons &p,size_t leading=0) {
    if(leading>=2&&leading<p.size()&&common_region_of(ConvexRationalPolygons(p.begin(),p.begin()+ptrdiff_t(leading))).empty())return {};
    return common_region_of(p);
}
// The complete boundary-anchor reduction, shared by both arithmetic backends.
// The independent restriction checker includes the zero-link subgradient disk.
// Step(anchor,edge,q,next) solves the anchored cycle q's own combinatorial
// type in closed form, with the anchor reflecting on that edge: the cycle is
// accepted if the global certificate proves it, and its anchor parameter is
// the next probe on the edge (Newton's role: when a probe has the type of the
// edge's minimizer, the next probe lands on it exactly). Midpoints and the
// simplest rationals keep the termination guarantee.
// Endpoint(anchor,left,right,a,b), when given, is Step 2's restricted
// endpoint test (ConvexCycleDoubleOptions::endpoint_certificate): it runs
// only on a piece about to be bisected, after the closed-form steps at its
// ends, and true skips the piece. The rational search keeps Restricted, which
// runs before the derivative signs.
template<class S,class Evaluate,class Restricted,class Refine,class Solved,class Step,class Endpoint=std::nullptr_t>
void search_intersecting_boundaries(const std::vector<std::vector<ConvexArithmeticPoint<S>>> &p,
        Evaluate evaluate,Restricted restricted,Refine refine,Solved solved,Step step,
        size_t &anchor_index,std::string &diagnostic,size_t *probes=nullptr,Endpoint endpoint=nullptr,
        std::optional<size_t> only=std::nullopt,bool ends_first=true,bool first_pass=true) {
    using P=ConvexArithmeticPoint<S>;using Polygon=std::vector<P>;const size_t k=p.size();
    // Check a representative boundary point of every region before spending
    // unbounded rational reconstruction work on one anchor's first edge.
    // These are the same anchored problems and exact acceptance conditions as
    // below; only their order changes. Reuse their contacts in the full search.
    std::vector<Polygon> first_contacts(k);
    // Every tour visits every region. A point is visited at itself, so the
    // anchored TPP at the point is the CTPP: it is searched alone, in both
    // steps (Step 3 then costs one exact TPP). A segment has no interior, so
    // it is always eligible (eligible_anchor), and is searched on one of its
    // two coincident edges.
    std::optional<size_t> point;
    for(size_t i=0;i<k&&!point;++i)if(p[i].size()==1)point=i;
    // With an eligible polygon (only), its boundary alone is searched
    // (Lemma 4.1); otherwise every boundary. first_pass: first one anchored
    // TPP at a vertex of every boundary, which seeds the contact refinement
    // (refine_contacts); Algorithm 4.1 alone has no use for it.
    std::vector<size_t> order,searched;
    if(point)order.push_back(*point);
    else if(only&&!first_pass)order.push_back(*only);
    else for(size_t i=0;i<k;++i)order.push_back(i);
    if(only&&!point)searched.push_back(*only);else searched=order;
    for(const size_t anchor:order)try {
        anchor_index=anchor;
        first_contacts[anchor]=evaluate(anchor,p[anchor].front());
        if(solved()||refine(first_contacts[anchor]))return;
    } catch(const std::runtime_error &error){diagnostic=error.what();}
    for(const size_t anchor:searched) {
        detail::cycle_checkpoint();
        anchor_index=anchor;
        // A point region is its own only anchor, already evaluated above.
        if(p[anchor].size()<2)continue;
        auto derivative=[&](const Polygon &q,const P &e) {
            size_t before=(anchor+k-1)%k,after=(anchor+1)%k;
            while(before!=anchor&&q[before]==q[anchor])before=(before+k-1)%k;
            while(after!=anchor&&q[after]==q[anchor])after=(after+1)%k;
            return direction_support_sign(q[anchor]-q[before],q[after]-q[anchor],e);
        };
        // The boundary is split into pieces at its crossings with the other
        // boundaries. Candidate edges of the other polygons come from a grid
        // of their bounding boxes, so splitting takes about linear time
        // instead of testing every pair of edges. The tests themselves are
        // exact, as before; the grid only skips pairs whose boxes, widened
        // by a relative margin, do not meet.
        const size_t edges=p[anchor].size()==2?1:p[anchor].size();
        auto as_double=[](const S &v)->double{if constexpr(std::is_same_v<S,double>)return v;else return v.template convert_to<double>();};
        struct Box{double x0,y0,x1,y1;};
        auto box=[&](const P &u,const P &w)->Box{
            const double ux=as_double(u.x),uy=as_double(u.y),wx=as_double(w.x),wy=as_double(w.y);
            Box b{std::min(ux,wx),std::min(uy,wy),std::max(ux,wx),std::max(uy,wy)};
            const double pad=1e-9*(1+std::max({std::abs(b.x0),std::abs(b.y0),std::abs(b.x1),std::abs(b.y1)}));
            b.x0-=pad;b.y0-=pad;b.x1+=pad;b.y1+=pad;return b;
        };
        std::vector<std::pair<size_t,size_t>> others;std::vector<Box> other_boxes;
        double lo_x=INFINITY,lo_y=INFINITY,hi_x=-INFINITY,hi_y=-INFINITY;
        for(size_t i=0;i<k;++i)if(i!=anchor)for(size_t h=0;h<p[i].size();++h) {
            others.push_back({i,h});other_boxes.push_back(box(p[i][h],p[i][(h+1)%p[i].size()]));
            const auto &b=other_boxes.back();lo_x=std::min(lo_x,b.x0);lo_y=std::min(lo_y,b.y0);hi_x=std::max(hi_x,b.x1);hi_y=std::max(hi_y,b.y1);
        }
        const size_t cells_per_side=std::max<size_t>(1,std::min<size_t>(4096,size_t(std::sqrt(double(others.size()+1)))));
        const double cell_w=std::max((hi_x-lo_x)/double(cells_per_side),1e-300),cell_h=std::max((hi_y-lo_y)/double(cells_per_side),1e-300);
        auto cell_range=[&](const Box &b,size_t &x0,size_t &x1,size_t &y0,size_t &y1) {
            auto clamp=[&](double v,double lo,double w){const double c=std::floor((v-lo)/w);return size_t(std::clamp(c,0.0,double(cells_per_side-1)));};
            x0=clamp(b.x0,lo_x,cell_w);x1=clamp(b.x1,lo_x,cell_w);y0=clamp(b.y0,lo_y,cell_h);y1=clamp(b.y1,lo_y,cell_h);
        };
        std::vector<std::vector<size_t>> grid(cells_per_side*cells_per_side);
        for(size_t o=0;o<others.size();++o) {
            size_t x0,x1,y0,y1;cell_range(other_boxes[o],x0,x1,y0,y1);
            for(size_t x=x0;x<=x1;++x)for(size_t y=y0;y<=y1;++y)grid[x*cells_per_side+y].push_back(o);
        }
        std::vector<size_t> stamp(others.size(),size_t(-1));
        auto split_edge=[&](size_t j) {
            const P a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
            std::vector<S> cuts{0,1};
            const Box mine=box(a,a+e);
            if(!(mine.x1<lo_x||mine.x0>hi_x||mine.y1<lo_y||mine.y0>hi_y)) {
                size_t x0,x1,y0,y1;cell_range(mine,x0,x1,y0,y1);
                for(size_t x=x0;x<=x1;++x)for(size_t y=y0;y<=y1;++y)for(const size_t o:grid[x*cells_per_side+y]) {
                    if(stamp[o]==j)continue;
                    stamp[o]=j;
                    const auto &ob=other_boxes[o];
                    if(ob.x1<mine.x0||ob.x0>mine.x1||ob.y1<mine.y0||ob.y0>mine.y1)continue;
                    const auto [i,h]=others[o];
                    const P b=p[i][h],f=p[i][(h+1)%p[i].size()]-b;const S determinant=e.cross(f);
                    if(determinant!=0) {
                        const S t=(b-a).cross(f)/determinant,u=(b-a).cross(e)/determinant;
                        if(t>0&&t<1&&u>=0&&u<=1)cuts.push_back(t);
                    } else if(e.cross(b-a)==0) {
                        const S t=(b-a).dot(e)/e.dot(e);if(t>0&&t<1)cuts.push_back(t);
                    }
                }
            }
            std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
            return cuts;
        };
        auto bisect_piece=[&](size_t j,S left,S right,const Polygon &end_left,const Polygon &end_right)->bool {
            const P a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
                if(restricted(anchor,end_left,a+e*left,a+e*right)||
                   restricted(anchor,end_right,a+e*left,a+e*right))return false;
                try {
                    if(derivative(end_left,e)>=0||derivative(end_right,e)<=0)return false;
                    std::optional<S> next;
                    auto inside=[&]{return next&&*next>left&&*next<right;};
                    // As in search_cycle: a fixed point at an end ends only this piece.
                    if(step(anchor,j,end_left,next)){return solved();}
                    if(!inside()&&step(anchor,j,end_right,next)){return solved();}
                    if constexpr(!std::is_same_v<Endpoint,std::nullptr_t>)
                        if(endpoint(anchor,end_left,end_right,a+e*left,a+e*right)){return solved();}
                    // A complete endpoint restriction certificate handles the
                    // nondifferentiable endpoints. Otherwise the minimizer is
                    // interior and the endpoint subgradients have strict signs.
                    std::size_t closed_form_run=0;
                    // The route at the right end of the bracket (see end_steps in search_cycle).
                    Polygon right_route=end_right;bool right_moved=true;
                    for(;;) {
                        detail::cycle_checkpoint();
                        auto probe=[&](const S &t) {
                            if(probes)++*probes;
                            const auto q=evaluate(anchor,a+e*t);
                            if(solved()||refine(q))return true;
                            const int d=derivative(q,e);
                            if(d<0)left=t;else if(d>0){right=t;right_route=q;right_moved=true;}
                            return d==0||step(anchor,j,q,next);
                        };
                        auto end_steps=[&] {
                            if constexpr(std::is_same_v<S,double>)return false;
                            else {
                                if(!right_moved)return false;
                                right_moved=false;
                                std::vector<S> proposals;std::optional<S> unused;
                                if(step(anchor,j,right_route,unused,&proposals))return true;
                                for(const S &t:proposals)if(t>left&&t<right&&probe(t))return true;
                                return false;
                            }
                        };
                        if(inside()&&closed_form_run<kClosedFormRun) {
                            ++closed_form_run;const S t=*next;next.reset();if(probe(t))break;continue;
                        }
                        closed_form_run=0;
                        next.reset();
                        if(end_steps())break;
                        const S midpoint=(left+right)/2;
                        if(midpoint==left||midpoint==right)break; // Binary64 representability only.
                        if(probe(midpoint))break;
                        const S candidate=simplest_rational(left,right);
                        if(candidate>left&&candidate<right&&probe(candidate))break;
                    }
                    if(solved())return true;
                } catch(const std::runtime_error &error){diagnostic=error.what();}
            return false;
        };
        std::vector<std::vector<S>> all_cuts(edges);
        for(size_t j=0;j<edges;++j)all_cuts[j]=split_edge(j);
        auto end_route=[&](const P &point,Polygon &route)->bool {
            try {
                if(point==p[anchor].front()&&!first_contacts[anchor].empty())route=first_contacts[anchor];
                else {route=evaluate(anchor,point);if(solved()||refine(route))return true;}
            } catch(const std::runtime_error &error){diagnostic=error.what();route.clear();}
            return false;
        };
        if(!ends_first) {
            // Edge by edge: the ends of each piece, then its search.
            for(size_t j=0;j<edges;++j) {
                const P a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
                const auto &cuts=all_cuts[j];std::vector<Polygon> ends(cuts.size());
                for(size_t r=0;r<cuts.size();++r)if(end_route(a+e*cuts[r],ends[r]))return;
                for(size_t r=1;r<cuts.size();++r)
                    if(!ends[r-1].empty()&&!ends[r].empty()&&bisect_piece(j,cuts[r-1],cuts[r],ends[r-1],ends[r]))return;
            }
            continue;
        }
        // The piece ends in boundary order; for a polygon the list is cyclic
        // and the end t=1 of an edge is the start of the next one.
        struct End{size_t edge;S t;P point;};
        std::vector<End> points;
        for(size_t j=0;j<edges;++j) {
            const P a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
            const size_t last=edges==1?all_cuts[j].size():all_cuts[j].size()-1;
            for(size_t r=0;r<last;++r)points.push_back({j,all_cuts[j][r],a+e*all_cuts[j][r]});
        }
        const bool cyclic=edges>1;const size_t n_points=points.size(),n_pieces=cyclic?n_points:n_points-1;
        std::vector<Polygon> routes(n_points);std::vector<char> tried(n_points,0),alive(n_pieces,1);
        auto piece_after=[&](size_t v)->std::optional<size_t>{if(cyclic||v+1<n_points)return v;return std::nullopt;};
        auto piece_before=[&](size_t v)->std::optional<size_t>{if(cyclic)return (v+n_pieces-1)%n_pieces;if(v>0)return v-1;return std::nullopt;};
        auto needed=[&](size_t v){
            if(tried[v])return false;
            const auto b=piece_before(v),a=piece_after(v);
            return (b&&alive[*b])||(a&&alive[*a]);
        };
        // Subgradient cuts (paper 1, Lemma 4.3): with u_in and u_out the
        // last nonzero segment before the anchor x and the first one after it,
        // every minimizer y has (u_in - u_out).(y - x) <= 0 when every other
        // polygon visited at x contains x in its interior (in particular when
        // none is), and a piece whose two ends lie strictly beyond is
        // discarded. The next end is the middle of the longest run of ends
        // still needed, so each cut removes at least half of it.
        auto strictly_inside=[&](const P &x,const Polygon &r) {
            const size_t n=r.size();
            if(n<3)return false; // a point or a segment has no interior
            for(size_t e=0;e<n;++e) {
                if constexpr(std::is_same_v<S,double>) {
                    if(exact_orientation(r[e].external(),r[(e+1)%n].external(),x.external())<=0)return false;
                } else if((r[(e+1)%n]-r[e]).cross(x-r[e])<=0)return false;
            }
            return true;
        };
        for(;;) {
            detail::cycle_checkpoint();
            size_t best_start=0,best_length=0;
            {
                size_t run=0,start=0;
                const size_t span=cyclic?2*n_points:n_points;
                for(size_t step=0;step<span;++step) {
                    const size_t v=step%n_points;
                    if(needed(v)){if(run==0)start=v;++run;if(run>best_length&&run<=n_points){best_length=run;best_start=start;}}
                    else run=0;
                }
            }
            if(best_length==0)break;
            const size_t v=(best_start+best_length/2)%n_points;
            tried[v]=1;
            if(end_route(points[v].point,routes[v]))return;
            const Polygon &q=routes[v];
            if(q.empty())continue;
            const P x=points[v].point;
            size_t before=(anchor+k-1)%k,after=(anchor+1)%k;
            while(after!=anchor&&q[after]==x)after=(after+1)%k;
            while(before!=anchor&&q[before]==x)before=(before+k-1)%k;
            if(after==anchor||before==anchor)continue;
            bool interior=true;
            for(size_t i=0;i<k&&interior;++i)if(i!=anchor&&q[i]==x)interior=strictly_inside(x,p[i]);
            if(!interior)continue;
            try {
                std::vector<int> side(n_points);
                for(size_t w=0;w<n_points;++w)side[w]=w==v||points[w].point==x?0:direction_support_sign(x-q[before],q[after]-x,points[w].point-x);
                for(size_t c=0;c<n_pieces;++c)if(alive[c]&&side[c]>0&&side[(c+1)%n_points]>0)alive[c]=0;
            } catch(const std::runtime_error &){}
        }
        for(size_t c=0;c<n_pieces;++c) {
            if(!alive[c])continue;
            const size_t w=(c+1)%n_points;
            if(routes[c].empty()||routes[w].empty())continue;
            const size_t j=points[c].edge;
            const S right=points[w].edge==j&&(!cyclic||w!=0||edges==1)&&points[w].t>points[c].t?points[w].t:S(1);
            if(bisect_piece(j,points[c].t,right,routes[c],routes[w]))return;
        }
    }
}
ConvexCycleResult solve_intersecting_cycle(ConvexCycleResult &result,const ConvexRationalPolygons &p,bool refine=true,
        const ConvexRationalPolygon &initial={},double cutoff=std::numeric_limits<double>::infinity(),bool bound_first=false,
        ConvexCycleAnchorChoice choice=ConvexCycleAnchorChoice::FewestVertices,
        bool straight=true,bool vertex_rebuild=true,bool ends_first=true) {
    const size_t k=p.size();
    const ConvexCycleCertificateGeometry prepared(p);
    auto solved=[&] {return result.status==ConvexCycleStatus::Optimal||result.status==ConvexCycleStatus::CertifiedBound;};
    auto consider=[&](const ConvexRationalPolygon &q) {
        detail::cycle_checkpoint();
        ++result.oracle_calls;++result.certificate_checks;
        const auto c=tpp_convex_verify_cycle_certificate(prepared,q,bound_first?cutoff:INFINITY);
        result.certificate_cutoff_skips+=c.optimality_check_skipped;
        if(c.status==ConvexCycleCertificateStatus::Optimal||
           (c.status==ConvexCycleCertificateStatus::Feasible&&c.lower_bound>=cutoff)) {
            result.contacts=q;result.certificate=c;
            result.status=c.status==ConvexCycleCertificateStatus::Optimal?ConvexCycleStatus::Optimal:ConvexCycleStatus::CertifiedBound;
            result.squared_link_lengths.clear();
            for(size_t i=0;i<k;++i){const auto d=q[(i+1)%k]-q[i];result.squared_link_lengths.push_back(d.dot(d));}
            return true;
        }
        if(c.status==ConvexCycleCertificateStatus::Feasible &&
           (result.contacts.empty()||c.upper_bound<result.certificate.upper_bound)) {
            result.contacts=q;result.certificate=c;
        }
        return false;
    };
    if(bound_first&&detail::cycle_cutoff_promising(p,initial,cutoff)&&consider(initial))return result;
    const auto intersection=common_region(p);
    if(!intersection.empty()){consider(ConvexRationalPolygon(k,intersection.front()));return result;}
    try {if(refine&&detail::CycleRefinement<R>::run(p,consider,initial))return result;}
    catch(const std::runtime_error &error){result.diagnostic=error.what();}

    // Preserve the cold finite proposal when an inherited configuration fails;
    // a poor hint must not force an otherwise easy solve into boundary maps.
    if(refine&&!initial.empty())try {
        if(detail::CycleRefinement<R>::run(p,consider))return result;
    } catch(const std::runtime_error &error){result.diagnostic=error.what();}

    ConvexRationalPolygons checked;
    if(refine&&detail::prepare_cycle_polygons(p,checked))return tpp_convex_solve_cycle_disjoint(p);

    // A containing polygon need not have an optimal boundary contact. The
    // general search visits every boundary; prefix/suffix visits at x are free.
    auto evaluate=[&](size_t anchor,const Point &x) {
        detail::cycle_checkpoint();
        ++result.anchored_solves;
        size_t first=1,last=k-1;
        while(first<=last&&rational_inside(x,p[(anchor+first)%k]))++first;
        while(last>=first&&rational_inside(x,p[(anchor+last)%k]))--last;
        ConvexRationalPolygons remaining;
        for(size_t r=first;r<=last;++r)remaining.push_back(p[(anchor+r)%k]);
        ConvexRationalPolygon q(k,x);
        if(!remaining.empty()) {
            const auto path=detail::solve_intersecting_map_contacts_exact(x,x,remaining);
            if(path.size()!=remaining.size())throw std::runtime_error("Incomplete intersecting cycle");
            for(size_t r=first;r<=last;++r)q[(anchor+r)%k]=path[r-first];
        }
        consider(q);
        if(solved())return q;
        // At a vertex of the anchor, as Step 2's vertex rebuild: the block
        // descriptor with the vertex pinned (a block of coincident visit
        // points pinned where two edges cross is in no polygon's descriptor).
        if(!refine&&vertex_rebuild) {
            const auto &region=p[anchor];
            for(size_t j=0;j<region.size();++j)if(region[j]==x) {
                auto feature=contact_features<R>(p,q);feature[anchor]=int(2*j);
                auto moved=q;auto blocks=feature;
                if(block_descriptor(p,moved,blocks)) {
                    bool built=false;
                    try {built=detail::CycleRefinement<R>::close(p,blocks,moved);} catch(const std::runtime_error &) {}
                    if(built&&consider(moved))return q;
                }
                break;
            }
        }
        auto fixed=p;fixed[anchor]={x};
        if(tpp_convex_verify_cycle_certificate(fixed,q).status!=ConvexCycleCertificateStatus::Optimal)
            throw std::runtime_error("Anchored path failed its exact support certificate");
        return q;
    };
    auto restricted=[&](size_t anchor,const ConvexRationalPolygon &q,const Point &a,const Point &b) {
        auto regions=p;regions[anchor]={a,b};
        return tpp_convex_verify_cycle_certificate(regions,q).status==ConvexCycleCertificateStatus::Optimal;
    };
    // proposals (the right end of a bracket, Theorem 4.1): every closed form
    // tried here, of the per-polygon and of the block descriptor.
    auto step=[&](size_t anchor,size_t j,const ConvexRationalPolygon &q,std::optional<R> &next,std::vector<R> *proposals=nullptr) {
        if(q.empty())return false;
        auto propose=[&](const Point &x){if(proposals)proposals->push_back((x-p[anchor][j]).dot(p[anchor][(j+1)%p[anchor].size()]-p[anchor][j])/
            (p[anchor][(j+1)%p[anchor].size()]-p[anchor][j]).dot(p[anchor][(j+1)%p[anchor].size()]-p[anchor][j]));};
        ++result.closed_form_steps;
        auto feature=contact_features<R>(p,q);feature[anchor]=int(2*j+1);
        const auto a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
        if(!refine) {
            // Without sweeps (Step 3 of Algorithm 4.1), a block of coincident
            // visit points also goes to the certificate with its block
            // descriptor, which can pin it where two edges cross. Only the
            // per-polygon descriptor proposes the next probe: its proposal
            // depends on the descriptor alone, which termination needs (a
            // block's pin depends on the current contacts).
            auto closed=q;bool built=false;
            try {built=detail::CycleRefinement<R>::close(p,feature,closed);} catch(const std::runtime_error &) {}
            if(built){next=(closed[anchor]-a).dot(e)/e.dot(e);propose(closed[anchor]);}
            if(built&&consider(closed))return true;
            // The anchor straight through the edge: it proposes the next
            // probe when the reflection does not close (descriptor alone).
            if(straight) {
                auto crossing=q;R u;
                if(close_straight<R>(p,feature,crossing,anchor,j,false,&u)) {
                    if(!built)next=u;
                    if(proposals)proposals->push_back(u);
                    if(consider(crossing))return true;
                }
            }
            auto moved=q;auto blocks=feature;bool built_blocks=false;
            const bool has_blocks=block_descriptor(p,moved,blocks);
            if(has_blocks)
                try {built_blocks=detail::CycleRefinement<R>::close(p,blocks,moved);} catch(const std::runtime_error &) {}
            if(built_blocks)propose(moved[anchor]);
            if(built_blocks&&consider(moved))return true;
            if(straight&&has_blocks) {
                auto crossing=q;auto f=feature;
                block_descriptor(p,crossing,f);
                if(close_straight<R>(p,f,crossing,anchor,j,false)) {
                    propose(crossing[anchor]);
                    if(consider(crossing))return true;
                }
            }
            return false;
        }
        auto closed=q;
        try {if(!detail::CycleRefinement<R>::close(p,feature,closed))return false;}
        catch(const std::runtime_error &){return false;}
        next=(closed[anchor]-a).dot(e)/e.dot(e);propose(closed[anchor]);
        return consider(closed);
    };
    search_intersecting_boundaries<R>(p,evaluate,restricted,
        [&](const ConvexRationalPolygon &q){return refine&&detail::CycleRefinement<R>::run(p,consider,q);},
        solved,step,result.anchor_polygon,result.diagnostic,&result.bisection_probes,nullptr,
        detail::eligible_anchor(p,choice),ends_first,refine);
    if(solved())return result;
    result.status=ConvexCycleStatus::OracleFailure;
    if(result.diagnostic.empty())result.diagnostic="No globally certified cycle constructed";
    return result;
}
} // namespace

ConvexCycleResult tpp_convex_solve_cycle(const ConvexRationalPolygons &input,const ConvexCycleOptions &options) {
    ConvexCycleResult result;
    return detail::run_cycle_execution(result,options,[&]() -> ConvexCycleResult {
    if(input.size()<2||std::isnan(options.lower_bound_cutoff))return {};
    if(!options.initial_contacts.empty()&&options.initial_contacts.size()!=input.size())return {};
    ConvexRationalPolygons normalized;
    try {detail::prepare_cycle_polygons(input,normalized,false);}
    catch(const std::invalid_argument &error) {
        result.diagnostic=error.what();return result;
    }
    try {return solve_intersecting_cycle(result,normalized,options.refine_contacts,options.initial_contacts,options.lower_bound_cutoff,options.bound_first,
        options.anchor_choice,options.straight_anchor,options.straight_anchor,options.anchor_ends_first);}
    catch(const std::exception &error) {
        result.status=ConvexCycleStatus::OracleFailure;
        result.diagnostic=error.what();return result;
    }

    });
}

ConvexCycleResult tpp_convex_solve_cycle(const std::vector<std::vector<Vector2>> &input,const ConvexCycleOptions &options) {
    ConvexRationalPolygons exact;
    for(const auto &p:input) {
        ConvexRationalPolygon q;
        for(auto v:p){if(!v.is_finite())return {};q.emplace_back(v);}
        exact.push_back(std::move(q));
    }
    return tpp_convex_solve_cycle(exact,options);
}
namespace {
// The exact path: binary64 construction whose candidates pass the exact
// certificate, with counted rational recoveries and the boundary search.
ConvexCycleDoubleResult solve_cycle_double_exact(ConvexCycleDoubleResult &result,
        const std::vector<std::vector<Vector2>> &input,const ConvexCycleDoubleOptions &options) {
    ConvexRationalPolygons normalized;
    std::vector<std::vector<Vector2>> binary_regions;
    std::optional<ConvexCycleCertificateGeometry> prepared;
    try {
        if(options.workspace) {
            prepared.emplace(options.workspace->prepare(input,options.interval_certificate));
            normalized=prepared->polygons();
        } else {
            for(const auto &poly:input)for(auto v:poly)if(!v.is_finite())return result;
            detail::prepare_cycle_polygons(input,normalized,false,&binary_regions);
            if(options.interval_certificate)prepared.emplace(input,true);
        }
    } catch(const std::invalid_argument &error){result.diagnostic=error.what();return result;}
    // refine_contacts=false with rational recovery: Algorithm 4.1 of paper 1,
    // with one prepared geometry for its exact certificates.
    const bool method=!options.refine_contacts&&options.recover_arithmetic_failures;
    if(method&&!options.proposal_only)try {
        // Pairwise disjoint polygons: one boundary, by the disjoint search
        // (pairwise disjoint polygons have no common point).
        // normalized is already normalized and checked.
        if(options.workspace?detail::cycle_polygons_disjoint(normalized):detail::cycle_polygons_disjoint(binary_regions))return solve_cycle_disjoint_double(result,input,normalized,options);
        if(!prepared)prepared.emplace(input,options.interval_certificate);
    } catch(const std::invalid_argument &error){result.diagnostic=error.what();return result;}
    auto verify=[&](const auto &q) {
        return prepared?tpp_convex_verify_cycle_certificate(*prepared,q):tpp_convex_verify_cycle_certificate(normalized,q);
    };
    std::vector<int> candidate_features;
    using Kernel=detail::CycleRefinement<double>;
    Kernel::Polygons p;for(const auto &poly:normalized){Kernel::Polygon q;for(const auto &v:poly)q.emplace_back(v.external());p.push_back(q);}
    Kernel::Polygon initial;for(const auto &v:options.initial_contacts)initial.emplace_back(v);
    auto consider=[&](std::vector<Vector2> q) {
        detail::cycle_checkpoint();
        ++result.oracle_calls;++result.certificate_checks;
        const double cut=(options.bound_first||options.interval_certificate)?options.lower_bound_cutoff:INFINITY;
        auto c=prepared?tpp_convex_verify_cycle_certificate(*prepared,q,cut,options.interval_certificate):tpp_convex_verify_cycle_certificate(input,q,cut,options.interval_certificate);
        if(c.status==ConvexCycleCertificateStatus::InvalidCandidate) {
            round_contacts_inward(normalized,q);c=prepared?tpp_convex_verify_cycle_certificate(*prepared,q,cut,options.interval_certificate):tpp_convex_verify_cycle_certificate(input,q,cut,options.interval_certificate);++result.certificate_checks;
        }
        result.certificate_cutoff_skips+=c.optimality_check_skipped;
        result.certificate_interval_uses+=c.interval_bounds_used;
        if(c.status==ConvexCycleCertificateStatus::Optimal||c.status==ConvexCycleCertificateStatus::Feasible) {
            if(c.status==ConvexCycleCertificateStatus::Optimal||c.lower_bound>=options.lower_bound_cutoff||result.contacts.empty()||c.upper_bound<result.certificate.upper_bound||
               (c.upper_bound==result.certificate.upper_bound&&c.lower_bound>result.certificate.lower_bound)) {
                result.contacts=q;result.certificate=c;result.active_features=candidate_features;
            }
            if(c.status==ConvexCycleCertificateStatus::Optimal){result.status=ConvexCycleStatus::Optimal;return true;}
            if(c.lower_bound>=options.lower_bound_cutoff){result.status=ConvexCycleStatus::CertifiedBound;return true;}
        }
        return false;
    };
    auto recover_cycle=[&] {
        detail::cycle_checkpoint();
        detail::CyclePhaseScope recovery(detail::CyclePhase::RationalRecovery);
        ++result.rational_cycle_recoveries;
        ConvexRationalPolygon warm;
        const auto &proposal=result.contacts.empty()?options.initial_contacts:result.contacts;
        for(const auto &q:proposal)warm.emplace_back(q);
        ConvexCycleResult recovered;
        // Step 3 of Algorithm 4.1 runs the rational search without sweeps.
        auto count=[&] {
            if(!method)return;
            result.rational_anchored_solves+=recovered.anchored_solves;
            result.rational_bisection_probes+=recovered.bisection_probes;
            result.rational_closed_form_steps+=recovered.closed_form_steps;
        };
        try {
            recovered=solve_intersecting_cycle(recovered,normalized,options.refine_contacts,warm,options.lower_bound_cutoff,options.bound_first,
                options.anchor_choice,options.straight_anchor,options.straight_anchor,options.anchor_ends_first);
        } catch(const detail::CycleInterrupted &) {
            count();
            if(recovered.certificate.status==ConvexCycleCertificateStatus::Feasible||
               recovered.certificate.status==ConvexCycleCertificateStatus::Optimal)
                result.certificate.lower_bound=std::max(result.certificate.lower_bound,recovered.certificate.lower_bound);
            throw;
        }
        count();
        result.certificate_cutoff_skips+=recovered.certificate_cutoff_skips;
        if(recovered.status!=ConvexCycleStatus::Optimal&&recovered.status!=ConvexCycleStatus::CertifiedBound){result.diagnostic=recovered.diagnostic;return false;}
        if(method&&recovered.status==ConvexCycleStatus::Optimal) {
            result.method_step=3;result.exact_contacts=recovered.contacts;result.exact_certificate=recovered.certificate;
        }
        std::vector<Vector2> q;for(const auto &v:recovered.contacts)q.push_back(v.external());
        consider(q);
        retain_exact_bound(result,recovered.certificate,options.lower_bound_cutoff);
        return !result.contacts.empty();
    };
    try {
        if(options.bound_first&&detail::cycle_cutoff_promising(normalized,options.initial_contacts,options.lower_bound_cutoff)) {
            ++result.initial_contact_checks;
            if(consider(options.initial_contacts)){++result.initial_contact_accepts;return result;}
        }
        auto joint=p.front();
        size_t leading=1; // the regions whose binary64 intersection was computed
        for(;leading<p.size()&&!joint.empty();++leading)joint=Kernel::intersect(std::move(joint),p[leading]);
        if(!joint.empty()&&consider(std::vector<Vector2>(p.size(),joint.front().external()))) {
            if(method){result.method_step=1;record_exact_contacts(result);}
            return result;
        }
        if(options.recover_arithmetic_failures&&!options.proposal_only) {
            // A common point may exist but have no binary64 representation.
            // Resolve that zero-cycle feature before attempting anchor search.
            const auto exact_joint=common_region(normalized,joint.empty()?leading:0);
            if(!exact_joint.empty()) {
                ++result.rational_cycle_recoveries;++result.oracle_calls;++result.certificate_checks;
                const auto q=ConvexRationalPolygon(p.size(),exact_joint.front());
                const auto exact_certificate=prepared?detail::cycle_optimality_certificate(*prepared,q):verify(q);
                if(exact_certificate.status==ConvexCycleCertificateStatus::Optimal) {
                    if(method){result.method_step=1;result.exact_contacts=q;result.exact_certificate=exact_certificate;}
                    if(consider(std::vector<Vector2>(p.size(),q.front().external())))return result;
                    if(!result.contacts.empty()||method){result.status=ConvexCycleStatus::FloatingPointLimit;return result;}
                }
            }
        }
        auto refine_candidate=[&](const Kernel::Polygon &q,const std::vector<int> &features,bool closed) {
            if(options.retain_active_features)candidate_features=features;
            std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());
            if(consider(candidate))return true;
            if(closed&&options.recover_arithmetic_failures&&!options.proposal_only) {
                // Filtered construction: reconstruct just the proposed feature
                // tuple, not a second optimization search. Its certificate
                // distinguishes rounding limits from a wrong active feature.
                detail::CyclePhaseScope recovery(detail::CyclePhase::RationalRecovery);
                ++result.rational_feature_recoveries;
                ConvexRationalPolygon lifted;for(const auto &v:candidate)lifted.emplace_back(v);
                if(detail::CycleRefinement<R>::close(normalized,features,lifted)) {
                    ++result.oracle_calls;++result.certificate_checks;
                    const auto exact_certificate=prepared?detail::cycle_optimality_certificate(*prepared,lifted):verify(lifted);
                    if(exact_certificate.status==ConvexCycleCertificateStatus::Optimal) {
                        candidate.clear();for(const auto &v:lifted)candidate.push_back(v.external());
                        consider(candidate);
                        retain_exact_bound(result,exact_certificate,options.lower_bound_cutoff);
                        if(!result.contacts.empty())return true;
                    }
                }
            }
            return false;
        };
        try {if(options.refine_contacts&&Kernel::run(p,refine_candidate,initial,options.initial_features))return result;}
        catch(const std::exception &error){result.diagnostic=error.what();}
        if(options.proposal_only) {
            result.status=ConvexCycleStatus::ProposalLimit;
            if(result.contacts.empty())result.certificate.upper_bound=INFINITY;
            return result;
        }
        ConvexRationalPolygons checked;
        if(options.refine_contacts&&detail::prepare_cycle_polygons(normalized,checked)) {
            auto disjoint=tpp_convex_solve_cycle_disjoint_double(input,options);
            disjoint.certificate_cutoff_skips+=result.certificate_cutoff_skips;
            disjoint.certificate_interval_uses+=result.certificate_interval_uses;
            disjoint.initial_contact_checks+=result.initial_contact_checks;
            disjoint.initial_contact_accepts+=result.initial_contact_accepts;
            return disjoint;
        }
        if(options.refine_contacts&&options.recover_arithmetic_failures)try {
            if(recover_cycle())return result;
        } catch(const std::exception &error){result.diagnostic=error.what();}
        auto solved=[&]{return result.status==ConvexCycleStatus::Optimal||result.status==ConvexCycleStatus::FloatingPointLimit||
                               result.status==ConvexCycleStatus::CertifiedBound||result.status==ConvexCycleStatus::GapClosed||
                               !result.exact_contacts.empty();};
        const bool gap_search=method&&options.boundary_search_gap&&options.max_gap>0;
        // Algorithm 4.1 with a tolerance: accept a binary64 candidate whose
        // certificate closes the gap, before any exact rebuild.
        auto gap_closes=[&](const Kernel::Polygon &q) {
            if(!gap_search)return false;
            std::vector<Vector2> contacts;for(const auto &v:q)contacts.push_back(v.external());
            ++result.oracle_calls;++result.certificate_checks;
            auto c=prepared?tpp_convex_verify_cycle_certificate(*prepared,contacts,INFINITY,options.interval_certificate):
                            tpp_convex_verify_cycle_certificate(input,contacts,INFINITY,options.interval_certificate);
            if(c.status==ConvexCycleCertificateStatus::InvalidCandidate) {
                round_contacts_inward(normalized,contacts);++result.certificate_checks;
                c=prepared?tpp_convex_verify_cycle_certificate(*prepared,contacts,INFINITY,options.interval_certificate):
                           tpp_convex_verify_cycle_certificate(input,contacts,INFINITY,options.interval_certificate);
            }
            if(!certificate_closes_gap(c,options.max_gap))return false;
            result.contacts=std::move(contacts);result.certificate=c;
            result.status=c.status==ConvexCycleCertificateStatus::Optimal?ConvexCycleStatus::Optimal:ConvexCycleStatus::GapClosed;
            return true;
        };
        DescriptorRebuilds rebuilds;ClosedFormFixedPoint fixed_point;rebuilds.pivots=options.descriptor_pivots;rebuilds.straight=options.straight_anchor;
        auto rebuild=[&](size_t anchor,const std::vector<int> &feature,const Kernel::Polygon &q) {
            ConvexRationalPolygon exact;ConvexCycleCertificateResult certificate;
            const int local=local_descriptor(p,q,anchor,kLocal)[anchor];
            if(!rebuilds(normalized,anchor,feature,local,q,[&](const ConvexRationalPolygon &c) {
                    ++result.oracle_calls;++result.certificate_checks;
                    // Kept only when Optimal: no reporting bounds otherwise.
                    return prepared?detail::cycle_optimality_certificate(*prepared,c):verify(c);
                },result.descriptor_rebuilds,exact,certificate))return false;
            std::vector<Vector2> rounded;for(const auto &v:exact)rounded.push_back(v.external());
            result.exact_contacts=std::move(exact);result.exact_certificate=certificate;
            consider(std::move(rounded));
            retain_exact_bound(result,certificate,options.lower_bound_cutoff);
            return true;
        };
        auto evaluate=[&](size_t anchor,const Kernel::P &x) {
            detail::cycle_checkpoint();
            if(method)++result.anchored_solves;
            const size_t k=p.size();size_t first=1,last=k-1;
            while(first<=last&&Kernel::inside(x,p[(anchor+first)%k]))++first;
            while(last>=first&&Kernel::inside(x,p[(anchor+last)%k]))--last;
            std::vector<std::vector<Vector2>> remaining;
            for(size_t r=first;r<=last;++r) {
                std::vector<Vector2> polygon;for(const auto &v:p[(anchor+r)%k])polygon.push_back(v.external());
                remaining.push_back(std::move(polygon));
            }
            Kernel::Polygon q(k,x);
            if(!remaining.empty()) {
                const auto path=[&] {
                    try {
                        if(!method)return detail::solve_intersecting_map_contacts_unchecked_double(x.external(),x.external(),remaining,false);
                        // Algorithm 4.1: the binary64 construction of the TPP flow.
                        auto chain=detail::double_candidate_chain(x.external(),x.external(),remaining);
                        if(chain.size()!=remaining.size()+2)throw std::runtime_error("Incomplete binary64 anchored chain");
                        return std::vector<Vector2>(chain.begin()+1,chain.end()-1);
                    }
                    catch(const std::exception &) {
                        if(!options.recover_arithmetic_failures)throw;
                        detail::CyclePhaseScope recovery(detail::CyclePhase::RationalRecovery);
                        ++result.rational_anchor_recoveries;
                        ConvexRationalPolygons regions;
                        for(size_t r=first;r<=last;++r)regions.push_back(normalized[(anchor+r)%k]);
                        const auto exact_path=detail::solve_intersecting_map_contacts_exact(Point(x.external()),Point(x.external()),regions);
                        std::vector<Vector2> path;for(const auto &v:exact_path)path.push_back(v.external());return path;
                    }
                }();
                if(path.size()!=remaining.size())throw std::runtime_error("Incomplete double intersecting cycle");
                for(size_t r=first;r<=last;++r)q[(anchor+r)%k]=Kernel::P(path[r-first]);
            }
            if(!method) {
                std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());consider(candidate);
                return q;
            }
            // At a vertex of the anchor polygon the anchored cycle with that
            // vertex pinned is rebuilt exactly, unless the vertex lies in no
            // other polygon (F is then differentiable along both edges near
            // it) and F decreases along one of them.
            if(const auto j=vertex_at(p[anchor],x)) {
                // F need not be differentiable at a vertex on the boundary of
                // another polygon, where membership changes. A vertex in the
                // interior of every other polygon that contains it is tested
                // by the signs along its two edges, with the last nonzero
                // segment before it and the first one after it: the
                // polygons visited at x then contain a neighborhood of x, so
                // the route through the nearest other visit points bounds F
                // from above near x and touches it at x, and F is
                // differentiable there with gradient u_in - u_out (the
                // argument of Lemma 4.2 in the plane).
                bool candidate=false;
                for(size_t i=0;i<k&&!candidate;++i) {
                    if(i==anchor||!Kernel::inside(x,p[i]))continue;
                    const auto &r=p[i];const size_t n=r.size();
                    if(n<3){candidate=true;break;} // a point or a segment has no interior
                    for(size_t e=0;e<n&&!candidate;++e)candidate=exact_orientation(r[e].external(),r[(e+1)%n].external(),x.external())<=0;
                }
                if(!candidate) {
                    const auto &a=p[anchor];const size_t m=a.size();
                    size_t after=1,before=1;
                    while(after<k&&q[(anchor+after)%k]==x)++after;
                    while(before<k&&q[(anchor+k-before)%k]==x)++before;
                    int forward=0,backward=0;
                    try {
                        if(after==k||before==k)throw std::runtime_error("No nonzero segment at the anchor");
                        const auto in=x-q[(anchor+k-before)%k],out=q[(anchor+after)%k]-x;
                        forward=direction_support_sign(in,out,a[(*j+1)%m]-a[*j]);
                        backward=direction_support_sign(in,out,a[*j]-a[(*j+m-1)%m]);
                    } catch(const std::runtime_error &) {forward=backward=0;}
                    candidate=forward>=0&&backward<=0;
                }
                if(candidate&&!gap_closes(q)&&!rebuild(anchor,local_descriptor(p,q,anchor,int(2**j)),q)
                   &&!remaining.empty()&&options.recover_arithmetic_failures) {
                    // The binary64 route only proposes. Contacts that meet at
                    // a crossing of two boundaries come out an ulp apart, so
                    // its descriptor misses that block; at a vertex the exact
                    // anchored TPP decides, and its route is rebuilt.
                    detail::CyclePhaseScope recovery(detail::CyclePhase::RationalRecovery);
                    ++result.rational_anchor_recoveries;
                    ConvexRationalPolygons regions;
                    for(size_t r=first;r<=last;++r)regions.push_back(normalized[(anchor+r)%k]);
                    const auto exact_path=detail::solve_intersecting_map_contacts_exact(Point(x.external()),Point(x.external()),regions);
                    auto exact=q;
                    for(size_t r=first;r<=last;++r)exact[(anchor+r)%k]=Kernel::P(exact_path[r-first].external());
                    rebuild(anchor,local_descriptor(p,exact,anchor,int(2**j)),exact);
                }
            }
            return q;
        };
        auto restricted=[&](size_t anchor,const Kernel::Polygon &q,const Kernel::P &a,const Kernel::P &b) {
            // Binary64 contacts rarely pass an exact test; Algorithm 4.1
            // decides a piece by the derivative signs at its ends.
            if(method)return false;
            auto regions=input;regions[anchor]={a.external(),b.external()};
            std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());
            return tpp_convex_verify_cycle_certificate(regions,candidate).status==ConvexCycleCertificateStatus::Optimal;
        };
        // The exact anchored route at the anchor point of q (the binary64 one
        // only proposes): contacts that meet at a crossing of two boundaries
        // come out an ulp apart in binary64, and no descriptor closes them.
        auto exact_route=[&](size_t anchor,const Kernel::Polygon &q) {
            const size_t k=p.size();const auto x=q[anchor];size_t first=1,last=k-1;
            while(first<=last&&Kernel::inside(x,p[(anchor+first)%k]))++first;
            while(last>=first&&Kernel::inside(x,p[(anchor+last)%k]))--last;
            auto exact=q;
            if(first>last)return exact;
            detail::CyclePhaseScope recovery(detail::CyclePhase::RationalRecovery);
            ++result.rational_anchor_recoveries;
            ConvexRationalPolygons regions;
            for(size_t r=first;r<=last;++r)regions.push_back(normalized[(anchor+r)%k]);
            const auto path=detail::solve_intersecting_map_contacts_exact(Point(x.external()),Point(x.external()),regions);
            for(size_t r=first;r<=last;++r)exact[(anchor+r)%k]=Kernel::P(path[r-first].external());
            return exact;
        };
        std::function<bool(size_t,size_t,const Kernel::Polygon &,std::optional<double> &)> step_self;
        bool exact_step=false;
        auto step=[&](size_t anchor,size_t j,const Kernel::Polygon &q,std::optional<double> &next) {
            if(q.empty())return false;
            if(method) {
                // The descriptor's closed form proposes the next probe, and
                // its exact rebuild goes to the certificate.
                ++result.closed_form_steps;
                const auto feature=local_descriptor(p,q,anchor,int(2*j+1));
                const auto a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
                // Only the per-polygon descriptor proposes the next probe (see
                // the rational step); the rebuild also tries the block one.
                auto closed=q;bool built=false,pivoted=false;snap_vertices(p,feature,closed);
                try {built=close_descriptor<double>(p,feature,closed,options.descriptor_pivots,&pivoted);} catch(const std::runtime_error &) {}
                if(built)next=(closed[anchor]-a).dot(e)/e.dot(e);
                if(built&&gap_closes(closed))return true;
                // Without a reflection, the anchor straight through the edge
                // proposes the next probe.
                auto straight=q;double u=0;bool crossed=false;
                if(!built&&options.straight_anchor)crossed=close_straight<double>(p,feature,straight,anchor,j,options.descriptor_pivots,&u);
                if(crossed) {
                    next=u;
                    if(gap_closes(straight))return true;
                }
                if(rebuild(anchor,feature,q))return true;
                if(crossed){auto f=feature;f[anchor]=-1;return fixed_point(anchor,j,f,*next);}
                if(built&&!pivoted)return fixed_point(anchor,j,feature,*next);
                // Nothing closed from the binary64 route: once, from the exact one.
                if(!exact_step&&options.recover_arithmetic_failures) {
                    exact_step=true;
                    const bool closed=step_self(anchor,j,exact_route(anchor,q),next);
                    exact_step=false;
                    return closed;
                }
                return false;
            }
            auto feature=contact_features<double>(p,q);feature[anchor]=int(2*j+1);
            auto closed=q;
            try {
                if(!Kernel::close(p,feature,closed))return false;
            } catch(const std::runtime_error &){return false;}
            const auto a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
            next=(closed[anchor]-a).dot(e)/e.dot(e);
            std::vector<Vector2> candidate;for(const auto &v:closed)candidate.push_back(v.external());
            return consider(candidate);
        };
        step_self=step;
        // Step 2's restricted endpoint certificate (endpoint_certificate).
        // The anchored cycle at an end is rebuilt exactly from its
        // descriptor (with pivots), with the anchor and every contact at the
        // anchor's point pinned there (kFixed): a cycle through the regions with the
        // piece [a,b] (binary64 points) as the anchor's region. If it passes
        // the exact certificate of that problem, no anchor on the piece gives
        // a shorter cycle, and the piece is skipped instead of bisected; the
        // global certificate then decides whether it is also the optimum.
        // Only the search order changes: Step 3 is complete.
        auto endpoint=[&](size_t anchor,const Kernel::Polygon &left_end,const Kernel::Polygon &right_end,
                          const Kernel::P &a,const Kernel::P &b) {
            if(!method||!options.endpoint_certificate)return false;
            constexpr int fixed=detail::CycleRefinement<R>::kFixed;
            ConvexRationalPolygons regions=normalized;
            regions[anchor]={Point(a.external()),Point(b.external())};
            for(const Kernel::Polygon *end:{&left_end,&right_end}) {
                const auto &q=*end;
                auto feature=local_descriptor(p,q,anchor,fixed);
                for(size_t i=0;i<q.size();++i)if(q[i]==q[anchor])feature[i]=fixed;
                ConvexRationalPolygon lifted;lifted.reserve(q.size());
                for(const auto &v:q)lifted.emplace_back(v.external());
                snap_vertices(normalized,feature,lifted);
                ++result.endpoint_certificates;
                try {if(!close_descriptor<R>(normalized,feature,lifted,true))continue;}
                catch(const std::runtime_error &) {continue;}
                ++result.certificate_checks;
                if(tpp_convex_verify_cycle_certificate(regions,lifted).status!=ConvexCycleCertificateStatus::Optimal)continue;
                ++result.endpoint_skips;
                ++result.oracle_calls;++result.certificate_checks;
                const auto certificate=prepared?detail::cycle_optimality_certificate(*prepared,lifted):verify(lifted);
                if(certificate.status==ConvexCycleCertificateStatus::Optimal) {
                    std::vector<Vector2> rounded;for(const auto &v:lifted)rounded.push_back(v.external());
                    result.exact_contacts=std::move(lifted);result.exact_certificate=certificate;
                    consider(std::move(rounded));
                    retain_exact_bound(result,certificate,options.lower_bound_cutoff);
                }
                return true;
            }
            return false;
        };
        const auto eligible=detail::eligible_anchor(normalized,options.anchor_choice);
        search_intersecting_boundaries<double>(p,evaluate,restricted,
            [&](const Kernel::Polygon &q){return options.refine_contacts&&Kernel::run(p,refine_candidate,q);},
            solved,step,result.anchor_polygon,result.diagnostic,method?&result.bisection_probes:nullptr,endpoint,
            eligible,options.anchor_ends_first,options.refine_contacts);
        if(solved()) {
            if(method){result.method_step=2;record_exact_contacts(result);}
            return result;
        }
    } catch(const std::exception &error){result.diagnostic=error.what();}
    if(options.proposal_only) {
        result.status=ConvexCycleStatus::ProposalLimit;
        if(result.contacts.empty())result.certificate.upper_bound=INFINITY;
        return result;
    }
    try {
        if(options.recover_arithmetic_failures) {
            if(!recover_cycle())result.status=ConvexCycleStatus::OracleFailure;
            return result;
        }
    } catch(const std::exception &error){result.status=ConvexCycleStatus::OracleFailure;result.diagnostic=error.what();return result;}
    result.status=result.contacts.empty()?ConvexCycleStatus::OracleFailure:ConvexCycleStatus::FloatingPointLimit;
    return result;
}
} // namespace

namespace {
// Algorithm 4.1 with a tolerance, in binary64 first (boundary_search_gap). The
// regions are normalized in binary64 with proved signs, as the binary64 stage
// does; the search is Step 2's (the binary64 TPP construction for each
// anchored cycle, closed-form steps, bisection; one boundary when the regions
// are proved pairwise apart, every boundary otherwise); and every candidate's
// bounds are proved by directed rounding (detail::CycleBounds). The candidates
// are those the exact path would certify (vertex candidates, and closed-form
// steps with the descriptors of DescriptorRebuilds) and the bisection probes
// that shorten the best anchored cycle; the duals of coincident blocks come
// from reachability (cycle_dual_by_reachability). Nothing is rational. True
// with GapClosed when a candidate's proved interval closes the gap; false
// leaves the call to the exact path. The cutoff is not used here.
bool float_first_boundary_search(ConvexCycleDoubleResult &result,const std::vector<std::vector<Vector2>> &input,
        const ConvexCycleDoubleOptions &options) {
    using Kernel=detail::CycleRefinement<double>;
    if(!detail::cycle_interval_environment()||!(options.max_gap>0))return false;
    const size_t k=input.size();
    std::vector<detail::FloatRegion> owned;owned.reserve(k);
    for(const auto &polygon:input) {
        auto region=detail::binary_region(polygon);
        // Points and segments, or undecided signs: the exact path.
        if(!region||region->size()<3)return false;
        owned.push_back(std::move(*region));
    }
    std::vector<const detail::FloatRegion*> regions;std::vector<Vector2> means;Vector2 reference{};
    for(const auto &region:owned){regions.push_back(&region);means.push_back(detail::vertex_mean(region));reference+=means.back()/double(k);}
    if(!reference.is_finite())return false;
    detail::CycleBounds bounds{regions,means,reference,options.max_gap,INFINITY};
    bounds.block_reachability=true;
    std::vector<DPolygon> p;
    for(const auto &region:owned){DPolygon q;for(const auto &v:region)q.emplace_back(v);p.push_back(std::move(q));}
    bool closed=false;
    auto prove=[&](const DPolygon &q) {
        std::vector<Vector2> contacts;contacts.reserve(k);for(const auto &v:q)contacts.push_back(v.external());
        ++result.certificate_checks;
        return closed=bounds.prove_proposal(contacts);
    };
    auto finish=[&](int step) {
        result.contacts=bounds.contacts;
        result.certificate.status=ConvexCycleCertificateStatus::Feasible;
        result.certificate.lower_bound=bounds.lower;result.certificate.upper_bound=bounds.upper;
        result.certificate.optimality_check_skipped=result.certificate.interval_bounds_used=true;
        result.status=ConvexCycleStatus::GapClosed;result.method_step=step;
        return true;
    };
    auto solved=[&]{return closed;};
    try {
        // Step 1: a common point gives the zero cycle.
        auto joint=p.front();
        for(size_t i=1;i<k&&!joint.empty();++i)joint=Kernel::intersect(std::move(joint),p[i]);
        if(!joint.empty()&&prove(DPolygon(k,joint.front())))return finish(1);
    } catch(const std::exception &) {}
    bool apart=true;
    for(size_t i=0;i<k&&apart;++i)for(size_t j=i+1;j<k&&apart;++j)
        apart=Kernel::separated_toward(p[i],p[j],means[j]-means[i])||Kernel::separated_toward(p[j],p[i],means[i]-means[j]);
    ClosedFormFixedPoint fixed_point;
    // The closed form of the anchored cycle's descriptor, with the anchor
    // reflecting on edge j: its anchor proposes the next probe, and the cycle
    // itself goes to the interval proof.
    auto closed_form=[&](size_t anchor,size_t j,const DPolygon &q,std::optional<double> &next) {
        ++result.closed_form_steps;
        const auto feature=local_descriptor(p,q,anchor,int(2*j+1));
        auto candidate=q;bool built=false;snap_vertices(p,feature,candidate);
        try {built=close_descriptor<double>(p,feature,candidate,options.descriptor_pivots);} catch(const std::runtime_error &) {}
        if(built) {
            const auto a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
            next=(candidate[anchor]-a).dot(e)/e.dot(e);
            if(prove(candidate))return true;
        }
        // The anchor straight through the edge (see close_straight).
        bool crossed=false;
        if(options.straight_anchor) {
            auto crossing=q;double u=0;
            crossed=close_straight<double>(p,feature,crossing,anchor,j,options.descriptor_pivots,&u);
            if(crossed&&!built)next=u;
            if(crossed&&prove(crossing))return true;
            auto f=feature;auto moved=q;
            if(block_descriptor(p,moved,f)&&close_straight<double>(p,f,moved,anchor,j,options.descriptor_pivots)&&prove(moved))return true;
        }
        // As the exact rebuild does (DescriptorRebuilds): the block
        // descriptor, then the anchor's own local feature, which differs when
        // an optimal cycle passes straight through the anchor polygon.
        auto attempt=[&](DPolygon c,const std::vector<int> &f) {
            snap_vertices(p,f,c);bool ok=false;
            try {ok=close_descriptor<double>(p,f,c,options.descriptor_pivots);} catch(const std::runtime_error &) {}
            return ok&&prove(c);
        };
        auto blocks=feature;auto moved=q;
        if(block_descriptor(p,moved,blocks)&&attempt(std::move(moved),blocks))return true;
        const auto own=local_descriptor(p,q,anchor,kLocal);
        if(own[anchor]!=feature[anchor]){auto f=feature;f[anchor]=own[anchor];if(attempt(q,f))return true;}
        // Unlike Step 2 of the exact path, a pivoted closed form's fixed point
        // also ends the piece here: an open search hands over to the exact
        // path, and ending early is what keeps the tolerant mode fast.
        if(!built&&crossed){auto f=feature;f[anchor]=-1;return fixed_point(anchor,j,f,*next);}
        return built&&fixed_point(anchor,j,feature,*next);
    };
    // The anchored cycle at x by the binary64 TPP construction, with the
    // polygons that contain x skipped; at a vertex where F does not decrease
    // along either edge (or that lies in another polygon) it goes to the proof.
    // Elsewhere, at a bisection probe or at a crossing with another polygon's
    // boundary, so does every cycle that shortens the best anchored one: a
    // minimum at a kink of F, which the closed-form steps never land on, then
    // ends the search once the gap closes.
    double best_anchored=INFINITY;
    auto anchored=[&](size_t anchor,const ConvexArithmeticPoint<double> &x) {
        detail::cycle_checkpoint();
        ++result.anchored_solves;
        size_t first=1,last=k-1;
        while(first<=last&&Kernel::inside(x,p[(anchor+first)%k]))++first;
        while(last>=first&&Kernel::inside(x,p[(anchor+last)%k]))--last;
        DPolygon q(k,x);
        if(first<=last) {
            std::vector<std::vector<Vector2>> remaining;
            for(size_t r=first;r<=last;++r)remaining.push_back(owned[(anchor+r)%k]);
            const auto chain=detail::double_candidate_chain(x.external(),x.external(),remaining);
            if(chain.size()!=remaining.size()+2)throw std::runtime_error("Incomplete binary64 anchored chain");
            for(size_t r=first;r<=last;++r)q[(anchor+r)%k]=ConvexArithmeticPoint<double>(chain[r-first+1]);
        }
        if(const auto j=vertex_at(p[anchor],x)) {
            bool candidate=false;
            for(size_t i=0;i<k&&!candidate;++i)candidate=i!=anchor&&Kernel::inside(x,p[i]);
            if(!candidate) {
                const auto &a=p[anchor];const size_t m=a.size();
                int forward=0,backward=0;
                try {
                    forward=support_sign(q,anchor,a[(*j+1)%m]-a[*j]);
                    backward=support_sign(q,anchor,a[*j]-a[(*j+m-1)%m]);
                } catch(const std::runtime_error &) {forward=backward=0;}
                candidate=forward>=0&&backward<=0;
            }
            if(candidate)prove(q);
            return q;
        }
        double length=0;
        for(size_t i=0;i<k;++i)length+=(q[(i+1)%k].external()-q[i].external()).length();
        if(length<best_anchored){best_anchored=length;prove(q);}
        return q;
    };
    try {
        if(apart) {
            // Pairwise apart: one boundary, the polygon with fewest vertices.
            result.anchor_polygon=size_t(std::min_element(p.begin(),p.end(),
                [](const auto &a,const auto &b){return a.size()<b.size();})-p.begin());
            const size_t anchor=result.anchor_polygon;
            search_cycle<double>(p,anchor,[&](const ConvexArithmeticPoint<double> &x){return anchored(anchor,x);},solved,
                &result.diagnostic,&result.bisection_probes,
                [&](size_t j,const DPolygon &q,std::optional<double> &next){return closed_form(anchor,j,q,next);});
        } else {
            // The eligible polygon, decided exactly on the binary64 regions.
            std::optional<size_t> only;
            if(options.anchor_choice!=ConvexCycleAnchorChoice::EveryBoundary) {
                std::vector<std::vector<Vector2>> binary;
                for(const auto &q:p){auto &r=binary.emplace_back();for(const auto &v:q)r.push_back(v.external());}
                only=detail::eligible_anchor(binary,options.anchor_choice);
            }
            search_intersecting_boundaries<double>(p,anchored,[](size_t,const DPolygon &,const auto &,const auto &){return false;},
                [](const DPolygon &){return false;},solved,closed_form,result.anchor_polygon,result.diagnostic,&result.bisection_probes,
                nullptr,only,options.anchor_ends_first,false);
        }
    } catch(const detail::CycleInterrupted &) {throw;}
    catch(const std::exception &error){result.diagnostic=error.what();}
    return closed&&finish(2);
}
} // namespace

ConvexCycleProposalResult tpp_convex_certify_cycle_proposal(const std::vector<std::vector<Vector2>> &input,
        const std::vector<Vector2> &contacts) {
    ConvexCycleProposalResult result;
    if(input.size()<2||contacts.size()!=input.size()||
       !std::all_of(contacts.begin(),contacts.end(),[](const Vector2 &v){return v.is_finite();})) {
        result.status=ConvexCycleStatus::InvalidInput;return result;
    }
    ConvexRationalPolygons normalized;std::vector<std::vector<Vector2>> binary;
    try {detail::prepare_cycle_polygons(input,normalized,false,&binary);}
    catch(const std::invalid_argument &error) {
        result.status=ConvexCycleStatus::InvalidInput;result.diagnostic=error.what();return result;
    }
    std::vector<DPolygon> p;
    for(const auto &polygon:binary){auto &q=p.emplace_back();for(const auto &v:polygon)q.emplace_back(v);}
    DPolygon q;for(const auto &v:contacts)q.emplace_back(v);
    const ConvexCycleCertificateGeometry prepared(normalized);
    const auto feature=local_descriptor(p,q,0,kLocal);
    DescriptorRebuilds rebuilds;rebuilds.pivots=true;rebuilds.straight=true;
    auto verify=[&](const ConvexRationalPolygon &c){return detail::cycle_optimality_certificate(prepared,c);};
    try {
        for(std::size_t anchor=0;anchor<p.size();++anchor) {
            ConvexRationalPolygon exact;ConvexCycleCertificateResult certificate;
            if(rebuilds(normalized,anchor,feature,feature[anchor],q,verify,result.rebuilds,exact,certificate)) {
                result.status=ConvexCycleStatus::Optimal;result.exact_contacts=std::move(exact);result.certificate=certificate;
                return result;
            }
        }
    } catch(const std::exception &error){result.diagnostic=error.what();}
    if(result.diagnostic.empty())result.diagnostic="No rebuilt descriptor passed the exact certificate";
    return result;
}

ConvexCycleDoubleResult tpp_convex_solve_cycle_double(const std::vector<std::vector<Vector2>> &input,
        const ConvexCycleDoubleOptions &options) {
    ConvexCycleDoubleResult result;
    return detail::run_cycle_execution(result,options,[&]() -> ConvexCycleDoubleResult {
    if(input.size()<2)return result;
    if(std::isnan(options.lower_bound_cutoff)||std::isnan(options.max_gap)||
       (!options.initial_contacts.empty()&&options.initial_contacts.size()!=input.size())||
       !std::all_of(options.initial_contacts.begin(),options.initial_contacts.end(),[](auto q){return q.is_finite();}))return result;
    // Binary64 stage: no rational preparation, no exact KKT and no recovery
    // when its proved interval closes the requested gap or reaches the cutoff.
    double float_lower=0;
    const bool boundary_gap=options.boundary_search_gap&&!options.refine_contacts&&options.recover_arithmetic_failures;
    if(options.max_gap>0&&boundary_gap) {
        // Algorithm 4.1 in binary64 first; the exact path below only when it
        // ends open, keeping the work counted.
        ConvexCycleDoubleResult fast;
        if(float_first_boundary_search(fast,input,options))return fast;
        result.anchored_solves+=fast.anchored_solves;result.bisection_probes+=fast.bisection_probes;
        result.closed_form_steps+=fast.closed_form_steps;result.certificate_checks+=fast.certificate_checks;
    }
    if(options.max_gap>0&&!boundary_gap) {
        ConvexCycleFloatOptions stage_options;
        stage_options.cutoff=options.lower_bound_cutoff;stage_options.max_gap=options.max_gap;
        stage_options.polish=options.float_polish;stage_options.workspace=options.workspace;
        if(options.region_ids.size()==input.size())stage_options.region_ids=&options.region_ids;
        if(!options.initial_contacts.empty())stage_options.initial_contacts=&options.initial_contacts;
        if(!options.initial_features.empty())stage_options.initial_features=&options.initial_features;
        const auto stage=tpp_convex_solve_cycle_float_certified(input,stage_options);
        result.float_candidates=stage.candidates;result.float_polish_attempted=stage.polish_attempted;
        result.float_newton_iterations=stage.newton_iterations;result.float_barrier_levels=stage.barrier_levels;
        const bool closed=stage.status==ConvexFloatOracleStatus::GapClosed||stage.status==ConvexFloatOracleStatus::CutoffReached;
        if(closed||stage.interrupted) {
            result.contacts=stage.contacts;
            result.certificate.status=ConvexCycleCertificateStatus::Feasible;
            result.certificate.lower_bound=stage.lower_bound;
            result.certificate.upper_bound=stage.upper_bound;
            result.certificate.optimality_check_skipped=result.certificate.interval_bounds_used=true;
        }
        if(closed) {
            result.float_interval_closed=stage.construction_closed;result.float_polish_closed=stage.polish_closed;
            result.status=stage.lower_bound>=options.lower_bound_cutoff?ConvexCycleStatus::CertifiedBound:ConvexCycleStatus::GapClosed;
            if(options.retain_active_features)result.active_features=stage.active_features;
            return result;
        }
        if(stage.interrupted)throw detail::CycleInterrupted{};
        if(stage.status==ConvexFloatOracleStatus::Open)float_lower=stage.lower_bound;
    }
    try {
        auto out=solve_cycle_double_exact(result,input,options);
        // A proved bound of the same constraints; it never promotes Optimal.
        if(float_lower>out.certificate.lower_bound&&out.status!=ConvexCycleStatus::Optimal) {
            out.certificate.lower_bound=float_lower;
            if(out.status==ConvexCycleStatus::FloatingPointLimit&&float_lower>=options.lower_bound_cutoff)
                out.status=ConvexCycleStatus::CertifiedBound;
        }
        return out;
    } catch(const detail::CycleInterrupted &) {
        result.certificate.lower_bound=std::max(result.certificate.lower_bound,float_lower);
        throw;
    }
    });
}
} // namespace tpp
