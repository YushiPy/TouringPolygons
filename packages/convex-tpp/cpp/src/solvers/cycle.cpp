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
    for(std::size_t j=0;j<m;++j) {
        try {
            const auto q=evaluate(anchor[j]);if(solved())return;
            at_vertex[j]=q;
            forward[j]=support_sign(q,anchor_index,anchor[(j+1)%m]-anchor[j]);
            backward[j]=support_sign(q,anchor_index,anchor[j]-anchor[(j+m-1)%m]);
            available[j]=true;
        } catch(const std::exception &error) {
            if(!floating_failure)throw;
            *floating_failure=error.what();
        }
    }
    for(std::size_t j=0;j<m;++j) {
        if(!available[j]||!available[(j+1)%m]||forward[j]>=0||backward[(j+1)%m]<=0)continue;
        const auto a=anchor[j],edge=anchor[(j+1)%m]-a;
        S left=0,right=1;std::optional<S> next;
        auto inside=[&]{return next&&*next>left&&*next<right;};
        auto probe=[&](const S &t) {
            if(probes)++*probes;
            const auto q=evaluate(a+edge*t);if(solved())return true;
            const int derivative=support_sign(q,anchor_index,edge);
            if(derivative<0)left=t;else if(derivative>0)right=t;
            return derivative==0||try_step(j,q,next);
        };
        try {
            if(try_step(j,at_vertex[j],next))return;
            if(!inside()&&try_step(j,at_vertex[(j+1)%m],next))return;
            for(;;) {
                detail::cycle_checkpoint();
                if(inside()){const S t=*next;next.reset();if(probe(t))break;continue;}
                next.reset();
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
            try {if(!detail::CycleRefinement<R>::close(polygons,features,candidate))return false;}
            catch(const std::runtime_error &){return false;}
            auto c=verify(candidate);
            if(c.status!=ConvexCycleCertificateStatus::Optimal)return false;
            exact=std::move(candidate);certificate=c;return true;
        };
        if(attempt(lifted,feature,0))return true;
        auto blocks=feature;auto moved=lifted;
        if(block_descriptor(polygons,moved,blocks)&&attempt(std::move(moved),blocks,1))return true;
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
            const auto c=tpp_convex_verify_cycle_certificate(prepared,q);
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
        auto step=[&](std::size_t j,const ConvexRationalPolygon &q,std::optional<R> &next) {
            ++result.closed_form_steps;
            auto feature=contact_features<R>(polygons,q);feature[result.anchor_polygon]=int(2*j+1);
            auto closed=q;
            try {
                if(!detail::CycleRefinement<R>::close(polygons,feature,closed))return false;
            } catch(const std::runtime_error &) {return false;}
            const auto a=(*anchor)[j],edge=(*anchor)[(j+1)%m]-a;
            next=(closed[result.anchor_polygon]-a).dot(edge)/edge.dot(edge);
            ++result.certificate_checks;
            const auto certificate=tpp_convex_verify_cycle_certificate(prepared,closed);
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
        DescriptorRebuilds rebuilds;ClosedFormFixedPoint fixed_point;
        auto rebuild=[&](std::size_t anchor,const std::vector<int> &feature,const DPolygon &q) {
            ConvexRationalPolygon exact;ConvexCycleCertificateResult certificate;
            const int local=local_descriptor(polygons,q,anchor,kLocal)[anchor];
            if(!rebuilds(normalized,anchor,feature,local,q,[&](const ConvexRationalPolygon &c) {
                    ++result.oracle_calls;++result.certificate_checks;
                    return tpp_convex_verify_cycle_certificate(geometry(),c);
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
            try {built=detail::CycleRefinement<double>::close(polygons,feature,closed);}
            catch(const std::runtime_error &) {}
            if(built) {
                const auto a=polygons[anchor][j],edge=polygons[anchor][(j+1)%m]-a;
                next=(closed[anchor]-a).dot(edge)/edge.dot(edge);
                if(gap_closes(closed))return true;
            }
            if(rebuild(anchor,feature,q))return true;
            return built&&fixed_point(anchor,j,feature,*next);
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
    ConvexRationalPolygons exact,normalized;
    for(const auto &polygon:input) {
        ConvexRationalPolygon p;
        for(auto v:polygon){if(!v.is_finite())return result;p.emplace_back(v);}
        exact.push_back(std::move(p));
    }
    try {
        if(!detail::prepare_cycle_polygons(exact,normalized)) {
            result.status=ConvexCycleStatus::UnsupportedIntersection;return result;
        }
    } catch(const std::invalid_argument &){return result;}
    return solve_cycle_disjoint_double(result,input,normalized,options);
    });
}

namespace {
ConvexRationalPolygon common_region(const ConvexRationalPolygons &p) {
    auto region=p.front();
    for(size_t i=1;i<p.size()&&!region.empty();++i)
        region=detail::CycleRefinement<R>::intersect(std::move(region),p[i]);
    return region;
}
// The complete boundary-anchor reduction, shared by both arithmetic backends.
// The independent restriction checker includes the zero-link subgradient disk.
// Step(anchor,edge,q,next) solves the anchored cycle q's own combinatorial
// type in closed form, with the anchor reflecting on that edge: the cycle is
// accepted if the global certificate proves it, and its anchor parameter is
// the next probe on the edge (Newton's role: when a probe has the type of the
// edge's minimizer, the next probe lands on it exactly). Midpoints and the
// simplest rationals keep the termination guarantee.
template<class S,class Evaluate,class Restricted,class Refine,class Solved,class Step>
void search_intersecting_boundaries(const std::vector<std::vector<ConvexArithmeticPoint<S>>> &p,
        Evaluate evaluate,Restricted restricted,Refine refine,Solved solved,Step step,
        size_t &anchor_index,std::string &diagnostic,size_t *probes=nullptr) {
    using P=ConvexArithmeticPoint<S>;using Polygon=std::vector<P>;const size_t k=p.size();
    // Check a representative boundary point of every region before spending
    // unbounded rational reconstruction work on one anchor's first edge.
    // These are the same anchored problems and exact acceptance conditions as
    // below; only their order changes. Reuse their contacts in the full search.
    std::vector<Polygon> first_contacts(k);
    for(size_t anchor=0;anchor<k;++anchor)try {
        anchor_index=anchor;
        first_contacts[anchor]=evaluate(anchor,p[anchor].front());
        if(solved()||refine(first_contacts[anchor]))return;
    } catch(const std::runtime_error &error){diagnostic=error.what();}
    for(size_t anchor=0;anchor<k;++anchor) {
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
        for(size_t j=0;j<p[anchor].size();++j) {
            const P a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
            std::vector<S> cuts{0,1};
            for(size_t i=0;i<k;++i)if(i!=anchor)for(size_t h=0;h<p[i].size();++h) {
                const P b=p[i][h],f=p[i][(h+1)%p[i].size()]-b;const S determinant=e.cross(f);
                if(determinant!=0) {
                    const S t=(b-a).cross(f)/determinant,u=(b-a).cross(e)/determinant;
                    if(t>0&&t<1&&u>=0&&u<=1)cuts.push_back(t);
                } else if(e.cross(b-a)==0) {
                    const S t=(b-a).dot(e)/e.dot(e);if(t>0&&t<1)cuts.push_back(t);
                }
            }
            std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
            std::vector<Polygon> ends(cuts.size());
            for(size_t r=0;r<cuts.size();++r)try {
                const P point=a+e*cuts[r];
                if(point==p[anchor].front()&&!first_contacts[anchor].empty())
                    ends[r]=first_contacts[anchor];
                else {
                    ends[r]=evaluate(anchor,point);if(solved()||refine(ends[r]))return;
                }
            } catch(const std::runtime_error &error){diagnostic=error.what();}
            for(size_t r=1;r<cuts.size();++r) {
                if(ends[r-1].empty()||ends[r].empty())continue;
                S left=cuts[r-1],right=cuts[r];
                if(restricted(anchor,ends[r-1],a+e*left,a+e*right)||
                   restricted(anchor,ends[r],a+e*left,a+e*right))continue;
                try {
                    if(derivative(ends[r-1],e)>=0||derivative(ends[r],e)<=0)continue;
                    std::optional<S> next;
                    auto inside=[&]{return next&&*next>left&&*next<right;};
                    if(step(anchor,j,ends[r-1],next))return;
                    if(!inside()&&step(anchor,j,ends[r],next))return;
                    // A complete endpoint restriction certificate handles the
                    // nondifferentiable endpoints. Otherwise the minimizer is
                    // interior and the endpoint subgradients have strict signs.
                    for(;;) {
                        detail::cycle_checkpoint();
                        auto probe=[&](const S &t) {
                            if(probes)++*probes;
                            const auto q=evaluate(anchor,a+e*t);
                            if(solved()||refine(q))return true;
                            const int d=derivative(q,e);
                            if(d<0)left=t;else if(d>0)right=t;
                            return d==0||step(anchor,j,q,next);
                        };
                        if(inside()){const S t=*next;next.reset();if(probe(t))break;continue;}
                        next.reset();
                        const S midpoint=(left+right)/2;
                        if(midpoint==left||midpoint==right)break; // Binary64 representability only.
                        if(probe(midpoint))break;
                        const S candidate=simplest_rational(left,right);
                        if(candidate>left&&candidate<right&&probe(candidate))break;
                    }
                    if(solved())return;
                } catch(const std::runtime_error &error){diagnostic=error.what();}
            }
        }
    }
}
ConvexCycleResult solve_intersecting_cycle(ConvexCycleResult &result,const ConvexRationalPolygons &p,bool refine=true,
        const ConvexRationalPolygon &initial={},double cutoff=std::numeric_limits<double>::infinity(),bool bound_first=false) {
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
        auto fixed=p;fixed[anchor]={x};
        if(tpp_convex_verify_cycle_certificate(fixed,q).status!=ConvexCycleCertificateStatus::Optimal)
            throw std::runtime_error("Anchored path failed its exact support certificate");
        return q;
    };
    auto restricted=[&](size_t anchor,const ConvexRationalPolygon &q,const Point &a,const Point &b) {
        auto regions=p;regions[anchor]={a,b};
        return tpp_convex_verify_cycle_certificate(regions,q).status==ConvexCycleCertificateStatus::Optimal;
    };
    auto step=[&](size_t anchor,size_t j,const ConvexRationalPolygon &q,std::optional<R> &next) {
        if(q.empty())return false;
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
            if(built)next=(closed[anchor]-a).dot(e)/e.dot(e);
            if(built&&consider(closed))return true;
            auto moved=q;auto blocks=feature;bool built_blocks=false;
            if(block_descriptor(p,moved,blocks))
                try {built_blocks=detail::CycleRefinement<R>::close(p,blocks,moved);} catch(const std::runtime_error &) {}
            return built_blocks&&consider(moved);
        }
        auto closed=q;
        try {if(!detail::CycleRefinement<R>::close(p,feature,closed))return false;}
        catch(const std::runtime_error &){return false;}
        next=(closed[anchor]-a).dot(e)/e.dot(e);
        return consider(closed);
    };
    search_intersecting_boundaries<R>(p,evaluate,restricted,
        [&](const ConvexRationalPolygon &q){return refine&&detail::CycleRefinement<R>::run(p,consider,q);},
        solved,step,result.anchor_polygon,result.diagnostic,&result.bisection_probes);
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
    try {return solve_intersecting_cycle(result,normalized,options.refine_contacts,options.initial_contacts,options.lower_bound_cutoff,options.bound_first);}
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
    ConvexRationalPolygons exact,normalized;
    std::optional<ConvexCycleCertificateGeometry> prepared;
    try {
        if(options.workspace) {
            prepared.emplace(options.workspace->prepare(input,options.interval_certificate));
            normalized=prepared->polygons();
        } else {
            for(const auto &poly:input) {
                ConvexRationalPolygon q;
                for(auto v:poly){if(!v.is_finite())return result;q.emplace_back(v);}
                exact.push_back(std::move(q));
            }
            detail::prepare_cycle_polygons(exact,normalized,false);
            if(options.interval_certificate)prepared.emplace(input,true);
        }
    } catch(const std::invalid_argument &error){result.diagnostic=error.what();return result;}
    // refine_contacts=false with rational recovery: Algorithm 4.1 of paper 1,
    // with one prepared geometry for its exact certificates.
    const bool method=!options.refine_contacts&&options.recover_arithmetic_failures;
    if(method&&!options.proposal_only)try {
        // Pairwise disjoint polygons: one boundary, by the disjoint search
        // (pairwise disjoint polygons have no common point).
        ConvexRationalPolygons checked;
        if(detail::prepare_cycle_polygons(normalized,checked))return solve_cycle_disjoint_double(result,input,checked,options);
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
            recovered=solve_intersecting_cycle(recovered,normalized,options.refine_contacts,warm,options.lower_bound_cutoff,options.bound_first);
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
        for(size_t i=1;i<p.size()&&!joint.empty();++i)joint=Kernel::intersect(std::move(joint),p[i]);
        if(!joint.empty()&&consider(std::vector<Vector2>(p.size(),joint.front().external()))) {
            if(method){result.method_step=1;record_exact_contacts(result);}
            return result;
        }
        if(options.recover_arithmetic_failures&&!options.proposal_only) {
            // A common point may exist but have no binary64 representation.
            // Resolve that zero-cycle feature before attempting anchor search.
            const auto exact_joint=common_region(normalized);
            if(!exact_joint.empty()) {
                ++result.rational_cycle_recoveries;++result.oracle_calls;++result.certificate_checks;
                const auto q=ConvexRationalPolygon(p.size(),exact_joint.front());
                const auto exact_certificate=verify(q);
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
                    const auto exact_certificate=verify(lifted);
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
        DescriptorRebuilds rebuilds;ClosedFormFixedPoint fixed_point;
        auto rebuild=[&](size_t anchor,const std::vector<int> &feature,const Kernel::Polygon &q) {
            ConvexRationalPolygon exact;ConvexCycleCertificateResult certificate;
            const int local=local_descriptor(p,q,anchor,kLocal)[anchor];
            if(!rebuilds(normalized,anchor,feature,local,q,[&](const ConvexRationalPolygon &c) {
                    ++result.oracle_calls;++result.certificate_checks;return verify(c);
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
                if(candidate&&!gap_closes(q))rebuild(anchor,local_descriptor(p,q,anchor,int(2**j)),q);
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
                auto closed=q;bool built=false;snap_vertices(p,feature,closed);
                try {built=Kernel::close(p,feature,closed);} catch(const std::runtime_error &) {}
                if(built)next=(closed[anchor]-a).dot(e)/e.dot(e);
                if(built&&gap_closes(closed))return true;
                if(rebuild(anchor,feature,q))return true;
                return built&&fixed_point(anchor,j,feature,*next);
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
        search_intersecting_boundaries<double>(p,evaluate,restricted,
            [&](const Kernel::Polygon &q){return options.refine_contacts&&Kernel::run(p,refine_candidate,q);},
            solved,step,result.anchor_polygon,result.diagnostic,method?&result.bisection_probes:nullptr);
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
        try {built=Kernel::close(p,feature,candidate);} catch(const std::runtime_error &) {}
        if(built) {
            const auto a=p[anchor][j],e=p[anchor][(j+1)%p[anchor].size()]-a;
            next=(candidate[anchor]-a).dot(e)/e.dot(e);
            if(prove(candidate))return true;
        }
        // As the exact rebuild does (DescriptorRebuilds): the block
        // descriptor, then the anchor's own local feature, which differs when
        // an optimal cycle passes straight through the anchor polygon.
        auto attempt=[&](DPolygon c,const std::vector<int> &f) {
            snap_vertices(p,f,c);bool ok=false;
            try {ok=Kernel::close(p,f,c);} catch(const std::runtime_error &) {}
            return ok&&prove(c);
        };
        auto blocks=feature;auto moved=q;
        if(block_descriptor(p,moved,blocks)&&attempt(std::move(moved),blocks))return true;
        const auto own=local_descriptor(p,q,anchor,kLocal);
        if(own[anchor]!=feature[anchor]){auto f=feature;f[anchor]=own[anchor];if(attempt(q,f))return true;}
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
            search_intersecting_boundaries<double>(p,anchored,[](size_t,const DPolygon &,const auto &,const auto &){return false;},
                [](const DPolygon &){return false;},solved,closed_form,result.anchor_polygon,result.diagnostic,&result.bisection_probes);
        }
    } catch(const detail::CycleInterrupted &) {throw;}
    catch(const std::exception &error){result.diagnostic=error.what();}
    return closed&&finish(2);
}
} // namespace

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
