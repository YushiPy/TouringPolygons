#include "tpp/convex/cycle.h"
#include "tpp/convex/detail/rational_disjoint.h"
#include "tpp/convex/detail/intersecting_maps.h"
#include "cycle_internal.h"
#include "disjoint_arithmetic.h"
#include "cycle_refinement.h"
#include <cmath>
#include <limits>

#include <algorithm>
#include <stdexcept>

namespace tpp {
namespace {
using R = ConvexRational;
using Point = ConvexRationalPoint;
using Integer = boost::multiprecision::cpp_int;

// Simplest rational in the closed interval. Continued fractions reconstruct a
// rational root exactly; they do not round it to a prescribed grid or tolerance.
template<class S> S simplest_rational(S left, S right) {
    using I=std::conditional_t<std::is_same_v<S,double>,double,Integer>;
    const S original_mid=(left+right)/2;
    std::vector<I> prefix;
    S result;
    for (;;) {
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
// Identical floating-anchor reduction for both arithmetic backends.
template<class S,class Evaluate,class Solved>
void search_cycle(const std::vector<std::vector<ConvexArithmeticPoint<S>>> &polygons,
                  std::size_t anchor_index,Evaluate evaluate,Solved solved, std::string *floating_failure=nullptr) {
    const auto &anchor=polygons[anchor_index];const auto m=anchor.size();
    std::vector<int> forward(m),backward(m);
    std::vector<bool> available(m,false);
    for(std::size_t j=0;j<m;++j) {
        try {
            const auto q=evaluate(anchor[j]);if(solved())return;
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
        S left=0,right=1;
        auto probe=[&](const S &t) {
            const auto q=evaluate(a+edge*t);if(solved())return true;
            const int derivative=support_sign(q,anchor_index,edge);
            if(derivative<0)left=t;else if(derivative>0)right=t;
            return derivative==0;
        };
        try {
            for(;;) {
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

} // namespace

ConvexCycleResult tpp_convex_solve_cycle_disjoint(const ConvexRationalPolygons &input) {
    ConvexCycleResult result;
    if (input.size() < 2) return result;
    ConvexRationalPolygons polygons;
    try {
        if (!detail::prepare_cycle_polygons(input, polygons)) {
            result.status = ConvexCycleStatus::UnsupportedIntersection;
            return result;
        }
    } catch (const std::invalid_argument &) { return result; }

    try {
        if(detail::CycleRefinement<R>::run(polygons,[&](const ConvexRationalPolygon &q) {
            ++result.oracle_calls;++result.certificate_checks;
            const auto c=tpp_convex_verify_cycle_certificate(polygons,q);
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
        ++result.oracle_calls;
        const auto path = detail::solve_rational_disjoint_exact(point, point, remaining);
        if (path.contacts.size() != k - 1) throw std::runtime_error("Incomplete anchored cycle");
        ConvexRationalPolygon contacts(k);
        contacts[result.anchor_polygon] = point;
        for (std::size_t j = 1; j < k; ++j)
            contacts[(result.anchor_polygon + j) % k] = path.contacts[j - 1];
        const auto certificate = tpp_convex_verify_cycle_certificate(input, contacts);
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
        search_cycle<R>(polygons,result.anchor_polygon,evaluate,solved);
        if(solved())return result;
    } catch (const std::runtime_error &error) {
        result.diagnostic=error.what();
        result.status = ConvexCycleStatus::OracleFailure;
        return result;
    }
    result.status = ConvexCycleStatus::OracleFailure;
    return result;
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

ConvexCycleDoubleResult tpp_convex_solve_cycle_disjoint_double(const std::vector<std::vector<Vector2>> &input,
        const ConvexCycleDoubleOptions &options) {
    ConvexCycleDoubleResult result;
    if(input.size()<2)return result;
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
    using DPoint=ConvexArithmeticPoint<double>;
    using DPolygon=std::vector<DPoint>;
    std::vector<DPolygon> polygons;
    for(const auto &p:normalized){DPolygon q;for(const auto &v:p)q.emplace_back(v.external());polygons.push_back(q);}
    result.anchor_polygon=std::size_t(std::min_element(polygons.begin(),polygons.end(),
        [](const auto &a,const auto &b){return a.size()<b.size();})-polygons.begin());
    const auto k=polygons.size();std::vector<DPolygon> remaining;
    for(std::size_t j=1;j<k;++j)remaining.push_back(polygons[(result.anchor_polygon+j)%k]);
    auto consider=[&](std::vector<Vector2> contacts) {
        auto certificate=tpp_convex_verify_cycle_certificate(input,contacts);++result.certificate_checks;
        if(certificate.status==ConvexCycleCertificateStatus::InvalidCandidate) {
            round_contacts_inward(normalized,contacts);
            certificate=tpp_convex_verify_cycle_certificate(input,contacts);++result.certificate_checks;
        }
        if(certificate.status!=ConvexCycleCertificateStatus::Feasible&&
           certificate.status!=ConvexCycleCertificateStatus::Optimal)
            throw std::runtime_error("Double candidate failed exact feasibility after rounding");
        if(result.contacts.empty()||certificate.upper_bound<result.certificate.upper_bound||
           (certificate.upper_bound==result.certificate.upper_bound&&certificate.lower_bound>result.certificate.lower_bound)||
           certificate.status==ConvexCycleCertificateStatus::Optimal) {
            result.contacts=std::move(contacts);result.certificate=certificate;
        }
        if(certificate.status==ConvexCycleCertificateStatus::Optimal)result.status=ConvexCycleStatus::Optimal;
    };
    auto solved=[&]{return result.status==ConvexCycleStatus::Optimal;};
    try {
        if(options.refine_contacts)detail::CycleRefinement<double>::run(polygons,
                [&](const DPolygon &q,const std::vector<int> &features,bool closed) {
            ++result.oracle_calls;
            std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());
            consider(candidate);if(solved())return true;
            if(closed&&options.recover_arithmetic_failures) {
                ++result.rational_feature_recoveries;
                ConvexRationalPolygon lifted;for(auto v:candidate)lifted.emplace_back(v);
                if(detail::CycleRefinement<R>::close(normalized,features,lifted)) {
                    ++result.oracle_calls;++result.certificate_checks;
                    if(tpp_convex_verify_cycle_certificate(normalized,lifted).status==ConvexCycleCertificateStatus::Optimal) {
                        candidate.clear();for(const auto &v:lifted)candidate.push_back(v.external());
                        ++result.oracle_calls;consider(candidate);
                        if(!solved())result.status=ConvexCycleStatus::FloatingPointLimit;
                        return true;
                    }
                }
            }
            return false;
        });
        if(solved()||result.status==ConvexCycleStatus::FloatingPointLimit)return result;
        // A repeated or exhausted feature proposal does not establish an
        // arithmetic limit. Continue the complete anchored search.
    } catch(const std::exception &error){result.diagnostic=error.what();}
    try {
        auto evaluate=[&](const DPoint &point) {
            ++result.oracle_calls;
            const auto path=[&] {
                try { return detail::DisjointArithmetic<double>::Solver(point,point,remaining).solve(true); }
                catch(const std::exception &) {
                    if(!options.recover_arithmetic_failures)throw;
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
            std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());
            consider(std::move(candidate));
            return q;
        };
        search_cycle<double>(polygons,result.anchor_polygon,evaluate,solved,&result.diagnostic);
        if(!solved())result.status=result.contacts.empty()?ConvexCycleStatus::OracleFailure:ConvexCycleStatus::FloatingPointLimit;
    } catch(const std::runtime_error &error) {
        result.status=ConvexCycleStatus::OracleFailure;result.diagnostic=error.what();
    }
    if(!solved()&&options.recover_arithmetic_failures)try {
        ++result.rational_cycle_recoveries;
        const auto recovered=tpp_convex_solve_cycle_disjoint(normalized);
        if(recovered.status==ConvexCycleStatus::Optimal) {
            std::vector<Vector2> q;for(const auto &v:recovered.contacts)q.push_back(v.external());
            ++result.oracle_calls;consider(q);
            if(!solved())result.status=ConvexCycleStatus::FloatingPointLimit;
        } else {result.status=ConvexCycleStatus::OracleFailure;result.diagnostic=recovered.diagnostic;}
    } catch(const std::exception &error){result.status=ConvexCycleStatus::OracleFailure;result.diagnostic=error.what();}
    return result;
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
template<class S,class Evaluate,class Restricted,class Refine,class Solved>
void search_intersecting_boundaries(const std::vector<std::vector<ConvexArithmeticPoint<S>>> &p,
        Evaluate evaluate,Restricted restricted,Refine refine,Solved solved,
        size_t &anchor_index,std::string &diagnostic) {
    using P=ConvexArithmeticPoint<S>;using Polygon=std::vector<P>;const size_t k=p.size();
    for(size_t anchor=0;anchor<k;++anchor) {
        anchor_index=anchor;
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
                ends[r]=evaluate(anchor,a+e*cuts[r]);if(solved()||refine(ends[r]))return;
            } catch(const std::runtime_error &error){diagnostic=error.what();}
            for(size_t r=1;r<cuts.size();++r) {
                if(ends[r-1].empty()||ends[r].empty())continue;
                S left=cuts[r-1],right=cuts[r];
                if(restricted(anchor,ends[r-1],a+e*left,a+e*right)||
                   restricted(anchor,ends[r],a+e*left,a+e*right))continue;
                try {
                    if(derivative(ends[r-1],e)>=0||derivative(ends[r],e)<=0)continue;
                    // A complete endpoint restriction certificate handles the
                    // nondifferentiable endpoints. Otherwise the minimizer is
                    // interior and the endpoint subgradients have strict signs.
                    for(;;) {
                        auto probe=[&](const S &t) {
                            const auto q=evaluate(anchor,a+e*t);
                            if(solved()||refine(q))return true;
                            const int d=derivative(q,e);
                            if(d<0)left=t;else if(d>0)right=t;
                            return d==0;
                        };
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
ConvexCycleResult solve_intersecting_cycle(const ConvexRationalPolygons &p,bool refine=true) {
    ConvexCycleResult result;const size_t k=p.size();
    auto consider=[&](const ConvexRationalPolygon &q) {
        ++result.oracle_calls;++result.certificate_checks;
        const auto c=tpp_convex_verify_cycle_certificate(p,q);
        if(c.status==ConvexCycleCertificateStatus::Optimal) {
            result.contacts=q;result.certificate=c;result.status=ConvexCycleStatus::Optimal;
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
    const auto intersection=common_region(p);
    if(!intersection.empty()){consider(ConvexRationalPolygon(k,intersection.front()));return result;}
    try {if(refine&&detail::CycleRefinement<R>::run(p,consider))return result;}
    catch(const std::runtime_error &error){result.diagnostic=error.what();}

    ConvexRationalPolygons checked;
    if(refine&&detail::prepare_cycle_polygons(p,checked))return tpp_convex_solve_cycle_disjoint(p);

    // A containing polygon need not have an optimal boundary contact. The
    // general search visits every boundary; prefix/suffix visits at x are free.
    auto evaluate=[&](size_t anchor,const Point &x) {
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
        if(result.status==ConvexCycleStatus::Optimal)return q;
        auto fixed=p;fixed[anchor]={x};
        if(tpp_convex_verify_cycle_certificate(fixed,q).status!=ConvexCycleCertificateStatus::Optimal)
            throw std::runtime_error("Anchored path failed its exact support certificate");
        return q;
    };
    auto restricted=[&](size_t anchor,const ConvexRationalPolygon &q,const Point &a,const Point &b) {
        auto regions=p;regions[anchor]={a,b};
        return tpp_convex_verify_cycle_certificate(regions,q).status==ConvexCycleCertificateStatus::Optimal;
    };
    search_intersecting_boundaries<R>(p,evaluate,restricted,
        [&](const ConvexRationalPolygon &q){return refine&&detail::CycleRefinement<R>::run(p,consider,q);},
        [&]{return result.status==ConvexCycleStatus::Optimal;},result.anchor_polygon,result.diagnostic);
    if(result.status==ConvexCycleStatus::Optimal)return result;
    result.status=ConvexCycleStatus::OracleFailure;
    if(result.diagnostic.empty())result.diagnostic="No globally certified cycle constructed";
    return result;
}
} // namespace

ConvexCycleResult tpp_convex_solve_cycle(const ConvexRationalPolygons &input,const ConvexCycleOptions &options) {
    if(input.size()<2)return {};
    ConvexRationalPolygons normalized;
    try {detail::prepare_cycle_polygons(input,normalized,false);}
    catch(const std::invalid_argument &error) {
        ConvexCycleResult result;result.diagnostic=error.what();return result;
    }
    try {return solve_intersecting_cycle(normalized,options.refine_contacts);}
    catch(const std::exception &error) {
        ConvexCycleResult result;result.status=ConvexCycleStatus::OracleFailure;
        result.diagnostic=error.what();return result;
    }
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
ConvexCycleDoubleResult tpp_convex_solve_cycle_double(const std::vector<std::vector<Vector2>> &input,
        const ConvexCycleDoubleOptions &options) {
    ConvexCycleDoubleResult result;if(input.size()<2)return result;
    ConvexRationalPolygons exact,normalized;
    for(const auto &p:input) {
        ConvexRationalPolygon q;for(auto v:p){if(!v.is_finite())return result;q.emplace_back(v);}exact.push_back(q);
    }
    try {
        detail::prepare_cycle_polygons(exact,normalized,false);
    } catch(const std::invalid_argument &error){result.diagnostic=error.what();return result;}
    using Kernel=detail::CycleRefinement<double>;
    Kernel::Polygons p;for(const auto &poly:normalized){Kernel::Polygon q;for(const auto &v:poly)q.emplace_back(v.external());p.push_back(q);}
    auto consider=[&](std::vector<Vector2> q) {
        ++result.oracle_calls;++result.certificate_checks;
        auto c=tpp_convex_verify_cycle_certificate(input,q);
        if(c.status==ConvexCycleCertificateStatus::InvalidCandidate) {
            round_contacts_inward(normalized,q);c=tpp_convex_verify_cycle_certificate(input,q);++result.certificate_checks;
        }
        if(c.status==ConvexCycleCertificateStatus::Optimal||c.status==ConvexCycleCertificateStatus::Feasible) {
            if(c.status==ConvexCycleCertificateStatus::Optimal||result.contacts.empty()||c.upper_bound<result.certificate.upper_bound||
               (c.upper_bound==result.certificate.upper_bound&&c.lower_bound>result.certificate.lower_bound)) {
                result.contacts=q;result.certificate=c;
            }
            if(c.status==ConvexCycleCertificateStatus::Optimal){result.status=ConvexCycleStatus::Optimal;return true;}
        }
        return false;
    };
    try {
        auto joint=p.front();
        for(size_t i=1;i<p.size()&&!joint.empty();++i)joint=Kernel::intersect(std::move(joint),p[i]);
        if(!joint.empty()&&consider(std::vector<Vector2>(p.size(),joint.front().external())))return result;
        if(options.recover_arithmetic_failures) {
            // A common point may exist but have no binary64 representation.
            // Resolve that zero-cycle feature before attempting anchor search.
            const auto exact_joint=common_region(normalized);
            if(!exact_joint.empty()) {
                ++result.rational_cycle_recoveries;++result.oracle_calls;++result.certificate_checks;
                const auto q=ConvexRationalPolygon(p.size(),exact_joint.front());
                if(tpp_convex_verify_cycle_certificate(normalized,q).status==ConvexCycleCertificateStatus::Optimal) {
                    if(consider(std::vector<Vector2>(p.size(),q.front().external())))return result;
                    if(!result.contacts.empty()){result.status=ConvexCycleStatus::FloatingPointLimit;return result;}
                }
            }
        }
        auto refine_candidate=[&](const Kernel::Polygon &q,const std::vector<int> &features,bool closed) {
            std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());
            if(consider(candidate))return true;
            if(closed&&options.recover_arithmetic_failures) {
                // Filtered construction: reconstruct just the proposed feature
                // tuple, not a second optimization search. Its certificate
                // distinguishes rounding limits from a wrong active feature.
                ++result.rational_feature_recoveries;
                ConvexRationalPolygon lifted;for(const auto &v:candidate)lifted.emplace_back(v);
                if(detail::CycleRefinement<R>::close(normalized,features,lifted)) {
                    ++result.oracle_calls;++result.certificate_checks;
                    if(tpp_convex_verify_cycle_certificate(normalized,lifted).status==ConvexCycleCertificateStatus::Optimal) {
                        candidate.clear();for(const auto &v:lifted)candidate.push_back(v.external());
                        if(consider(candidate))return true;
                        if(!result.contacts.empty()) {result.status=ConvexCycleStatus::FloatingPointLimit;return true;}
                    }
                }
            }
            return false;
        };
        try {if(options.refine_contacts&&Kernel::run(p,refine_candidate))return result;}
        catch(const std::exception &error){result.diagnostic=error.what();}
        ConvexRationalPolygons checked;
        if(options.refine_contacts&&detail::prepare_cycle_polygons(normalized,checked))return tpp_convex_solve_cycle_disjoint_double(input,options);
        auto evaluate=[&](size_t anchor,const Kernel::P &x) {
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
                    try {return detail::solve_intersecting_map_contacts_unchecked_double(x.external(),x.external(),remaining,false);}
                    catch(const std::exception &) {
                        if(!options.recover_arithmetic_failures)throw;
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
            std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());consider(candidate);
            return q;
        };
        auto restricted=[&](size_t anchor,const Kernel::Polygon &q,const Kernel::P &a,const Kernel::P &b) {
            auto regions=input;regions[anchor]={a.external(),b.external()};
            std::vector<Vector2> candidate;for(const auto &v:q)candidate.push_back(v.external());
            return tpp_convex_verify_cycle_certificate(regions,candidate).status==ConvexCycleCertificateStatus::Optimal;
        };
        search_intersecting_boundaries<double>(p,evaluate,restricted,
            [&](const Kernel::Polygon &q){return options.refine_contacts&&Kernel::run(p,refine_candidate,q);},
            [&]{return result.status==ConvexCycleStatus::Optimal||result.status==ConvexCycleStatus::FloatingPointLimit;},
            result.anchor_polygon,result.diagnostic);
        if(result.status==ConvexCycleStatus::Optimal||result.status==ConvexCycleStatus::FloatingPointLimit)return result;
    } catch(const std::exception &error){result.diagnostic=error.what();}
    try {
        if(options.recover_arithmetic_failures) {
            ++result.rational_cycle_recoveries;
            const auto recovered=solve_intersecting_cycle(normalized);
            if(recovered.status==ConvexCycleStatus::Optimal) {
                std::vector<Vector2> q;for(const auto &v:recovered.contacts)q.push_back(v.external());
                if(consider(q))return result;
                result.status=result.contacts.empty()?ConvexCycleStatus::OracleFailure:ConvexCycleStatus::FloatingPointLimit;
                return result;
            }
            result.status=ConvexCycleStatus::OracleFailure;result.diagnostic=recovered.diagnostic;return result;
        }
    } catch(const std::exception &error){result.status=ConvexCycleStatus::OracleFailure;result.diagnostic=error.what();return result;}
    result.status=result.contacts.empty()?ConvexCycleStatus::OracleFailure:ConvexCycleStatus::FloatingPointLimit;
    return result;
}
} // namespace tpp
