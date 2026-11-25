#include "cycle_execution.h"
#include "tpp/convex/cycle_certificate.h"
#include "cycle_internal.h"
#include "cycle_zero_certificate.h"
#include "cycle_interval.h"
#include "binary_certificate.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace {
using Rational = tpp::ConvexRational;
using Integer = tpp::ConvexInteger;

using Point = tpp::ConvexRationalPoint;
using Polygon = std::vector<Point>;

Polygon exact_convex_polygon(const Polygon &input) {
	Polygon polygon;
	polygon.reserve(input.size());
	for (const Point &exact : input) {
		if (polygon.empty() || polygon.back().x != exact.x || polygon.back().y != exact.y)
			polygon.push_back(std::move(exact));
	}
	if (polygon.size() > 1 && polygon.front().x == polygon.back().x && polygon.front().y == polygon.back().y)
		polygon.pop_back();
	if (polygon.empty()) throw std::invalid_argument("Empty polygon");
    if (polygon.size() <= 2) return polygon;

	Rational area = 0;
	for (std::size_t i = 0; i < polygon.size(); ++i)
		area += polygon[i].cross(polygon[(i + 1) % polygon.size()]);
	if (area == 0) throw std::invalid_argument("Polygon has zero area");

    if (area < 0) std::reverse(polygon.begin(), polygon.end());
    size_t winding=0;
    for(size_t i=0;i<polygon.size();++i) {
        const auto a=polygon[i]-polygon[(i+polygon.size()-1)%polygon.size()];
        const auto b=polygon[(i+1)%polygon.size()]-polygon[i];
        const Rational turn=a.cross(b);
        if(turn<0 || (turn==0&&a.dot(b)<=0))throw std::invalid_argument("Nonconvex or backtracking boundary");
        if(a.y<=0&&b.y>0)++winding;
    }
    if(winding!=1)throw std::invalid_argument("Multiply-wound boundary");
	return polygon;
}

// binary, when given, holds the same vertices as binary64 values.
bool inside(const Point &point, const Polygon &polygon, const std::vector<Vector2> *binary = nullptr,
            bool filtered = false) {
    if(polygon.size()==1)return point==polygon[0];
    if(polygon.size()==2) {
        const Point e=polygon[1]-polygon[0],d=point-polygon[0];
        return e.cross(d)==0 && d.dot(e)>=0 && d.dot(e)<=e.dot(e);
    }
    // Binary64 enclosures decide most sides; the rest are exact.
    std::vector<tpp::detail::IntervalPoint> v;v.reserve(polygon.size());
    if(binary&&binary->size()==polygon.size())for(const auto &p:*binary)v.emplace_back(p);
    else for(const auto &p:polygon)v.push_back(tpp::detail::enclose_rational(p,filtered));
    const auto q=tpp::detail::enclose_rational(point,filtered);
	for (std::size_t i = 0; i < polygon.size(); ++i) {
        if(const auto side=tpp::detail::filtered_side(v[i],v[(i+1)%v.size()],q)) {
            if(*side<0)return false;
            continue;
        }
		const Point edge = polygon[(i + 1) % polygon.size()] - polygon[i];
		if (edge.cross(point - polygon[i]) < 0) return false;
	}
	return true;
}

// Reporting precision only: scale relative to each norm, including values
// outside binary64's exponent range. No bound width is an acceptance test.
Rational rational_sqrt_bound(const Rational &squared,bool upper) {
    if(squared==0)return 0;
    Integer top=boost::multiprecision::numerator(squared);
    Integer bottom=boost::multiprecision::denominator(squared);
    const long exponent=long(boost::multiprecision::msb(top))-long(boost::multiprecision::msb(bottom));
    const long half=exponent>=0?exponent/2:-((-exponent+1)/2);
    const long shift=96-half;
    if(shift>=0)top<<=2*shift;else bottom<<=-2*shift;
    const Integer quotient=top/bottom;
    Integer root=sqrt(quotient);
    if(upper&&root*root*bottom!=top)++root;
    if(shift>=0)return Rational(root)/Rational(Integer(1)<<shift);
    return Rational(root<<(-shift));
}
Rational rational_sqrt_lower(const Rational &squared) {return rational_sqrt_bound(squared,false);}
Rational rational_sqrt_upper(const Rational &squared) {return rational_sqrt_bound(squared,true);}

double rounded_lower(const Rational &value) {
	double result = value.convert_to<double>();
	if (std::isinf(result)) return result > 0 ? std::numeric_limits<double>::max() : result;
	while (Rational(result) > value)
		result = std::nextafter(result, -std::numeric_limits<double>::infinity());
	return result;
}

double rounded_upper(const Rational &value) {
	double result = value.convert_to<double>();
	if (std::isinf(result)) return result < 0 ? -std::numeric_limits<double>::max() : result;
	while (Rational(result) < value)
		result = std::nextafter(result, std::numeric_limits<double>::infinity());
	return result;
}

// The sums of rational_sqrt_lower and of rational_sqrt_upper over the
// squared links, as exact rationals: one integer root per link (the upper
// root is the lower one, plus one unless it is exact), each squared length
// reduced once from its numerators and denominators, and dyadic sums in
// integers. The same values as summing those functions.
std::pair<Rational,Rational> primal_bounds(const std::vector<Point> &contacts) {
    using boost::multiprecision::numerator;
    using boost::multiprecision::denominator;
    std::vector<std::pair<Integer,long>> lower,upper;
    lower.reserve(contacts.size());upper.reserve(contacts.size());
    long most=0;bool any=false;
    for(std::size_t i=0;i<contacts.size();++i) {
        tpp::detail::cycle_checkpoint();
        const Point d=contacts[(i+1)%contacts.size()]-contacts[i];
        const Integer &nx=numerator(d.x),&dx=denominator(d.x),&ny=numerator(d.y),&dy=denominator(d.y);
        Integer top=nx*nx*dy*dy+ny*ny*dx*dx,bottom=dx*dx*dy*dy;
        if(top==0)continue;
        const Integer common=boost::multiprecision::gcd(top,bottom);
        if(common>1){top/=common;bottom/=common;}
        const long exponent=long(boost::multiprecision::msb(top))-long(boost::multiprecision::msb(bottom));
        const long half=exponent>=0?exponent/2:-((-exponent+1)/2);
        const long shift=96-half;
        if(shift>=0)top<<=2*shift;else bottom<<=-2*shift;
        const Integer quotient=top/bottom;
        Integer root=sqrt(quotient);
        const bool exact=root*root*bottom==top;
        upper.emplace_back(exact?root:Integer(root+1),shift);
        lower.emplace_back(std::move(root),shift);
        most=any?std::max(most,shift):shift;any=true;
    }
    auto sum=[&](const std::vector<std::pair<Integer,long>> &terms)->Rational {
        if(terms.empty())return 0;
        Integer total=0;
        for(const auto &[root,shift]:terms)total+=root<<(most-shift);
        if(most>=0)return Rational(total)/Rational(Integer(1)<<most);
        return Rational(Integer(total<<(-most)));
    };
    return {sum(lower),sum(upper)};
}

Rational dual_lower(const std::vector<Point> &contacts, const std::vector<Polygon> &polygons) {
	std::vector<Point> directions;
	directions.reserve(contacts.size());
	for (std::size_t i = 0; i < contacts.size(); ++i) {
        tpp::detail::cycle_checkpoint();
		const Point difference = contacts[(i + 1) % contacts.size()] - contacts[i];
		const Rational squared = difference.dot(difference);
		if (squared == 0) {
			directions.emplace_back(); // Any vector in the unit disk is dual-feasible.
		} else {
			directions.push_back(difference * (Rational(1) / rational_sqrt_upper(squared)));
		}
	}
	Rational lower = 0;
	for (std::size_t i = 0; i < contacts.size(); ++i) {
        tpp::detail::cycle_checkpoint();
		const Point coefficient = directions[(i + contacts.size() - 1) % contacts.size()] - directions[i];
		Rational support = coefficient.dot(polygons[i].front());
		for (std::size_t j = 1; j < polygons[i].size(); ++j)
			support = std::min(support, coefficient.dot(polygons[i][j]));
		lower += support;
	}
	return lower;
}

Vector2 floating_point(const Point &point) {return point.external();}
Vector2 floating_point(const Vector2 &point) {return point;}
template<class Contacts> bool cutoff_promising(const std::vector<Polygon> &polygons,
        const Contacts &contacts,double cutoff) {
    if(cutoff==INFINITY||std::isnan(cutoff))return false;
    const size_t n=contacts.size();
    if(!n||n!=polygons.size())return false;
    if(cutoff<=0)return true;
    const auto origin=floating_point(contacts.front());
    std::vector<Vector2> directions;directions.reserve(n);
    for(size_t i=0;i<n;++i) {
        const auto delta=floating_point(contacts[(i+1)%n])-floating_point(contacts[i]);
        const double norm=std::hypot(delta.x,delta.y);
        if(!std::isfinite(norm))return true; // Fall back to the exact test.
        directions.push_back(norm==0?Vector2{}:delta/norm);
    }
    double lower=0;
    for(size_t i=0;i<n;++i) {
        const auto normal=directions[(i+n-1)%n]-directions[i];
        double support=INFINITY;
        for(const auto &v:polygons[i]) {
            const double candidate=(v.external()-origin).dot(normal);
            if(!std::isfinite(candidate))return true;
            support=std::min(support,candidate);
        }
        lower+=support;
    }
    // This value is never exported, used as a bound, or allowed to prune.
    return !std::isfinite(lower)||lower>=cutoff;
}

using Interval=tpp::detail::CycleInterval;
using IntervalPoint=tpp::detail::IntervalPoint;
struct IntervalBounds {double dual=0,primal_lower=0,primal_upper=0;};
bool interval_inside(Vector2 q,const std::vector<Vector2> &p,const Polygon &exact,size_t &predicates) {
    if(p.size()<3) {++predicates;return inside(Point(q),exact);}
    return tpp::detail::interval_convex_contains(q,p,exact,predicates);
}
std::optional<IntervalBounds> interval_cycle_bounds(const std::vector<std::vector<Vector2>> &p,
                                                    const std::vector<Vector2> &q) {
    const size_t n=q.size();
    std::vector<IntervalPoint> directions;directions.reserve(n);
    Interval length;
    for(size_t i=0;i<n;++i) {
        if(q[i]==q[(i+1)%n]) {directions.emplace_back();continue;}
        const auto d=IntervalPoint(q[(i+1)%n])-IntervalPoint(q[i]);
        const auto norm=(d.x.square()+d.y.square()).sqrt();
        if(!norm.finite()||norm.lo<=0)return {};
        length=length+norm;
        // The EXACT difference divided by this upper norm is in the unit disk.
        // These intervals enclose that vector; their endpoints are not the dual.
        directions.emplace_back(d.x.divided_by(norm.hi),d.y.divided_by(norm.hi));
    }
    Interval dual;
    const IntervalPoint origin(q.front());
    for(size_t i=0;i<n;++i) {
        const auto normal=directions[(i+n-1)%n]-directions[i];
        Interval support(INFINITY);
        for(const auto &v:p[i]) {
            const auto term=normal.dot(IntervalPoint(v)-origin);
            support.lo=std::min(support.lo,term.lo);support.hi=std::min(support.hi,term.hi);
        }
        dual=dual+support;
    }
    if(!length.finite()||!dual.finite())return {};
    return IntervalBounds{std::max(0.0,dual.lo),std::max(0.0,length.lo),length.hi};
}

} // namespace

namespace tpp {

bool detail::cycle_cutoff_promising(const ConvexRationalPolygons &p,const ConvexRationalPolygon &q,double cut) {
    return cutoff_promising(p,q,cut);
}
bool detail::cycle_cutoff_promising(const ConvexRationalPolygons &p,const std::vector<Vector2> &q,double cut) {
    return cutoff_promising(p,q,cut);
}

ConvexCycleCertificateGeometry::ConvexCycleCertificateGeometry(const ConvexRationalPolygons &input) {
    for(const auto &p:input) polygons_.push_back(exact_convex_polygon(p));
}
ConvexCycleCertificateGeometry::ConvexCycleCertificateGeometry(const std::vector<std::vector<Vector2>> &input,bool binary_geometry) {
    std::vector<Vector2> region;
    for(const auto &p:input) {
        for(auto v:p)if(!v.is_finite())throw std::invalid_argument("Nonfinite polygon");
        // exact_convex_polygon on the exact values, by binary64 predicates.
        switch(detail::binary_convex_region(p,region,false)) {
            case detail::BinaryRegionDefect::None:break;
            case detail::BinaryRegionDefect::Empty:throw std::invalid_argument("Empty polygon");
            case detail::BinaryRegionDefect::ZeroArea:throw std::invalid_argument("Polygon has zero area");
            case detail::BinaryRegionDefect::NotConvex:case detail::BinaryRegionDefect::Backtracks:
                throw std::invalid_argument("Nonconvex or backtracking boundary");
            case detail::BinaryRegionDefect::Winding:throw std::invalid_argument("Multiply-wound boundary");
        }
        Polygon exact;exact.reserve(region.size());
        for(auto v:region)exact.emplace_back(v);
        polygons_.push_back(std::move(exact));
        if(binary_geometry) {
            // The exact vertices are these binary64 values, with -0 as +0,
            // as converting the rational vertex back would give.
            for(auto &v:region){if(v.x==0)v.x=0.0;if(v.y==0)v.y=0.0;}
            binary_polygons_.push_back(region);
        }
    }
}
ConvexCycleCertificateGeometry ConvexCycleWorkspace::prepare(const std::vector<std::vector<Vector2>> &input,bool binary_geometry) {
    ConvexCycleCertificateGeometry result(ConvexRationalPolygons{});
    for(const auto &p:input) {
        Key key;
        for(auto v:p) { if(!v.is_finite())throw std::invalid_argument("Nonfinite polygon");key.emplace_back(v.x,v.y); }
        auto found=polygons_.find(key);
        if(found==polygons_.end()) {
            ConvexCycleCertificateGeometry one(std::vector<std::vector<Vector2>>{p});
            ConvexRationalPolygons strict;
            detail::prepare_cycle_polygons(one.polygons(),strict,false);
            found=polygons_.emplace(std::move(key),std::move(strict.front())).first;
        }
        result.polygons_.push_back(found->second);
        if(binary_geometry) {
            auto binary=binary_polygons_.find(found->first);
            if(binary==binary_polygons_.end()) {
                std::vector<Vector2> points;
                for(const auto &v:found->second)points.push_back(v.external());
                binary=binary_polygons_.emplace(found->first,std::move(points)).first;
            }
            result.binary_polygons_.push_back(binary->second);
        }
    }
    return result;
}
ConvexRationalPolygon tpp_convex_cycle_dual_directions(const std::vector<Vector2> &contacts,
        const ConvexRationalPolygon &inherited) {
    if(!inherited.empty()&&inherited.size()!=contacts.size())throw std::invalid_argument("Dual size mismatch");
    for(const auto &v:inherited)if(v.dot(v)>1)throw std::invalid_argument("Dual vector outside unit disk");
    ConvexRationalPolygon result;
    for(size_t i=0;i<contacts.size();++i) {
        if(!contacts[i].is_finite())throw std::invalid_argument("Nonfinite contact");
        const Point d=Point(contacts[(i+1)%contacts.size()])-Point(contacts[i]);
        const Rational squared=d.dot(d);
        result.push_back(squared==0?(inherited.empty()?Point{}:inherited[i]):d*(Rational(1)/rational_sqrt_upper(squared)));
    }
    return result;
}
double tpp_convex_cycle_dual_bound(const ConvexCycleCertificateGeometry &geometry,
        const ConvexRationalPolygon &u) {
    const auto &p=geometry.polygons();
    if(p.empty()||p.size()!=u.size())throw std::invalid_argument("Dual size mismatch");
    for(const auto &v:u)if(v.dot(v)>1)throw std::invalid_argument("Dual vector outside unit disk");
    Rational lower=0;
    for(size_t i=0;i<u.size();++i) {
        const auto normal=u[(i+u.size()-1)%u.size()]-u[i];
        Rational term=normal.dot(p[i].front());
        for(const auto &v:p[i])term=std::min(term,normal.dot(v));
        lower+=term;
    }
    return rounded_lower(std::max(Rational(0),lower));
}

static ConvexCycleCertificateResult verify_rational_cycle(
	const ConvexCycleCertificateGeometry &geometry,
	const ConvexRationalPolygon &input_contacts, double lower_bound_cutoff,
    const std::optional<IntervalBounds> &bounds = {}, bool membership_checked = false,
    bool optimal_bounds_only = false) {
	const auto &input_polygons=geometry.polygons();
    const auto &binary=geometry.binary_polygons();
    // Binary64 enclosures need a verified environment in this thread (the
    // rounding mode and flush-to-zero are per thread): checked per certificate.
    const bool filtered=tpp::detail::cycle_interval_environment();
	ConvexCycleCertificateResult result;
	if (input_polygons.empty() || std::isnan(lower_bound_cutoff)) return result;
	if (input_contacts.size() != input_polygons.size()) {
		result.status = ConvexCycleCertificateStatus::InvalidCandidate;
		return result;
	}

	try {
		const auto &polygons=input_polygons;
		const auto &contacts=input_contacts;
		for (std::size_t i = 0; !membership_checked && i < contacts.size(); ++i) {
            tpp::detail::cycle_checkpoint();
			if (!inside(contacts[i], polygons[i], binary.size()==polygons.size()?&binary[i]:nullptr, filtered)) {
				result.status = ConvexCycleCertificateStatus::InvalidCandidate;
				return result;
			}
		}
		result.status = ConvexCycleCertificateStatus::Feasible;
        result.interval_bounds_used=bounds.has_value();

		if (contacts.size() == 1) {
			result.status = ConvexCycleCertificateStatus::Optimal;
			result.lower_bound = result.upper_bound = 0;
			return result;
		}

        const bool try_bound=!bounds&&cutoff_promising(polygons,contacts,lower_bound_cutoff);
        if(try_bound) {
            result.lower_bound=rounded_lower(std::max(Rational(0),dual_lower(contacts,polygons)));
            if(result.lower_bound>=lower_bound_cutoff) {
                result.upper_bound=rounded_upper(primal_bounds(contacts).second);
                result.optimality_check_skipped=true;
                return result; // Feasible with a dual bound, never Optimal.
            }
        }
        if(detail::cycle_support_certificate(polygons,contacts,result.exact_predicate_evaluations,
                                             binary.size()==polygons.size()?&binary:nullptr))
            result.status=ConvexCycleCertificateStatus::Optimal;

        if(bounds) {
            result.lower_bound=result.status==ConvexCycleCertificateStatus::Optimal?bounds->primal_lower:bounds->dual;
            result.upper_bound=bounds->primal_upper;
            return result;
        }

    if(optimal_bounds_only&&result.status!=ConvexCycleCertificateStatus::Optimal) {
        // The trivial bounds: the caller discards a candidate that is not optimal.
        result.lower_bound=0;result.upper_bound=std::numeric_limits<double>::infinity();
        return result;
    }
	const auto [length_lower,upper] = primal_bounds(contacts);
	if (result.status == ConvexCycleCertificateStatus::Optimal) {
			result.lower_bound = rounded_lower(length_lower);
			result.upper_bound = rounded_upper(upper);
		} else {
			// Zero is always a valid dual bound (all dual vectors zero).
			if(!try_bound)result.lower_bound = rounded_lower(std::max(Rational(0),dual_lower(contacts, polygons)));
			result.upper_bound = rounded_upper(upper);
		}
		return result;
	} catch (const std::exception &) {
		result.status = ConvexCycleCertificateStatus::InvalidInput;
		return result;
	}
}

ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
    const ConvexCycleCertificateGeometry &geometry,const ConvexRationalPolygon &contacts,double cutoff) {
    detail::CyclePhaseScope certification(detail::CyclePhase::Certification);
    tpp::detail::cycle_checkpoint();
    return verify_rational_cycle(geometry,contacts,cutoff);
}

ConvexCycleCertificateResult detail::cycle_optimality_certificate(
    const ConvexCycleCertificateGeometry &geometry,const ConvexRationalPolygon &contacts) {
    detail::CyclePhaseScope certification(detail::CyclePhase::Certification);
    tpp::detail::cycle_checkpoint();
    return verify_rational_cycle(geometry,contacts,INFINITY,{},false,true);
}

ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
        const ConvexRationalPolygons &polygons,const ConvexRationalPolygon &contacts,double lower_bound_cutoff) {
    detail::CyclePhaseScope certification(detail::CyclePhase::Certification);
    tpp::detail::cycle_checkpoint();
    if(polygons.empty())return {};
    if(polygons.size()!=contacts.size()) { ConvexCycleCertificateResult r;r.status=ConvexCycleCertificateStatus::InvalidCandidate;return r; }
    try {return tpp_convex_verify_cycle_certificate(ConvexCycleCertificateGeometry(polygons),contacts,lower_bound_cutoff);}
    catch(const std::exception &) {return {};}
}
ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
        const ConvexCycleCertificateGeometry &geometry,const std::vector<Vector2> &contacts,double lower_bound_cutoff,bool interval_filter) {
    detail::CyclePhaseScope certification(detail::CyclePhase::Certification);
    tpp::detail::cycle_checkpoint();
    std::optional<IntervalBounds> bounds;
    bool membership_checked=false;size_t predicates=0;
    if(geometry.polygons().empty()||std::isnan(lower_bound_cutoff))return {};
    if(contacts.size()!=geometry.polygons().size()||
       !std::all_of(contacts.begin(),contacts.end(),[](auto q){return q.is_finite();})) {
        ConvexCycleCertificateResult r;r.status=ConvexCycleCertificateStatus::InvalidCandidate;return r;
    }
    if(interval_filter&&detail::cycle_interval_environment()&&geometry.binary_polygons().size()==contacts.size()) {
        for(size_t i=0;i<contacts.size();++i)if(!interval_inside(contacts[i],geometry.binary_polygons()[i],geometry.polygons()[i],predicates)) {
            ConvexCycleCertificateResult r;r.status=ConvexCycleCertificateStatus::InvalidCandidate;
            r.exact_predicate_evaluations=predicates;return r;
        }
        membership_checked=true;
        bounds=interval_cycle_bounds(geometry.binary_polygons(),contacts);
        if(contacts.size()>1&&bounds&&std::isfinite(lower_bound_cutoff)&&bounds->dual>=lower_bound_cutoff) {
            ConvexCycleCertificateResult r;r.status=ConvexCycleCertificateStatus::Feasible;
            r.lower_bound=bounds->dual;r.upper_bound=bounds->primal_upper;
            r.optimality_check_skipped=true;r.interval_bounds_used=true;
            r.exact_predicate_evaluations=predicates;return r;
        }
        // An inconclusive cutoff uses the original rational proof. An infinite
        // cutoff permits interval reporting after exact KKT, never early success.
        if(bounds&&std::isfinite(lower_bound_cutoff)&&bounds->primal_upper>=lower_bound_cutoff)bounds.reset();
    }
    ConvexRationalPolygon exact;
    for(auto q:contacts) {
        if(!q.is_finite()) {ConvexCycleCertificateResult r;r.status=ConvexCycleCertificateStatus::InvalidCandidate;return r;}
        exact.emplace_back(q);
    }
    auto result=verify_rational_cycle(geometry,exact,lower_bound_cutoff,bounds,membership_checked);
    result.exact_predicate_evaluations+=predicates;
    return result;
}

ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
    const std::vector<std::vector<Vector2>> &polygons, const std::vector<Vector2> &contacts,double lower_bound_cutoff,bool interval_filter) {
    detail::CyclePhaseScope certification(detail::CyclePhase::Certification);
    tpp::detail::cycle_checkpoint();
    if(interval_filter) {
        try {return tpp_convex_verify_cycle_certificate(ConvexCycleCertificateGeometry(polygons,true),contacts,lower_bound_cutoff,true);}
        catch(const std::exception &) {return {};}
    }
    ConvexCycleCertificateResult invalid;
    if (polygons.empty()) return invalid;
    if (contacts.size()!=polygons.size()) {
        invalid.status=ConvexCycleCertificateStatus::InvalidCandidate;
        return invalid;
    }
    ConvexRationalPolygons exact_polygons;
    for (const auto &polygon:polygons) {
        ConvexRationalPolygon exact;
        for (auto point:polygon) {
            if (!point.is_finite()) return invalid;
            exact.emplace_back(point);
        }
        exact_polygons.push_back(std::move(exact));
    }
    ConvexRationalPolygon exact_contacts;
    for (auto point:contacts) {
        if (!point.is_finite()) {
            invalid.status=ConvexCycleCertificateStatus::InvalidCandidate;
            return invalid;
        }
        exact_contacts.emplace_back(point);
    }
    return tpp_convex_verify_cycle_certificate(exact_polygons,exact_contacts,lower_bound_cutoff);
}

} // namespace tpp
