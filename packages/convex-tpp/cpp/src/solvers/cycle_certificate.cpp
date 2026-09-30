#include "tpp/convex/cycle_certificate.h"
#include "cycle_internal.h"
#include "cycle_zero_certificate.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
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

bool inside(const Point &point, const Polygon &polygon) {
    if(polygon.size()==1)return point==polygon[0];
    if(polygon.size()==2) {
        const Point e=polygon[1]-polygon[0],d=point-polygon[0];
        return e.cross(d)==0 && d.dot(e)>=0 && d.dot(e)<=e.dot(e);
    }
	for (std::size_t i = 0; i < polygon.size(); ++i) {
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

Rational primal_upper(const std::vector<Point> &contacts) {
	Rational upper = 0;
	for (std::size_t i = 0; i < contacts.size(); ++i) {
		const Point difference = contacts[(i + 1) % contacts.size()] - contacts[i];
		upper += rational_sqrt_upper(difference.dot(difference));
	}
	return upper;
}

Rational dual_lower(const std::vector<Point> &contacts, const std::vector<Polygon> &polygons) {
	std::vector<Point> directions;
	directions.reserve(contacts.size());
	for (std::size_t i = 0; i < contacts.size(); ++i) {
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
ConvexCycleCertificateGeometry::ConvexCycleCertificateGeometry(const std::vector<std::vector<Vector2>> &input) {
    for(const auto &p:input) {
        Polygon exact;
        for(auto v:p) { if(!v.is_finite())throw std::invalid_argument("Nonfinite polygon");exact.emplace_back(v); }
        polygons_.push_back(exact_convex_polygon(exact));
    }
}
ConvexCycleCertificateGeometry ConvexCycleWorkspace::prepare(const std::vector<std::vector<Vector2>> &input) {
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

ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
	const ConvexCycleCertificateGeometry &geometry,
	const ConvexRationalPolygon &input_contacts, double lower_bound_cutoff) {
	const auto &input_polygons=geometry.polygons();
	ConvexCycleCertificateResult result;
	if (input_polygons.empty() || std::isnan(lower_bound_cutoff)) return result;
	if (input_contacts.size() != input_polygons.size()) {
		result.status = ConvexCycleCertificateStatus::InvalidCandidate;
		return result;
	}

	try {
		const auto &polygons=input_polygons;
		const auto &contacts=input_contacts;
		for (std::size_t i = 0; i < contacts.size(); ++i) {
			if (!inside(contacts[i], polygons[i])) {
				result.status = ConvexCycleCertificateStatus::InvalidCandidate;
				return result;
			}
		}
		result.status = ConvexCycleCertificateStatus::Feasible;

		if (contacts.size() == 1) {
			result.status = ConvexCycleCertificateStatus::Optimal;
			result.lower_bound = result.upper_bound = 0;
			return result;
		}

        const bool try_bound=cutoff_promising(polygons,contacts,lower_bound_cutoff);
        if(try_bound) {
            result.lower_bound=rounded_lower(std::max(Rational(0),dual_lower(contacts,polygons)));
            if(result.lower_bound>=lower_bound_cutoff) {
                result.upper_bound=rounded_upper(primal_upper(contacts));
                result.optimality_check_skipped=true;
                return result; // Feasible with a dual bound, never Optimal.
            }
        }
        if(detail::cycle_support_certificate(polygons,contacts,result.exact_predicate_evaluations))
            result.status=ConvexCycleCertificateStatus::Optimal;

	const Rational upper = primal_upper(contacts);
	if (result.status == ConvexCycleCertificateStatus::Optimal) {
			Rational length_lower = 0;
			for (std::size_t i = 0; i < contacts.size(); ++i) {
				const Point edge = contacts[(i + 1) % contacts.size()] - contacts[i];
				length_lower += rational_sqrt_lower(edge.dot(edge));
			}
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
        const ConvexRationalPolygons &polygons,const ConvexRationalPolygon &contacts,double lower_bound_cutoff) {
    if(polygons.empty())return {};
    if(polygons.size()!=contacts.size()) { ConvexCycleCertificateResult r;r.status=ConvexCycleCertificateStatus::InvalidCandidate;return r; }
    try {return tpp_convex_verify_cycle_certificate(ConvexCycleCertificateGeometry(polygons),contacts,lower_bound_cutoff);}
    catch(const std::exception &) {return {};}
}
ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
        const ConvexCycleCertificateGeometry &geometry,const std::vector<Vector2> &contacts,double lower_bound_cutoff) {
    ConvexRationalPolygon exact;
    for(auto q:contacts) {
        if(!q.is_finite()) {ConvexCycleCertificateResult r;r.status=ConvexCycleCertificateStatus::InvalidCandidate;return r;}
        exact.emplace_back(q);
    }
    return tpp_convex_verify_cycle_certificate(geometry,exact,lower_bound_cutoff);
}

ConvexCycleCertificateResult tpp_convex_verify_cycle_certificate(
    const std::vector<std::vector<Vector2>> &polygons, const std::vector<Vector2> &contacts,double lower_bound_cutoff) {
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
