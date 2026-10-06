#pragma once

#include "tpp/geometry/vec2.h"
#ifdef TPP_USE_GMP_RATIONAL
#include <boost/multiprecision/gmp.hpp>
#else
#include <boost/multiprecision/cpp_int.hpp>
#endif
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <vector>

namespace tpp {
#ifdef TPP_USE_GMP_RATIONAL
using ConvexInteger = boost::multiprecision::mpz_int;
using ConvexRational = boost::multiprecision::mpq_rational;
#else
using ConvexInteger = boost::multiprecision::cpp_int;
using ConvexRational = boost::multiprecision::cpp_rational;
#endif

// Round to the nearest double, ties to even: the value Boost's convert_to
// returns. mpq_get_d truncates toward zero; in the normal range the result is
// that value or its neighbour away from zero, chosen by an exact comparison
// with their midpoint. Thread-local scratch rationals keep their storage.
inline double convex_nearest_double(const ConvexRational &q) {
#ifdef TPP_USE_GMP_RATIONAL
    const double toward=mpq_get_d(q.backend().data());
    if(std::isnormal(toward)) {
        const double away=std::nextafter(toward,toward>0?INFINITY:-INFINITY);
        if(std::isfinite(away)) {
            thread_local ConvexRational midpoint,other;
            midpoint=toward;
            if(q==midpoint)return toward;
            other=away;
            mpq_add(midpoint.backend().data(),midpoint.backend().data(),other.backend().data());
            mpq_div_2exp(midpoint.backend().data(),midpoint.backend().data(),1);
            const int beyond=q.compare(midpoint)*(toward>0?1:-1);
            if(beyond<0)return toward;
            if(beyond>0)return away;
            return std::bit_cast<std::uint64_t>(toward)&1?away:toward;
        }
    }
#endif
    return q.template convert_to<double>();
}

// Exact coordinates are part of the result, not rounded binary64 witnesses.
template<class Scalar>
struct ConvexArithmeticPoint {
    Scalar x = 0, y = 0;
    ConvexArithmeticPoint() = default;
    ConvexArithmeticPoint(Scalar x_, Scalar y_) : x(std::move(x_)), y(std::move(y_)) {}
    explicit ConvexArithmeticPoint(Vector2 p) : x(p.x), y(p.y) {}
    ConvexArithmeticPoint operator+(const ConvexArithmeticPoint &p) const { return {x+p.x,y+p.y}; }
    ConvexArithmeticPoint operator-(const ConvexArithmeticPoint &p) const { return {x-p.x,y-p.y}; }
    ConvexArithmeticPoint operator-() const { return {-x,-y}; }
    ConvexArithmeticPoint operator*(const Scalar &s) const { return {x*s,y*s}; }
    Scalar cross(const ConvexArithmeticPoint &p) const { return x*p.y-y*p.x; }
    Scalar dot(const ConvexArithmeticPoint &p) const { return x*p.x+y*p.y; }
    bool operator==(const ConvexArithmeticPoint &) const = default;
    bool zero() const { return x==0 && y==0; }
    // Presentation only: this conversion need not preserve boundary membership.
    Vector2 external() const {
        if constexpr (std::is_same_v<Scalar,double>) return {x,y};
        else if constexpr (std::is_same_v<Scalar,ConvexRational>) return {convex_nearest_double(x),convex_nearest_double(y)};
        else return {x.template convert_to<double>(),y.template convert_to<double>()};
    }
};
using ConvexRationalPoint = ConvexArithmeticPoint<ConvexRational>;
using ConvexRationalPolygon = std::vector<ConvexRationalPoint>;
using ConvexRationalPolygons = std::vector<ConvexRationalPolygon>;
} // namespace tpp
