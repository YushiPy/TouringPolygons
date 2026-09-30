#pragma once

#include "vector2.h"
#ifdef TPP_USE_GMP_RATIONAL
#include <boost/multiprecision/gmp.hpp>
#else
#include <boost/multiprecision/cpp_int.hpp>
#endif
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
        else return {x.template convert_to<double>(),y.template convert_to<double>()};
    }
};
using ConvexRationalPoint = ConvexArithmeticPoint<ConvexRational>;
using ConvexRationalPolygon = std::vector<ConvexRationalPoint>;
using ConvexRationalPolygons = std::vector<ConvexRationalPolygon>;
} // namespace tpp
