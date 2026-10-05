#pragma once

#include "tpp/convex/rational.h"
#include <algorithm>
#include <stdexcept>

namespace tpp::unordered_detail {

// Binary64 coordinates and their exact differences are dyadic rationals.
// Scaling one polygon to a common power-of-two denominator lets support
// comparisons use integers; only the winning dot product is canonicalized.
class DyadicSupportPolygon {
    using I = ConvexInteger;
    using R = ConvexRational;
    using P = ConvexRationalPoint;
    struct IntegerPoint { I x, y; };
    std::vector<IntegerPoint> vertices;
    I scale = 1;
public:
    DyadicSupportPolygon(const std::vector<Vector2> &polygon, Vector2 origin) {
        if(polygon.empty() || !origin.is_finite())throw std::invalid_argument("Invalid support geometry");
        const P exact_origin(origin);
        scale=std::max(I(denominator(exact_origin.x)),I(denominator(exact_origin.y)));
        ConvexRationalPolygon exact;exact.reserve(polygon.size());
        for(auto v:polygon) {
            if(!v.is_finite())throw std::invalid_argument("Nonfinite support vertex");
            exact.emplace_back(v);
            scale=std::max(scale,I(denominator(exact.back().x)));
            scale=std::max(scale,I(denominator(exact.back().y)));
        }
        // Translate after exact integer scaling, avoiding a rational
        // normalization for every vertex coordinate.
        const I ox=numerator(exact_origin.x)*(scale/denominator(exact_origin.x));
        const I oy=numerator(exact_origin.y)*(scale/denominator(exact_origin.y));
        vertices.reserve(exact.size());
        for(const auto &v:exact)
            vertices.push_back({numerator(v.x)*(scale/denominator(v.x))-ox,
                                numerator(v.y)*(scale/denominator(v.y))-oy});
    }

    R support(const P &normal) const {
        if(normal.zero())return R(0);
        const I dx=denominator(normal.x),dy=denominator(normal.y);
        I x=numerator(normal.x),y=numerator(normal.y),divisor=dx;
        if(dx!=dy) { x*=dy;y*=dx;divisor*=dy; }
        I best=vertices.front().x*x+vertices.front().y*y;
        for(size_t i=1;i<vertices.size();++i) {
            I value=vertices[i].x*x+vertices[i].y*y;
            if(value<best)best=std::move(value);
        }
        return R(best)/R(scale*divisor);
    }
};
} // namespace tpp::unordered_detail
