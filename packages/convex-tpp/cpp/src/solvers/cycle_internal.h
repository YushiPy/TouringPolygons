#pragma once
#include "tpp/convex/rational.h"

namespace tpp::detail {
// Shares exact halfplanes and winding conventions with rational_disjoint.cpp.
bool prepare_cycle_polygons(const ConvexRationalPolygons &input,
                            ConvexRationalPolygons &normalized, bool check_disjoint = true);

// Sign(p/sqrt(a2) - q/sqrt(b2)), without square roots or approximate signs.
inline int cycle_normalized_difference_sign(const ConvexRational &p, const ConvexRational &a2,
                                           const ConvexRational &q, const ConvexRational &b2) {
    if(p>=0 && q<=0)return p==0 && q==0?0:1;
    if(p<=0 && q>=0)return p==0 && q==0?0:-1;
    const ConvexRational left=p*p*b2,right=q*q*a2;
    if(left==right)return 0;
    if(p>0)return left>right?1:-1;
    return left<right?1:-1;
}
} // namespace tpp::detail
