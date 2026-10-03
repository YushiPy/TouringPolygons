#pragma once
#include "zero_contact_certificate.h"

namespace tpp::detail {
// Shares exact halfplanes and winding conventions with rational_disjoint.cpp.
bool prepare_cycle_polygons(const ConvexRationalPolygons &input,
                            ConvexRationalPolygons &normalized, bool check_disjoint = true);

// Floating proposal filter only. Either answer may be wrong under rounding:
// true must still pass the exact certificate; false retains the ordinary solve.
bool cycle_cutoff_promising(const ConvexRationalPolygons &,const ConvexRationalPolygon &,double);
bool cycle_cutoff_promising(const ConvexRationalPolygons &,const std::vector<Vector2> &,double);

// Shared exact radical sign; kept as the cycle-internal name for callers.
inline int cycle_normalized_difference_sign(const ConvexRational &p, const ConvexRational &a2,
                                           const ConvexRational &q, const ConvexRational &b2) {
    return convex_normalized_difference_sign(p,a2,q,b2);
}
} // namespace tpp::detail
