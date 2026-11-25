#pragma once
#include "zero_contact_certificate.h"

namespace tpp {
class ConvexCycleCertificateGeometry;
struct ConvexCycleCertificateResult;
}
namespace tpp::detail {
// tpp_convex_verify_cycle_certificate without a cutoff, for a caller that
// keeps the candidate only when it is Optimal: the same status, and the
// reporting bounds only for Optimal (otherwise 0 and +infinity).
ConvexCycleCertificateResult cycle_optimality_certificate(
    const ConvexCycleCertificateGeometry &, const ConvexRationalPolygon &);

// Shares exact halfplanes and winding conventions with rational_disjoint.cpp.
bool prepare_cycle_polygons(const ConvexRationalPolygons &input,
                            ConvexRationalPolygons &normalized, bool check_disjoint = true);
// Binary64 regions: the same normalization, checks and output as the
// rational overload on their exact values, decided by exact binary64
// predicates (the 128-bit determinant, rationals only when it cannot).
// binary, when given, receives the same normalized regions in binary64.
bool prepare_cycle_polygons(const std::vector<std::vector<Vector2>> &input,
                            ConvexRationalPolygons &normalized, bool check_disjoint = true,
                            std::vector<std::vector<Vector2>> *binary = nullptr);
// Normalized regions (prepare_cycle_polygons): whether they are polygons of
// positive area that are pairwise strictly separated.
bool cycle_polygons_disjoint(const ConvexRationalPolygons &normalized);
// The same test on normalized binary64 regions, by exact binary64 predicates.
bool cycle_polygons_disjoint(const std::vector<std::vector<Vector2>> &normalized);
// One binary64 region, as prepare_cycle_polygons (strict: without collinear
// vertices) or the certificate geometry (keeping them) normalizes it: no
// repeated vertices, counterclockwise. None, or what makes it invalid.
enum class BinaryRegionDefect { None, Empty, ZeroArea, NotConvex, Backtracks, Winding };
BinaryRegionDefect binary_convex_region(const std::vector<Vector2> &input, std::vector<Vector2> &region, bool strict);

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
