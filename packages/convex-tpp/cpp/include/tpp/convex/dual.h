#pragma once
#include "tpp/geometry/vec2.h"
#include <vector>

namespace tpp {
// Certified bounds for inserting one region in every gap of an open path.
// Contacts only propose new dual directions; they need not be feasible.
// An empty result declines unsupported arithmetic or a dual hint whose
// unit-disk membership cannot be proved with the interval filter.
std::vector<double> tpp_convex_binary_dual_insertion_bounds(
    Vector2 start,Vector2 target,const std::vector<Vector2> &contacts,
    const std::vector<const std::vector<Vector2> *> &regions,
    const std::vector<Vector2> &inserted,const std::vector<Vector2> &proposals,
    const std::vector<Vector2> &dual);
}
