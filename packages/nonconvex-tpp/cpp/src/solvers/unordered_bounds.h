#pragma once

#include "unordered_geometry.h"

namespace tpp::unordered_detail {
	// Lower bounds for all insertion positions. References may be infeasible;
	// only the fixed endpoints and the ordered region definitions matter.
	std::vector<double> insertion_lower_bounds(const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const Polygon &inserted);
}
