#pragma once

#include "vector2.h"

#include <vector>

namespace tpp::unordered_detail {
	using Polygon = std::vector<Vector2>;

	struct Contact {
		double distance;
		double position;
	};

	double path_length(const Polygon &path);
	Polygon convex_hull(Polygon polygon);
	Contact contact(const Polygon &path, const Polygon &polygon, double tolerance);
}
