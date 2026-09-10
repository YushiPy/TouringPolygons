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
	// Exact one-contact minimization over the polygon boundary, with a feasible
	// preferred point retained when it has the same objective (e.g. pass-through).
	Vector2 best_contact(Vector2 left, Vector2 right, const Polygon &polygon, Vector2 preferred);
}
