#pragma once

#include "vector2.h"

#include <cstddef>
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
	enum class PerimeterSamplingWorkModel { AdjacentPairs, AllPairs };
	double perimeter_sampling_work_budget(double log2_complexity);
	std::vector<size_t> choose_perimeter_sample_point_counts(
		const std::vector<Polygon> &polygons, double work_budget,
		PerimeterSamplingWorkModel model);
	Polygon evenly_spaced_perimeter_points(const Polygon &polygon, size_t point_count);
	// Exact one-contact minimization over the polygon boundary, with a feasible
	// preferred point retained when it has the same objective (e.g. pass-through).
	Vector2 best_contact(Vector2 left, Vector2 right, const Polygon &polygon, Vector2 preferred);
}
