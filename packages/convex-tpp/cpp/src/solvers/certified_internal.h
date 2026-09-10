#pragma once

#include "tpp/convex/certified.h"

#include <vector>

namespace tpp::certified_detail {
	using Polygon = std::vector<Vector2>;

	double path_length(const Polygon &path);
	double dual_bound(const Polygon &contacts, const std::vector<Polygon> &polygons, double smoothing = 0);
	bool recover_contacts(const Polygon &path, const std::vector<Polygon> &polygons, Polygon &contacts);
	bool repair_contacts(const Polygon &path, const std::vector<Polygon> &polygons, Polygon &contacts);

	CertifiedConvexTppResult refine_long_double(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &polygons,
		double tolerance, double scale, double safety, double cutoff, CertifiedConvexTppResult result
	);
	CertifiedConvexTppResult refine_extended_precision(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &polygons,
		double tolerance, double scale, double safety, double cutoff, CertifiedConvexTppResult result
	);
}
