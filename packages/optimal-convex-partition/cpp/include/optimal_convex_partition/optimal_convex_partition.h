#pragma once

#include <vector>

namespace optimal_convex_partition {

	struct Point {
		double x = 0.0;
		double y = 0.0;

		constexpr bool operator==(const Point &other) const noexcept {
			return x == other.x && y == other.y;
		}
	};

	using Polygon = std::vector<Point>;
	using Partition = std::vector<Polygon>;

	// Standalone port of CGAL::optimal_convex_partition_2 for simple polygons.
	// The output order intentionally follows CGAL's implementation.
	Partition decompose_polygon(const Polygon &polygon);
}
