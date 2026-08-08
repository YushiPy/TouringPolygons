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

	// Placeholder for the standalone port of CGAL::optimal_convex_partition_2.
	// Input polygons are expected to be simple. The final implementation should
	// match CGAL's output order exactly for the same input vertex order.
	Partition decompose_polygon(const Polygon &polygon);
}
