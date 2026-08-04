
#include "vector2.h"
#include "common.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>
#include <CGAL/point_generators_2.h>

using K      = CGAL::Exact_predicates_inexact_constructions_kernel;
using Traits = CGAL::Partition_traits_2<K>;
using Poly   = Traits::Polygon_2;
using Point  = Traits::Point_2;
using PolyList = std::list<Poly>;

using std::vector;


namespace tpp {

	std::vector<std::vector<Vector2>> decompose_polygon(const std::vector<Vector2> &polygon) {

		vector<Point> points;

		for (const auto &v : polygon) {
			points.emplace_back(v.x, v.y);
		}

		Poly poly(points.begin(), points.end());

		if (poly.orientation() == CGAL::CLOCKWISE) {
			poly.reverse_orientation();
		}

		// Ensure the polygon is simple (no self-intersections)
		assert(poly.is_simple());

		PolyList pieces;
		CGAL::optimal_convex_partition_2(poly.vertices_begin(), poly.vertices_end(), std::back_inserter(pieces));

		// Validate the partitioning
		assert(CGAL::convex_partition_is_valid_2(
			poly.vertices_begin(), poly.vertices_end(),
			pieces.begin(), pieces.end()
		));

		vector<vector<Vector2>> result;

		for (const auto& p : pieces) {
		
			vector<Vector2> piece;
		
			for (auto v = p.vertices_begin(); v != p.vertices_end(); ++v) {
				piece.emplace_back(v->x(), v->y());
			}

			result.push_back(std::move(piece));
		}

		return result;
	}

}
