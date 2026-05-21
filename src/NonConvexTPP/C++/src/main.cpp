
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>
#include <CGAL/point_generators_2.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>

using K      = CGAL::Exact_predicates_inexact_constructions_kernel;
using Traits = CGAL::Partition_traits_2<K>;
using Poly   = Traits::Polygon_2;
using Point  = Traits::Point_2;
using PolyList = std::list<Poly>;

int main() {
	// Simple concave polygon (star/arrow shape), CCW orientation required
	std::vector<Point> pts = {
		{0, 4}, {-1, 1}, {-4, 0}, {-1, -1},
		{0, -4}, {1, -1}, {4, 0}, {1, 1},
	};

	Poly poly(pts.begin(), pts.end());

	// CGAL requires CCW; flip if needed
	if (poly.orientation() == CGAL::CLOCKWISE)
		poly.reverse_orientation();

	assert(poly.is_simple());
	std::cout << "Input: " << poly.size() << " vertices\n";

	// Run Hertel-Mehlhorn approximation (O(n), ≤4× optimal pieces)
	// Alternatives: greene_approx_convex_partition_2, optimal_convex_partition_2
	PolyList pieces;
	//CGAL::approx_convex_partition_2(poly.vertices_begin(), poly.vertices_end(), std::back_inserter(pieces));
	//CGAL::greene_approx_convex_partition_2(poly.vertices_begin(), poly.vertices_end(), std::back_inserter(pieces));
	CGAL::optimal_convex_partition_2(poly.vertices_begin(), poly.vertices_end(), std::back_inserter(pieces));

	// Validate
	assert(CGAL::convex_partition_is_valid_2(
		poly.vertices_begin(), poly.vertices_end(),
		pieces.begin(), pieces.end()
	));

	std::cout << "Decomposed into " << pieces.size() << " convex pieces:\n";
	int i = 1;
	for (const auto& p : pieces) {
		std::cout << "  Piece " << i++ << ":";
		for (auto v = p.vertices_begin(); v != p.vertices_end(); ++v)
			std::cout << " (" << v->x() << "," << v->y() << ")";
		std::cout << "\n";
	}
}