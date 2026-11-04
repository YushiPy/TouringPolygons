#include "chains.h"

#include <cmath>

namespace steiner_convex_partition::detail {

	ConvexChains::ConvexChains(const PolygonData &polygon)
		: polygon_(polygon) {
		const int n = polygon.n;
		const int start = polygon.c > 0 ? polygon.notch_vertex.front() : 0;
		Chain current {start, 0};
		for (int s = 0; s < n; ++s) {
			const int e = start + s;
			if (current.edges > 0) {
				// Edge e (from vertex e) extends the chain if the chain turns left or
				// goes straight at e and stays within half a turn of its first edge.
				const Vec2 &a0 = polygon.at(current.first);
				const Vec2 &b0 = polygon.at(current.first + 1);
				const Vec2 &a = polygon.at(e);
				const Vec2 &b = polygon.at(e + 1);
				const bool left = exact::orient(polygon.at(e - 1), a, b) >= 0;
				const int turned = exact::cross_sign(a0, b0, a, b);
				const bool within = turned > 0 || (turned == 0 && exact::dot_sign(a0, b0, a, b) > 0);
				if (!left || !within) {
					chains_.push_back(current);
					current = {polygon.wrap(e), 0};
				}
			}
			++current.edges;
		}
		chains_.push_back(current);
		// A chain query evaluates about three binary searches per chain; a scan
		// evaluates two signs per edge.
		const double per_chain = 3.0 * std::log2(1.0 + static_cast<double>(n) / static_cast<double>(chains_.size())) + 2.0;
		scan_ = per_chain * static_cast<double>(chains_.size()) > 2.0 * static_cast<double>(n);
	}
}
