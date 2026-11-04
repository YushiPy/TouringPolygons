#pragma once

// Superranges (Chazelle and Dobkin 1985, Section 4.2): what a notch sees,
// compressed to the directions of the pseudo-notches. Between two consecutive
// such directions the visible boundary is a portion of a single convex chain,
// so a point is located by two binary searches and tested against one edge.

#include "chains.h"
#include "internal.h"

#include <vector>

namespace steiner_convex_partition::detail {

	// Endpoints of the maximal convex chains between consecutive notches
	// (Lemma 8): every chain with its chord is a convex polygon. All notches are
	// pseudo-notches; there are at most 2(c + 1).
	std::vector<int> pseudo_notches(const PolygonData &polygon);

	// A point of the boundary in the forward order from a notch v: 2 * (u - v) for
	// a vertex u, 2 * (e - v) + 1 for a point inside edge (e, e + 1), indices
	// taken modulo n. -1 means no hit.
	using Offset = int;

	class Superranges {
	public:
		Superranges(const PolygonData &polygon, const std::vector<int> &pseudo);

		// Is the double point s strictly visible from notch vertex v: the open
		// segment (v, s) and s in the interior of the polygon?
		bool sees(int v, const Vec2 &s) const;

		// Is notch vertex b strictly visible from notch vertex a (a proper
		// diagonal)?
		bool sees_vertex(int a, int b) const;

		// The boundary from notch vertex a forward to notch vertex b: the target of
		// the most counterclockwise direction from a toward a visible point of it,
		// when b is not visible from a (-1 otherwise). A Steiner point joined to a
		// and b must be strictly counterclockwise of it.
		int last_direction(int a, int b) const;

		// The same boundary seen from b: the target of the most clockwise
		// direction from b toward a visible point of it, when a is not visible from
		// b (-1 otherwise). A Steiner point must be strictly clockwise of it.
		int first_direction(int a, int b) const;

		std::size_t directions() const;

	private:
		struct Direction {
			int target = -1;      // vertex the ray points at
			Offset contact = -1;  // first boundary point on the exact ray
			int contact_vertex = -1;
			int contact_edge = -1;
			Offset before = -1;   // hit of a ray turned slightly clockwise
			int before_edge = -1;
			Offset after = -1;    // hit of a ray turned slightly counterclockwise
			int after_edge = -1;
		};

		struct Range {
			int vertex = -1;
			// Directions in counterclockwise order: the edge to the next vertex,
			// the pseudo-notches strictly inside the corner, the edge to the
			// previous vertex.
			std::vector<Direction> directions;
		};

		const PolygonData &polygon_;
		ConvexChains chains_;
		std::vector<Range> ranges_;  // by notch index
		// last_direction and first_direction for every ordered pair of notches
		// (Lemma 14 needs them in constant time), by [a * c + b] in notch indices.
		std::vector<int> last_;
		std::vector<int> first_;

		int compute_last_direction(int a, int b) const;
		int compute_first_direction(int a, int b) const;

		const Range &range_of(int vertex) const;
		Offset vertex_offset(int v, int u) const;
		Offset edge_offset(int v, int e) const;
		// Counterclockwise order of directions from v inside its corner, starting
		// at the edge to the next vertex: negative, zero or positive.
		int compare_directions(int v, const Vec2 &p, const Vec2 &q) const;
		Direction trace(int v, int target, bool want_before, bool want_after) const;
		bool sees_in_cell(int v, const Range &range, std::size_t cell, const Vec2 &s) const;
	};
}
