#pragma once

// Predicates on homogeneous points (see exact.h), used where points may be
// rational: cut endpoints on edges and the validation of a partition.

#include "exact.h"

#include <cstddef>
#include <string>
#include <vector>

namespace steiner_convex_partition::exact {

	// o strictly between a and b, for collinear points.
	inline bool strictly_between(const HPoint &a, const HPoint &o, const HPoint &b) {
		return dot_sign(o, a, o, b) < 0;
	}

	// o on the closed segment [a, b], for collinear points.
	inline bool on_closed(const HPoint &a, const HPoint &o, const HPoint &b) {
		return dot_sign(o, a, o, b) <= 0;
	}

	inline bool open_segment_meets_closed(const HPoint &p, const HPoint &q, const HPoint &u, const HPoint &w) {
		const int o1 = orient(p, q, u);
		const int o2 = orient(p, q, w);
		if (o1 * o2 > 0) return false;
		if (o1 == 0 && o2 == 0) {
			if (strictly_between(p, u, q) || strictly_between(p, w, q)) return true;
			return on_closed(u, p, w) && on_closed(u, q, w);
		}
		const int o3 = orient(u, w, p);
		const int o4 = orient(u, w, q);
		if (o3 * o4 > 0) return false;
		if (o1 * o2 < 0 && o3 * o4 < 0) return true;
		if (o1 == 0 && strictly_between(p, u, q)) return true;
		if (o2 == 0 && strictly_between(p, w, q)) return true;
		if (o3 == 0 && strictly_between(u, p, w)) return true;
		if (o4 == 0 && strictly_between(u, q, w)) return true;
		return false;
	}

	inline bool closed_segments_meet(const HPoint &p, const HPoint &q, const HPoint &u, const HPoint &w) {
		const int o1 = orient(p, q, u);
		const int o2 = orient(p, q, w);
		const int o3 = orient(u, w, p);
		const int o4 = orient(u, w, q);
		if (o1 * o2 < 0 && o3 * o4 < 0) return true;
		if (o1 == 0 && on_closed(p, u, q)) return true;
		if (o2 == 0 && on_closed(p, w, q)) return true;
		if (o3 == 0 && on_closed(u, p, w)) return true;
		if (o4 == 0 && on_closed(u, q, w)) return true;
		return false;
	}

	// Counterclockwise angle from a - o to b - o is in (0, 180].
	inline bool gap_at_most_half_turn(const HPoint &o, const HPoint &a, const HPoint &b) {
		const int s = orient(o, a, b);
		if (s > 0) return true;
		if (s < 0) return false;
		return dot_sign(o, a, o, b) < 0;
	}

	// Direction o -> target strictly inside the corner (prev, o, next) of a
	// counterclockwise polygon.
	inline bool strictly_inside_corner(const HPoint &prev, const HPoint &o, const HPoint &next, const HPoint &target) {
		const int turn = orient(prev, o, next);
		if (turn > 0) return orient(o, next, target) > 0 && orient(o, target, prev) > 0;
		if (turn == 0) return orient(o, next, target) > 0;
		return !(orient(o, prev, target) >= 0 && orient(o, target, next) >= 0);
	}

	// Weakly convex, counterclockwise, positive area, one revolution.
	inline bool is_weakly_convex(const std::vector<HPoint> &piece, std::string *reason) {
		const std::size_t m = piece.size();
		if (m < 3) {
			if (reason) *reason = "piece with fewer than three vertices";
			return false;
		}
		bool any_turn = false;
		int upward = 0;
		for (std::size_t k = 0; k < m; ++k) {
			const HPoint &a = piece[k];
			const HPoint &b = piece[(k + 1) % m];
			const HPoint &c = piece[(k + 2) % m];
			if (same_point(a, b)) {
				if (reason) *reason = "piece with a repeated vertex";
				return false;
			}
			const int o = orient(a, b, c);
			if (o < 0) {
				if (reason) *reason = "piece with a reflex vertex";
				return false;
			}
			if (o == 0 && dot_sign(a, b, b, c) <= 0) {
				if (reason) *reason = "piece folds back on itself";
				return false;
			}
			if (o > 0) any_turn = true;
			// Count the edge directions that cross from the lower half-plane of
			// directions into the upper one; a convex polygon does it exactly once.
			const BigInt dy1 = b.y * a.w - a.y * b.w;
			const BigInt dx1 = b.x * a.w - a.x * b.w;
			const BigInt dy2 = c.y * b.w - b.y * c.w;
			const BigInt dx2 = c.x * b.w - b.x * c.w;
			const bool lower1 = dy1.sign() < 0 || (dy1.sign() == 0 && dx1.sign() < 0);
			const bool lower2 = dy2.sign() < 0 || (dy2.sign() == 0 && dx2.sign() < 0);
			if (lower1 && !lower2) ++upward;
		}
		if (!any_turn) {
			if (reason) *reason = "piece has zero area";
			return false;
		}
		if (upward != 1) {
			if (reason) *reason = "piece winds more than once";
			return false;
		}
		return true;
	}

	struct ValidationReport {
		bool ok = false;
		bool exact_area = false;
		std::string reason;
	};

	// Exact check that the pieces form a convex partition of the simple polygon
	// `boundary` (counterclockwise).
	ValidationReport validate_partition(const std::vector<HPoint> &boundary, const std::vector<std::vector<HPoint>> &pieces, const Frame &frame);
}
