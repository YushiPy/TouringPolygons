#pragma once

// X4-patterns of the Section 4 algorithm (Chazelle and Dobkin 1985, Lemmas
// 16-21). Every X4-pattern that matters can be taken loose (Lemma 16), and a
// loose one admits the configuration where each notch's arm runs along the
// extension r_v of its edge to the next vertex. The Steiner points are then
// fixed: A = r_k meet r_i joins the notches k, i and B = r_l meet r_j joins l,
// j. For an interval (i..j) the set E(i, j) pairs every notch k outside it with
// a notch l inside it such that (k, i, l, j) is such a pattern; it is computed
// in one pass over the candidates on r_i and r_j.

#include "chains.h"
#include "internal.h"

#include <optional>
#include <vector>

namespace steiner_convex_partition::detail {

	class LooseX4 {
	public:
		explicit LooseX4(const PolygonData &polygon);

		struct Match {
			int outer;  // notch index paired with the interval's first notch
			int inner;  // notch index paired with the interval's last notch
		};

		// E(i, j) for the notches first, ..., first + len - 1 (indices modulo c):
		// for notches k outside, a notch l inside with candidate[l - first] set,
		// such that the loose pattern (k, i, l, j) is valid. A k without a match
		// has no such l.
		std::vector<Match> compute(int first, int len, const std::vector<char> &candidate);

		// The loose pattern (k, i, l, j) checked from scratch, exactly; notch
		// indices in counterclockwise order, A joins k and i, B joins l and j.
		// `direct` scans every edge for the middle edge instead of using the
		// convex chains (for cross-checks).
		bool valid_exactly(int k, int i, int l, int j, bool direct = false) const;

		// Double approximations of A and B, and for each a direction that moves it
		// off the collinear arms into the open angles at its two notches.
		struct Points {
			Vec2 a;
			Vec2 b;
			Vec2 a_shift;
			Vec2 b_shift;
		};
		std::optional<Points> points(int k, int i, int l, int j) const;

		std::size_t intervals = 0;   // E(i, j) with candidates on both rays
		std::size_t separated = 0;   // the boundary cuts every middle edge
		std::size_t fallbacks = 0;   // degenerate quadrilateral, pairs tested one by one
		// Matchings by the linear algorithm of Lemmas 19 and 20 ([0]) and by the
		// general sweep because a claim of the text failed: [1] g1 or g2
		// decreased, [2] a B accepts A's that are not a prefix, [3] the angle at A
		// accepts B's that are not a prefix, [4] the chosen pair has crossing arms.
		std::size_t matchings[5] = {0, 0, 0, 0, 0};

	private:
		struct RPoint {
			exact::HPoint exact;
			long double x = 0.0L;
			long double y = 0.0L;
			long double error = 0.0L;
			bool is_double = false;
			Vec2 v;
		};

		struct Crossing {
			int other = -1;  // r_other meets r_self at point, usable as (other, self)
			RPoint point;
		};

		const PolygonData &polygon_;
		int c_ = 0;
		ConvexChains chains_;
		exact::Frame frame_;
		std::vector<RPoint> vertices_;
		std::vector<RPoint> hits_;                       // by notch: end of r_v
		std::vector<std::vector<Crossing>> crossings_;   // by notch s, sorted along r_s
		std::vector<std::vector<int>> crossing_index_;   // [s][t] into crossings_[s], or -1

		RPoint approximate(const exact::HPoint &h) const;
		const RPoint &vertex(int v) const;
		const RPoint &notch(int t) const;
		const RPoint &after(int t) const;  // the vertex next to notch t
		const Crossing *crossing(int first, int second) const;

		// The line through two rational points, for ConvexChains queries.
		struct RLine {
			const LooseX4 &owner;
			const RPoint &p;
			const RPoint &q;
			int side(int v) const;
			int turn(int u, int v) const;
		};

		static int orient(const RPoint &a, const RPoint &b, const RPoint &c);
		static int cross_sign(const RPoint &a, const RPoint &b, const RPoint &c, const RPoint &d);
		static int dot_sign(const RPoint &a, const RPoint &b, const RPoint &c, const RPoint &d);
		static bool closed_segments_meet(const RPoint &p, const RPoint &q, const RPoint &u, const RPoint &w);
		static bool same_point(const RPoint &a, const RPoint &b);

		RPoint first_contact(int t) const;
		bool arms_disjoint(int k, int i, int l, int j, const RPoint &a, const RPoint &b) const;
		bool middle_edge_clear(const RPoint &a, const RPoint &b, bool direct) const;
	};
}
