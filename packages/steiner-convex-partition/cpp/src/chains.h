#pragma once

// The boundary cut into convex pieces: the convex chains of Lemma 8, each cut
// further where it has turned by half a turn. A piece turns left (or goes
// straight) at every inner vertex and turns less than half a turn in all, so
// the signed distance of its vertices to any line is unimodal: the line meets
// it in at most two places, found by binary search. There are O(c) pieces, so
// a line query costs O(c log n) instead of a scan of the n edges.

#include "internal.h"

#include <vector>

namespace steiner_convex_partition::detail {

	class ConvexChains {
	public:
		explicit ConvexChains(const PolygonData &polygon);

		struct Chain {
			int first = 0;  // first vertex
			int edges = 0;  // the chain is first, ..., first + edges (indices modulo n)
		};

		const std::vector<Chain> &chains() const {
			return chains_;
		}

		// Appends the edges e (edge e joins vertices e and e + 1) that meet a
		// line: every edge crossing it and every edge with an endpoint on it,
		// except the inner edges of a run of consecutive vertices on the line (the
		// edges at both ends of such a run are kept). An edge may come twice.
		//
		// `line.side(v)` is the exact side of vertex v (-1, 0, 1) and
		// `line.turn(u, v)` the sign of the cross product of the line's direction
		// with the vector from vertex u to vertex v.
		//
		// When the chains are many for the polygon's size, scanning is cheaper and
		// every edge is appended: the cost is O(min(n, c log n)).
		template <class Line>
		void edges_meeting(const Line &line, std::vector<int> &out) const {
			if (scan_) {
				for (int e = 0; e < polygon_.n; ++e) out.push_back(e);
				return;
			}
			for (const Chain &chain : chains_) chain_edges(chain, line, out);
		}

		// The line through two double points.
		struct PointLine {
			const PolygonData &polygon;
			Vec2 p;
			Vec2 q;

			int side(int v) const {
				return exact::orient(p, q, polygon.at(v));
			}

			int turn(int u, int v) const {
				return exact::cross_sign(p, q, polygon.at(u), polygon.at(v));
			}
		};

		void edges_meeting_line(const Vec2 &p, const Vec2 &q, std::vector<int> &out) const {
			edges_meeting(PointLine {polygon_, p, q}, out);
		}

	private:
		const PolygonData &polygon_;
		std::vector<Chain> chains_;
		bool scan_ = false;

		template <class Line>
		void chain_edges(const Chain &chain, const Line &line, std::vector<int> &out) const {
			const int k = chain.edges;
			const int first = chain.first;
			auto side = [&](int t) {
				return line.side(polygon_.wrap(first + t));
			};
			auto turn = [&](int t) {
				return line.turn(polygon_.wrap(first + t), polygon_.wrap(first + t + 1));
			};
			auto edge = [&](int t) {
				if (t >= 0 && t < k) out.push_back(polygon_.wrap(first + t));
			};
			// First t in [lo, hi) where a monotone (false, then true) predicate holds.
			auto first_true = [](int lo, int hi, auto holds) {
				while (lo < hi) {
					const int mid = lo + (hi - lo) / 2;
					if (holds(mid)) hi = mid;
					else lo = mid + 1;
				}
				return lo;
			};
			// The edge directions span less than half a turn, so the turns against
			// the line read s..s 0..0 -s..-s (any part may be empty): the side of
			// the vertices is strictly monotone, constant, strictly monotone.
			const int t0 = turn(0);
			const int i1 = first_true(1, k, [&](int t) { return turn(t) != t0; });
			int i2 = i1;
			if (t0 != 0 && i1 < k && turn(i1) == 0) i2 = first_true(i1 + 1, k, [&](int t) { return turn(t) != 0; });
			struct Phase {
				int lo;
				int hi;
				int sign;
			};
			Phase phases[3];
			int count = 0;
			phases[count++] = {0, i1, t0};
			if (i2 > i1) phases[count++] = {i1, i2, 0};
			if (i2 < k) phases[count++] = {i2, k, turn(i2)};
			for (int index = 0; index < count; ++index) {
				const Phase &phase = phases[index];
				if (phase.sign == 0) {
					if (side(phase.lo) != 0) continue;
					// A run of vertices on the line: the edges at both of its ends.
					edge(phase.lo - 1);
					edge(phase.lo);
					edge(phase.hi - 1);
					edge(phase.hi);
					continue;
				}
				// Sides strictly increasing (sign 1) or decreasing over lo..hi: the
				// first vertex on or past the line.
				const int a = first_true(phase.lo, phase.hi + 1, [&](int t) { return side(t) * phase.sign >= 0; });
				if (a > phase.hi) continue;
				const int s = side(a);
				if (s == 0) {
					edge(a - 1);
					edge(a);
				} else if (a > phase.lo) {
					edge(a - 1);
				}
			}
		}
	};
}
