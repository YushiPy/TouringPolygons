#include "loose_x4.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace steiner_convex_partition::detail {

	namespace {
		// Bound on the relative error of a rational point's long double
		// approximation (Frame::to_long_double keeps 64 bits of each term).
		constexpr long double approximation_error = 1.0e-17L;
		// Bound on the rounding of a predicate evaluated in long double, relative
		// to the squared magnitude of its inputs.
		constexpr long double evaluation_error = 1.0e-17L;

		// The index range [first, last] of `count` points on a line where a
		// half-plane predicate holds; empty when last < first.
		template <class Predicate>
		std::pair<int, int> half_line(int count, Predicate holds) {
			if (count <= 0) return {0, -1};
			const bool front = holds(0);
			const bool back = holds(count - 1);
			if (front && back) return {0, count - 1};
			if (!front && !back) return {0, -1};
			int lo = 0;
			int hi = count - 1;
			while (hi - lo > 1) {
				const int mid = (lo + hi) / 2;
				if (holds(mid) == front) lo = mid;
				else hi = mid;
			}
			return front ? std::pair {0, lo} : std::pair {hi, count - 1};
		}
	}

	LooseX4::RPoint LooseX4::approximate(const exact::HPoint &h) const {
		RPoint p;
		p.exact = h;
		p.x = frame_.to_long_double(h.x, h.w);
		p.y = frame_.to_long_double(h.y, h.w);
		p.error = approximation_error * (std::fabs(p.x) + std::fabs(p.y));
		return p;
	}

	const LooseX4::RPoint &LooseX4::vertex(int v) const {
		return vertices_[static_cast<std::size_t>(polygon_.wrap(v))];
	}

	const LooseX4::RPoint &LooseX4::notch(int t) const {
		return vertex(polygon_.notch_vertex[static_cast<std::size_t>(t)]);
	}

	const LooseX4::RPoint &LooseX4::after(int t) const {
		return vertex(polygon_.notch_vertex[static_cast<std::size_t>(t)] + 1);
	}

	const LooseX4::Crossing *LooseX4::crossing(int first, int second) const {
		const int index = crossing_index_[static_cast<std::size_t>(second)][static_cast<std::size_t>(first)];
		return index < 0 ? nullptr : &crossings_[static_cast<std::size_t>(second)][static_cast<std::size_t>(index)];
	}

	int LooseX4::orient(const RPoint &a, const RPoint &b, const RPoint &c) {
		if (a.is_double && b.is_double && c.is_double) return exact::orient(a.v, b.v, c.v);
		const long double bx = b.x - a.x;
		const long double by = b.y - a.y;
		const long double cx = c.x - a.x;
		const long double cy = c.y - a.y;
		const long double det = bx * cy - by * cx;
		const long double eab = a.error + b.error;
		const long double eac = a.error + c.error;
		const long double scale = std::fabs(a.x) + std::fabs(a.y) + std::fabs(b.x) + std::fabs(b.y) + std::fabs(c.x) + std::fabs(c.y);
		const long double bound = (std::fabs(bx) + std::fabs(by)) * eac + (std::fabs(cx) + std::fabs(cy)) * eab
			+ 2.0L * eab * eac + evaluation_error * scale * scale;
		if (det > bound) return 1;
		if (det < -bound) return -1;
		return exact::orient(a.exact, b.exact, c.exact);
	}

	// sign((b - a) x (d - c)).
	int LooseX4::cross_sign(const RPoint &a, const RPoint &b, const RPoint &c, const RPoint &d) {
		if (a.is_double && b.is_double && c.is_double && d.is_double) return exact::cross_sign(a.v, b.v, c.v, d.v);
		const long double ux = b.x - a.x;
		const long double uy = b.y - a.y;
		const long double wx = d.x - c.x;
		const long double wy = d.y - c.y;
		const long double value = ux * wy - uy * wx;
		const long double eu = a.error + b.error;
		const long double ew = c.error + d.error;
		const long double scale = std::fabs(a.x) + std::fabs(a.y) + std::fabs(b.x) + std::fabs(b.y)
			+ std::fabs(c.x) + std::fabs(c.y) + std::fabs(d.x) + std::fabs(d.y);
		const long double bound = (std::fabs(ux) + std::fabs(uy)) * ew + (std::fabs(wx) + std::fabs(wy)) * eu
			+ 2.0L * eu * ew + evaluation_error * scale * scale;
		if (value > bound) return 1;
		if (value < -bound) return -1;
		const exact::HPoint &ha = a.exact;
		const exact::HPoint &hb = b.exact;
		const exact::HPoint &hc = c.exact;
		const exact::HPoint &hd = d.exact;
		const exact::BigInt bx = hb.x * ha.w - ha.x * hb.w;
		const exact::BigInt by = hb.y * ha.w - ha.y * hb.w;
		const exact::BigInt dx = hd.x * hc.w - hc.x * hd.w;
		const exact::BigInt dy = hd.y * hc.w - hc.y * hd.w;
		return (bx * dy - by * dx).sign();
	}

	int LooseX4::RLine::side(int v) const {
		return orient(p, q, owner.vertex(v));
	}

	int LooseX4::RLine::turn(int u, int v) const {
		return cross_sign(p, q, owner.vertex(u), owner.vertex(v));
	}

	int LooseX4::dot_sign(const RPoint &a, const RPoint &b, const RPoint &c, const RPoint &d) {
		if (a.is_double && b.is_double && c.is_double && d.is_double) return exact::dot_sign(a.v, b.v, c.v, d.v);
		const long double ux = b.x - a.x;
		const long double uy = b.y - a.y;
		const long double wx = d.x - c.x;
		const long double wy = d.y - c.y;
		const long double value = ux * wx + uy * wy;
		const long double eu = a.error + b.error;
		const long double ew = c.error + d.error;
		const long double scale = std::fabs(a.x) + std::fabs(a.y) + std::fabs(b.x) + std::fabs(b.y)
			+ std::fabs(c.x) + std::fabs(c.y) + std::fabs(d.x) + std::fabs(d.y);
		const long double bound = (std::fabs(ux) + std::fabs(uy)) * ew + (std::fabs(wx) + std::fabs(wy)) * eu
			+ 2.0L * eu * ew + evaluation_error * scale * scale;
		if (value > bound) return 1;
		if (value < -bound) return -1;
		return exact::dot_sign(a.exact, b.exact, c.exact, d.exact);
	}

	bool LooseX4::same_point(const RPoint &a, const RPoint &b) {
		if (a.is_double && b.is_double) return a.v == b.v;
		const long double slack = a.error + b.error + evaluation_error * (std::fabs(a.x) + std::fabs(a.y) + std::fabs(b.x) + std::fabs(b.y));
		if (std::fabs(a.x - b.x) > slack || std::fabs(a.y - b.y) > slack) return false;
		return exact::same_point(a.exact, b.exact);
	}

	bool LooseX4::closed_segments_meet(const RPoint &p, const RPoint &q, const RPoint &u, const RPoint &w) {
		auto on_closed = [](const RPoint &a, const RPoint &o, const RPoint &b) {
			return dot_sign(o, a, o, b) <= 0;
		};
		const int o1 = orient(p, q, u);
		const int o2 = orient(p, q, w);
		if (o1 * o2 > 0) return false;
		const int o3 = orient(u, w, p);
		const int o4 = orient(u, w, q);
		if (o3 * o4 > 0) return false;
		if (o1 * o2 < 0 && o3 * o4 < 0) return true;
		if (o1 == 0 && on_closed(p, u, q)) return true;
		if (o2 == 0 && on_closed(p, w, q)) return true;
		if (o3 == 0 && on_closed(u, p, w)) return true;
		if (o4 == 0 && on_closed(u, q, w)) return true;
		return false;
	}

	// First boundary point on the ray from notch t away from its next vertex:
	// the end of r_t.
	LooseX4::RPoint LooseX4::first_contact(int t) const {
		const int n = polygon_.n;
		const int v = polygon_.notch_vertex[static_cast<std::size_t>(t)];
		const RPoint &o = vertex(v);
		const RPoint &u = vertex(v + 1);
		const RPoint *best = nullptr;
		std::vector<const RPoint *> candidates;
		auto ahead = [&](const Vec2 &x) {
			return exact::dot_sign(u.v, o.v, o.v, x) > 0;
		};
		std::vector<int> crossed_edges;
		std::vector<int> edges;
		chains_.edges_meeting_line(u.v, o.v, edges);
		std::sort(edges.begin(), edges.end());
		edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
		for (int e : edges) {
			if (e == v || (e + 1) % n == v) continue;
			const RPoint &a = vertex(e);
			const RPoint &b = vertex(e + 1);
			const int oa = exact::orient(u.v, o.v, a.v);
			const int ob = exact::orient(u.v, o.v, b.v);
			if (oa == 0 && ahead(a.v)) candidates.push_back(&a);
			if (ob == 0 && ahead(b.v)) candidates.push_back(&b);
			if (oa * ob < 0) {
				const int along = exact::cross_sign(u.v, o.v, a.v, b.v);
				if (along != 0 && exact::orient(o.v, a.v, b.v) == along) crossed_edges.push_back(e);
			}
		}
		std::vector<RPoint> crossings;
		crossings.reserve(crossed_edges.size());
		for (int e : crossed_edges) {
			crossings.push_back(approximate(exact::line_intersection(u.exact, o.exact, vertex(e).exact, vertex(e + 1).exact)));
		}
		for (const RPoint &p : crossings) candidates.push_back(&p);
		for (const RPoint *p : candidates) {
			// p strictly nearer to o than best along the ray.
			if (best == nullptr || dot_sign(*p, *best, u, o) > 0) best = p;
		}
		// A notch always meets the boundary ahead: its corner is reflex, so the
		// ray starts inside the polygon.
		return best != nullptr ? *best : o;
	}

	LooseX4::LooseX4(const PolygonData &polygon)
		: polygon_(polygon), c_(polygon.c), chains_(polygon) {
		for (const Vec2 &p : polygon.pts) frame_.include(p);
		vertices_.resize(static_cast<std::size_t>(polygon.n));
		for (int v = 0; v < polygon.n; ++v) {
			RPoint &p = vertices_[static_cast<std::size_t>(v)];
			p.v = polygon.pts[static_cast<std::size_t>(v)];
			p.x = p.v.x;
			p.y = p.v.y;
			p.is_double = true;
			p.exact = frame_.point(p.v);
		}
		const std::size_t cs = static_cast<std::size_t>(c_);
		hits_.reserve(cs);
		for (int t = 0; t < c_; ++t) hits_.push_back(first_contact(t));

		crossings_.assign(cs, {});
		crossing_index_.assign(cs, std::vector<int>(cs, -1));
		for (int s = 0; s < c_; ++s) {
			const RPoint &vs = notch(s);
			const RPoint &us = after(s);
			std::vector<Crossing> &list = crossings_[static_cast<std::size_t>(s)];
			for (int t = 0; t < c_; ++t) {
				if (t == s) continue;
				const RPoint &vt = notch(t);
				const RPoint &ut = after(t);
				// Usable as (t, s): v_t strictly left of r_s, v_s strictly right of r_t,
				// and the rays meet ahead of both notches.
				if (exact::orient(us.v, vs.v, vt.v) <= 0) continue;
				if (exact::orient(ut.v, vt.v, vs.v) >= 0) continue;
				if (exact::cross_sign(us.v, vs.v, ut.v, vt.v) >= 0) continue;
				RPoint point = approximate(exact::line_intersection(us.exact, vs.exact, ut.exact, vt.exact));
				if (dot_sign(point, hits_[static_cast<std::size_t>(s)], us, vs) <= 0) continue;
				if (dot_sign(point, hits_[static_cast<std::size_t>(t)], ut, vt) <= 0) continue;
				list.push_back({t, std::move(point)});
			}
			std::sort(list.begin(), list.end(), [&](const Crossing &x, const Crossing &y) {
				const int along = dot_sign(x.point, y.point, us, vs);
				if (along != 0) return along > 0;
				return x.other < y.other;
			});
			for (std::size_t index = 0; index < list.size(); ++index) {
				crossing_index_[static_cast<std::size_t>(s)][static_cast<std::size_t>(list[index].other)] = static_cast<int>(index);
			}
		}
	}

	bool LooseX4::arms_disjoint(int k, int i, int l, int j, const RPoint &a, const RPoint &b) const {
		for (int x : {k, i}) {
			for (int y : {l, j}) {
				if (closed_segments_meet(notch(x), a, notch(y), b)) return false;
			}
		}
		return true;
	}

	bool LooseX4::middle_edge_clear(const RPoint &a, const RPoint &b, bool direct) const {
		const long double slack = a.error + b.error;
		const long double min_x = std::min(a.x, b.x) - slack;
		const long double max_x = std::max(a.x, b.x) + slack;
		const long double min_y = std::min(a.y, b.y) - slack;
		const long double max_y = std::max(a.y, b.y) + slack;
		auto strictly_between = [](const RPoint &p, const RPoint &o, const RPoint &q) {
			return dot_sign(o, p, o, q) < 0;
		};
		// Only edges meeting the line through a and b can meet the segment.
		std::vector<int> edges;
		if (direct) {
			for (int e = 0; e < polygon_.n; ++e) edges.push_back(e);
		} else {
			chains_.edges_meeting(RLine {*this, a, b}, edges);
		}
		for (int e : edges) {
			const RPoint &u = vertex(e);
			const RPoint &w = vertex(e + 1);
			if (std::max(u.x, w.x) < min_x || std::min(u.x, w.x) > max_x) continue;
			if (std::max(u.y, w.y) < min_y || std::min(u.y, w.y) > max_y) continue;
			// Does the open segment (a, b) meet the closed edge [u, w]?
			const int o1 = orient(a, b, u);
			const int o2 = orient(a, b, w);
			if (o1 * o2 > 0) continue;
			if (o1 == 0 && o2 == 0) {
				if (strictly_between(a, u, b) || strictly_between(a, w, b)) return false;
				if (dot_sign(a, u, a, w) <= 0 && dot_sign(b, u, b, w) <= 0) return false;
				continue;
			}
			const int o3 = orient(u, w, a);
			const int o4 = orient(u, w, b);
			if (o3 * o4 > 0) continue;
			if (o1 * o2 < 0 && o3 * o4 < 0) return false;
			if (o1 == 0 && strictly_between(a, u, b)) return false;
			if (o2 == 0 && strictly_between(a, w, b)) return false;
			if (o3 == 0 && strictly_between(u, a, w)) return false;
			if (o4 == 0 && strictly_between(u, b, w)) return false;
		}
		return true;
	}

	bool LooseX4::valid_exactly(int k, int i, int l, int j, bool direct) const {
		const Crossing *ca = crossing(k, i);
		const Crossing *cb = crossing(l, j);
		if (ca == nullptr || cb == nullptr) return false;
		const RPoint &a = ca->point;
		const RPoint &b = cb->point;
		if (same_point(a, b)) return false;
		const RPoint &vk = notch(k);
		const RPoint &vi = notch(i);
		const RPoint &vl = notch(l);
		const RPoint &vj = notch(j);
		// Each Steiner point in the closed triangle of its three neighbors: all
		// its angles are at most a half turn.
		if (orient(vk, vi, b) <= 0) return false;
		if (orient(vk, vi, a) < 0 || orient(vi, b, a) < 0 || orient(b, vk, a) < 0) return false;
		if (orient(vl, vj, a) <= 0) return false;
		if (orient(vl, vj, b) < 0 || orient(vj, a, b) < 0 || orient(a, vl, b) < 0) return false;
		if (!arms_disjoint(k, i, l, j, a, b)) return false;
		return middle_edge_clear(a, b, direct);
	}

	std::optional<LooseX4::Points> LooseX4::points(int k, int i, int l, int j) const {
		const Crossing *ca = crossing(k, i);
		const Crossing *cb = crossing(l, j);
		if (ca == nullptr || cb == nullptr) return std::nullopt;
		auto right_normal = [&](int t) {
			const Vec2 &v = notch(t).v;
			const Vec2 &u = after(t).v;
			const long double dx = static_cast<long double>(v.x) - u.x;
			const long double dy = static_cast<long double>(v.y) - u.y;
			const long double length = std::hypot(dx, dy);
			return std::pair {dy / length, -dx / length};
		};
		auto shift = [&](int x, int y) {
			const auto [ax, ay] = right_normal(x);
			const auto [bx, by] = right_normal(y);
			return Vec2 {static_cast<double>(ax + bx), static_cast<double>(ay + by)};
		};
		Points result;
		result.a = {static_cast<double>(ca->point.x), static_cast<double>(ca->point.y)};
		result.b = {static_cast<double>(cb->point.x), static_cast<double>(cb->point.y)};
		result.a_shift = shift(k, i);
		result.b_shift = shift(l, j);
		return result;
	}

	std::vector<LooseX4::Match> LooseX4::compute(int first, int len, const std::vector<char> &candidate) {
		std::vector<Match> result;
		const int c = c_;
		const int i = ((first % c) + c) % c;
		const int j = (i + len - 1) % c;
		auto offset = [&](int t) {
			return ((t - i) % c + c) % c;
		};

		// Candidates on r_i: outer notches k, with A weakly right of r_j (the angle
		// at B between its arm to v_j and the middle edge is at most a half turn).
		std::vector<const Crossing *> as;
		for (const Crossing &cr : crossings_[static_cast<std::size_t>(i)]) {
			if (offset(cr.other) < len) continue;
			if (orient(after(j), notch(j), cr.point) > 0) continue;
			as.push_back(&cr);
		}
		if (as.empty()) return result;
		// Candidates on r_j: inner notches l allowed by relation (2), with B
		// weakly right of r_i.
		std::vector<const Crossing *> bs;
		for (const Crossing &cr : crossings_[static_cast<std::size_t>(j)]) {
			const int d = offset(cr.other);
			if (d < 1 || d > len - 2 || !candidate[static_cast<std::size_t>(d)]) continue;
			if (orient(after(i), notch(i), cr.point) > 0) continue;
			bs.push_back(&cr);
		}
		if (bs.empty()) return result;
		++intervals;
		const int p = static_cast<int>(as.size());
		const int q = static_cast<int>(bs.size());

		auto pairwise = [&] {
			++fallbacks;
			for (const Crossing *ca : as) {
				for (const Crossing *cb : bs) {
					if (valid_exactly(ca->other, i, cb->other, j)) {
						result.push_back({ca->other, cb->other});
						break;
					}
				}
			}
			return result;
		};

		// The quadrilateral A_1 A_p B_1 B_q holding every middle edge.
		const RPoint &a1 = as.front()->point;
		const RPoint &ap = as.back()->point;
		const RPoint &b1 = bs.front()->point;
		const RPoint &bq = bs.back()->point;
		std::vector<const RPoint *> corners;
		for (const RPoint *x : {&a1, &ap, &b1, &bq}) {
			if (!corners.empty() && same_point(*corners.back(), *x)) continue;
			corners.push_back(x);
		}
		if (corners.size() > 1 && same_point(*corners.back(), *corners.front())) corners.pop_back();
		if (corners.size() < 3) return pairwise();
		const int sigma = orient(*corners[0], *corners[1], *corners[2]);
		if (sigma == 0) return pairwise();
		const std::size_t m = corners.size();
		for (std::size_t t = 0; t < m; ++t) {
			if (orient(*corners[t], *corners[(t + 1) % m], *corners[(t + 2) % m]) != sigma) return pairwise();
		}
		auto in_quadrilateral = [&](const RPoint &w) {
			for (std::size_t t = 0; t < m; ++t) {
				if (orient(*corners[t], *corners[(t + 1) % m], w) == -sigma) return false;
			}
			return true;
		};

		// Walk the boundary from v_i (outside the quadrilateral) and sort the
		// pieces inside it by the free side they hang from: X = B_q A_1 or
		// Y = A_p B_1 (the boundary never meets r_i or r_j there). A piece touching
		// both cuts every middle edge; otherwise a middle edge is clear iff the
		// vertices of X-pieces lie strictly on X's side of it and those of
		// Y-pieces strictly on Y's side (the safety chains C and D are their
		// hulls).
		const int n = polygon_.n;
		const int start = polygon_.notch_vertex[static_cast<std::size_t>(i)];
		if (in_quadrilateral(vertex(start))) return pairwise();
		// Events in boundary order from v_i: the edges meeting X or Y (found
		// through the convex chains) and the chain ends inside the quadrilateral.
		// Between events the boundary stays on one side of the quadrilateral, and
		// every notch is a chain end, so this is the walk restricted to what the
		// safety region depends on.
		std::vector<int> candidates;
		chains_.edges_meeting(RLine {*this, bq, a1}, candidates);
		chains_.edges_meeting(RLine {*this, ap, b1}, candidates);
		std::sort(candidates.begin(), candidates.end());
		candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
		struct Event {
			int key;     // 2 * offset of a vertex, 2 * offset + 1 of an edge
			int index;   // vertex or edge
			int sides;   // for an edge: 1 meets X, 2 meets Y
		};
		auto position = [&](int v) {
			return ((v - start) % n + n) % n;
		};
		std::vector<Event> events;
		for (int e : candidates) {
			const RPoint &a = vertex(e);
			const RPoint &b = vertex(e + 1);
			const int sides = (closed_segments_meet(a, b, bq, a1) ? 1 : 0) | (closed_segments_meet(a, b, ap, b1) ? 2 : 0);
			if (sides != 0) events.push_back({2 * position(e) + 1, e, sides});
		}
		for (const ConvexChains::Chain &chain : chains_.chains()) {
			if (in_quadrilateral(vertex(chain.first))) events.push_back({2 * position(chain.first), chain.first, 0});
		}
		std::sort(events.begin(), events.end(), [](const Event &x, const Event &y) {
			return x.key < y.key;
		});
		std::vector<Vec2> x_points;
		std::vector<Vec2> y_points;
		std::vector<int> piece;
		int sides = 0;
		bool in_piece = false;
		for (const Event &event : events) {
			if (event.key % 2 == 0) {
				if (!in_piece) return pairwise();
				piece.push_back(event.index);
				continue;
			}
			in_piece = true;
			sides |= event.sides;
			if (in_quadrilateral(vertex(event.index + 1))) continue;
			if (sides == 3) {
				++separated;
				return result;
			}
			std::vector<Vec2> &target = sides == 1 ? x_points : y_points;
			for (int w : piece) target.push_back(polygon_.pts[static_cast<std::size_t>(w)]);
			piece.clear();
			sides = 0;
			in_piece = false;
		}
		if (in_piece) return pairwise();

		// Obstacle hulls as their lower and upper chains: a linear function is
		// unimodal along each, so its minimum over a hull takes O(log h) signs.
		struct Hull {
			std::vector<RPoint> chain[2];
		};
		auto build_hull = [&](std::vector<Vec2> points) {
			Hull result;
			std::sort(points.begin(), points.end(), [](const Vec2 &a, const Vec2 &b) {
				return a.x < b.x || (a.x == b.x && a.y < b.y);
			});
			points.erase(std::unique(points.begin(), points.end()), points.end());
			for (int side = 0; side < 2; ++side) {
				std::vector<Vec2> h;
				for (std::size_t t = 0; t < points.size(); ++t) {
					const Vec2 &x = side == 0 ? points[t] : points[points.size() - 1 - t];
					while (h.size() >= 2 && exact::orient(h[h.size() - 2], h.back(), x) <= 0) h.pop_back();
					h.push_back(x);
				}
				for (const Vec2 &x : h) {
					RPoint r;
					r.v = x;
					r.x = x.x;
					r.y = x.y;
					r.is_double = true;
					r.exact = frame_.point(x);
					result.chain[side].push_back(std::move(r));
				}
			}
			return result;
		};
		const Hull x_hull = build_hull(std::move(x_points));
		const Hull y_hull = build_hull(std::move(y_points));
		// Are all points of the hull strictly on side `side` of the line a -> b?
		// The minimum of side * orient(a, b, .) over a chain is at its first
		// vertex, its last one, or where its decreasing run ends.
		auto all_on_side = [&](const Hull &hull, const RPoint &a, const RPoint &b, int side) {
			for (const std::vector<RPoint> &chain : hull.chain) {
				const int m = static_cast<int>(chain.size());
				if (m == 0) continue;
				int lo = 0;
				int hi = m - 1;
				while (lo < hi) {
					const int mid = lo + (hi - lo) / 2;
					if (side * cross_sign(a, b, chain[static_cast<std::size_t>(mid)], chain[static_cast<std::size_t>(mid) + 1]) >= 0) hi = mid;
					else lo = mid + 1;
				}
				for (int t : {0, lo, m - 1}) {
					if (orient(a, b, chain[static_cast<std::size_t>(t)]) != side) return false;
				}
			}
			return true;
		};

		auto alpha = [&](int ka, int lb) {
			const int k = as[static_cast<std::size_t>(ka)]->other;
			return orient(after(k), notch(k), bs[static_cast<std::size_t>(lb)]->point) >= 0;
		};
		auto beta = [&](int lb, int ka) {
			const int l = bs[static_cast<std::size_t>(lb)]->other;
			return orient(after(l), notch(l), as[static_cast<std::size_t>(ka)]->point) >= 0;
		};
		auto below_x = [&](int ka, int lb) {
			return all_on_side(x_hull, as[static_cast<std::size_t>(ka)]->point, bs[static_cast<std::size_t>(lb)]->point, sigma);
		};
		auto above_y = [&](int ka, int lb) {
			return all_on_side(y_hull, as[static_cast<std::size_t>(ka)]->point, bs[static_cast<std::size_t>(lb)]->point, -sigma);
		};
		auto acceptable = [&](int ka, int lb) {
			const Crossing *ca = as[static_cast<std::size_t>(ka)];
			const Crossing *cb = bs[static_cast<std::size_t>(lb)];
			return !same_point(ca->point, cb->point) && arms_disjoint(ca->other, i, cb->other, j, ca->point, cb->point);
		};

		// Lemmas 19 and 20 in linear time. For a fixed A, the B's whose middle
		// edge passes the X-obstacles are a prefix and those passing the
		// Y-obstacles a suffix (the two sides of the quadrilateral grow and shrink
		// monotonically); the text's claims are that both bounds (g2 and g1)
		// never decrease along r_i, that each B accepts a prefix of the A's (A
		// before f(B)) and that the angle at A accepts a prefix of the B's. Then
		// x(A) = the first B at or after g1(A) with f(B) >= A, found by one pointer.
		// Each claim is checked as it is used; if one fails, the interval is
		// matched by the general sweep below instead.
		const int reason = [&]() {
			std::vector<int> g1(static_cast<std::size_t>(p));
			std::vector<int> g2(static_cast<std::size_t>(p));
			int last_below = -1;
			int first_above = 0;
			for (int ka = 0; ka < p; ++ka) {
				if (last_below >= 0 && !below_x(ka, last_below)) return 1;
				while (last_below + 1 < q && below_x(ka, last_below + 1)) ++last_below;
				if (first_above > 0 && above_y(ka, first_above - 1)) return 1;
				while (first_above < q && !above_y(ka, first_above)) ++first_above;
				g1[static_cast<std::size_t>(ka)] = first_above;
				g2[static_cast<std::size_t>(ka)] = last_below;
			}
			std::vector<int> f(static_cast<std::size_t>(q), -1);
			for (int lb = 0; lb < q; ++lb) {
				const auto accepted = half_line(p, [&](int ka) { return beta(lb, ka); });
				if (accepted.second < accepted.first) continue;
				if (accepted.first != 0) return 2;
				f[static_cast<std::size_t>(lb)] = accepted.second;
			}
			std::vector<Match> found;
			int lb = 0;
			for (int ka = 0; ka < p; ++ka) {
				lb = std::max(lb, g1[static_cast<std::size_t>(ka)]);
				while (lb < q && f[static_cast<std::size_t>(lb)] < ka) ++lb;
				if (lb >= q) break;
				if (lb > g2[static_cast<std::size_t>(ka)]) continue;
				if (!alpha(ka, lb)) {
					if (alpha(ka, q - 1)) return 3;
					continue;
				}
				if (!acceptable(ka, lb)) return 4;
				found.push_back({as[static_cast<std::size_t>(ka)]->other, bs[static_cast<std::size_t>(lb)]->other});
			}
			result = std::move(found);
			return 0;
		}();
		++matchings[static_cast<std::size_t>(reason)];
		if (reason == 0) return result;
		result.clear();

		// The general sweep: for each A_k the B's allowed by the angle at A
		// (alpha) and by the safety region (gamma); for each B_l the A's allowed
		// by the angle at B (beta). Each is an interval of the sorted candidates.
		std::vector<std::pair<int, int>> range(static_cast<std::size_t>(p));
		for (int ka = 0; ka < p; ++ka) {
			const auto by_angle = half_line(q, [&](int lb) { return alpha(ka, lb); });
			const auto by_x = half_line(q, [&](int lb) { return below_x(ka, lb); });
			const auto by_y = half_line(q, [&](int lb) { return above_y(ka, lb); });
			range[static_cast<std::size_t>(ka)] = {
				std::max({by_angle.first, by_x.first, by_y.first}),
				std::min({by_angle.second, by_x.second, by_y.second}),
			};
		}
		std::vector<std::vector<int>> opens(static_cast<std::size_t>(p));
		std::vector<std::vector<int>> closes(static_cast<std::size_t>(p));
		for (int lb = 0; lb < q; ++lb) {
			const auto accepted = half_line(p, [&](int ka) { return beta(lb, ka); });
			if (accepted.second < accepted.first) continue;
			opens[static_cast<std::size_t>(accepted.first)].push_back(lb);
			closes[static_cast<std::size_t>(accepted.second)].push_back(lb);
		}
		// Sweep the A's, keeping the B's whose beta interval holds the current
		// one; take the first acceptable such B in the A's range.
		std::set<int> active;
		for (int ka = 0; ka < p; ++ka) {
			for (int lb : opens[static_cast<std::size_t>(ka)]) active.insert(lb);
			const auto [lo, hi] = range[static_cast<std::size_t>(ka)];
			for (auto it = active.lower_bound(lo); it != active.end() && *it <= hi; ++it) {
				if (!acceptable(ka, *it)) continue;
				result.push_back({as[static_cast<std::size_t>(ka)]->other, bs[static_cast<std::size_t>(*it)]->other});
				break;
			}
			for (int lb : closes[static_cast<std::size_t>(ka)]) active.erase(lb);
		}
		return result;
	}
}
