#include "superrange.h"

#include <algorithm>

namespace steiner_convex_partition::detail {

	using exact::dot_sign;
	using exact::orient;

	std::vector<int> pseudo_notches(const PolygonData &polygon) {
		std::vector<int> result;
		const int c = polygon.c;
		if (c == 0) return result;
		const int n = polygon.n;
		for (int t = 0; t < c; ++t) {
			const int start = polygon.notch_vertex[static_cast<std::size_t>(t)];
			const int end = polygon.notch_vertex[static_cast<std::size_t>((t + 1) % c)];
			// Walk from start to end (once around when there is a single notch).
			const int steps = ((end - start) % n + n) % n == 0 ? n : ((end - start) % n + n) % n;
			result.push_back(start);
			int first = start;
			int last = (start + 1) % n;
			for (int s = 2; s <= steps; ++s) {
				const int next = (start + s) % n;
				const Vec2 &y1 = polygon.at(first);
				const Vec2 &y2 = polygon.at(first + 1);
				const Vec2 &yk = polygon.at(last);
				const Vec2 &yn = polygon.at(next);
				// The chain first..last extended by next, closed by its chord, must
				// stay convex: no reflex turn at next nor at first.
				const bool convex = orient(yk, yn, y1) >= 0 && orient(yn, y1, y2) >= 0;
				if (!convex) {
					result.push_back(last);
					first = last;
				}
				last = next;
			}
		}
		std::sort(result.begin(), result.end());
		result.erase(std::unique(result.begin(), result.end()), result.end());
		return result;
	}

	namespace {

		struct Hit {
			int kind = 0;  // 0 none, 1 vertex on the ray, 2 edge crossing, 3 edge near a vertex
			int vertex = -1;
			int edge = -1;
			int other = -1;  // for kind 3: the edge's endpoint off the ray
		};

		// Is h1 strictly before h2 along the ray from v through q? `turn` is +1 for
		// a ray turned slightly counterclockwise, -1 clockwise (kind 3 only).
		bool before(const PolygonData &polygon, const Vec2 &v, const Vec2 &q, const Hit &h1, const Hit &h2, int turn) {
			if (h2.kind == 0) return h1.kind != 0;
			if (h1.kind == 0) return false;
			const bool point1 = h1.kind != 2;
			const bool point2 = h2.kind != 2;
			if (point1 && point2) {
				if (h1.vertex != h2.vertex) {
					return dot_sign(polygon.at(h1.vertex), polygon.at(h2.vertex), v, q) > 0;
				}
				if (h1.kind == 3 && h2.kind == 3) {
					// Two edges leave the touched vertex on the turned side; the ray meets
					// first the one at the larger angle from its direction.
					const Vec2 &u = polygon.at(h1.vertex);
					const int o = orient(u, polygon.at(h2.other), polygon.at(h1.other));
					return turn > 0 ? o > 0 : o < 0;
				}
				return false;
			}
			if (point1 != point2) {
				// A point on the ray against the crossing of an edge: the point comes
				// first when it lies on v's side of the edge's line.
				const Hit &p = point1 ? h1 : h2;
				const Hit &e = point1 ? h2 : h1;
				const Vec2 &a = polygon.at(e.edge);
				const Vec2 &b = polygon.at(e.edge + 1);
				const int side_v = orient(a, b, v);
				const int side_p = orient(a, b, polygon.at(p.vertex));
				const bool point_first = side_p == side_v;
				return point1 ? point_first : !point_first;
			}
			// Two disjoint edges crossing the ray.
			const Vec2 &a1 = polygon.at(h1.edge);
			const Vec2 &b1 = polygon.at(h1.edge + 1);
			const Vec2 &a2 = polygon.at(h2.edge);
			const Vec2 &b2 = polygon.at(h2.edge + 1);
			const int s1 = orient(a1, b1, v);
			const int p = orient(a1, b1, a2);
			const int r = orient(a1, b1, b2);
			if ((p == -s1 || p == 0) && (r == -s1 || r == 0) && !(p == 0 && r == 0)) return true;
			const int s2 = orient(a2, b2, v);
			const int x = orient(a2, b2, a1);
			const int y = orient(a2, b2, b1);
			if ((x == s2 || x == 0) && (y == s2 || y == 0) && !(x == 0 && y == 0)) return true;
			return false;
		}
	}

	Offset Superranges::vertex_offset(int v, int u) const {
		const int n = polygon_.n;
		return 2 * (((u - v) % n + n) % n);
	}

	Offset Superranges::edge_offset(int v, int e) const {
		const int n = polygon_.n;
		return 2 * (((e - v) % n + n) % n) + 1;
	}

	int Superranges::compare_directions(int v, const Vec2 &p, const Vec2 &q) const {
		const Vec2 &o = polygon_.at(v);
		const Vec2 &next = polygon_.next(v);
		auto half = [&](const Vec2 &x) {
			const int s = orient(o, next, x);
			if (s > 0) return 0;
			if (s == 0 && dot_sign(o, next, o, x) > 0) return 0;
			return 1;
		};
		const int hp = half(p);
		const int hq = half(q);
		if (hp != hq) return hp < hq ? -1 : 1;
		const int s = orient(o, p, q);
		return s > 0 ? -1 : (s < 0 ? 1 : 0);
	}

	Superranges::Direction Superranges::trace(int v, int target, bool want_before, bool want_after) const {
		const int n = polygon_.n;
		const Vec2 &o = polygon_.at(v);
		const Vec2 &q = polygon_.at(target);
		Hit contact;
		Hit after;
		Hit earlier;
		auto offer = [&](Hit &best, const Hit &h, int turn) {
			if (before(polygon_, o, q, h, best, turn)) best = h;
		};
		auto in_front = [&](const Vec2 &x) {
			return dot_sign(o, q, o, x) > 0;
		};
		// Only edges meeting the ray's line can be hit: O(c log n) of them to look
		// at, through the convex chains.
		std::vector<int> edges;
		chains_.edges_meeting_line(o, q, edges);
		for (int e : edges) {
			if (e == v || (e + 1) % n == v) continue;
			const Vec2 &a = polygon_.at(e);
			const Vec2 &b = polygon_.at(e + 1);
			const int oa = orient(o, q, a);
			const int ob = orient(o, q, b);
			if (oa == 0 && ob == 0) {
				if (in_front(a)) offer(contact, Hit {1, e, e, -1}, 0);
				if (in_front(b)) offer(contact, Hit {1, (e + 1) % n, e, -1}, 0);
				continue;
			}
			if (oa == 0 && in_front(a)) {
				offer(contact, Hit {1, e, e, -1}, 0);
				if (ob > 0) offer(after, Hit {3, e, e, (e + 1) % n}, 1);
				if (ob < 0) offer(earlier, Hit {3, e, e, (e + 1) % n}, -1);
			}
			if (ob == 0 && in_front(b)) {
				offer(contact, Hit {1, (e + 1) % n, e, -1}, 0);
				if (oa > 0) offer(after, Hit {3, (e + 1) % n, e, e}, 1);
				if (oa < 0) offer(earlier, Hit {3, (e + 1) % n, e, e}, -1);
			}
			if (oa * ob < 0) {
				const int along = exact::cross_sign(o, q, a, b);
				if (along != 0 && orient(o, a, b) == along) {
					const Hit h {2, -1, e, -1};
					offer(contact, h, 0);
					offer(after, h, 1);
					offer(earlier, h, -1);
				}
			}
		}
		Direction d;
		d.target = target;
		if (contact.kind != 0) {
			if (contact.kind == 1) {
				d.contact = vertex_offset(v, contact.vertex);
				d.contact_vertex = contact.vertex;
			} else {
				d.contact = edge_offset(v, contact.edge);
				d.contact_edge = contact.edge;
			}
		}
		if (want_before && earlier.kind != 0) {
			d.before = edge_offset(v, earlier.edge);
			d.before_edge = earlier.edge;
		}
		if (want_after && after.kind != 0) {
			d.after = edge_offset(v, after.edge);
			d.after_edge = after.edge;
		}
		return d;
	}

	Superranges::Superranges(const PolygonData &polygon, const std::vector<int> &pseudo)
		: polygon_(polygon), chains_(polygon) {
		ranges_.resize(static_cast<std::size_t>(polygon.c));
		for (int t = 0; t < polygon.c; ++t) {
			const int v = polygon.notch_vertex[static_cast<std::size_t>(t)];
			Range &range = ranges_[static_cast<std::size_t>(t)];
			range.vertex = v;
			const Vec2 &o = polygon.at(v);
			std::vector<int> targets;
			for (int u : pseudo) {
				if (u == v) continue;
				if (strictly_inside_corner(polygon, v, polygon.at(u))) targets.push_back(u);
			}
			std::sort(targets.begin(), targets.end(), [&](int a, int b) {
				const int order = compare_directions(v, polygon.at(a), polygon.at(b));
				if (order != 0) return order < 0;
				return dot_sign(polygon.at(a), polygon.at(b), o, polygon.at(b)) > 0;  // nearer first
			});
			std::vector<int> unique;
			for (int u : targets) {
				if (!unique.empty() && compare_directions(v, polygon.at(unique.back()), polygon.at(u)) == 0) continue;
				unique.push_back(u);
			}
			const int n = polygon.n;
			range.directions.push_back(trace(v, (v + 1) % n, false, true));
			range.directions.front().before = 0;
			for (int u : unique) range.directions.push_back(trace(v, u, true, true));
			range.directions.push_back(trace(v, (v - 1 + n) % n, true, false));
			range.directions.back().after = 2 * n;
		}
		const std::size_t cs = static_cast<std::size_t>(polygon.c);
		last_.assign(cs * cs, -1);
		first_.assign(cs * cs, -1);
		for (int a = 0; a < polygon.c; ++a) {
			for (int b = 0; b < polygon.c; ++b) {
				if (a == b) continue;
				const int va = polygon.notch_vertex[static_cast<std::size_t>(a)];
				const int vb = polygon.notch_vertex[static_cast<std::size_t>(b)];
				const std::size_t index = static_cast<std::size_t>(a) * cs + static_cast<std::size_t>(b);
				last_[index] = compute_last_direction(va, vb);
				first_[index] = compute_first_direction(va, vb);
			}
		}
	}

	std::size_t Superranges::directions() const {
		std::size_t total = 0;
		for (const Range &range : ranges_) total += range.directions.size();
		return total;
	}

	const Superranges::Range &Superranges::range_of(int vertex) const {
		return ranges_[static_cast<std::size_t>(polygon_.vertex_notch[static_cast<std::size_t>(vertex)])];
	}

	bool Superranges::sees_in_cell(int v, const Range &range, std::size_t cell, const Vec2 &s) const {
		const int n = polygon_.n;
		const int first = range.directions[cell].after_edge;
		const int last = range.directions[cell + 1].before_edge;
		if (first < 0 || last < 0) return false;
		const int span = ((last - first) % n + n) % n;
		int lo = 0;
		int hi = span;
		while (lo < hi) {
			const int mid = (lo + hi) / 2;
			if (compare_directions(v, s, polygon_.at(first + mid + 1)) <= 0) hi = mid;
			else lo = mid + 1;
		}
		const int edge = first + lo;
		return orient(polygon_.at(edge), polygon_.at(edge + 1), s) > 0;
	}

	bool Superranges::sees(int v, const Vec2 &s) const {
		if (s == polygon_.at(v)) return false;
		if (!strictly_inside_corner(polygon_, v, s)) return false;
		const Range &range = range_of(v);
		const std::vector<Direction> &dirs = range.directions;
		// Last direction not after s's direction (the first one always is).
		std::size_t lo = 0;
		std::size_t hi = dirs.size() - 1;
		while (lo + 1 < hi) {
			const std::size_t mid = (lo + hi) / 2;
			if (compare_directions(v, polygon_.at(dirs[mid].target), s) <= 0) lo = mid;
			else hi = mid;
		}
		const Direction &d = dirs[lo];
		if (lo > 0 && compare_directions(v, polygon_.at(d.target), s) == 0) {
			// On the ray itself: before its first contact with the boundary.
			const Vec2 &o = polygon_.at(v);
			if (d.contact_vertex >= 0) return dot_sign(s, polygon_.at(d.contact_vertex), o, polygon_.at(d.target)) > 0;
			if (d.contact_edge >= 0) {
				const Vec2 &a = polygon_.at(d.contact_edge);
				const Vec2 &b = polygon_.at(d.contact_edge + 1);
				const int side = orient(a, b, s);
				return side != 0 && side == orient(a, b, o);
			}
			return false;
		}
		return sees_in_cell(v, range, lo, s);
	}

	bool Superranges::sees_vertex(int a, int b) const {
		if (a == b) return false;
		const Vec2 &target = polygon_.at(b);
		if (!strictly_inside_corner(polygon_, a, target)) return false;
		const Range &range = range_of(a);
		const std::vector<Direction> &dirs = range.directions;
		std::size_t lo = 1;
		std::size_t hi = dirs.size() - 1;
		while (lo < hi) {
			const std::size_t mid = (lo + hi) / 2;
			if (compare_directions(a, polygon_.at(dirs[mid].target), target) < 0) lo = mid + 1;
			else hi = mid;
		}
		if (lo >= dirs.size() - 1) return false;
		if (compare_directions(a, polygon_.at(dirs[lo].target), target) != 0) return false;
		return dirs[lo].contact_vertex == b;
	}

	namespace {
		// Index of the direction whose jump (offset before, offset after) strictly
		// contains `offset`, or -1.
		template <class Directions>
		int jump_containing(const Directions &dirs, int offset) {
			std::size_t lo = 0;
			std::size_t hi = dirs.size();
			while (lo < hi) {
				const std::size_t mid = (lo + hi) / 2;
				if (dirs[mid].after > offset) hi = mid;
				else lo = mid + 1;
			}
			if (lo >= dirs.size()) return -1;
			if (dirs[lo].before < offset && offset < dirs[lo].after) return static_cast<int>(lo);
			return -1;
		}
	}

	int Superranges::last_direction(int a, int b) const {
		const std::size_t c = static_cast<std::size_t>(polygon_.c);
		return last_[static_cast<std::size_t>(polygon_.vertex_notch[static_cast<std::size_t>(a)]) * c
			+ static_cast<std::size_t>(polygon_.vertex_notch[static_cast<std::size_t>(b)])];
	}

	int Superranges::first_direction(int a, int b) const {
		const std::size_t c = static_cast<std::size_t>(polygon_.c);
		return first_[static_cast<std::size_t>(polygon_.vertex_notch[static_cast<std::size_t>(a)]) * c
			+ static_cast<std::size_t>(polygon_.vertex_notch[static_cast<std::size_t>(b)])];
	}

	int Superranges::compute_last_direction(int a, int b) const {
		if (sees_vertex(a, b)) return -1;
		const Range &range = range_of(a);
		const int k = jump_containing(range.directions, vertex_offset(a, b));
		return k < 0 ? -1 : range.directions[static_cast<std::size_t>(k)].target;
	}

	int Superranges::compute_first_direction(int a, int b) const {
		if (sees_vertex(b, a)) return -1;
		const Range &range = range_of(b);
		const int k = jump_containing(range.directions, vertex_offset(b, a));
		return k < 0 ? -1 : range.directions[static_cast<std::size_t>(k)].target;
	}
}
