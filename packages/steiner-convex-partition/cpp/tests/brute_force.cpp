// One-sided optimality check on small polygons. An independent search finds
// X2-patterns, X3-patterns (by sampling Steiner points on a fine barycentric
// grid, each sample checked exactly) and three-notch chains of diagonals, then
// packs them with its own non-crossing interval DP. If that packing ever holds
// more patterns than the decomposition found, the decomposition missed one.
//
// usage: steiner_convex_partition_brute_force [count] [seed]

#include "../src/exact.h"

#include <steiner_convex_partition/steiner_convex_partition.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

namespace scp = steiner_convex_partition;
using steiner_convex_partition::exact::Vec2;
using steiner_convex_partition::exact::gap_at_most_half_turn;
using steiner_convex_partition::exact::open_segment_meets_closed;
using steiner_convex_partition::exact::orient;

namespace {

	struct Poly {
		std::vector<Vec2> pts;  // counterclockwise
		std::vector<int> notches;

		int n() const {
			return static_cast<int>(pts.size());
		}

		const Vec2 &at(int v) const {
			return pts[static_cast<std::size_t>((v % n() + n()) % n())];
		}
	};

	Poly make(const scp::Polygon &polygon) {
		Poly p;
		long double area = 0.0L;
		for (std::size_t i = 0; i < polygon.size(); ++i) {
			const scp::Point &a = polygon[i];
			const scp::Point &b = polygon[(i + 1) % polygon.size()];
			area += static_cast<long double>(a.x) * b.y - static_cast<long double>(b.x) * a.y;
		}
		for (const scp::Point &q : polygon) p.pts.push_back({q.x, q.y});
		if (area < 0) std::reverse(p.pts.begin(), p.pts.end());
		for (int v = 0; v < p.n(); ++v) {
			if (orient(p.at(v - 1), p.at(v), p.at(v + 1)) < 0) p.notches.push_back(v);
		}
		return p;
	}

	bool segment_clear(const Poly &p, const Vec2 &a, const Vec2 &b) {
		for (int e = 0; e < p.n(); ++e) {
			if (open_segment_meets_closed(a, b, p.at(e), p.at(e + 1))) return false;
		}
		return true;
	}

	bool on_boundary(const Poly &p, const Vec2 &x) {
		for (int e = 0; e < p.n(); ++e) {
			const Vec2 &u = p.at(e);
			const Vec2 &w = p.at(e + 1);
			if (orient(u, w, x) == 0 && std::min(u.x, w.x) <= x.x && x.x <= std::max(u.x, w.x)
				&& std::min(u.y, w.y) <= x.y && x.y <= std::max(u.y, w.y)) return true;
		}
		return false;
	}

	// Arms from vertex v toward the targets remove its reflex angle; targets are
	// listed counterclockwise from the edge to the next vertex.
	bool resolves(const Poly &p, int v, const std::vector<Vec2> &targets) {
		Vec2 previous = p.at(v + 1);
		for (const Vec2 &t : targets) {
			if (!gap_at_most_half_turn(p.at(v), previous, t)) return false;
			previous = t;
		}
		return gap_at_most_half_turn(p.at(v), previous, p.at(v - 1));
	}

	bool inside_corner(const Poly &p, int v, const Vec2 &t) {
		const Vec2 &o = p.at(v);
		return !(orient(o, p.at(v - 1), t) >= 0 && orient(o, t, p.at(v + 1)) >= 0);
	}

	bool diagonal(const Poly &p, int a, int b) {
		return inside_corner(p, a, p.at(b)) && segment_clear(p, p.at(a), p.at(b));
	}

	bool x3(const Poly &p, int a, int b, int c, int grid) {
		const Vec2 &A = p.at(a);
		const Vec2 &B = p.at(b);
		const Vec2 &C = p.at(c);
		if (orient(A, B, C) <= 0) return false;
		for (int i = 1; i < grid; ++i) {
			for (int j = 1; i + j < grid; ++j) {
				const double u = static_cast<double>(i) / grid;
				const double w = static_cast<double>(j) / grid;
				const Vec2 s {A.x + u * (B.x - A.x) + w * (C.x - A.x), A.y + u * (B.y - A.y) + w * (C.y - A.y)};
				if (orient(A, B, s) < 0 || orient(B, C, s) < 0 || orient(C, A, s) < 0) continue;
				if (!resolves(p, a, {s}) || !resolves(p, b, {s}) || !resolves(p, c, {s})) continue;
				if (on_boundary(p, s)) continue;
				if (segment_clear(p, A, s) && segment_clear(p, B, s) && segment_clear(p, C, s)) return true;
			}
		}
		return false;
	}

	// Maximum number of non-interleaving patterns, each a sorted list of notch
	// positions in 0..c-1, by the plain interval DP on the cut circle.
	int pack(int c, const std::vector<std::vector<int>> &patterns) {
		std::vector<std::vector<int>> s(static_cast<std::size_t>(c + 1), std::vector<int>(static_cast<std::size_t>(c + 1), 0));
		auto value = [&](int i, int j) {
			return j < i ? 0 : s[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
		};
		for (int len = 2; len <= c; ++len) {
			for (int i = 0; i + len - 1 < c; ++i) {
				const int j = i + len - 1;
				int best = 0;
				for (int k = i; k < j; ++k) best = std::max(best, value(i, k) + value(k + 1, j));
				for (const std::vector<int> &q : patterns) {
					if (q.front() != i || q.back() != j) continue;
					int total = 1;
					for (std::size_t t = 0; t + 1 < q.size(); ++t) total += value(q[t] + 1, q[t + 1] - 1);
					best = std::max(best, total);
				}
				s[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = best;
			}
		}
		return c >= 2 ? s[0][static_cast<std::size_t>(c - 1)] : 0;
	}

	int brute_force(const Poly &p, int grid) {
		const int c = static_cast<int>(p.notches.size());
		std::vector<std::vector<int>> patterns;
		auto v = [&](int t) {
			return p.notches[static_cast<std::size_t>(t)];
		};
		for (int a = 0; a < c; ++a) {
			for (int b = a + 1; b < c; ++b) {
				if (diagonal(p, v(a), v(b)) && resolves(p, v(a), {p.at(v(b))}) && resolves(p, v(b), {p.at(v(a))})) {
					patterns.push_back({a, b});
				}
			}
		}
		for (int a = 0; a < c; ++a) {
			for (int b = a + 1; b < c; ++b) {
				for (int d = b + 1; d < c; ++d) {
					if (x3(p, v(a), v(b), v(d), grid)) {
						patterns.push_back({a, b, d});
						continue;
					}
					// Chains of two diagonals through the middle notch of any of the three
					// positions: the middle notch has two arms in boundary order.
					const int ids[3] = {a, b, d};
					for (int m = 0; m < 3; ++m) {
						const int mid = v(ids[m]);
						const int x = v(ids[(m + 1) % 3]);
						const int y = v(ids[(m + 2) % 3]);
						if (!diagonal(p, mid, x) || !diagonal(p, mid, y)) continue;
						if (!resolves(p, x, {p.at(mid)}) || !resolves(p, y, {p.at(mid)})) continue;
						if (resolves(p, mid, {p.at(x), p.at(y)})) {
							patterns.push_back({a, b, d});
							break;
						}
					}
				}
			}
		}
		// Every rotation of the cut gives the same optimum; use them all.
		int best = 0;
		for (int r = 0; r < std::max(c, 1); ++r) {
			std::vector<std::vector<int>> rotated;
			for (std::vector<int> q : patterns) {
				for (int &t : q) t = (t - r + c) % c;
				std::sort(q.begin(), q.end());
				rotated.push_back(q);
			}
			best = std::max(best, pack(c, rotated));
		}
		return best;
	}

	// A convex polygon whose sides carry thin V-notches with random depth, width
	// and tilt: notches that see each other only through Steiner points.
	scp::Polygon random_v_notches(std::mt19937 &rng, int sides) {
		std::uniform_real_distribution<double> unit(0.0, 1.0);
		std::vector<double> angles(static_cast<std::size_t>(sides));
		for (double &a : angles) a = unit(rng) * 2.0 * std::numbers::pi;
		std::sort(angles.begin(), angles.end());
		std::vector<scp::Point> corners;
		for (double a : angles) corners.push_back({std::round(100.0 * std::cos(a)), std::round(100.0 * std::sin(a))});
		scp::Polygon polygon;
		for (std::size_t k = 0; k < corners.size(); ++k) {
			const scp::Point &p0 = corners[k];
			const scp::Point &p1 = corners[(k + 1) % corners.size()];
			polygon.push_back(p0);
			const double length = std::hypot(p1.x - p0.x, p1.y - p0.y);
			if (length < 30.0 || unit(rng) < 0.25) continue;
			const double t = 0.3 + 0.4 * unit(rng);
			const scp::Point base {p0.x + t * (p1.x - p0.x), p0.y + t * (p1.y - p0.y)};
			const double ux = (p1.x - p0.x) / length;
			const double uy = (p1.y - p0.y) / length;
			// Inward normal of a counterclockwise side, tilted at random.
			const double tilt = (unit(rng) - 0.5) * 1.2;
			const double nx = -uy * std::cos(tilt) + ux * std::sin(tilt);
			const double ny = ux * std::cos(tilt) + uy * std::sin(tilt);
			const double depth = 15.0 + 35.0 * unit(rng);
			const double width = 0.5 + 3.0 * unit(rng);
			polygon.push_back({std::round((base.x - ux * width) * 8.0) / 8.0, std::round((base.y - uy * width) * 8.0) / 8.0});
			polygon.push_back({std::round((base.x + nx * depth) * 8.0) / 8.0, std::round((base.y + ny * depth) * 8.0) / 8.0});
			polygon.push_back({std::round((base.x + ux * width) * 8.0) / 8.0, std::round((base.y + uy * width) * 8.0) / 8.0});
		}
		return polygon;
	}

	scp::Polygon random_star(std::mt19937 &rng, int n) {
		std::uniform_real_distribution<double> unit(0.0, 1.0);
		std::vector<double> angles(static_cast<std::size_t>(n));
		for (double &a : angles) a = unit(rng) * 2.0 * std::numbers::pi;
		std::sort(angles.begin(), angles.end());
		scp::Polygon polygon;
		for (double a : angles) {
			const double r = (unit(rng) < 0.5 ? 25.0 : 70.0) + 30.0 * unit(rng);
			polygon.push_back({std::round(r * std::cos(a)), std::round(r * std::sin(a))});
		}
		return polygon;
	}
}

int main(int argc, char **argv) {
	const int count = argc > 1 ? std::atoi(argv[1]) : 200;
	std::mt19937 rng(argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 2024u);
	int checked = 0;
	int missed = 0;
	int ties = 0;
	int ahead = 0;
	int with_steiner = 0;
	for (int t = 0; t < count; ++t) {
		const scp::Polygon polygon = t % 2 == 0
			? random_star(rng, 8 + static_cast<int>(rng() % 8))
			: random_v_notches(rng, 4 + static_cast<int>(rng() % 6));
		scp::Result result;
		try {
			result = scp::decompose(polygon);
		} catch (const std::invalid_argument &) {
			continue;
		}
		if (result.notch_count > 9) continue;
		const Poly p = make(polygon);
		const int brute = brute_force(p, 48);
		++checked;
		const int ours = static_cast<int>(result.pattern_count);
		if (result.interior_steiner_points > 0) ++with_steiner;
		if (brute > ours) {
			++missed;
			std::cerr << "case " << t << ": brute force packs " << brute << " patterns, the decomposition " << ours << '\n';
			for (const scp::Point &q : polygon) std::cerr << "  " << q.x << ' ' << q.y << '\n';
		} else if (brute == ours) {
			++ties;
		} else {
			++ahead;
		}
	}
	std::cout << "checked " << checked << " polygons: brute force ahead on " << missed << ", tied on " << ties
		<< ", behind on " << ahead << " (patterns it does not search); " << with_steiner
		<< " decompositions use interior Steiner points\n";
	return missed == 0 ? 0 : 1;
}
