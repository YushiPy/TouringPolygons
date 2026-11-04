// Superrange queries against brute force on random polygons, many of them
// rectilinear with integer coordinates (collinear vertices everywhere).

#include "../src/internal.h"
#include "../src/superrange.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

namespace scp = steiner_convex_partition;
using namespace steiner_convex_partition::detail;

namespace {

	scp::Polygon random_star(std::mt19937 &rng, int n) {
		std::uniform_real_distribution<double> unit(0.0, 1.0);
		std::vector<double> angles(static_cast<std::size_t>(n));
		for (double &a : angles) a = unit(rng) * 2.0 * std::numbers::pi;
		std::sort(angles.begin(), angles.end());
		scp::Polygon polygon;
		for (double a : angles) {
			const double r = (unit(rng) < 0.5 ? 20.0 : 60.0) + 40.0 * unit(rng);
			polygon.push_back({std::round(r * std::cos(a)), std::round(r * std::sin(a))});
		}
		return polygon;
	}

	scp::Polygon random_skyline(std::mt19937 &rng, int columns) {
		std::uniform_int_distribution<int> top_dist(5, 10);
		std::uniform_int_distribution<int> bottom_dist(0, 4);
		scp::Polygon polygon;
		std::vector<int> top(static_cast<std::size_t>(columns));
		std::vector<int> bottom(static_cast<std::size_t>(columns));
		for (int i = 0; i < columns; ++i) {
			top[static_cast<std::size_t>(i)] = top_dist(rng);
			bottom[static_cast<std::size_t>(i)] = bottom_dist(rng);
		}
		for (int i = 0; i < columns; ++i) {
			polygon.push_back({static_cast<double>(i), static_cast<double>(bottom[static_cast<std::size_t>(i)])});
			polygon.push_back({static_cast<double>(i + 1), static_cast<double>(bottom[static_cast<std::size_t>(i)])});
		}
		for (int i = columns - 1; i >= 0; --i) {
			polygon.push_back({static_cast<double>(i + 1), static_cast<double>(top[static_cast<std::size_t>(i)])});
			polygon.push_back({static_cast<double>(i), static_cast<double>(top[static_cast<std::size_t>(i)])});
		}
		return polygon;
	}

	bool brute_sees(const PolygonData &p, int v, const Vec2 &s) {
		if (s == p.at(v)) return false;
		if (!strictly_inside_corner(p, v, s)) return false;
		if (point_on_boundary(p, s)) return false;
		return open_segment_clear(p, p.at(v), s);
	}
}

int main(int argc, char **argv) {
	const int count = argc > 1 ? std::atoi(argv[1]) : 300;
	std::mt19937 rng(argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 99u);
	long point_checks = 0;
	long vertex_checks = 0;
	long mismatches = 0;
	long visible = 0;
	for (int t = 0; t < count; ++t) {
		const scp::Polygon input = t % 2 ? random_star(rng, 8 + static_cast<int>(rng() % 20)) : random_skyline(rng, 3 + static_cast<int>(rng() % 9));
		PolygonData p;
		try {
			p = prepare_polygon(input);
		} catch (const std::exception &) {
			continue;
		}
		if (p.c == 0) continue;
		const std::vector<int> pseudo = pseudo_notches(p);
		const Superranges sr(p, pseudo, true);
		double min_x = p.pts[0].x, max_x = min_x, min_y = p.pts[0].y, max_y = min_y;
		for (const Vec2 &q : p.pts) {
			min_x = std::min(min_x, q.x);
			max_x = std::max(max_x, q.x);
			min_y = std::min(min_y, q.y);
			max_y = std::max(max_y, q.y);
		}
		std::uniform_int_distribution<int> grid_x(static_cast<int>(2 * min_x), static_cast<int>(2 * max_x));
		std::uniform_int_distribution<int> grid_y(static_cast<int>(2 * min_y), static_cast<int>(2 * max_y));
		for (int v : p.notch_vertex) {
			for (int k = 0; k < 400; ++k) {
				// Half-integer grid points: on many lines through vertices.
				const Vec2 s {grid_x(rng) / 2.0, grid_y(rng) / 2.0};
				const bool fast = sr.sees(v, s);
				const bool slow = brute_sees(p, v, s);
				++point_checks;
				visible += slow;
				if (fast != slow) {
					++mismatches;
					if (mismatches <= 10) {
						std::cerr << "sees mismatch: case " << t << " notch " << v << " point (" << s.x << ", " << s.y << ") fast " << fast << " slow " << slow << '\n';
					}
				}
			}
			for (int b : p.notch_vertex) {
				if (b == v) continue;
				const bool fast = sr.sees_vertex(v, b);
				const bool slow = strictly_inside_corner(p, v, p.at(b)) && strictly_inside_corner(p, b, p.at(v))
					&& open_segment_clear(p, p.at(v), p.at(b));
				++vertex_checks;
				if (fast != slow) {
					++mismatches;
					if (mismatches <= 10) std::cerr << "sees_vertex mismatch: case " << t << " " << v << " -> " << b << " fast " << fast << " slow " << slow << '\n';
				}
			}
		}
	}
	std::cout << "superrange tests: " << point_checks << " point checks (" << visible << " visible), "
		<< vertex_checks << " notch pairs, " << mismatches << " mismatches\n";
	return mismatches == 0 ? 0 : 1;
}
