#include "../src/exact.h"

#include <steiner_convex_partition/steiner_convex_partition.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace scp = steiner_convex_partition;

namespace {

	int failures = 0;
	int checks = 0;

	void expect(bool condition, const std::string &what) {
		++checks;
		if (!condition) {
			++failures;
			std::cerr << "FAILED: " << what << '\n';
		}
	}

	std::size_t count_notches(const scp::Polygon &polygon) {
		const std::size_t n = polygon.size();
		long double area = 0.0L;
		for (std::size_t i = 0; i < n; ++i) {
			const scp::Point &a = polygon[i];
			const scp::Point &b = polygon[(i + 1) % n];
			area += static_cast<long double>(a.x) * b.y - static_cast<long double>(b.x) * a.y;
		}
		std::size_t notches = 0;
		for (std::size_t i = 0; i < n; ++i) {
			const scp::Point &a = polygon[(i + n - 1) % n];
			const scp::Point &b = polygon[i];
			const scp::Point &c = polygon[(i + 1) % n];
			const long double turn = (static_cast<long double>(b.x) - a.x) * (static_cast<long double>(c.y) - a.y)
				- (static_cast<long double>(b.y) - a.y) * (static_cast<long double>(c.x) - a.x);
			if (area > 0 ? turn < 0 : turn > 0) ++notches;
		}
		return notches;
	}

	std::size_t rounded_boundary_points = 0;

	// A rounded cut endpoint on a polygon edge must lie on or outside it, so
	// that the pieces cover the polygon.
	void check_cover(const scp::Polygon &input, const scp::Result &result, const std::string &name) {
		using steiner_convex_partition::exact::Vec2;
		std::vector<Vec2> polygon;
		for (const scp::Point &p : input) {
			if (!polygon.empty() && polygon.back() == Vec2 {p.x, p.y}) continue;
			polygon.push_back({p.x, p.y});
		}
		long double area = 0.0L;
		for (std::size_t i = 0; i < polygon.size(); ++i) {
			const Vec2 &a = polygon[i];
			const Vec2 &b = polygon[(i + 1) % polygon.size()];
			area += static_cast<long double>(a.x) * b.y - static_cast<long double>(b.x) * a.y;
		}
		if (area < 0) std::reverse(polygon.begin(), polygon.end());
		for (const scp::Polygon &piece : result.pieces) {
			for (const scp::Point &point : piece) {
				const Vec2 q {point.x, point.y};
				if (std::find(polygon.begin(), polygon.end(), q) != polygon.end()) continue;
				for (std::size_t e = 0; e < polygon.size(); ++e) {
					const Vec2 &a = polygon[e];
					const Vec2 &b = polygon[(e + 1) % polygon.size()];
					const long double length = std::hypot(static_cast<long double>(b.x) - a.x, static_cast<long double>(b.y) - a.y);
					const long double cross = (static_cast<long double>(b.x) - a.x) * (static_cast<long double>(q.y) - a.y)
						- (static_cast<long double>(b.y) - a.y) * (static_cast<long double>(q.x) - a.x);
					const long double along = ((static_cast<long double>(q.x) - a.x) * (static_cast<long double>(b.x) - a.x)
						+ (static_cast<long double>(q.y) - a.y) * (static_cast<long double>(b.y) - a.y)) / (length * length);
					if (std::fabs(cross) > 1.0e-9L * length * std::max(1.0L, length) || along < 0.0L || along > 1.0L) continue;
					if (!result.exact_coordinates) ++rounded_boundary_points;
					expect(steiner_convex_partition::exact::orient(a, b, q) <= 0, name + ": boundary point on or outside its edge");
				}
			}
		}
	}

	scp::Result run_checked(const scp::Polygon &polygon, const std::string &name, const scp::Options &options = {}) {
		scp::Result result;
		try {
			result = scp::decompose(polygon, options);
		} catch (const std::exception &error) {
			expect(false, name + ": threw " + error.what());
			return result;
		}
		std::string reason;
		expect(result.validated, name + ": validated");
		check_cover(polygon, result, name);
		if (result.exact_coordinates) {
			expect(scp::is_convex_partition(polygon, result.pieces, &reason), name + ": independent check (" + reason + ")");
		}
		expect(result.pieces.size() == result.notch_count + 1 - result.pattern_count, name + ": piece count is c + 1 - p");
		expect(result.rotation_min_patterns == result.rotation_max_patterns, name + ": all rotations agree");
		expect(result.pieces.size() <= result.notch_count + 1, name + ": at most the naive count");
		expect(result.pieces.size() >= 1 + (result.notch_count + 1) / 2, name + ": at least the lower bound");
		return result;
	}

	scp::Polygon regular_with_v_notches(int sides, double radius, double depth, double half_width) {
		// A regular polygon whose every side carries a V-shaped notch pointing at
		// the center. Narrow notches cannot see each other inside their cones, so
		// no diagonal removes two of them; a Steiner point at the center removes
		// three.
		scp::Polygon polygon;
		for (int s = 0; s < sides; ++s) {
			const double a0 = 2.0 * std::numbers::pi * s / sides;
			const double a1 = 2.0 * std::numbers::pi * (s + 1) / sides;
			const scp::Point p0 {radius * std::cos(a0), radius * std::sin(a0)};
			const scp::Point p1 {radius * std::cos(a1), radius * std::sin(a1)};
			const scp::Point mid {(p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0};
			const double length = std::hypot(p1.x - p0.x, p1.y - p0.y);
			const double ux = (p1.x - p0.x) / length;
			const double uy = (p1.y - p0.y) / length;
			const double inward_x = -mid.x / std::hypot(mid.x, mid.y);
			const double inward_y = -mid.y / std::hypot(mid.x, mid.y);
			polygon.push_back(p0);
			polygon.push_back({mid.x - ux * half_width, mid.y - uy * half_width});
			polygon.push_back({mid.x + inward_x * depth, mid.y + inward_y * depth});
			polygon.push_back({mid.x + ux * half_width, mid.y + uy * half_width});
		}
		return polygon;
	}

	scp::Polygon random_star(std::mt19937 &rng, int n, bool integer) {
		std::uniform_real_distribution<double> unit(0.0, 1.0);
		std::vector<double> angles(static_cast<std::size_t>(n));
		for (double &a : angles) a = unit(rng) * 2.0 * std::numbers::pi;
		std::sort(angles.begin(), angles.end());
		scp::Polygon polygon;
		for (double a : angles) {
			const double r = 20.0 + 80.0 * unit(rng);
			double x = r * std::cos(a);
			double y = r * std::sin(a);
			if (integer) {
				x = std::round(x);
				y = std::round(y);
			}
			if (!polygon.empty() && polygon.back() == scp::Point {x, y}) continue;
			polygon.push_back({x, y});
		}
		return polygon;
	}

	// Rectilinear polygon between a random floor and ceiling over unit columns.
	scp::Polygon random_skyline(std::mt19937 &rng, int columns) {
		std::uniform_int_distribution<int> top_dist(6, 12);
		std::uniform_int_distribution<int> bottom_dist(0, 5);
		std::vector<int> top(static_cast<std::size_t>(columns));
		std::vector<int> bottom(static_cast<std::size_t>(columns));
		for (int i = 0; i < columns; ++i) {
			top[static_cast<std::size_t>(i)] = top_dist(rng);
			bottom[static_cast<std::size_t>(i)] = bottom_dist(rng);
		}
		scp::Polygon polygon;
		auto push = [&](double x, double y) {
			const scp::Point p {x, y};
			if (!polygon.empty() && polygon.back() == p) return;
			polygon.push_back(p);
		};
		// Bottom from left to right.
		for (int i = 0; i < columns; ++i) {
			push(i, bottom[static_cast<std::size_t>(i)]);
			push(i + 1, bottom[static_cast<std::size_t>(i)]);
		}
		for (int i = columns - 1; i >= 0; --i) {
			push(i + 1, top[static_cast<std::size_t>(i)]);
			push(i, top[static_cast<std::size_t>(i)]);
		}
		// Drop collinear repeats (keep only corners).
		scp::Polygon cleaned;
		const std::size_t n = polygon.size();
		for (std::size_t i = 0; i < n; ++i) {
			const scp::Point &a = polygon[(i + n - 1) % n];
			const scp::Point &b = polygon[i];
			const scp::Point &c = polygon[(i + 1) % n];
			const double turn = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
			if (turn != 0.0) cleaned.push_back(b);
		}
		return cleaned;
	}

	void test_basic() {
		const scp::Polygon square {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
		scp::Result r = run_checked(square, "square");
		expect(r.pieces.size() == 1, "square: one piece");

		const scp::Polygon clockwise {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
		r = run_checked(clockwise, "clockwise square");
		expect(r.pieces.size() == 1, "clockwise square: one piece");

		const scp::Polygon l_shape {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}};
		r = run_checked(l_shape, "L-shape");
		expect(r.pieces.size() == 2, "L-shape: two pieces");

		// Two notches facing each other across a corridor: one diagonal removes
		// both (an X2-pattern).
		const scp::Polygon h_shape {
			{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 0}, {3, 0}, {3, 3}, {2, 3}, {2, 2}, {1, 2}, {1, 3}, {0, 3},
		};
		r = run_checked(h_shape, "H-shape");
		expect(r.pieces.size() == 3, "H-shape: three pieces");

		// Repeated and straight vertices, the first vertex repeated at the end.
		const scp::Polygon messy {{0, 0}, {1, 0}, {1, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 1.5}, {1, 2}, {0, 2}, {0, 1}, {0, 0}};
		r = run_checked(messy, "L-shape with repeated and straight vertices");
		expect(r.pieces.size() == 2, "L-shape with repeated and straight vertices: two pieces");

		bool threw = false;
		try {
			scp::decompose({{0, 0}, {1, 1}, {1, 0}, {0, 1}});
		} catch (const std::invalid_argument &) {
			threw = true;
		}
		expect(threw, "bow tie: rejected as not simple");
	}

	void test_v_notches() {
		for (int sides : {3, 4, 5, 6}) {
			const scp::Polygon polygon = regular_with_v_notches(sides, 100.0, 30.0, 4.0);
			const std::string name = "regular " + std::to_string(sides) + "-gon with V-notches";
			const scp::Result with = run_checked(polygon, name);
			scp::Options no_steiner;
			no_steiner.steiner_points = false;
			no_steiner.x4_patterns = false;
			const scp::Result without = run_checked(polygon, name + " (diagonals only)", no_steiner);
			expect(with.pieces.size() <= without.pieces.size(), name + ": Steiner points never hurt");
			if (sides == 3) {
				expect(with.pieces.size() == 3, name + ": three pieces with one Steiner point");
				expect(without.pieces.size() == 4, name + ": four pieces with diagonals only");
				expect(with.interior_steiner_points == 1, name + ": one Steiner point");
			}
			std::cout << name << ": " << with.pieces.size() << " pieces with Steiner points, "
				<< without.pieces.size() << " with diagonals only\n";
		}
	}

	void test_x4() {
		// Four thin V-notches: the cones of the two top notches meet near (0, 1)
		// and those of the two bottom notches near (0, -1), and no cone reaches
		// another notch. Only an X4-pattern (two adjacent Steiner points) joins them.
		const double d = 0.13;
		const scp::Polygon polygon {
			{-10, -10}, {-4.5 - d, -10}, {-3, -7}, {-4.5 + d, -10},
			{4.5 - d, -10}, {3, -7}, {4.5 + d, -10}, {10, -10},
			{10, 10}, {4.5 + d, 10}, {3, 7}, {4.5 - d, 10},
			{-4.5 + d, 10}, {-3, 7}, {-4.5 - d, 10}, {-10, 10},
		};
		const scp::Result with = run_checked(polygon, "X4 square");
		expect(with.pieces.size() == 4, "X4 square: four pieces");
		expect(with.patterns.size() == 1 && with.patterns[0].kind == scp::PatternKind::x4, "X4 square: one X4-pattern");
		scp::Options no_x4;
		no_x4.x4_patterns = false;
		const scp::Result without = run_checked(polygon, "X4 square without X4", no_x4);
		expect(without.pieces.size() == 5, "X4 square without X4: five pieces");
		std::cout << "X4 square: " << with.pieces.size() << " pieces, " << without.pieces.size() << " without X4-patterns\n";
	}

	void test_random(int count, unsigned seed) {
		std::mt19937 rng(seed);
		int better = 0;
		for (int t = 0; t < count; ++t) {
			const int n = 6 + static_cast<int>(rng() % 22);
			const bool integer = t % 2 == 0;
			const scp::Polygon polygon = random_star(rng, n, integer);
			if (polygon.size() < 3) continue;
			scp::Result with;
			try {
				with = scp::decompose(polygon);
			} catch (const std::invalid_argument &) {
				continue;  // rounding may make a star polygon degenerate
			}
			const std::string name = "random star " + std::to_string(t);
			with = run_checked(polygon, name);
			scp::Options no_steiner;
			no_steiner.steiner_points = false;
			no_steiner.x4_patterns = false;
			const scp::Result without = run_checked(polygon, name + " (diagonals only)", no_steiner);
			expect(with.pieces.size() <= without.pieces.size(), name + ": Steiner points never hurt");
			expect(with.notch_count == count_notches(polygon), name + ": notch count");
			if (with.pieces.size() < without.pieces.size()) ++better;
		}
		std::cout << "random stars: Steiner points removed a piece in " << better << " of " << count << '\n';
	}

	void test_skylines(int count, unsigned seed) {
		std::mt19937 rng(seed);
		for (int t = 0; t < count; ++t) {
			const scp::Polygon polygon = random_skyline(rng, 3 + static_cast<int>(rng() % 8));
			run_checked(polygon, "skyline " + std::to_string(t));
		}
	}

	void test_validator() {
		const scp::Polygon square {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
		std::string reason;
		expect(scp::is_convex_partition(square, {{{0, 0}, {1, 0}, {1, 2}, {0, 2}}, {{1, 0}, {2, 0}, {2, 2}, {1, 2}}}, &reason), "validator accepts two halves");
		expect(!scp::is_convex_partition(square, {{{0, 0}, {1, 0}, {1, 2}, {0, 2}}}, &reason), "validator rejects a missing half");
		expect(!scp::is_convex_partition(square, {{{0, 0}, {2, 0}, {2, 2}, {0, 2}}, {{1, 0}, {2, 0}, {2, 2}, {1, 2}}}, &reason), "validator rejects overlap");
		const scp::Polygon l_shape {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}};
		expect(!scp::is_convex_partition(l_shape, {l_shape}, &reason), "validator rejects a reflex piece");
	}
}

int main(int argc, char **argv) {
	const int random_count = argc > 1 ? std::atoi(argv[1]) : 300;
	test_basic();
	test_validator();
	test_v_notches();
	test_x4();
	test_random(random_count, 12345u);
	test_skylines(random_count, 777u);
	std::cout << "rounded cut endpoints on the boundary checked: " << rounded_boundary_points << '\n';
	std::cout << checks - failures << '/' << checks << " checks passed\n";
	return failures == 0 ? 0 : 1;
}
