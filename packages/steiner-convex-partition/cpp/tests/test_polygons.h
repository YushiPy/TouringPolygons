#pragma once

// Polygon sources shared by the tests: random generators and the reader of
// TouringPolygons fixture files.

#include <steiner_convex_partition/steiner_convex_partition.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

namespace test_polygons {

	namespace scp = steiner_convex_partition;

	inline std::size_t count_notches(const scp::Polygon &polygon) {
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

	// Vertices at sorted random angles and random radii around the origin.
	inline scp::Polygon random_star(std::mt19937 &rng, int n, bool integer) {
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
	inline scp::Polygon random_skyline(std::mt19937 &rng, int columns) {
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
		for (int i = 0; i < columns; ++i) {
			push(i, bottom[static_cast<std::size_t>(i)]);
			push(i + 1, bottom[static_cast<std::size_t>(i)]);
		}
		for (int i = columns - 1; i >= 0; --i) {
			push(i + 1, top[static_cast<std::size_t>(i)]);
			push(i, top[static_cast<std::size_t>(i)]);
		}
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

	// A convex polygon whose sides carry thin V-shaped notches pointing roughly
	// inward. Narrow notches see little besides the interior, so they are
	// resolved by Steiner points: X3- and X4-patterns are common.
	inline scp::Polygon random_v_notches(std::mt19937 &rng, int sides, bool integer) {
		std::uniform_real_distribution<double> unit(0.0, 1.0);
		std::vector<double> angles(static_cast<std::size_t>(sides));
		for (double &a : angles) a = unit(rng) * 2.0 * std::numbers::pi;
		std::sort(angles.begin(), angles.end());
		const double scale = integer ? 1000.0 : 100.0;
		scp::Polygon polygon;
		auto push = [&](double x, double y) {
			if (integer) {
				x = std::round(x);
				y = std::round(y);
			}
			const scp::Point p {x, y};
			if (!polygon.empty() && polygon.back() == p) return;
			polygon.push_back(p);
		};
		for (int s = 0; s < sides; ++s) {
			const double a0 = angles[static_cast<std::size_t>(s)];
			const double a1 = angles[static_cast<std::size_t>((s + 1) % sides)] + (s + 1 == sides ? 2.0 * std::numbers::pi : 0.0);
			const double x0 = scale * std::cos(a0);
			const double y0 = scale * std::sin(a0);
			const double x1 = scale * std::cos(a1);
			const double y1 = scale * std::sin(a1);
			push(x0, y0);
			const double length = std::hypot(x1 - x0, y1 - y0);
			if (length < 0.2 * scale || unit(rng) < 0.2) continue;
			const double ux = (x1 - x0) / length;
			const double uy = (y1 - y0) / length;
			const double t = 0.3 + 0.4 * unit(rng);
			const double mx = x0 + t * (x1 - x0);
			const double my = y0 + t * (y1 - y0);
			const double half = (0.005 + 0.03 * unit(rng)) * scale;
			const double depth = (0.2 + 0.5 * unit(rng)) * scale;
			const double tilt = (unit(rng) - 0.5) * 1.2;
			// Inward normal of a counterclockwise side, turned by the tilt.
			const double nx = -uy * std::cos(tilt) + ux * std::sin(tilt);
			const double ny = ux * std::cos(tilt) + uy * std::sin(tilt);
			push(mx - ux * half, my - uy * half);
			push(mx + nx * depth, my + ny * depth);
			push(mx + ux * half, my + uy * half);
		}
		return polygon;
	}

	template <class T>
	bool read_value(std::ifstream &input, T &value) {
		input.read(reinterpret_cast<char *>(&value), sizeof(T));
		return static_cast<bool>(input);
	}

	// Every polygon of a TouringPolygons fixture file (start, target, polygons,
	// solution per instance).
	inline std::vector<scp::Polygon> read_fixture(const std::string &path) {
		struct FixturePoint {
			double x;
			double y;
		};
		std::ifstream input(path, std::ios::binary);
		std::vector<scp::Polygon> polygons;
		if (!input) {
			std::cerr << "cannot open " << path << '\n';
			return polygons;
		}
		while (input.peek() != EOF) {
			FixturePoint start {};
			FixturePoint target {};
			std::size_t count = 0;
			if (!read_value(input, start) || !read_value(input, target) || !read_value(input, count)) break;
			for (std::size_t p = 0; p < count; ++p) {
				std::size_t n = 0;
				read_value(input, n);
				scp::Polygon polygon(n);
				for (scp::Point &point : polygon) {
					FixturePoint f {};
					read_value(input, f);
					point = {f.x, f.y};
				}
				polygons.push_back(std::move(polygon));
			}
			std::size_t solution = 0;
			read_value(input, solution);
			for (std::size_t s = 0; s < solution; ++s) {
				FixturePoint f {};
				read_value(input, f);
			}
		}
		return polygons;
	}
}
