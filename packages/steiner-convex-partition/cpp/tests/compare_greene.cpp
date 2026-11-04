// Compares the Steiner decomposition with the standalone port of Greene's
// algorithm (optimal_convex_partition) on TouringPolygons fixture files.
//
// usage: steiner_convex_partition_compare [--max-notches K] [--wins FILE] file.bin...
//
// For every polygon: both piece counts, the time of each, and our exact
// validation. A polygon where we use more pieces than Greene is a failure.

#include "../src/exact.h"

#include <optimal_convex_partition/optimal_convex_partition.h>
#include <steiner_convex_partition/steiner_convex_partition.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace scp = steiner_convex_partition;
namespace ocp = optimal_convex_partition;

namespace {

	struct FixturePoint {
		double x;
		double y;
	};

	template <class T>
	bool read_value(std::ifstream &input, T &value) {
		input.read(reinterpret_cast<char *>(&value), sizeof(T));
		return static_cast<bool>(input);
	}

	std::vector<scp::Polygon> read_fixture(const std::string &path) {
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

	// Rounded cut endpoints on a polygon edge lie on or outside it (so the pieces
	// cover the polygon); returns the number of such points that do not.
	std::size_t uncovered_boundary_points(const scp::Polygon &input, const scp::Result &result, std::size_t &checked) {
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
		std::size_t bad = 0;
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
					++checked;
					if (steiner_convex_partition::exact::orient(a, b, q) > 0) ++bad;
				}
			}
		}
		return bad;
	}

	double seconds_since(std::chrono::steady_clock::time_point start) {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}

	const char *kind_name(scp::PatternKind kind) {
		switch (kind) {
			case scp::PatternKind::x2: return "X2";
			case scp::PatternKind::diagonals: return "Y-diagonals";
			case scp::PatternKind::steiner: return "Y-Steiner";
			case scp::PatternKind::x4: return "X4";
		}
		return "?";
	}
}

int main(int argc, char **argv) {
	std::size_t max_notches = 1000000;
	std::string wins_path;
	std::vector<std::string> files;
	for (int a = 1; a < argc; ++a) {
		const std::string arg = argv[a];
		if (arg == "--max-notches" && a + 1 < argc) {
			max_notches = static_cast<std::size_t>(std::atol(argv[++a]));
		} else if (arg == "--wins" && a + 1 < argc) {
			wins_path = argv[++a];
		} else {
			files.push_back(arg);
		}
	}
	if (files.empty()) {
		std::cerr << "usage: " << argv[0] << " [--max-notches K] [--wins FILE] file.bin...\n";
		return 2;
	}

	std::ofstream wins;
	if (!wins_path.empty()) wins.open(wins_path);

	int status = 0;
	for (const std::string &file : files) {
		const std::vector<scp::Polygon> polygons = read_fixture(file);
		std::size_t compared = 0;
		std::size_t skipped = 0;
		std::size_t invalid = 0;
		std::size_t better = 0;
		std::size_t better_interior = 0;
		std::size_t better_boundary = 0;
		std::size_t better_vertices = 0;
		std::size_t worse = 0;
		std::size_t inexact = 0;
		std::size_t boundary_checked = 0;
		std::size_t uncovered = 0;
		std::size_t inconsistent = 0;
		std::size_t greene_total = 0;
		std::size_t ours_total = 0;
		double greene_time = 0.0;
		double ours_time = 0.0;
		double slowest = 0.0;
		std::size_t slowest_index = 0;
		std::map<std::string, std::size_t> kinds;

		for (std::size_t index = 0; index < polygons.size(); ++index) {
			const scp::Polygon &polygon = polygons[index];
			ocp::Polygon greene_input;
			for (const scp::Point &p : polygon) greene_input.push_back({p.x, p.y});

			if (count_notches(polygon) > max_notches) {
				++skipped;
				continue;
			}
			scp::Result ours;
			auto start = std::chrono::steady_clock::now();
			try {
				ours = scp::decompose(polygon);
			} catch (const std::invalid_argument &) {
				++skipped;
				continue;
			} catch (const std::exception &error) {
				std::cerr << file << " polygon " << index << ": ours failed: " << error.what() << '\n';
				++invalid;
				status = 1;
				continue;
			}
			const double ours_seconds = seconds_since(start);

			ocp::Partition greene;
			start = std::chrono::steady_clock::now();
			try {
				greene = ocp::decompose_polygon(greene_input);
			} catch (const std::exception &error) {
				std::cerr << file << " polygon " << index << ": Greene failed: " << error.what() << '\n';
				++skipped;
				continue;
			}
			const double greene_seconds = seconds_since(start);

			++compared;
			greene_time += greene_seconds;
			ours_time += ours_seconds;
			if (ours_seconds > slowest) {
				slowest = ours_seconds;
				slowest_index = index;
			}
			if (!ours.validated) {
				std::cerr << file << " polygon " << index << ": not validated\n";
				++invalid;
				status = 1;
			}
			if (!ours.exact_coordinates) {
				++inexact;
				const std::size_t bad = uncovered_boundary_points(polygon, ours, boundary_checked);
				if (bad) {
					std::cerr << file << " polygon " << index << ": " << bad << " rounded boundary points inside the polygon\n";
					uncovered += bad;
					status = 1;
				}
			}
			if (ours.rotation_min_patterns != ours.rotation_max_patterns) ++inconsistent;
			for (const scp::PatternSummary &pattern : ours.patterns) ++kinds[kind_name(pattern.kind)];
			greene_total += greene.size();
			ours_total += ours.pieces.size();
			if (ours.pieces.size() > greene.size()) {
				++worse;
				status = 1;
				std::cerr << file << " polygon " << index << ": ours " << ours.pieces.size()
					<< " pieces, Greene " << greene.size() << " (n=" << polygon.size()
					<< ", c=" << ours.notch_count << ")\n";
			} else if (ours.pieces.size() < greene.size()) {
				++better;
				const char *source = "vertices only";
				if (ours.interior_steiner_points > 0) {
					++better_interior;
					source = "interior Steiner points";
				} else if (ours.cut_endpoints > 0) {
					++better_boundary;
					source = "cut endpoints on edges";
				} else {
					++better_vertices;
				}
				std::cout << "  better: polygon " << index << ": " << ours.pieces.size() << " pieces against Greene's "
					<< greene.size() << " (n=" << polygon.size() << ", c=" << ours.notch_count
					<< ", interior Steiner points " << ours.interior_steiner_points
					<< ", cut endpoints " << ours.cut_endpoints << "; " << source << ")\n";
				if (wins) {
					wins << "# " << file << " polygon " << index << " ours " << ours.pieces.size()
						<< " greene " << greene.size() << " interior " << ours.interior_steiner_points
						<< " cut_endpoints " << ours.cut_endpoints << '\n';
					wins << std::setprecision(17);
					for (const scp::Point &p : polygon) wins << p.x << ' ' << p.y << '\n';
					wins << "pieces\n";
					for (const scp::Polygon &piece : ours.pieces) {
						for (const scp::Point &p : piece) wins << p.x << ' ' << p.y << ' ';
						wins << '\n';
					}
					wins << "greene\n";
					for (const ocp::Polygon &piece : greene) {
						for (const ocp::Point &p : piece) wins << p.x << ' ' << p.y << ' ';
						wins << '\n';
					}
					wins << "end\n";
				}
			}
		}

		std::cout << file << ":\n"
			<< "  compared " << compared << " polygons (skipped " << skipped << ")\n"
			<< "  pieces: ours " << ours_total << ", Greene " << greene_total << '\n'
			<< "  ours better on " << better << " (with interior Steiner points " << better_interior
			<< ", with cut endpoints on edges only " << better_boundary << ", with polygon vertices only " << better_vertices
			<< "), worse on " << worse << ", invalid " << invalid << '\n'
			<< "  inexact cut endpoints in " << inexact << " (rounded boundary points checked " << boundary_checked
			<< ", inside the polygon " << uncovered << "), rotation disagreement in " << inconsistent << '\n'
			<< "  time: ours " << std::fixed << std::setprecision(2) << ours_time << " s, Greene " << greene_time
			<< " s; slowest ours " << slowest << " s (polygon " << slowest_index << ")\n"
			<< std::defaultfloat << "  patterns:";
		for (const auto &[name, count] : kinds) std::cout << ' ' << name << '=' << count;
		std::cout << '\n';
	}
	return status;
}
