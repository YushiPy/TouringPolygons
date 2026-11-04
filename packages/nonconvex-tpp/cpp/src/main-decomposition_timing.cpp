// Times the convex decomposition of every nonconvex polygon of each instance in
// a fixture file, with Greene's dynamic program and with Chazelle and Dobkin's
// Steiner decomposition, through tpp::decompose_polygon (the call the branch
// and bound makes lazily).
//
// usage: tpp-decomposition-timing FILE.bin [REPEATS]
// stdout: one CSV line per instance, then totals.

#include "tpp/nonconvex/decomposition.h"
#include "tests.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

	bool is_nonconvex(const std::vector<Vector2> &polygon) {
		const size_t n = polygon.size();
		if (n < 4) return false;
		double area = 0.0;
		for (size_t i = 0; i < n; ++i) area += polygon[i].cross(polygon[(i + 1) % n]);
		for (size_t i = 0; i < n; ++i) {
			const Vector2 &a = polygon[(i + n - 1) % n];
			const Vector2 &b = polygon[i];
			const Vector2 &c = polygon[(i + 1) % n];
			const double turn = (b - a).cross(c - b);
			if (area > 0 ? turn < 0 : turn > 0) return true;
		}
		return false;
	}

	struct Timing {
		double seconds = std::numeric_limits<double>::infinity();
		size_t pieces = 0;
		bool failed = false;
	};

	// Best of `repeats` runs over all the instance's nonconvex polygons.
	Timing time_method(const std::vector<std::vector<Vector2>> &polygons, tpp::ConvexDecomposition method, int repeats) {
		Timing timing;
		for (int r = 0; r < repeats; ++r) {
			size_t pieces = 0;
			const auto began = std::chrono::steady_clock::now();
			for (const auto &polygon : polygons) {
				try {
					pieces += tpp::decompose_polygon(polygon, method).size();
				} catch (const std::exception &) {
					timing.failed = true;
				}
			}
			const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
			timing.seconds = std::min(timing.seconds, seconds);
			timing.pieces = pieces;
		}
		return timing;
	}
}

int main(int argc, char **argv) {
	if (argc < 2) {
		std::cerr << "usage: " << argv[0] << " FILE.bin [REPEATS]\n";
		return 2;
	}
	const int repeats = argc > 2 ? std::max(1, std::atoi(argv[2])) : 3;
	const auto cases = tpp::load_test_cases(argv[1]);
	std::cout << "case,nonconvex_polygons,greene_pieces,steiner_pieces,greene_seconds,steiner_seconds\n";
	double greene_total = 0.0;
	double steiner_total = 0.0;
	size_t greene_pieces = 0;
	size_t steiner_pieces = 0;
	size_t failures = 0;
	std::cout << std::setprecision(9);
	for (size_t index = 0; index < cases.size(); ++index) {
		std::vector<std::vector<Vector2>> nonconvex;
		for (const auto &polygon : cases[index].polygons) {
			if (is_nonconvex(polygon)) nonconvex.push_back(polygon);
		}
		const Timing greene = time_method(nonconvex, tpp::ConvexDecomposition::greene, repeats);
		const Timing steiner = time_method(nonconvex, tpp::ConvexDecomposition::steiner, repeats);
		failures += greene.failed + steiner.failed;
		greene_total += greene.seconds;
		steiner_total += steiner.seconds;
		greene_pieces += greene.pieces;
		steiner_pieces += steiner.pieces;
		std::cout << index << ',' << nonconvex.size() << ',' << greene.pieces << ',' << steiner.pieces << ','
			<< greene.seconds << ',' << steiner.seconds << '\n';
	}
	std::cout << "# total pieces: greene " << greene_pieces << ", steiner " << steiner_pieces
		<< "; total seconds (best of " << repeats << "): greene " << greene_total << ", steiner " << steiner_total
		<< "; failed decompositions " << failures << '\n';
	return failures == 0 ? 0 : 1;
}
