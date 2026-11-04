// Differential test of the two algorithms. The Section 4 version (cubic)
// decides X4-patterns exactly in their loose configuration (Lemma 16), while
// the Section 3 version (polynomial) tries a fixed set of candidate Steiner
// points; every other pattern is decided the same way. So the cubic result is
// never worse, and is better exactly when the candidates miss an X4-pattern.
// Values of single intervals may differ either way: reducing an X4-pattern to
// a loose one may bring in a notch outside the interval.
//
// The cubic run also cross-checks every superrange answer and every set E(i, j)
// against direct computations.
//
// usage: steiner_convex_partition_cubic_differential [count] [seed] [--max-notches K] [file.bin...]

#include "test_polygons.h"

#include <steiner_convex_partition/steiner_convex_partition.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace scp = steiner_convex_partition;

namespace {

	struct Totals {
		std::size_t compared = 0;
		std::size_t skipped = 0;
		std::size_t worse = 0;
		std::size_t better = 0;
		std::size_t failures = 0;
		std::size_t pieces = 0;
		double polynomial_seconds = 0.0;
		double cubic_seconds = 0.0;
		std::size_t polynomial_junctions = 0;
		std::size_t cubic_junctions = 0;
		std::size_t polynomial_x4 = 0;
		std::size_t cubic_x4 = 0;
		std::size_t x4_patterns = 0;
	};

	double seconds_since(std::chrono::steady_clock::time_point start) {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}

	void compare(const scp::Polygon &polygon, const std::string &name, Totals &totals) {
		scp::Options polynomial;
		scp::Options cubic;
		cubic.algorithm = scp::Algorithm::cubic;
		cubic.cross_check = true;
		scp::Result a;
		auto start = std::chrono::steady_clock::now();
		try {
			a = scp::decompose(polygon, polynomial);
		} catch (const std::invalid_argument &) {
			++totals.skipped;
			return;
		} catch (const std::exception &error) {
			std::cerr << name << ": polynomial threw " << error.what() << '\n';
			++totals.failures;
			return;
		}
		const double polynomial_seconds = seconds_since(start);
		scp::Result b;
		start = std::chrono::steady_clock::now();
		try {
			b = scp::decompose(polygon, cubic);
		} catch (const std::exception &error) {
			std::cerr << name << ": cubic threw " << error.what() << '\n';
			++totals.failures;
			return;
		}
		const double cubic_seconds = seconds_since(start);
		++totals.compared;
		totals.pieces += a.pieces.size();
		totals.polynomial_seconds += polynomial_seconds;
		totals.cubic_seconds += cubic_seconds;
		totals.polynomial_junctions += a.junction_searches;
		totals.cubic_junctions += b.junction_searches;
		totals.polynomial_x4 += a.x4_tests;
		totals.cubic_x4 += b.x4_tests;
		for (const scp::PatternSummary &pattern : a.patterns) totals.x4_patterns += pattern.kind == scp::PatternKind::x4;
		if (!b.validated) {
			std::cerr << name << ": cubic result not validated\n";
			++totals.failures;
		}
		if (b.pieces.size() > a.pieces.size() || b.rotation_min_patterns != b.rotation_max_patterns) {
			std::cerr << name << ": polynomial " << a.pieces.size() << " pieces, cubic " << b.pieces.size()
				<< " (rotations " << b.rotation_min_patterns << ".." << b.rotation_max_patterns
				<< ", n=" << polygon.size() << ", c=" << a.notch_count << ")\n";
			++totals.worse;
		} else if (b.pieces.size() < a.pieces.size()) {
			std::cout << "  cubic better on " << name << ": " << b.pieces.size() << " pieces against " << a.pieces.size() << '\n';
			++totals.better;
		}
	}

	void report(const std::string &label, const Totals &totals) {
		std::cout << label << ": compared " << totals.compared << " (skipped " << totals.skipped << "), "
			<< "cubic worse on " << totals.worse << ", better on " << totals.better << ", " << totals.failures << " failures, "
			<< totals.pieces << " pieces\n"
			<< "  time: polynomial " << std::fixed << std::setprecision(2) << totals.polynomial_seconds
			<< " s, cubic " << totals.cubic_seconds << " s" << std::defaultfloat << '\n'
			<< "  junction searches: polynomial " << totals.polynomial_junctions << ", cubic " << totals.cubic_junctions
			<< "; X4 tests: polynomial " << totals.polynomial_x4 << ", cubic " << totals.cubic_x4
			<< "; X4-patterns used " << totals.x4_patterns << '\n';
	}
}

int main(int argc, char **argv) {
	int count = 200;
	unsigned seed = 4242u;
	std::size_t max_notches = 1000000;
	std::vector<std::string> files;
	int positional = 0;
	for (int a = 1; a < argc; ++a) {
		const std::string arg = argv[a];
		if (arg == "--max-notches" && a + 1 < argc) {
			max_notches = static_cast<std::size_t>(std::atol(argv[++a]));
		} else if (arg.ends_with(".bin")) {
			files.push_back(arg);
		} else if (positional == 0) {
			count = std::atoi(argv[a]);
			++positional;
		} else {
			seed = static_cast<unsigned>(std::atoi(argv[a]));
			++positional;
		}
	}

	bool ok = true;
	std::mt19937 rng(seed);
	Totals random;
	for (int t = 0; t < count; ++t) {
		scp::Polygon polygon;
		switch (t % 5) {
			case 0: polygon = test_polygons::random_star(rng, 6 + static_cast<int>(rng() % 30), false); break;
			case 1: polygon = test_polygons::random_star(rng, 6 + static_cast<int>(rng() % 30), true); break;
			case 2: polygon = test_polygons::random_skyline(rng, 3 + static_cast<int>(rng() % 10)); break;
			case 3: polygon = test_polygons::random_v_notches(rng, 4 + static_cast<int>(rng() % 9), false); break;
			default: polygon = test_polygons::random_v_notches(rng, 4 + static_cast<int>(rng() % 9), true); break;
		}
		if (polygon.size() < 3) continue;
		compare(polygon, "random " + std::to_string(t), random);
	}
	report("random polygons", random);
	ok = ok && random.worse == 0 && random.failures == 0;

	for (const std::string &file : files) {
		Totals totals;
		const std::vector<scp::Polygon> polygons = test_polygons::read_fixture(file);
		for (std::size_t index = 0; index < polygons.size(); ++index) {
			if (test_polygons::count_notches(polygons[index]) > max_notches) {
				++totals.skipped;
				continue;
			}
			compare(polygons[index], file + " polygon " + std::to_string(index), totals);
		}
		report(file, totals);
		ok = ok && totals.worse == 0 && totals.failures == 0;
	}
	return ok ? 0 : 1;
}
