#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Minimum convex decomposition of a simple polygon with Steiner points, after
// B. Chazelle and D. P. Dobkin, "Decomposing a polygon into its convex parts"
// (STOC 1979) and "Optimal convex decompositions" (Computational Geometry,
// G. T. Toussaint ed., North-Holland, 1985).
//
// The decomposition removes every notch (reflex vertex) with X-patterns: trees
// of added segments whose leaves and degree-2 nodes are notches and whose
// degree-3 nodes are Steiner points. With p compatible patterns the partition
// has c + 1 - p convex pieces, where c is the number of notches; the dynamic
// program maximizes p over Y-patterns and X4-patterns, which suffice.

namespace steiner_convex_partition {

	struct Point {
		double x = 0.0;
		double y = 0.0;

		constexpr bool operator==(const Point &other) const noexcept {
			return x == other.x && y == other.y;
		}
	};

	using Polygon = std::vector<Point>;
	using Partition = std::vector<Polygon>;

	enum class Algorithm {
		// Procedure ConvDec of Section 3 of the 1985 paper: every geometric test
		// scans the polygon (O(n) per Steiner point, every quadruple for X4).
		polynomial,
		// Section 4: the same dynamic program with geometric tests answered from
		// precomputed superranges, and X4-patterns decided exactly in their loose
		// configuration through the sets E(i, j). Never more pieces than the
		// polynomial version, whose X4 search tries a fixed set of candidates.
		cubic,
	};

	struct Options {
		Algorithm algorithm = Algorithm::polynomial;
		// Testing aid: in the cubic algorithm, also check every Steiner point and
		// diagonal it accepts or rejects with the polynomial algorithm's direct
		// tests, and (up to 40 notches) every set E(i, j) against a search of all
		// quadruples; throw std::logic_error when they disagree.
		bool cross_check = false;
		// Patterns with Steiner points (X3 and larger Y-patterns). Without them
		// only diagonal patterns between notches remain.
		bool steiner_points = true;
		// X4-patterns: two adjacent Steiner points (only with steiner_points).
		bool x4_patterns = true;
		// Exact validation of the returned partition: every piece convex, inside
		// the polygon, pairwise interior-disjoint, and the areas add up.
		bool validate = true;
	};

	enum class PatternKind {
		x2,         // one diagonal between two notches
		diagonals,  // a Y-pattern made of diagonals only (a chain of notches)
		steiner,    // a Y-pattern with at least one Steiner point (X3 included)
		x4,         // two adjacent Steiner points joining four notches
	};

	struct PatternSummary {
		PatternKind kind = PatternKind::x2;
		// Polygon vertex indices of the notches, in the input's indexing.
		std::vector<std::size_t> notches;
		std::size_t steiner_points = 0;
	};

	struct Result {
		// Counterclockwise convex pieces. A vertex is exact unless it is a point
		// where a cut meets an edge and that point is not a double; such points
		// are rounded here and `exact_coordinates` is false. All pieces share each
		// rounded point, on the polygon boundary it is rounded to the outside, and
		// a piece with a rounded point is replaced by its convex hull: the pieces
		// are then a convex cover of the polygon, exceeding it and overlapping by
		// at most rounding errors.
		Partition pieces;
		std::size_t notch_count = 0;
		// Number p of patterns used; pieces.size() == notch_count + 1 - p.
		std::size_t pattern_count = 0;
		std::vector<PatternSummary> patterns;
		// Steiner points inside the polygon, and new points on its boundary or on
		// earlier cuts (endpoints of the cuts that remove the remaining notches).
		std::size_t interior_steiner_points = 0;
		std::size_t cut_endpoints = 0;
		bool exact_coordinates = true;
		// Set when Options::validate ran and the partition passed.
		bool validated = false;
		// The dynamic program solves the circular problem from every starting
		// notch; in exact arithmetic all starts agree. Both values are reported
		// as a consistency diagnostic.
		std::size_t rotation_min_patterns = 0;
		std::size_t rotation_max_patterns = 0;
		// Work counters of the dynamic program: Steiner point searches and X4
		// tests, and how many found a pattern.
		std::size_t junction_searches = 0;
		std::size_t junction_found = 0;
		std::size_t x4_tests = 0;
		std::size_t x4_found = 0;
	};

	// Throws std::invalid_argument when the polygon has fewer than three distinct
	// vertices or is not simple. Either orientation is accepted.
	Result decompose(const Polygon &polygon, const Options &options = {});

	// The pieces of decompose(polygon).
	Partition decompose_polygon(const Polygon &polygon);

	// Exact check that `pieces` is a convex partition of `polygon`; on failure,
	// `reason` (when given) says why. Pieces are checked as given, in doubles.
	bool is_convex_partition(const Polygon &polygon, const Partition &pieces, std::string *reason = nullptr);
}
