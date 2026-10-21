#pragma once

#include "tpp/geometry/vec2.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tpp::unordered_detail {
	using Polygon = std::vector<Vector2>;

	struct Contact {
		double distance;
		double position;
		double squared_distance = 0;
		// Closest point of the region found for a positive distance. Any point
		// of the region bounds later distances from above (visit anchors).
		Vector2 polygon_point{};
	};
	struct ContactEdge {
		Vector2 start, end, direction;
		double squared;
		ContactEdge(Vector2 start, Vector2 end);
	};
	struct ContactSegment : ContactEdge {
		Vector2 minimum, maximum;
		ContactSegment(Vector2 start, Vector2 end);
	};
	// Immutable preparation; both overloads below use the same query algorithm.
	struct PreparedContactPolygon {
		Vector2 minimum, maximum;
		std::vector<ContactEdge> edges;
		explicit PreparedContactPolygon(const Polygon &polygon);
	};
	struct PreparedContactPath {
		std::vector<ContactSegment> segments;
		explicit PreparedContactPath(const Polygon &path = {});
		void prepare(const Polygon &path);
	};
	class SegmentContactCache {
		using Key = std::array<uint64_t,4>;
		// Direct-mapped: each (segment, polygon) pair has one slot, and a
		// colliding pair overwrites it. Sized once per search; no allocation
		// per query and no wholesale eviction.
		struct Entry {
			Key segment{};
			size_t polygon = std::numeric_limits<size_t>::max();
			Contact contact{0, 0};
		};
		std::vector<Entry> table_;
        std::vector<const PreparedContactPolygon *> polygons_;
        std::optional<uint64_t> tolerance_bits_;
		void invalidate();
	public:
		size_t queries=0,hits=0;
		// upper_distance, when finite, is an upper bound of the path's
		// distance to the polygon (e.g. path_point_distance to one of its
		// points). Segments whose boxes lie beyond it, with a rounding margin,
		// cannot hold the closest or a touching contact and are skipped; the
		// result is the one of a full scan.
		Contact query(const PreparedContactPath &path,const PreparedContactPolygon &polygon,
			size_t polygon_index,size_t polygon_count,double tolerance,
			double upper_distance=std::numeric_limits<double>::infinity());
	};

	double path_length(const Polygon &path);
	Polygon convex_hull(Polygon polygon);
	Contact contact(const Polygon &path, const Polygon &polygon, double tolerance);
	Contact contact(const PreparedContactPath &path, const PreparedContactPolygon &polygon, double tolerance);
	// The polygon counterclockwise and strictly convex, if it is (else nothing).
	std::optional<Polygon> counterclockwise_convex(const Polygon &polygon);
	// Euclidean distance between two counterclockwise strictly convex polygons
	// by GJK, each support an O(log m) binary search: a floating estimate of
	// the distance contact() measures between them, for heuristics only.
	double convex_distance(const Polygon &a, const Polygon &b);
	// Distance from a point to a prepared path, with the projection used by
	// contact(); an upper bound for the path's distance to any region holding it.
	double path_point_distance(const PreparedContactPath &path, Vector2 point);
	enum class PerimeterSamplingWorkModel { AdjacentPairs, AllPairs };
	double perimeter_sampling_work_budget(double log2_complexity);
	std::vector<size_t> choose_perimeter_sample_point_counts(
		const std::vector<Polygon> &polygons, double work_budget,
		PerimeterSamplingWorkModel model);
	Polygon evenly_spaced_perimeter_points(const Polygon &polygon, size_t point_count);
	// Exact one-contact minimization over the polygon boundary, with a feasible
	// preferred point retained when it has the same objective (e.g. pass-through).
	Vector2 best_contact(Vector2 left, Vector2 right, const Polygon &polygon, Vector2 preferred);
}
