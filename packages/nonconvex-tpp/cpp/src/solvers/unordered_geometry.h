#pragma once

#include "tpp/geometry/vec2.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tpp::unordered_detail {
	using Polygon = std::vector<Vector2>;

	struct Contact {
		double distance;
		double position;
		double squared_distance = 0;
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
		struct Hash { size_t operator()(const Key &key) const; };
		std::unordered_map<Key,std::vector<std::optional<Contact>>,Hash> entries_;
        std::vector<const PreparedContactPolygon *> polygons_;
        std::optional<uint64_t> tolerance_bits_;
	public:
		size_t queries=0,hits=0;
		Contact query(const PreparedContactPath &path,const PreparedContactPolygon &polygon,
			size_t polygon_index,size_t polygon_count,double tolerance);
	};

	double path_length(const Polygon &path);
	Polygon convex_hull(Polygon polygon);
	Contact contact(const Polygon &path, const Polygon &polygon, double tolerance);
	Contact contact(const PreparedContactPath &path, const PreparedContactPolygon &polygon, double tolerance);
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
