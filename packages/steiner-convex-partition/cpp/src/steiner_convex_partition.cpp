#include "steiner_convex_partition/steiner_convex_partition.h"

#include "exact_geometry.h"
#include "internal.h"

#include <stdexcept>
#include <utility>

namespace steiner_convex_partition {

	Result decompose(const Polygon &polygon, const Options &options) {
		const detail::PolygonData data = detail::prepare_polygon(polygon);
		const detail::DpOutput dp = detail::run_dynamic_program(data, options);
		const detail::AssemblyOutput assembly = detail::assemble(data, dp.patterns, options.validate);
		if (!assembly.failure.empty()) {
			throw std::logic_error("steiner_convex_partition: " + assembly.failure);
		}

		Result result;
		result.pieces = assembly.pieces;
		result.notch_count = static_cast<std::size_t>(data.c);
		result.pattern_count = dp.patterns.size();
		result.cut_endpoints = assembly.cut_endpoints;
		result.exact_coordinates = assembly.exact_coordinates;
		result.validated = assembly.validated;
		result.rotation_min_patterns = static_cast<std::size_t>(dp.rotation_min);
		result.rotation_max_patterns = static_cast<std::size_t>(dp.rotation_max);
		result.junction_searches = dp.junction_calls;
		result.junction_found = dp.junction_successes;
		result.x4_tests = dp.x4_calls;
		result.x4_found = dp.x4_successes;
		for (const detail::PatternGeometry &pattern : dp.patterns) {
			PatternSummary summary;
			summary.kind = pattern.kind;
			summary.steiner_points = pattern.steiner_points;
			for (int v : pattern.notch_vertices) {
				summary.notches.push_back(data.original_index[static_cast<std::size_t>(v)]);
			}
			result.interior_steiner_points += pattern.steiner_points;
			result.patterns.push_back(std::move(summary));
		}
		return result;
	}

	Partition decompose_polygon(const Polygon &polygon) {
		return decompose(polygon).pieces;
	}

	bool is_convex_partition(const Polygon &polygon, const Partition &pieces, std::string *reason) {
		detail::PolygonData data;
		try {
			data = detail::prepare_polygon(polygon);
		} catch (const std::invalid_argument &error) {
			if (reason) *reason = error.what();
			return false;
		}
		exact::Frame frame;
		for (const exact::Vec2 &p : data.pts) frame.include(p);
		for (const Polygon &piece : pieces) {
			for (const Point &p : piece) frame.include(exact::Vec2 {p.x, p.y});
		}
		std::vector<exact::HPoint> boundary;
		for (const exact::Vec2 &p : data.pts) boundary.push_back(frame.point(p));
		std::vector<std::vector<exact::HPoint>> rings;
		for (const Polygon &piece : pieces) {
			std::vector<exact::HPoint> ring;
			for (const Point &p : piece) {
				const exact::Vec2 v {p.x, p.y};
				if (!ring.empty() && exact::same_point(ring.back(), frame.point(v))) continue;
				ring.push_back(frame.point(v));
			}
			while (ring.size() > 1 && exact::same_point(ring.front(), ring.back())) ring.pop_back();
			rings.push_back(std::move(ring));
		}
		const exact::ValidationReport report = exact::validate_partition(boundary, rings, frame);
		if (!report.ok && reason) *reason = report.reason;
		return report.ok;
	}
}
