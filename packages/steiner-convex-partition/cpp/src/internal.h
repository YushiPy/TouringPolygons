#pragma once

#include "exact.h"
#include "steiner_convex_partition/steiner_convex_partition.h"

#include <cstddef>
#include <string>
#include <vector>

namespace steiner_convex_partition::detail {

	using exact::Vec2;

	// A simple polygon in counterclockwise order with its notches.
	struct PolygonData {
		std::vector<Vec2> pts;
		std::vector<std::size_t> original_index;
		int n = 0;
		std::vector<int> notch_vertex;
		std::vector<int> vertex_notch;
		int c = 0;

		int wrap(int v) const {
			const int r = v % n;
			return r < 0 ? r + n : r;
		}

		const Vec2 &at(int v) const {
			return pts[static_cast<std::size_t>(wrap(v))];
		}

		const Vec2 &next(int v) const {
			return at(v + 1);
		}

		const Vec2 &prev(int v) const {
			return at(v - 1);
		}
	};

	// Normalizes orientation, drops repeated consecutive vertices and checks
	// simplicity; throws std::invalid_argument on bad input.
	PolygonData prepare_polygon(const Polygon &polygon);

	// A pattern node: a polygon vertex (vertex >= 0) or a Steiner point.
	struct NodeRef {
		int vertex = -1;
		Vec2 point;
	};

	struct PatternEdge {
		NodeRef a;
		NodeRef b;
	};

	struct PatternGeometry {
		PatternKind kind = PatternKind::x2;
		std::vector<int> notch_vertices;
		std::vector<PatternEdge> edges;
		std::size_t steiner_points = 0;
	};

	struct DpOutput {
		std::vector<PatternGeometry> patterns;
		int rotation_min = 0;
		int rotation_max = 0;
		std::size_t junction_calls = 0;
		std::size_t junction_successes = 0;
		std::size_t x4_calls = 0;
		std::size_t x4_successes = 0;
		// With Options::cross_check: S(i, j) by [i * (c + 1) + length].
		std::vector<int> table;
	};

	DpOutput run_dynamic_program(const PolygonData &polygon, const Options &options);

	struct AssemblyOutput {
		Partition pieces;
		std::size_t cut_endpoints = 0;
		bool exact_coordinates = true;
		bool validated = false;
		std::string failure;
	};

	AssemblyOutput assemble(const PolygonData &polygon, const std::vector<PatternGeometry> &patterns, bool validate);

	// Geometry shared by the dynamic program and the assembly.
	bool strictly_inside_corner(const PolygonData &polygon, int vertex, const Vec2 &target);
	bool open_segment_clear(const PolygonData &polygon, const Vec2 &a, const Vec2 &b);
	bool point_on_boundary(const PolygonData &polygon, const Vec2 &point);
}
