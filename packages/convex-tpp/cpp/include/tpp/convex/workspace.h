#pragma once

#include "vector2.h"

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tpp {

	using Cone = std::pair<Vector2, Vector2>;

	struct ConvexTppWorkspaceView {
		std::span<size_t> polygon_offsets;
		std::span<uint8_t> first_contact;
		std::span<Cone> cones;
	};

	inline size_t total_vertex_count(const std::vector<std::vector<Vector2>> &polygons) {
		size_t total = 0;

		for (const auto &polygon : polygons) {
			total += polygon.size();
		}

		return total;
	}

	class DynamicConvexTppWorkspace {
		public:
		std::vector<size_t> polygon_offsets;
		std::vector<uint8_t> first_contact;
		std::vector<Cone> cones;

		void reserve(size_t max_polygons, size_t max_total_vertices);
		ConvexTppWorkspaceView prepare(size_t polygon_count, size_t total_vertices);
		ConvexTppWorkspaceView view();
	};

	template <size_t MaxPolygons, size_t MaxTotalVertices>
	class StaticConvexTppWorkspace {
		public:
		std::array<size_t, MaxPolygons + 1> polygon_offsets;
		std::array<uint8_t, MaxTotalVertices> first_contact;
		std::array<Cone, MaxTotalVertices> cones;

		ConvexTppWorkspaceView view() {
			return {
				std::span<size_t>(polygon_offsets),
				std::span<uint8_t>(first_contact),
				std::span<Cone>(cones),
			};
		}
	};
}
