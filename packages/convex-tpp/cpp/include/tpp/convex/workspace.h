#pragma once

#include "tpp/geometry/vec2.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tpp {
	struct ConvexHybridCache;

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
		// Prepared exact polygons for repeated safe-hybrid calls in one search.
		std::shared_ptr<ConvexHybridCache> hybrid_cache;
		// Diagnostic ablation: reuse exact geometry, but repeat pair dispatch.
		bool cache_disjoint_dispatch = true;
		// Reuse the binary coordinates of the same normalized exact polygons.
		bool cache_interval_geometry = true;
		// Borrow immutable exact geometry instead of copying it into each call.
		bool borrow_hybrid_geometry = true;
		bool bound_before_optimality = false;
		// Additional feasible dual proposal for short/coincident contact blocks.
		bool interpolated_zero_dual = false;
        bool retain_binary_dual = false;
		// With a positive gap or a finite cutoff, keep the hybrid interval
		// proof but replace everything after it (exact replay, KKT, filtered
		// and rational recovery) with the binary64 polish of
		// tpp_convex_solve_float_certified; open calls still finish in the
		// hybrid oracle. Off here; the unordered search turns it on.
		bool float_recovery = false;

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
