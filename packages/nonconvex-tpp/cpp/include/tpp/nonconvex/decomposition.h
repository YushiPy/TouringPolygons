#pragma once

#include "tpp/geometry/common.h"

#include <optional>
#include <string_view>
#include <vector>

namespace tpp {

	enum class ConvexDecomposition {
		// Greene's dynamic program (optimal_convex_partition): pieces have only
		// polygon vertices.
		greene,
		// Chazelle and Dobkin (steiner_convex_partition): minimum number of pieces
		// with Steiner points inside the polygon and on its edges.
		steiner,
	};

	std::optional<ConvexDecomposition> parse_convex_decomposition(std::string_view name);
	const char *convex_decomposition_name(ConvexDecomposition method);

	// Convex pieces whose union is the polygon. With `steiner`, a cut endpoint on
	// an edge that is not a double is rounded on or outside the polygon and the
	// affected pieces are their convex hulls, so the pieces still cover the
	// polygon (they may then exceed it by a rounding error). A polygon that the
	// Steiner decomposition rejects as not simple gets Greene's.
	std::vector<std::vector<Vector2>> decompose_polygon(
		const std::vector<Vector2> &polygon,
		ConvexDecomposition method = ConvexDecomposition::greene
	);
}
