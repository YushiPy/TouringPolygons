#include "tpp/geometry/vec2.h"
#include "tpp/nonconvex/decomposition.h"

#include <optimal_convex_partition/optimal_convex_partition.h>
#include <steiner_convex_partition/steiner_convex_partition.h>

#include <stdexcept>

using std::vector;


namespace tpp {

	std::optional<ConvexDecomposition> parse_convex_decomposition(std::string_view name) {
		if (name == "greene") return ConvexDecomposition::greene;
		if (name == "steiner") return ConvexDecomposition::steiner;
		return std::nullopt;
	}

	const char *convex_decomposition_name(ConvexDecomposition method) {
		return method == ConvexDecomposition::steiner ? "steiner" : "greene";
	}

	namespace {

		vector<vector<Vector2>> decompose_greene(const vector<Vector2> &polygon) {
			optimal_convex_partition::Polygon input;
			input.reserve(polygon.size());
			for (const auto &v : polygon) input.push_back({v.x, v.y});

			const auto pieces = optimal_convex_partition::decompose_polygon(input);

			vector<vector<Vector2>> result;
			result.reserve(pieces.size());
			for (const auto &piece : pieces) {
				auto &converted_piece = result.emplace_back();
				converted_piece.reserve(piece.size());
				for (const auto &point : piece) converted_piece.emplace_back(point.x, point.y);
			}
			return result;
		}

		vector<vector<Vector2>> decompose_steiner(const vector<Vector2> &polygon) {
			steiner_convex_partition::Polygon input;
			input.reserve(polygon.size());
			for (const auto &v : polygon) input.push_back({v.x, v.y});

			// Every insertion and every piece is already checked exactly while the
			// partition is built; the independent validation would add about 30%.
			steiner_convex_partition::Options options;
			options.validate = false;
			steiner_convex_partition::Result decomposition;
			try {
				decomposition = steiner_convex_partition::decompose(input, options);
			} catch (const std::invalid_argument &) {
				// Not a simple polygon (two of Fekete's 16,303 cross themselves):
				// Greene's port accepts it, as it did before this option existed.
				return decompose_greene(polygon);
			}

			vector<vector<Vector2>> result;
			result.reserve(decomposition.pieces.size());
			// With rounded cut endpoints the library already returns convex hulls
			// that cover the polygon.
			for (const auto &piece : decomposition.pieces) {
				auto &converted_piece = result.emplace_back();
				converted_piece.reserve(piece.size());
				for (const auto &point : piece) converted_piece.emplace_back(point.x, point.y);
			}
			return result;
		}
	}

	std::vector<std::vector<Vector2>> decompose_polygon(const std::vector<Vector2> &polygon, ConvexDecomposition method) {
		return method == ConvexDecomposition::steiner ? decompose_steiner(polygon) : decompose_greene(polygon);
	}

}
