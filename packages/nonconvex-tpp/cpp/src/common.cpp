
#include "vector2.h"
#include "common.h"

#include <optimal_convex_partition/optimal_convex_partition.h>

using std::vector;


namespace tpp {

	std::vector<std::vector<Vector2>> decompose_polygon(const std::vector<Vector2> &polygon) {

		optimal_convex_partition::Polygon input;
		input.reserve(polygon.size());

		for (const auto &v : polygon) {
			input.push_back({v.x, v.y});
		}

		const auto pieces = optimal_convex_partition::decompose_polygon(input);

		vector<vector<Vector2>> result;
		result.reserve(pieces.size());

		for (const auto &piece : pieces) {
			auto &converted_piece = result.emplace_back();
			converted_piece.reserve(piece.size());

			for (const auto &point : piece) {
				converted_piece.emplace_back(point.x, point.y);
			}
		}

		return result;
	}

}
