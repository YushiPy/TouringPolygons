#pragma once

#include "tpp/convex/solution.h"

#include <algorithm>
#include <optional>
#include <vector>

namespace tpp::detail {

	// The disjoint cores assume CCW boundaries. Normalize only when necessary;
	// already-CCW inputs retain the existing solver and storage representation.
	inline std::optional<std::vector<std::vector<Vector2>>> normalized_winding(
			const std::vector<std::vector<Vector2>> &polygons) {
		auto clockwise = [](const std::vector<Vector2> &p) {
			for (size_t j = 0; j < p.size(); ++j) {
				const auto &a = p[j], &b = p[(j + 1) % p.size()], &c = p[(j + 2) % p.size()];
				const long double turn = ((long double)b.x - a.x) * ((long double)c.y - b.y)
					- ((long double)b.y - a.y) * ((long double)c.x - b.x);
				if (turn != 0) return turn < 0;
			}
			return false;
		};
		std::optional<std::vector<std::vector<Vector2>>> result;
		for (size_t i = 0; i < polygons.size(); ++i) {
			if (!clockwise(polygons[i])) continue;
			if (!result) result = polygons;
			std::reverse((*result)[i].begin(), (*result)[i].end());
		}
		return result;
	}
}
