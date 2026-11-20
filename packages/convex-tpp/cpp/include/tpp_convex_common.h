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
		// By the signed area, so that a reversed polygon is never clockwise
		// again: the callers normalize and call themselves. The first nonzero
		// turn, used before 2026-11-20, could stay negative after reversal on a
		// region with repeated vertices (a segment written as [a, b, a]), and
		// the call recursed until the stack overflowed. Zero area (a point or
		// a segment) is left as it is, and so is an area within rounding of
		// zero (three nearly collinear vertices): the sum is taken in another
		// order after reversal, and its sign need not flip.
		auto clockwise = [](const std::vector<Vector2> &p) {
			long double area = 0, magnitude = 0;
			for (size_t j = 0; j < p.size(); ++j) {
				const auto &a = p[j], &b = p[(j + 1) % p.size()];
				const long double term = (long double)a.x * b.y - (long double)a.y * b.x;
				area += term;
				magnitude += term < 0 ? -term : term;
			}
			return area < -1e-12L * magnitude;
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
