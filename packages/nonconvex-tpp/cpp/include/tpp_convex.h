
#pragma once

#include "vector2.h"

#include <vector>
#include <functional>

namespace tpp {

	using Solver = std::function<std::vector<Vector2>(const Vector2&, const Vector2&, const std::vector<std::vector<Vector2>>&)>;

	std::vector<Vector2> tpp_convex_solve(const Vector2 &start, const Vector2 &target, const std::vector<std::vector<Vector2>> &polygons);
}
