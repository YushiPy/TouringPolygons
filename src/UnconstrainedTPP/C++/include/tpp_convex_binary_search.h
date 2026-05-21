
#pragma once

#include <vector>
#include "vector2.h"

namespace tpp {
	std::vector<Vector2> tpp_convex_solve_binary_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
}

