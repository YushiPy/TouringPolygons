
#pragma once

#include "tpp_convex_linear_search.h"
#include "tpp_convex_binary_search.h"
#include "tpp_convex_tamc.h"

#include <functional>

namespace tpp {

	using Solver = std::function<std::vector<Vector2>(const Vector2&, const Vector2&, const std::vector<std::vector<Vector2>>&)>;

	auto tpp_convex_solve = tpp_convex_solve_binary_search;
}
