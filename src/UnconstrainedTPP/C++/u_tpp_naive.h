#pragma once

#include "vector2.h"
#include <vector>

std::vector<Vector2> tpp_solve(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);
