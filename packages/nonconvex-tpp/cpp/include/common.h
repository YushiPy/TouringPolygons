#pragma once

#include "tpp/geometry/common.h"

#include <vector>

namespace tpp {

	std::vector<std::vector<Vector2>> decompose_polygon(const std::vector<Vector2> &polygon);
}

