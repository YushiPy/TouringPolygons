
#include "vector2.h"
#include <vector>

namespace tpp {
	
	/*
	Checks if the given `solution` is a valid solution to the TPP instance 
	defined by `start`, `target`, and `polygons`. 
	
	A valid solution must start:
	- Start at `start`.
	- End at `target`.
	- Have no consecutive collinear points.
	- Visit the polygons in order without skipping any.
	- Bends must occour at only the boundary of the polygons.
	- If a bend occours at a vertex, it must leave at a proper angle.
	- If a bend occours at an edge, it must follow the reflection rule.
	*/
	bool is_valid_solution(const Vector2 &start, const Vector2 & target, const std::vector<std::vector<Vector2>> &polygons, const std::vector<Vector2> &solution);
}
