
#include "vector2.h"

#include <vector>
#include <tuple>

namespace tpp {

	/*
	Generates a random TPP instance with the given number of polygons and vertices per polygon.
	The polygons are generated as regular polygons with random centers and radii, and are placed in a grid-like pattern to avoid overlaps.
	The start and target point are placed at the top-left and bottom-right corners of the grid, respectively.
	*/
	std::tuple<Vector2, Vector2, std::vector<std::vector<Vector2>>> generate_random_test(const std::vector<size_t> &polygon_sizes);

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
