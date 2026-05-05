
#pragma once

#include "vector2.h"

#include <vector>
#include <tuple>

namespace tpp {

	/*
	Generates a random TPP instance with the given number of polygons and vertices per polygon.
	The polygons are generated as regular polygons with random centers and radii, and are placed in a grid-like pattern to avoid overlaps.
	The start and target point are placed at the top-left and bottom-right corners of the grid, respectively.
	*/
	std::tuple<Vector2, Vector2, std::vector<std::vector<Vector2>>> generate_test(const std::vector<size_t> &polygon_sizes);

	/*
	Generates a TPP instance with the same parameters as `generate_test()`, 
	but with the polygons placed in a way that makes the problem more difficult to solve.

	We place the polygons in a line, which makes every query go all the way to the start point,
	this should make the time complexity closer to the worst case of the algorithm.

	The `shuffle` parameter can be set to `true` to randomize the order of the polygon sizes.
	*/
	std::tuple<Vector2, Vector2, std::vector<std::vector<Vector2>>> generate_test_bad(const std::vector<size_t> &polygon_sizes, bool shuffle = false);

	/*
	Generates a random TPP instance with the same parameters as `generate_test()`,
	but with the polygons placed in a way that makes the problem easier to solve.

	We place the polygons such that every path comes from a vertex, thus every query only
	needs to locate a single point.

	The `shuffle` parameter can be set to `true` to randomize the order of the polygon sizes.
	*/
	std::tuple<Vector2, Vector2, std::vector<std::vector<Vector2>>> generate_test_good(const std::vector<size_t> &polygon_sizes, bool shuffle = false);

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

	/*
	Plots the given `solution` to a TPP instance defined by `start`, `target`, and `polygons` using matplotlib. 
	This function calls Python through the command line to generate the plot, so it requires Python and matplotlib to be installed on the system.
	*/
	void plot_solution(const Vector2 &start, const Vector2 &target, const std::vector<std::vector<Vector2>> &polygons, const std::vector<Vector2> &solution);
}
