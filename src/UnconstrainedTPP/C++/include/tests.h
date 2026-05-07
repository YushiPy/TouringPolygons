
#pragma once

#include "vector2.h"

#include <vector>
#include <tuple>

namespace tpp {

	struct TestCase {
		Vector2 start, target;
		std::vector<std::vector<Vector2>> polygons;
		std::vector<Vector2> solution;
	};

	/*
	Sets the seed for the random number generator used in the test case generation functions (`generate_test()`).
	By default, the seed is set to the current time since epoch, which means that the generated test cases 
	will be different each time the program is run.

	Setting a fixed seed can be useful for debugging or for generating the same test cases across 
	different runs of the program.
	*/
	void set_rng_seed(unsigned int seed);

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

	/*
	Encodes the given TPP instance defined by `start`, `target`, `polygons` and its `solution` into a 
	binary format that can be saved to a file or transmitted over a network.
	
	The encoding format is as follows:
	- 16 bytes: Start point (2 doubles)
	- 16 bytes: Target point (2 doubles)
	- 8 bytes: Number of polygons (size_t)
	- For each polygon:
		- 8 bytes: Number of vertices (size_t)
		- For each vertex:
			- 16 bytes: Vertex position (2 doubles)
	- 8 bytes: Number of points in the solution (size_t)
	- For each point in the solution:
		- 16 bytes: Point position (2 doubles)
	*/
	std::vector<std::byte> encode_test(const Vector2 &start, const Vector2 &target, const std::vector<std::vector<Vector2>> &polygons, const std::vector<Vector2> &solution = {});

	/*
	Decodes a TPP instance and its solution from a `data` buffer and updates the `offset` to
	the position after the decoded data.

	The input `data` is expected to be in the following format:
	- 16 bytes: Start point (2 doubles)
	- 16 bytes: Target point (2 doubles)
	- 8 bytes: Number of polygons (size_t)
	- For each polygon:
		- 8 bytes: Number of vertices (size_t)
		- For each vertex:
			- 16 bytes: Vertex position (2 doubles)
	- 8 bytes: Number of points in the solution (size_t)
	- For each point in the solution:
		- 16 bytes: Point position (2 doubles)

	Otherwise, the behavior is undefined (e.g. if the data is malformed or does not follow the expected format).
	*/
	TestCase decode_test(const std::byte *data, size_t &offset);

	/*
	Decodes a TPP instance and its solution from a binary input stream.

	The input `data` is expected to be in the following format:
	- 16 bytes: Start point (2 doubles)
	- 16 bytes: Target point (2 doubles)
	- 8 bytes: Number of polygons (size_t)
	- For each polygon:
		- 8 bytes: Number of vertices (size_t)
		- For each vertex:
			- 16 bytes: Vertex position (2 doubles)
	- 8 bytes: Number of points in the solution (size_t)
	- For each point in the solution:
		- 16 bytes: Point position (2 doubles)

	Otherwise, the behavior is undefined (e.g. if the data is malformed or does not follow the expected format).
	*/
	TestCase decode_test(std::istream &ifs);
}
