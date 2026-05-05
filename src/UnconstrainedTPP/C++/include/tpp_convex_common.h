
#pragma once

#include "vector2.h"
#include "common.h"


namespace tpp {

	class Solution {

		public:

		const Vector2 &start;
		const Vector2 &target;
		const std::vector<std::vector<Vector2>> &polygons;

		std::vector<std::vector<bool>> first_contact;
		std::vector<std::vector<std::pair<Vector2, Vector2>>> cones;

		Solution(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons);

		/*
		Returns `cones[i][j] = (ray1, ray2)`. If the rays haven't been computed yet, computes the cone of vertex `j` in polygon `i` 
		and stores it in `cones[i][j]` before returning it. It also uses this value to update the `first_contact` array.
		
		`cones` should be initialized with `(Vector2.NAN, Vector2.NAN)` for all vertices before calling
		any of the `locate_point` or `query` functions, and should only
		be accessed through this function to ensure that the cones are properly computed and cached.
		*/
		std::pair<Vector2, Vector2> &get_cone(size_t i, size_t j);

		/*
		Returns the `i`-path to `point`, i.e. the shortest path from
		`start` that visits all polygons from `1` to `i` (inclusive) in order and ends at `point`.

		The last point, i.e. `point` is not included in the returned path, 
		is it the caller's responsibility to add it to the path if needed.
		
		The function should be called with an empty `accumulator` vector, 
		and the path will be constructed in the `accumulator` vector.
		*/
		void query_full(const Vector2 &point, size_t i, std::vector<Vector2> &accumulator);

		/*
		Returns the `i`-path to `point`, i.e. the shortest path from
		`start` that visits all polygons from `1` to `i` (inclusive) in order and ends at `point`.
		*/
		std::vector<Vector2> query_full(const Vector2& point, size_t i);

		/*
		Returns the last step of the `i`-path to `point`, i.e. the point that precedes
		`point` in the shortest path from `start` that visits all polygons from `1` to `i` (inclusive) 
		in order and ends at `point`.
		*/
		Vector2 query(const Vector2& point, size_t i);

		/*
		Returns the shortest path from `start` to `target` that visits all polygons in `polygons` in order.
		*/
		std::vector<Vector2> solve();

		/*
		Locates `point` in the shortest last step map of `i (0 to k)` 
		and returns the index of the region as follows:
		- `2n` if the point is in the region of vertex `n`
		- `2n + 1` if the point is between vertices `n` and `n + 1`.
		- `-1` if the point is in the pass through region.
		*/
		virtual int64_t locate_point(const Vector2& point, size_t i) = 0;

		/*
		This function is called before the last call to `query_full` in `solve`, 
		and can be used to preload any data that is needed for the queries, such as the cones.

		Depending on the implementation, this function could be empty, making it a pure recursive solution,
		or it could be used to compute the cones for all vertices in all polygons, 
		making it a dinamic programming solution.
		*/
		virtual void preload_cones() {}

		virtual ~Solution() = default;
	};
}
