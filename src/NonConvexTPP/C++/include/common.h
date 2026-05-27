
#pragma once

#include "vector2.h"

#include <vector>
#include <iostream>
#include <format>


namespace std {

	// std::vector<Vector2> -> {{x1, y1}, ..., {xn, yn}}
	template<>
	struct std::formatter<std::vector<Vector2>> : std::formatter<Vector2> {
		auto format(const std::vector<Vector2>& vec, std::format_context& ctx) const {
			auto out = std::format_to(ctx.out(), "{{");
			for (size_t i = 0; i < vec.size(); ++i) {
				if (i > 0) out = std::format_to(out, ", ");
				out = std::formatter<Vector2>::format(vec[i], ctx);
			}
			return std::format_to(out, "}}");
		}
	};

	// std::vector<std::vector<Vector2>> -> {{{x1, y1}, ...}, ..., {{xn, yn}, ...}}
	template<>
	struct std::formatter<std::vector<std::vector<Vector2>>> : std::formatter<std::vector<Vector2>> {
		auto format(const std::vector<std::vector<Vector2>>& mat, std::format_context& ctx) const {
			auto out = std::format_to(ctx.out(), "{{");
			for (size_t i = 0; i < mat.size(); ++i) {
				if (i > 0) out = std::format_to(out, ", ");
				out = std::formatter<std::vector<Vector2>>::format(mat[i], ctx);
			}
			return std::format_to(out, "}}");
		}
	};
}

namespace tpp {

	/*
	Returns the intersection point of segments (start1, end1) and (start2, end2) if they intersect, 
	otherwise returns a "reasonable point".

	If the segments are parallel, returns `start1` as a "reasonable point".
	If the segments do not intersect, returns the intersection point of the lines extended
	from the segments, which is a "reasonable point" for the purposes of the algorithm.
	*/
	Vector2 segment_segment_intersection(const Vector2& start1, const Vector2& end1, const Vector2& start2, const Vector2& end2);

	/*
	Returns the intersection point of segments (start1, end1) and (start2, end2) if they intersect, otherwise returns `Vector2::INF`.
	*/
	Vector2 segment_segment_intersection_safe(const Vector2& start1, const Vector2& end1, const Vector2& start2, const Vector2& end2);

	/*
	Returns the reflection of `point` across the line defined by `vertex1` and `vertex2`.
	*/
	Vector2 reflect_segment(const Vector2& point, const Vector2& vertex1, const Vector2& vertex2);


	/*
	Check if `point` is inside the cone defined by `vertex` and rays `ray1` and `ray2`. 
	The rays are in counter-clockwise order.
	*/
	bool point_in_cone(const Vector2& point, const Vector2& vertex, const Vector2& ray1, const Vector2& ray2);

	/*
	Check if `point` is inside the edge region defined by `ray1` coming from `vertex1` and `ray2` coming from `vertex2`.
	The rays and edge are in counter-clockwise order.
	*/
	bool point_in_edge(const Vector2& point, const Vector2& vertex1, const Vector2& ray1, const Vector2& vertex2, const Vector2& ray2);

	/*
	Check if a `point` is inside the cone defined by `vertex` and rays `ray1` and `ray2`.
	The rays are in counter-clockwise order. 

	This version is used for binary search and considers the case where `ray1` and `ray2` point in the same direction.
	*/
	bool point_in_cone_plus(const Vector2& point, const Vector2& vertex, const Vector2& ray1, const Vector2& ray2);

	/*
	Check if `point` is inside the edge region defined by `ray1` coming from `vertex1` and `ray2` coming from `vertex2`.
	The rays and edge are in counter-clockwise order.

	This version is used for binary search and considers all four cases for the direction of `ray1` and `ray2` 
	with respect to the edge from `vertex1` to `vertex2`. It also considers the case where `ray1` and `ray2` point in the same direction.
	*/
	bool point_in_edge_plus(const Vector2& point, const Vector2& vertex1, const Vector2& vertex2, const Vector2& ray1, const Vector2& ray2);


	/*
	Removes collinear points from a sequence of points.
	*/
	std::vector<Vector2> remove_collinear_points(const std::vector<Vector2>& points);

	/*
	Decomposes a simple polygon into convex pieces using CGAL's optimal convex partitioning algorithm.
	Returns a list of convex pieces, where each piece is represented as a list of its vertices in counter-clockwise order.

	The input polygon must be simple (no self-intersections) and can be either clockwise or counter-clockwise. 
	Furthermore, the last vertex of the input polygon should not be the same as the first vertex, as CGAL expects a sequence of distinct vertices.
	*/
	std::vector<std::vector<Vector2>> decompose_polygon(const std::vector<Vector2> &polygon);

	// operator<< overloads for iostream
	inline std::ostream& operator<<(std::ostream& os, const std::vector<Vector2>& vec) {
		return os << std::format("{}", vec);
	}

	inline std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<Vector2>>& mat) {
		return os << std::format("{}", mat);
	}
}
