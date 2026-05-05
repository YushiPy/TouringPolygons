
#pragma once

#include <vector>

namespace tpp {

	/*
	Determines the intersection point of two line segments (start1, end1) and (start2, end2).
	Returns the intersection point if they intersect, otherwise returns `Vector2::INF`.
	*/
	Vector2 segment_segment_intersection(const Vector2& start1, const Vector2& end1, const Vector2& start2, const Vector2& end2);

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

}
