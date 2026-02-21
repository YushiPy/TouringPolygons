
#include "vector2.h"

#include <print>
#define print(...) std::println(__VA_ARGS__)

Vector2 segment_segment_intersection(const Vector2 &start1, const Vector2 &end1, const Vector2 &start2, const Vector2 &end2) {
	/*
	Returns the intersection point of two line segments defined by `start1` to `end1` and `start2` to `end2`.
	If the segments are parallel or do not intersect, returns a vector with `NAN` components
	*/

	Vector2 dir1 = end1 - start1;
	Vector2 dir2 = end2 - start2;
	
	double det = dir1.x * dir2.y - dir1.y * dir2.x;

	if (det == 0) {
		return Vector2{NAN, NAN}; // Parallel lines
	}
	
	double t = ((start2 - start1).x * dir2.y - (start2 - start1).y * dir2.x) / det;
	double u = ((start2 - start1).x * dir1.y - (start2 - start1).y * dir1.x) / det;
	
	if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
		return start1 + t * dir1; // Intersection point
	}
	
	return Vector2{NAN, NAN}; // No intersection
}

int main() {
	Vector2 v(1, 2);
	print("{}", v);
}
