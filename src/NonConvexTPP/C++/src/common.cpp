
#include "vector2.h"
#include "common.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>
#include <CGAL/point_generators_2.h>

#include <cmath>

#define EPSILON 1e-8
#define EPSILON_SQUARED (EPSILON * EPSILON)


using K      = CGAL::Exact_predicates_inexact_constructions_kernel;
using Traits = CGAL::Partition_traits_2<K>;
using Poly   = Traits::Polygon_2;
using Point  = Traits::Point_2;
using PolyList = std::list<Poly>;

using std::vector;


namespace tpp {

	// Geometry functions
	Vector2 segment_segment_intersection(const Vector2& start1, const Vector2& end1, const Vector2& start2, const Vector2& end2) {

		Vector2 direction1 = end1 - start1;
		Vector2 direction2 = end2 - start2;

		double cross = direction1.cross(direction2);

		if (cross == 0) {
			return start1;
		}

		double rate1 = (start2 - start1).cross(direction2) / cross;
		double rate2 = (start2 - start1).cross(direction1) / cross;

		return start1 + direction1 * rate1;
	}

	Vector2 segment_segment_intersection_safe(const Vector2& start1, const Vector2& end1, const Vector2& start2, const Vector2& end2) {

		Vector2 direction1 = end1 - start1;
		Vector2 direction2 = end2 - start2;

		double cross = direction1.cross(direction2);

		if (cross == 0) {
			return Vector2::INF;
		}

		double rate1 = (start2 - start1).cross(direction2) / cross;
		double rate2 = (start2 - start1).cross(direction1) / cross;

		if (0 <= rate1 && rate1 <= 1 && 0 <= rate2 && rate2 <= 1) {
			return start1 + direction1 * rate1;
		}
		
		return Vector2::INF;
	}

	Vector2 reflect_segment(const Vector2& point, const Vector2& vertex1, const Vector2& vertex2) {
		return vertex1 + (point - vertex1).reflect(vertex2 - vertex1);
	}


	// Convex TPP location functions
	bool point_in_cone(const Vector2& point, const Vector2& vertex, const Vector2& ray1, const Vector2& ray2) {

		bool c1 = ray1.cross(point - vertex) >= -EPSILON_SQUARED;
		bool c2 = ray2.cross(point - vertex) <= EPSILON_SQUARED;

		if (ray1.cross(ray2) >= 0) {
			return c1 && c2;
		} else {
			return c1 || c2;
		}
	}

	bool point_in_edge(const Vector2& point, const Vector2& vertex1, const Vector2& vertex2, const Vector2& ray1, const Vector2& ray2) {
		return ray1.cross(point - vertex1) > 0 && ray2.cross(point - vertex2) < 0 && (vertex2 - vertex1).cross(point - vertex1) <= 0;
	}

	bool point_in_cone_plus(const Vector2& point, const Vector2& vertex, const Vector2& ray1, const Vector2& ray2) {

		if (ray1.is_same_direction(ray2)) {
			return (point - vertex).is_same_direction(ray1);
		}

		if (ray1.cross(ray2) < 0) {
			return ray1.cross(point - vertex) >= -EPSILON_SQUARED || ray2.cross(point - vertex) <= EPSILON_SQUARED;
		} else {
			return ray1.cross(point - vertex) >= -EPSILON_SQUARED && ray2.cross(point - vertex) <= EPSILON_SQUARED;
		}
	}

	bool point_in_edge_plus(const Vector2& point, const Vector2& vertex1, const Vector2& vertex2, const Vector2& ray1, const Vector2& ray2) {

		if (vertex1 == vertex2) {
			return point_in_cone_plus(point, vertex1, ray1, ray2);
		}

		Vector2 dv = vertex2 - vertex1;

		if (ray1.is_same_direction(dv) || ray2.is_same_direction(-dv)) {
			return false;
		}

		Vector2 p1 = point - vertex1;
		Vector2 p2 = point - vertex2;

		if (dv.cross(ray1) < 0) {
			if (dv.cross(ray2) < 0) {
				return ray1.cross(p1) >= -EPSILON_SQUARED && ray2.cross(p2) <= EPSILON_SQUARED && dv.cross(p1) <= EPSILON_SQUARED;
			} else {
				if (dv.cross(p1) < 0) {
					return ray1.cross(p1) >= -EPSILON_SQUARED;
				} else {
					return ray2.cross(p2) <= EPSILON_SQUARED;
				}
			}
		} else {
			if (dv.cross(ray2) < 0) {
				if (dv.cross(p2) < 0) {
					return ray2.cross(p2) <= EPSILON_SQUARED;
				} else {
					return ray1.cross(p1) >= -EPSILON_SQUARED;
				}
			} else {
				return ray1.cross(p1) >= -EPSILON_SQUARED || ray2.cross(p2) <= EPSILON_SQUARED || dv.cross(p1) <= EPSILON_SQUARED;
			}
		}
	}


	// Polygon functions
	std::vector<Vector2> remove_collinear_points(const std::vector<Vector2>& points) {

		if (points.size() <= 2) {
			return points;
		}

		std::vector<Vector2> cleaned = {points[0], points[1]};

		for (size_t i = 2; i < points.size(); i++) {

			auto a = cleaned[cleaned.size() - 2];
			auto b = cleaned[cleaned.size() - 1];
			auto candidate = points[i];

			auto v1 = b - a;
			auto v2 = candidate - b;

			if (std::fabs(v1.cross(v2)) < EPSILON_SQUARED && v1.dot(v2) >= 0) {
				cleaned.back() = candidate;
			} else {
				cleaned.push_back(candidate);
			}
		}

		return cleaned;
	}

	
	std::vector<std::vector<Vector2>> decompose_polygon(const std::vector<Vector2> &polygon) {

		vector<Point> points;

		for (const auto &v : polygon) {
			points.emplace_back(v.x, v.y);
		}

		Poly poly(points.begin(), points.end());

		if (poly.orientation() == CGAL::CLOCKWISE) {
			poly.reverse_orientation();
		}

		// Ensure the polygon is simple (no self-intersections)
		assert(poly.is_simple());

		PolyList pieces;
		CGAL::optimal_convex_partition_2(poly.vertices_begin(), poly.vertices_end(), std::back_inserter(pieces));

		// Validate the partitioning
		assert(CGAL::convex_partition_is_valid_2(
			poly.vertices_begin(), poly.vertices_end(),
			pieces.begin(), pieces.end()
		));

		vector<vector<Vector2>> result;

		for (const auto& p : pieces) {
		
			vector<Vector2> piece;
		
			for (auto v = p.vertices_begin(); v != p.vertices_end(); ++v) {
				piece.emplace_back(v->x(), v->y());
			}

			result.push_back(std::move(piece));
		}

		return result;
	}

}

#undef EPSILON
#undef EPSILON_SQUARED
