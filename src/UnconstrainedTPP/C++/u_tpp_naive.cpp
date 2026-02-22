
#include "vector2.h"

#include <vector>
#include <utility>
#include <print>
#include <format>
#define print(...) std::println(__VA_ARGS__)


using std::vector;
using std::pair;

Vector2 segment_segment_intersection(const Vector2& start1, const Vector2& end1, const Vector2& start2, const Vector2& end2) {
	/*
	Returns the intersection point of segments (start1, end1) and (start2, end2) if they intersect, otherwise returns `Vector2::INF`.
	*/

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

bool point_in_cone(const Vector2& point, const Vector2& vertex, const Vector2& ray1, const Vector2& ray2) {
	/*
	Check if `point` is inside the cone defined by `vertex` and rays `ray1` and `ray2`. 
	The rays are in counter-clockwise order.
	*/

	if (ray1.cross(ray2) == 0 && ray1.dot(ray2) >= 0) {
		return ray1.cross(point - vertex) == 0 && ray1.dot(point - vertex) >= 0;
	}

	if (ray1.cross(ray2) >= 0) {
		return ray1.cross(point - vertex) >= 0 && ray2.cross(point - vertex) <= 0;
	} else {
		return ray1.cross(point - vertex) >= 0 || ray2.cross(point - vertex) <= 0;
	}
}

bool point_in_edge(const Vector2& point, const Vector2& vertex1, const Vector2& ray1, const Vector2& vertex2, const Vector2& ray2) {
	/*
	Check if `point` is inside the edge region defined by `ray1` coming from `vertex1` and `ray2` coming from `vertex2`.
	The rays and edge are in counter-clockwise order.
	*/
	return ray1.cross(point - vertex1) > 0 && ray2.cross(point - vertex2) < 0 && (vertex2 - vertex1).cross(point - vertex1) <= 0;
}

Vector2 reflect_segment(const Vector2& point, const Vector2& vertex1, const Vector2& vertex2) {
	/*
	Returns the reflection of `point` across the line defined by `vertex1` and `vertex2`.
	*/
	return vertex1 + (point - vertex1).reflect(vertex2 - vertex1);
}


class Solution {

	public:

	const Vector2 start;
	const Vector2 target;
	const vector<vector<Vector2>> polygons;

	vector<vector<bool>> first_contact;
	vector<vector<pair<Vector2, Vector2>>> cones;

	Solution(const Vector2& start, const Vector2& target, const vector<vector<Vector2>>& polygons) : start(start), target(target), polygons(polygons), first_contact(), cones() {}

	int locate_point(const Vector2& point, int i) const {
		/*
		Locates `point` in the shortest last step map of `i` and returns the index of the region as follows:
		- `2n` if the point is in the region of vertex `n`
		- `2n + 1` if the point is between vertices `n` and `n + 1`.
		- `-1` if the point is in the pass through region.
		*/

		auto polygon = polygons[i - 1];
		auto first_contact = this->first_contact[i - 1];
		auto cones = this->cones[i - 1];
		
		for (size_t j = 0; j < polygon.size(); j++) {

			if (!first_contact[j] && !first_contact[(j - 1 + first_contact.size()) % first_contact.size()]) {
				continue;
			}

			auto v = polygon[j];
			auto [ray1, ray2] = cones[j];

			if (point_in_cone(point, v, ray1, ray2)) {
				return 2 * j;
			}
		}

		for (size_t j = 0; j < polygon.size(); j++) {

			if (!first_contact[j]) {
				continue;
			}

			auto v1 = polygon[j];
			auto v2 = polygon[(j + 1) % polygon.size()];

			auto ray1 = cones[j].second;
			auto ray2 = cones[(j + 1) % cones.size()].first;

			if (point_in_edge(point, v1, ray1, v2, ray2)) {
				return 2 * j + 1;
			}
		}

		return -1;
	}

	vector<Vector2> query_full(const Vector2& point, int i) const {
		/*
		Returns the `i`-path to `point`.
		*/

		if (i == 0) {
			return {start, point};
		}

		int location = locate_point(point, i);

		if (location == -1) {
			return query_full(point, i - 1);
		}
		
		auto polygon = polygons[i - 1];
		int pos = location / 2;
		
		if (location % 2 == 0) {
			auto path = query_full(polygon[pos], i - 1);
			path.push_back(point);
			return path;
		}

		auto v1 = polygon[pos];
		auto v2 = polygon[(pos + 1) % polygon.size()];

		auto reflected = reflect_segment(point, v1, v2);

		auto path = query_full(reflected, i - 1);
		path.pop_back();

		auto last = path.back();
		auto intersection = segment_segment_intersection(last, reflected, v1, v2);

		if (!intersection.is_finite()) {
			throw std::runtime_error(
				std::format("Intersection not found for point {} in polygon {} at edge {}", point, i, pos)
			);
		}

		path.push_back(intersection);
		path.push_back(point);

		return path;
	}

	Vector2 query(const Vector2& point, int i) const {
		/*
		Returns the last step of the `i`-path to `point`.
		*/

		if (i == 0) {
			return start;
		}

		int location = locate_point(point, i);

		if (location == -1) {
			return query(point, i - 1);
		}
		
		auto polygon = polygons[i - 1];
		int pos = location / 2;
		
		if (location % 2 == 0) {
			return polygon[pos];
		}

		auto v1 = polygon[pos];
		auto v2 = polygon[(pos + 1) % polygon.size()];

		auto reflected = reflect_segment(point, v1, v2);
		
		auto last = query(reflected, i - 1);
		auto intersection = segment_segment_intersection(last, reflected, v1, v2);

		if (!intersection.is_finite()) {
			throw std::runtime_error(
				std::format("Intersection not found for point {} in polygon {} at edge {}", point, i, pos)
			);
		}

		return intersection;
	}

	vector<bool> get_first_contact_region(int i) const {
		/*
		Returns the first contact region of polygon `i`.
		*/

		vector<bool> region;
		auto polygon = polygons[i - 1];

		for (size_t j = 0; j < polygon.size(); j++) {

			auto v1 = polygon[j];
			auto v2 = polygon[(j + 1) % polygon.size()];

			auto last = query(v1, i - 1);

			region.push_back((v2 - v1).cross(last - v1) < 0);
		}

		return region;
	}

	vector<pair<Vector2, Vector2>> get_last_step_map(int i) const {
		/*Returns the last step map of polygon `i`.*/
		
		vector<pair<Vector2, Vector2>> map;
		auto polygon = polygons[i - 1];
		auto first_contact = this->first_contact[i - 1];

		for (size_t j = 0; j < polygon.size(); j++) {

			auto before = polygon[(j - 1 + polygon.size()) % polygon.size()];
			auto vertex = polygon[j];
			auto after = polygon[(j + 1) % polygon.size()];
			
			auto last = query(vertex, i - 1);
			auto diff = (vertex - last);

			auto ray1 = diff.reflect(vertex - before);
			auto ray2 = diff.reflect(vertex - after);
	
			
			if (!first_contact[((j - 1) + first_contact.size()) % first_contact.size()]) {
				ray1 = diff;
			}
			
			if (!first_contact[j]) {
				ray2 = diff;
			}

			map.emplace_back(ray1.normalized(), ray2.normalized());
		}

		return map;
	}

	vector<Vector2> solve() {
		/*
		Returns the shortest path from `start` to `target` that visits all polygons in order.
		*/

		this->first_contact.clear();
		this->cones.clear();

		for (size_t i = 1; i <= polygons.size(); i++) {
			first_contact.push_back(get_first_contact_region(i));
			cones.push_back(get_last_step_map(i));
		}

		return query_full(target, polygons.size());
	}
};


int main() {

	/*
	(
	Vector2(5, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(2, 4), Vector2(1, 4), Vector2(-1, 1)]),
		Polygon2([Vector2(3, 3), Vector2(4, 3), Vector2(4, 4), Vector2(3, 4)]),
		Polygon2([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)]),
	]
	)
	*/

	Vector2 start(5, 1);
	Vector2 target(7, 3);
	vector<vector<Vector2>> polygons = {
		{Vector2(3, 0), Vector2(2, 4), Vector2(1, 4), Vector2(-1, 1)},
		{Vector2(3, 3), Vector2(4, 3), Vector2(4, 4), Vector2(3, 4)},
		{Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)},
	};

	Solution solution(start, target, polygons);

	auto path = solution.solve();
	print("Path:");
	for (const auto& point : path) {
		print("{}", point);
	}
	for (const auto& [ray1, ray2] : solution.cones[0]) {
		// print("Cone: {} to {}", ray1, ray2);
	}
}
