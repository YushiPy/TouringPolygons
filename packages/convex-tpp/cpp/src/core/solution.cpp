
#include "vector2.h"
#include "common.h"
#include "tpp_convex_common.h"

#include <algorithm>
#include <span>

using std::vector;

namespace {

	constexpr double LOCAL_EPSILON = 1e-8;
	constexpr double LOCAL_EPSILON_SQUARED = LOCAL_EPSILON * LOCAL_EPSILON;

	bool point_in_convex_polygon_closed(const Vector2 &point, const vector<Vector2> &polygon) {
		if (polygon.empty()) {
			return false;
		}

		bool has_positive = false;
		bool has_negative = false;

		for (size_t i = 0; i < polygon.size(); i++) {
			const auto &v1 = polygon[i];
			const auto &v2 = polygon[(i + 1) % polygon.size()];
			const double cross = (v2 - v1).cross(point - v1);

			if (cross > LOCAL_EPSILON_SQUARED) {
				has_positive = true;
			} else if (cross < -LOCAL_EPSILON_SQUARED) {
				has_negative = true;
			}

			if (has_positive && has_negative) {
				return false;
			}
		}

		return true;
	}

}

namespace tpp {

	void DynamicConvexTppWorkspace::reserve(size_t max_polygons, size_t max_total_vertices) {
		polygon_offsets.reserve(max_polygons + 1);
		first_contact.reserve(max_total_vertices);
		cones.reserve(max_total_vertices);
	}

	ConvexTppWorkspaceView DynamicConvexTppWorkspace::prepare(size_t polygon_count, size_t total_vertices) {
		polygon_offsets.resize(polygon_count + 1);
		first_contact.resize(total_vertices);
		cones.resize(total_vertices);
		return view();
	}

	ConvexTppWorkspaceView DynamicConvexTppWorkspace::view() {
		return {
			std::span<size_t>(polygon_offsets),
			std::span<uint8_t>(first_contact),
			std::span<Cone>(cones),
		};
	}

	Solution::Solution(const Vector2& start, const Vector2& target, const vector<vector<Vector2>>& polygons) :
		start(start),
		target(target),
		polygons(polygons),
		owned_workspace(),
		external_workspace(std::nullopt),
		workspace()
	{}

	Solution::Solution(
		const Vector2& start,
		const Vector2& target,
		const vector<vector<Vector2>>& polygons,
		ConvexTppWorkspaceView workspace
	) :
		start(start),
		target(target),
		polygons(polygons),
		owned_workspace(),
		external_workspace(workspace),
		workspace(workspace)
	{}

	void Solution::initialize_storage() {
		size_t vertex_count = 0;

		for (size_t i = 0; i < polygons.size(); i++) {
			vertex_count += polygons[i].size();
		}

		if (external_workspace.has_value()) {
			workspace = *external_workspace;

			if (workspace.polygon_offsets.size() < polygons.size() + 1) {
				throw std::invalid_argument("ConvexTppWorkspaceView polygon_offsets span is too small.");
			}

			if (workspace.first_contact.size() < vertex_count) {
				throw std::invalid_argument("ConvexTppWorkspaceView first_contact span is too small.");
			}

			if (workspace.cones.size() < vertex_count) {
				throw std::invalid_argument("ConvexTppWorkspaceView cones span is too small.");
			}
		} else {
			owned_workspace.polygon_offsets.resize(polygons.size() + 1);
			owned_workspace.first_contact.resize(vertex_count);
			owned_workspace.cones.resize(vertex_count);
			workspace = owned_workspace.view();
		}

		workspace.polygon_offsets[0] = 0;

		for (size_t i = 0; i < polygons.size(); i++) {
			workspace.polygon_offsets[i + 1] = workspace.polygon_offsets[i] + polygons[i].size();
		}

		std::fill(workspace.first_contact.begin(), workspace.first_contact.begin() + vertex_count, false);
		std::fill(workspace.cones.begin(), workspace.cones.begin() + vertex_count, Cone{Vector2::NaN, Vector2::NaN});
	}

	size_t Solution::storage_index(size_t i, size_t j) const {
		return workspace.polygon_offsets[i] + j;
	}

	bool Solution::is_first_contact(size_t i, size_t j) const {
		return workspace.first_contact[storage_index(i, j)] != 0;
	}

	void Solution::set_first_contact(size_t i, size_t j, bool value) {
		workspace.first_contact[storage_index(i, j)] = value;
	}

	Cone &Solution::cone_slot(size_t i, size_t j) {
		return workspace.cones[storage_index(i, j)];
	}

	void Solution::build_cone(size_t i, size_t j, const Vector2 &last) {
		
		auto j_prev = (j - 1 + polygons[i].size()) % polygons[i].size();
		auto j_next = (j + 1) % polygons[i].size();

		const auto &polygon = polygons[i];

		const auto before = polygon[j_prev];
		const auto vertex = polygon[j];
		const auto after = polygon[j_next];

		const auto diff = (vertex - last).normalized(); // Normalizing is not necessary, but keeps numerical tolerances stable.

		auto ray1 = diff.reflect(vertex - before);
		auto ray2 = diff.reflect(vertex - after);

		set_first_contact(i, j_prev, diff.cross(vertex - before) < 0);
		set_first_contact(i, j, diff.cross(vertex - after) > 0);

		if (!is_first_contact(i, j_prev)) {
			ray1 = diff;
		}

		if (!is_first_contact(i, j)) {
			ray2 = diff;
		}

		cone_slot(i, j) = {ray1, ray2};
	}

	void Solution::build_cone(size_t i, size_t j) {
		const auto last = query(polygons[i][j], i);
		build_cone(i, j, last);
	}

	Cone& Solution::get_cone(size_t i, size_t j) {

		auto &slot = cone_slot(i, j);

		if (slot.first.is_nan()) {
			build_cone(i, j);
		}

		return slot;
	}

	void Solution::query_full(const Vector2 &point, size_t i, vector<Vector2> &accumulator) {
		
		if (i == 0) {
			accumulator.push_back(start);
			return;
		}

		if (point.is_equal_approx(target, LOCAL_EPSILON) && point_in_convex_polygon_closed(point, polygons[i - 1])) {
			query_full(point, i - 1, accumulator);
			return;
		}

		auto location = locate_point(point, i);

		if (location == -1) {
			query_full(point, i - 1, accumulator);
			return;
		}

		const auto &polygon = polygons[i - 1];
		auto vertex_index = location / 2;

		if (location % 2 == 0) {
			const auto &vertex = polygon[vertex_index];
			query_full(vertex, i - 1, accumulator);
			accumulator.push_back(vertex);
			return;
		}

		const auto &v1 = polygon[vertex_index];
		const auto &v2 = polygon[(vertex_index + 1) % polygon.size()];

		const auto &reflected = point.reflect_line(v1, v2);

		query_full(reflected, i - 1, accumulator);

		const auto &last = accumulator.back();
		const auto &intersection = tpp::segment_segment_intersection(last, reflected, v1, v2);

		if (!intersection.is_finite()) {
			throw std::runtime_error(
				std::format("Intersection not found for point {} in polygon {} at edge {}", point, i, vertex_index)
			);
		}

		accumulator.push_back(intersection);

		return;
	}

	vector<Vector2> Solution::query_full(const Vector2& point, size_t i) {

		vector<Vector2> path;
		query_full(point, i, path);
		path.push_back(point);

		return path;
	}

	Vector2 Solution::query(const Vector2& point, size_t i) {

		if (i == 0) {
			return start;
		}

		if (point.is_equal_approx(target, LOCAL_EPSILON) && point_in_convex_polygon_closed(point, polygons[i - 1])) {
			return query(point, i - 1);
		}

		auto location = locate_point(point, i);

		if (location == -1) {
			return query(point, i - 1);
		}

		const auto &polygon = polygons[i - 1];
		auto vertex_index = location / 2;
		
		if (location % 2 == 0) {
			return polygon[vertex_index];
		}

		const auto &v1 = polygon[vertex_index];
		const auto &v2 = polygon[(vertex_index + 1) % polygon.size()];

		const auto reflected = point.reflect_line(v1, v2);
		const auto last = query(reflected, i - 1);
		const auto intersection = tpp::segment_segment_intersection(last, reflected, v1, v2);

		return intersection;
	}
	
	vector<Vector2> Solution::solve() {
		return solve(PreloadPolicy::Lazy);
	}

	vector<Vector2> Solution::solve(PreloadPolicy preload_policy) {
		vector<Vector2> output;
		solve(preload_policy, output);
		return output;
	}

	void Solution::solve(PreloadPolicy preload_policy, vector<Vector2> &output) {
		
		initialize_storage();

		if (preload_policy == PreloadPolicy::Eager) {
			for (size_t i = 0; i < polygons.size(); i++) {
				for (size_t j = 0; j < polygons[i].size(); j++) {
					build_cone(i, j);
				}
			}
		} else {
			preload_cones();
		}

		output.clear();
		query_full(target, polygons.size(), output);
		output.push_back(target);
		tpp::remove_collinear_points_inplace(output);
	}

	double Solution::solve_length() {
		return solve_length(PreloadPolicy::Lazy);
	}

	double Solution::solve_length(PreloadPolicy preload_policy) {

		initialize_storage();

		if (preload_policy == PreloadPolicy::Eager) {
			for (size_t i = 0; i < polygons.size(); i++) {
				for (size_t j = 0; j < polygons[i].size(); j++) {
					build_cone(i, j);
				}
			}
		} else {
			preload_cones();
		}

		return query_length(target, polygons.size());
	}

	double Solution::query_length(const Vector2& point, size_t i) {

		if (i == 0) {
			return start.distance_to(point);
		}

		if (point.is_equal_approx(target, LOCAL_EPSILON) && point_in_convex_polygon_closed(point, polygons[i - 1])) {
			return query_length(point, i - 1);
		}

		const auto location = locate_point(point, i);

		if (location == -1) {
			return query_length(point, i - 1);
		}

		const auto &polygon = polygons[i - 1];
		const auto vertex_index = location / 2;

		if (location % 2 == 0) {
			const auto &vertex = polygon[vertex_index];
			return query_length(vertex, i - 1) + vertex.distance_to(point);
		}

		const auto &v1 = polygon[vertex_index];
		const auto &v2 = polygon[(vertex_index + 1) % polygon.size()];
		const auto reflected = point.reflect_line(v1, v2);

		return query_length(reflected, i - 1);
	}
}
