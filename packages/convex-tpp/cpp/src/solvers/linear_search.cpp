
#include "vector2.h"
#include "common.h"
#include "tpp_convex_common.h"
#include "tpp_convex.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

using std::vector;
using std::pair;

namespace {

	constexpr double LOCAL_EPSILON = 1e-8;
	constexpr double LOCAL_EPSILON_SQUARED = LOCAL_EPSILON * LOCAL_EPSILON;

	bool point_on_segment(const Vector2 &point, const Vector2 &a, const Vector2 &b);

	bool point_in_convex_polygon_closed(const Vector2 &point, const vector<Vector2> &polygon) {
		bool has_positive = false;
		bool has_negative = false;

		for (size_t j = 0; j < polygon.size(); j++) {
			const auto &v1 = polygon[j];
			const auto &v2 = polygon[(j + 1) % polygon.size()];
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

	bool point_in_convex_polygon_open(const Vector2 &point, const vector<Vector2> &polygon) {
		if (!point_in_convex_polygon_closed(point, polygon)) {
			return false;
		}

		for (size_t j = 0; j < polygon.size(); j++) {
			if (point_on_segment(point, polygon[j], polygon[(j + 1) % polygon.size()])) {
				return false;
			}
		}

		return true;
	}

	bool point_on_segment(const Vector2 &point, const Vector2 &a, const Vector2 &b) {
		return std::fabs((b - a).cross(point - a)) <= LOCAL_EPSILON_SQUARED
			&& (point - a).dot(point - b) <= LOCAL_EPSILON_SQUARED;
	}

	bool segment_enters_polygon_before_endpoint(const Vector2 &from, const Vector2 &to, const vector<Vector2> &polygon) {
		if (point_in_convex_polygon_closed(from, polygon)) {
			return !from.is_equal_approx(to, LOCAL_EPSILON);
		}

		for (size_t j = 0; j < polygon.size(); j++) {
			const auto intersection = tpp::segment_segment_intersection_safe(from, to, polygon[j], polygon[(j + 1) % polygon.size()]);

			if (intersection.is_finite() && !intersection.is_equal_approx(to, LOCAL_EPSILON)) {
				return true;
			}
		}

		return false;
	}

	bool polygons_intersect_or_touch(const vector<Vector2> &a, const vector<Vector2> &b) {
		auto bounds = [](const vector<Vector2> &polygon) {
			double min_x = polygon.front().x;
			double max_x = polygon.front().x;
			double min_y = polygon.front().y;
			double max_y = polygon.front().y;

			for (const auto &point : polygon) {
				min_x = std::min(min_x, point.x);
				max_x = std::max(max_x, point.x);
				min_y = std::min(min_y, point.y);
				max_y = std::max(max_y, point.y);
			}

			return std::tuple(min_x, max_x, min_y, max_y);
		};

		const auto [a_min_x, a_max_x, a_min_y, a_max_y] = bounds(a);
		const auto [b_min_x, b_max_x, b_min_y, b_max_y] = bounds(b);

		if (
			a_max_x < b_min_x - LOCAL_EPSILON ||
			b_max_x < a_min_x - LOCAL_EPSILON ||
			a_max_y < b_min_y - LOCAL_EPSILON ||
			b_max_y < a_min_y - LOCAL_EPSILON
		) {
			return false;
		}

		for (size_t i = 0; i < a.size(); i++) {
			for (size_t j = 0; j < b.size(); j++) {
				if (tpp::segment_segment_intersection_safe(a[i], a[(i + 1) % a.size()], b[j], b[(j + 1) % b.size()]).is_finite()) {
					return true;
				}
			}
		}

		return point_in_convex_polygon_closed(a.front(), b) || point_in_convex_polygon_closed(b.front(), a);
	}

	bool polygons_are_pairwise_disjoint(const vector<vector<Vector2>> &polygons) {
		for (size_t i = 0; i < polygons.size(); i++) {
			for (size_t j = i + 1; j < polygons.size(); j++) {
				if (polygons_intersect_or_touch(polygons[i], polygons[j])) {
					return false;
				}
			}
		}

		return true;
	}

	double path_length(const vector<Vector2> &path) {
		double length = 0.0;

		for (size_t i = 1; i < path.size(); i++) {
			length += path[i - 1].distance_to(path[i]);
		}

		return length;
	}

	double polygon_area2(const vector<Vector2> &polygon) {
		double area = 0.0;

		for (size_t i = 0; i < polygon.size(); i++) {
			const auto &current = polygon[i];
			const auto &next = polygon[(i + 1) % polygon.size()];
			area += current.cross(next);
		}

		return area;
	}

	bool line_segment_polygon_interval(
		const Vector2 &from,
		const Vector2 &to,
		const vector<Vector2> &polygon,
		double &entry,
		double &exit
	) {
		const auto direction = to - from;

		if (direction.length_squared() <= LOCAL_EPSILON_SQUARED) {
			if (!point_in_convex_polygon_closed(from, polygon)) {
				return false;
			}

			entry = 0.0;
			exit = 0.0;
			return true;
		}

		const double orientation_sign = polygon_area2(polygon) >= 0.0 ? 1.0 : -1.0;
		entry = 0.0;
		exit = 1.0;

		for (size_t i = 0; i < polygon.size(); i++) {
			const auto &a = polygon[i];
			const auto &b = polygon[(i + 1) % polygon.size()];
			const double numerator = orientation_sign * (b - a).cross(from - a);
			const double denominator = orientation_sign * (b - a).cross(direction);

			if (std::fabs(denominator) <= LOCAL_EPSILON_SQUARED) {
				if (numerator < -LOCAL_EPSILON_SQUARED) {
					return false;
				}

				continue;
			}

			const double boundary = -numerator / denominator;

			if (denominator > 0.0) {
				entry = std::max(entry, boundary);
			} else {
				exit = std::min(exit, boundary);
			}

			if (entry > exit + LOCAL_EPSILON) {
				return false;
			}
		}

		entry = std::clamp(entry, 0.0, 1.0);
		exit = std::clamp(exit, 0.0, 1.0);
		return entry <= exit + LOCAL_EPSILON;
	}

	double two_point_distance_sum(const Vector2 &left, const Vector2 &right, const Vector2 &point) {
		return left.distance_to(point) + point.distance_to(right);
	}

	Vector2 closest_point_on_segment(const Vector2 &point, const Vector2 &a, const Vector2 &b) {
		const auto edge = b - a;
		const double length_squared = edge.length_squared();

		if (length_squared <= LOCAL_EPSILON_SQUARED) {
			return a;
		}

		const double rate = std::clamp((point - a).dot(edge) / length_squared, 0.0, 1.0);
		return a + edge * rate;
	}

	Vector2 line_line_intersection(
		const Vector2 &a,
		const Vector2 &b,
		const Vector2 &c,
		const Vector2 &d
	) {
		const auto first_direction = b - a;
		const auto second_direction = d - c;
		const double denominator = first_direction.cross(second_direction);

		if (std::fabs(denominator) <= LOCAL_EPSILON_SQUARED) {
			return b;
		}

		return a + first_direction * ((c - a).cross(second_direction) / denominator);
	}

	vector<Vector2> intersect_convex_polygons(vector<Vector2> subject, const vector<Vector2> &clipper) {
		const double orientation_sign = polygon_area2(clipper) >= 0.0 ? 1.0 : -1.0;

		for (size_t i = 0; i < clipper.size(); i++) {
			const auto &edge_start = clipper[i];
			const auto &edge_end = clipper[(i + 1) % clipper.size()];
			const auto input = std::move(subject);
			subject.clear();

			if (input.empty()) {
				break;
			}

			auto previous = input.back();
			bool previous_inside = orientation_sign * (edge_end - edge_start).cross(previous - edge_start) >= -LOCAL_EPSILON_SQUARED;

			for (const auto &current : input) {
				const bool current_inside = orientation_sign * (edge_end - edge_start).cross(current - edge_start) >= -LOCAL_EPSILON_SQUARED;

				if (current_inside) {
					if (!previous_inside) {
						subject.push_back(line_line_intersection(previous, current, edge_start, edge_end));
					}

					subject.push_back(current);
				} else if (previous_inside) {
					subject.push_back(line_line_intersection(previous, current, edge_start, edge_end));
				}

				previous = current;
				previous_inside = current_inside;
			}
		}

		tpp::remove_collinear_points_inplace(subject);
		return subject;
	}

	Vector2 closest_point_on_polygon(const Vector2 &point, const vector<Vector2> &polygon) {
		if (point_in_convex_polygon_closed(point, polygon)) {
			return point;
		}

		Vector2 best = polygon.front();
		double best_distance = point.distance_squared_to(best);

		for (size_t i = 0; i < polygon.size(); i++) {
			const auto candidate = closest_point_on_segment(point, polygon[i], polygon[(i + 1) % polygon.size()]);
			const double distance = point.distance_squared_to(candidate);

			if (distance < best_distance) {
				best = candidate;
				best_distance = distance;
			}
		}

		return best;
	}

	Vector2 snap_near_polygon_boundary(const Vector2 &point, const vector<Vector2> &polygon) {
		Vector2 best = point;
		double best_distance = std::numeric_limits<double>::infinity();

		for (size_t i = 0; i < polygon.size(); i++) {
			const auto candidate = closest_point_on_segment(point, polygon[i], polygon[(i + 1) % polygon.size()]);
			const double distance = point.distance_to(candidate);

			if (distance < best_distance) {
				best = candidate;
				best_distance = distance;
			}
		}

		return best_distance <= LOCAL_EPSILON ? best : point;
	}

	Vector2 best_point_between_on_edge(
		const Vector2 &left,
		const Vector2 &right,
		const Vector2 &edge_start,
		const Vector2 &edge_end
	) {
		const auto edge = edge_end - edge_start;
		const double edge_length_squared = edge.length_squared();

		if (edge_length_squared <= LOCAL_EPSILON_SQUARED) {
			return edge_start;
		}

		Vector2 best = edge_start;
		double best_value = two_point_distance_sum(left, right, best);

		auto consider = [&](const Vector2 &candidate) {
			const double rate = (candidate - edge_start).dot(edge) / edge_length_squared;

			if (rate < -LOCAL_EPSILON || rate > 1.0 + LOCAL_EPSILON) {
				return;
			}

			const auto clamped_candidate = edge_start.lerp(edge_end, std::clamp(rate, 0.0, 1.0));
			const double value = two_point_distance_sum(left, right, clamped_candidate);

			if (value < best_value) {
				best = clamped_candidate;
				best_value = value;
			}
		};

		consider(edge_end);

		const auto left_right = right - left;
		const double straight_denominator = left_right.cross(edge);

		if (std::fabs(straight_denominator) > LOCAL_EPSILON_SQUARED) {
			consider(line_line_intersection(left, right, edge_start, edge_end));
		}

		const auto reflected_left = left.reflect_line(edge_start, edge_end);
		const auto reflected_direction = right - reflected_left;
		const double reflected_denominator = reflected_direction.cross(edge);

		if (std::fabs(reflected_denominator) > LOCAL_EPSILON_SQUARED) {
			consider(line_line_intersection(reflected_left, right, edge_start, edge_end));
		}

		return best;
	}

	Vector2 best_point_between(
		const Vector2 &left,
		const Vector2 &right,
		const vector<Vector2> &polygon,
		const Vector2 *preferred = nullptr
	) {
		double entry = 0.0;
		double exit = 0.0;

		if (line_segment_polygon_interval(left, right, polygon, entry, exit)) {
			if (preferred != nullptr) {
				const auto direction = right - left;
				const double length_squared = direction.length_squared();

				if (length_squared > LOCAL_EPSILON_SQUARED) {
					const double rate = std::clamp((*preferred - left).dot(direction) / length_squared, entry, exit);
					return left.lerp(right, rate);
				}
			}

			return left.lerp(right, (entry + exit) / 2.0);
		}

		Vector2 best = polygon.front();
		double best_value = two_point_distance_sum(left, right, best);

		for (size_t i = 0; i < polygon.size(); i++) {
			const auto candidate = best_point_between_on_edge(left, right, polygon[i], polygon[(i + 1) % polygon.size()]);
			const double value = two_point_distance_sum(left, right, candidate);

			if (value < best_value) {
				best = candidate;
				best_value = value;
			}
		}

		return best;
	}

	bool path_touches_polygons_in_order(
		const vector<Vector2> &path,
		const vector<vector<Vector2>> &polygons
	) {
		size_t next_polygon = 0;

		for (size_t i = 1; i < path.size() && next_polygon < polygons.size(); i++) {
			double min_rate = 0.0;

			while (next_polygon < polygons.size()) {
				double entry = 0.0;
				double exit = 0.0;

				if (!line_segment_polygon_interval(path[i - 1], path[i], polygons[next_polygon], entry, exit)
					|| exit + LOCAL_EPSILON < min_rate) {
					break;
				}

				min_rate = std::max(min_rate, entry);
				next_polygon++;
			}
		}

		return next_polygon == polygons.size();
	}

	vector<Vector2> optimize_intersecting_convex_sequence(
		const Vector2 &start,
		const Vector2 &target,
		const vector<vector<Vector2>> &contact_sets
	) {
		if (contact_sets.empty()) {
			return {start, target};
		}

		if (contact_sets.size() == 1) {
			vector<Vector2> path = {start, best_point_between(start, target, contact_sets.front()), target};
			tpp::remove_collinear_points_inplace(path);
			return path;
		}

		vector<Vector2> contacts;
		contacts.reserve(contact_sets.size());

		for (const auto &contact_set : contact_sets) {
			contacts.push_back(closest_point_on_polygon(target, contact_set));
		}

		constexpr size_t MAX_ITERATIONS = 20000;
		constexpr double CONVERGENCE_EPSILON = 1e-13;

		for (size_t iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
			double max_move_squared = 0.0;

			for (size_t i = 0; i < contacts.size(); i++) {
				const auto &left = i == 0 ? start : contacts[i - 1];
				const auto &right = i + 1 == contacts.size() ? target : contacts[i + 1];
				const auto next = best_point_between(left, right, contact_sets[i], &contacts[i]);
				max_move_squared = std::max(max_move_squared, contacts[i].distance_squared_to(next));
				contacts[i] = next;
			}

			for (size_t reverse_index = contacts.size(); reverse_index-- > 0;) {
				const auto &left = reverse_index == 0 ? start : contacts[reverse_index - 1];
				const auto &right = reverse_index + 1 == contacts.size() ? target : contacts[reverse_index + 1];
				const auto next = best_point_between(left, right, contact_sets[reverse_index], &contacts[reverse_index]);
				max_move_squared = std::max(max_move_squared, contacts[reverse_index].distance_squared_to(next));
				contacts[reverse_index] = next;
			}

			if (max_move_squared <= CONVERGENCE_EPSILON * CONVERGENCE_EPSILON) {
				break;
			}
		}

		for (size_t i = 0; i < contacts.size(); i++) {
			contacts[i] = snap_near_polygon_boundary(contacts[i], contact_sets[i]);
		}

		constexpr double MERGE_EPSILON = 1e-6;

		vector<Vector2> path;
		path.reserve(contacts.size() + 2);
		path.push_back(start);

		for (size_t i = 0; i < contacts.size();) {
			size_t end = i;

			while (end + 1 < contacts.size() && contacts[end].distance_to(contacts[end + 1]) <= MERGE_EPSILON) {
				end++;
			}

			if (end == i) {
				if (!contacts[i].is_equal_approx(path.back(), LOCAL_EPSILON)) {
					path.push_back(contacts[i]);
				}

				i++;
				continue;
			}

			auto intersection = contact_sets[i];

			for (size_t j = i + 1; j <= end; j++) {
				intersection = intersect_convex_polygons(std::move(intersection), contact_sets[j]);
			}

			if (intersection.empty()) {
				for (size_t j = i; j <= end; j++) {
					if (!contacts[j].is_equal_approx(path.back(), LOCAL_EPSILON)) {
						path.push_back(contacts[j]);
					}
				}
			} else {
				const auto &right = end + 1 == contacts.size() ? target : contacts[end + 1];
				const auto merged = best_point_between(path.back(), right, intersection);

				if (!merged.is_equal_approx(path.back(), LOCAL_EPSILON)) {
					path.push_back(merged);
				}
			}

			i = end + 1;
		}

		path.push_back(target);
		tpp::remove_collinear_points_inplace(path);
		return path;
	}

}

class SolutionLinearSearchDisjoint : public tpp::Solution {

	using tpp::Solution::Solution;

	protected:

	int64_t locate_point(const Vector2& point, size_t i) override {

		const auto &polygon = polygons[i - 1];

		for (size_t j = 0; j < polygon.size(); j++) {

			const auto &v = polygon[j];
			const auto &[ray1, ray2] = get_cone(i - 1, j);

			size_t prev = (j + polygon.size() - 1) % polygon.size();

			if (!is_first_contact(i - 1, j) && !is_first_contact(i - 1, prev)) {
				continue;
			}

			if (tpp::point_in_cone(point, v, ray1, ray2)) {
				return 2 * j;
			}
		}

		for (size_t j = 0; j < polygon.size(); j++) {

			if (!is_first_contact(i - 1, j)) {
				continue;
			}

			const auto &v1 = polygon[j];
			const auto &v2 = polygon[(j + 1) % polygon.size()];

			const auto &ray1 = get_cone(i - 1, j).second;
			const auto &ray2 = get_cone(i - 1, (j + 1) % polygon.size()).first;

			if (tpp::point_in_edge(point, v1, v2, ray1, ray2)) {
				return 2 * j + 1;
			}
		}

		return -1;
	}
};

class SolutionLinearSearchIntersecting : public SolutionLinearSearchDisjoint {

	using SolutionLinearSearchDisjoint::SolutionLinearSearchDisjoint;

	vector<uint8_t> intersects_previous_cache;

	bool intersects_previous_polygon(size_t polygon_index) {
		if (intersects_previous_cache.size() != polygons.size()) {
			intersects_previous_cache.assign(polygons.size(), 2);
		}

		auto &cached = intersects_previous_cache[polygon_index];

		if (cached != 2) {
			return cached != 0;
		}

		for (size_t h = 0; h < polygon_index; h++) {
			if (polygons_intersect_or_touch(polygons[h], polygons[polygon_index])) {
				cached = 1;
				return true;
			}
		}

		cached = 0;
		return false;
	}

	int64_t locate_point(const Vector2& point, size_t i) override {
		const auto polygon_index = i - 1;
		const auto &polygon = polygons[polygon_index];

		if (point_in_convex_polygon_open(point, polygon)) {
			return -1;
		}

		if (intersects_previous_polygon(polygon_index)) {
			const auto previous = query(point, i - 1);

			if (segment_enters_polygon_before_endpoint(previous, point, polygon)) {
				return -1;
			}
		}

		return SolutionLinearSearchDisjoint::locate_point(point, i);
	}
};

namespace {

	vector<Vector2> solve_intersecting_convex_sequence(
		const Vector2 &start,
		const Vector2 &target,
		const vector<vector<Vector2>> &polygons
	) {
		vector<Vector2> best_path;
		double best_length = std::numeric_limits<double>::infinity();

		auto consider_path = [&](vector<Vector2> candidate_path) {
			if (!path_touches_polygons_in_order(candidate_path, polygons)) {
				return false;
			}

			const double candidate_length = path_length(candidate_path);

			if (!std::isfinite(best_length)
				|| candidate_length + LOCAL_EPSILON < best_length
				|| (candidate_length <= best_length + LOCAL_EPSILON && candidate_path.size() < best_path.size())) {
				best_path = std::move(candidate_path);
				best_length = candidate_length;
			}

			return true;
		};

		if (consider_path({start, target})) {
			return best_path;
		}

		for (const auto &polygon : polygons) {
			consider_path(optimize_intersecting_convex_sequence(start, target, {polygon}));
		}

		for (size_t i = 0; i < polygons.size(); i++) {
			auto intersection = polygons[i];

			for (size_t j = i + 1; j < polygons.size(); j++) {
				intersection = intersect_convex_polygons(std::move(intersection), polygons[j]);

				if (intersection.empty()) {
					break;
				}

				consider_path(optimize_intersecting_convex_sequence(start, target, {intersection}));
			}
		}

		if (std::isfinite(best_length)) {
			return best_path;
		}

		return SolutionLinearSearchIntersecting(start, target, polygons).solve(tpp::PreloadPolicy::Lazy);
	}
}

namespace tpp {

	void tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			SolutionLinearSearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
		} else {
			output = solve_intersecting_convex_sequence(start, target, polygons);
		}
	}

	void tpp_convex_solve_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		SolutionLinearSearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Lazy, output);
	}

	void tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, ConvexTppWorkspaceView workspace, std::vector<Vector2>& output) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			SolutionLinearSearchDisjoint(start, target, polygons, workspace).solve(PreloadPolicy::Eager, output);
		} else {
			output = solve_intersecting_convex_sequence(start, target, polygons);
		}
	}

	void tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_linear_search_lazy(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	void tpp_convex_solve_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_linear_search_disjoint(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	void tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons, DynamicConvexTppWorkspace& workspace, std::vector<Vector2>& output) {
		tpp_convex_solve_linear_search_eager(start, target, polygons, workspace.prepare(polygons.size(), total_vertex_count(polygons)), output);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionLinearSearchDisjoint(start, target, polygons).solve(PreloadPolicy::Lazy);
		} else {
			return solve_intersecting_convex_sequence(start, target, polygons);
		}
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionLinearSearchDisjoint(start, target, polygons).solve(PreloadPolicy::Lazy);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionLinearSearchDisjoint(start, target, polygons).solve(PreloadPolicy::Eager);
		} else {
			return solve_intersecting_convex_sequence(start, target, polygons);
		}
	}

	double tpp_convex_solve_length_linear_search_lazy(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionLinearSearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Lazy);
		} else {
			return path_length(tpp_convex_solve_linear_search_lazy(start, target, polygons));
		}
	}

	double tpp_convex_solve_length_linear_search_disjoint(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return SolutionLinearSearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Lazy);
	}

	double tpp_convex_solve_length_linear_search_eager(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		if (polygons_are_pairwise_disjoint(polygons)) {
			return SolutionLinearSearchDisjoint(start, target, polygons).solve_length(PreloadPolicy::Eager);
		} else {
			return path_length(tpp_convex_solve_linear_search_eager(start, target, polygons));
		}
	}

	std::vector<Vector2> tpp_convex_solve_linear_search(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_linear_search_lazy(start, target, polygons);
	}

	std::vector<Vector2> tpp_convex_solve_linear_search_dp(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		return tpp_convex_solve_linear_search_eager(start, target, polygons);
	}
}
