#include "unordered_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace {
	using tpp::unordered_detail::Polygon;

	Vector2 project(Vector2 point, Vector2 start, Vector2 end) {
		const auto direction = end - start;
		return start + direction * (direction.length_squared() == 0 ? 0
			: std::clamp((point - start).dot(direction) / direction.length_squared(), 0.0, 1.0));
	}

	bool inside(Vector2 point, const Polygon &polygon, double tolerance) {
		bool result = false;
		for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
			const auto a = polygon[j], b = polygon[i];
			if ((point - project(point, a, b)).length_squared() <= tolerance * tolerance) return true;
			if ((a.y > point.y) != (b.y > point.y)
				&& point.x < a.x + (b.x - a.x) * (point.y - a.y) / (b.y - a.y)) result = !result;
		}
		return result;
	}
}

namespace tpp::unordered_detail {
	Vector2 best_contact(Vector2 left, Vector2 right, const Polygon &polygon, Vector2 preferred) {
		if (inside(left, polygon, 0)) return left;
		if (inside(right, polygon, 0)) return right;
		auto cost = [&](Vector2 p) { return left.distance_to(p) + right.distance_to(p); };
		Vector2 best = preferred;
		double value = cost(best);
		for (size_t j = 0; j < polygon.size(); ++j) {
			const auto a = polygon[j], b = polygon[(j + 1) % polygon.size()], edge = b - a;
			const double squared = edge.length_squared();
			if (squared == 0) continue;
			const double h_left = std::abs(edge.cross(left - a)), h_right = std::abs(edge.cross(right - a));
			// Reflect one endpoint into the opposite halfplane. The joining line
			// meets the edge at this weighted average of endpoint projections.
			const double weight = h_left + h_right == 0 ? 0 : h_left / (h_left + h_right);
			const double rate = std::clamp(((1 - weight) * (left - a).dot(edge)
				+ weight * (right - a).dot(edge)) / squared, 0.0, 1.0);
			const auto candidate = rate == 0 ? a : rate == 1 ? b : a + rate * edge;
			const double candidate_value = cost(candidate);
			if (candidate_value < value) { best = candidate; value = candidate_value; }
		}
		return best;
	}

	double path_length(const Polygon &path) {
		double result = 0;
		for (size_t i = 1; i < path.size(); ++i) result += path[i - 1].distance_to(path[i]);
		return result;
	}

	Polygon convex_hull(Polygon polygon) {
		std::sort(polygon.begin(), polygon.end(), [](auto a, auto b) { return std::tie(a.x, a.y) < std::tie(b.x, b.y); });
		polygon.erase(std::unique(polygon.begin(), polygon.end(), [](auto a, auto b) { return a.x == b.x && a.y == b.y; }), polygon.end());
		Polygon hull;
		for (auto vertex : polygon) {
			while (hull.size() > 1 && (hull.back() - hull[hull.size() - 2]).cross(vertex - hull.back()) <= 0) hull.pop_back();
			hull.push_back(vertex);
		}
		const size_t lower = hull.size();
		for (size_t i = polygon.size() - 1; i-- > 0;) {
			while (hull.size() > lower && (hull.back() - hull[hull.size() - 2]).cross(polygon[i] - hull.back()) <= 0) hull.pop_back();
			hull.push_back(polygon[i]);
		}
		hull.pop_back();
		return hull;
	}

	Contact contact(const Polygon &path, const Polygon &polygon, double tolerance) {
		Contact best{std::numeric_limits<double>::infinity(), 0};
		double best_squared = std::numeric_limits<double>::infinity();
		const double tolerance_squared = tolerance * tolerance;
		Vector2 minimum = polygon.front(), maximum = polygon.front();
		for (auto p : polygon) {
			minimum.x = std::min(minimum.x, p.x); minimum.y = std::min(minimum.y, p.y);
			maximum.x = std::max(maximum.x, p.x); maximum.y = std::max(maximum.y, p.y);
		}
		for (size_t i = 1; i < path.size(); ++i) {
			const auto a = path[i - 1], b = path[i], direction = b - a;
			const double dx = std::max({0.0, minimum.x - std::max(a.x, b.x), std::min(a.x, b.x) - maximum.x});
			const double dy = std::max({0.0, minimum.y - std::max(a.y, b.y), std::min(a.y, b.y) - maximum.y});
			// A segment whose bounding box is farther away than the best actual
			// contact cannot change either coverage or the closest distance.
			if (dx * dx + dy * dy > best_squared) continue;
			if (inside(a, polygon, tolerance)) return {0, double(i - 1)};
			double first = std::numeric_limits<double>::infinity();
			for (size_t j = 0; j < polygon.size(); ++j) {
				const auto c = polygon[j], e = polygon[(j + 1) % polygon.size()], edge = e - c;
				const double denominator = direction.cross(edge);
				if (denominator != 0) {
					const double t = (c - a).cross(edge) / denominator;
					const double u = (c - a).cross(direction) / denominator;
					if (t >= 0 && t <= 1 && u >= 0 && u <= 1) first = std::min(first, t);
				}
				for (const auto &[p, q] : {std::pair{a, project(a, c, e)}, std::pair{b, project(b, c, e)},
					std::pair{project(c, a, b), c}, std::pair{project(e, a, b), e}}) {
					const double squared = (p - q).length_squared();
					if (squared <= tolerance_squared || squared < best_squared) {
						const double t = direction.length_squared() == 0 ? 0 : (p - a).dot(direction) / direction.length_squared();
						if (squared <= tolerance_squared) first = std::min(first, t);
						if (squared < best_squared) { best_squared = squared; best.position = double(i - 1) + t; }
					}
				}
			}
			if (std::isfinite(first)) return {0, double(i - 1) + first};
			if (inside(b, polygon, tolerance)) return {0, double(i)};
		}
		best.distance = std::sqrt(best_squared);
		return best;
	}
}
