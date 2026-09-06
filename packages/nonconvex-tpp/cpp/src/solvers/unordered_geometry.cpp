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
			if (point.distance_to(project(point, a, b)) <= tolerance) return true;
			if ((a.y > point.y) != (b.y > point.y)
				&& point.x < a.x + (b.x - a.x) * (point.y - a.y) / (b.y - a.y)) result = !result;
		}
		return result;
	}
}

namespace tpp::unordered_detail {
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
		for (size_t i = 1; i < path.size(); ++i) {
			const auto a = path[i - 1], b = path[i], direction = b - a;
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
					const double distance = p.distance_to(q);
					const double t = direction.length_squared() == 0 ? 0 : (p - a).dot(direction) / direction.length_squared();
					if (distance <= tolerance) first = std::min(first, t);
					if (distance < best.distance) best = {distance, double(i - 1) + t};
				}
			}
			if (std::isfinite(first)) return {0, double(i - 1) + first};
			if (inside(b, polygon, tolerance)) return {0, double(i)};
		}
		return best;
	}
}
