#include "tests.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
	using Real = long double;
	struct Point { Real x, y; };
	Point subtract(const Vector2 &a, const Vector2 &b) {
		return {Real(a.x) - b.x, Real(a.y) - b.y};
	}
	Real cross(Point a, Point b) { return a.x * b.y - a.y * b.x; }
	Real norm(Point p) { return std::hypot(p.x, p.y); }

	Real dot(Point a, Point b) { return a.x * b.x + a.y * b.y; }

	// A region as half-planes n.(x - p) >= 0, each widened by the tolerance.
	struct HalfPlane { Point normal; Vector2 point; };

	// A polygon with nonempty interior: one half-plane per edge. A segment:
	// its two sides and a cap at each end, so that its line is not accepted.
	// A point: the four axis half-planes, a box of side twice the tolerance.
	std::vector<HalfPlane> half_planes(const std::vector<Vector2> &region, Real sign) {
		std::vector<HalfPlane> result;
		if (sign != 0) {
			for (size_t j = 0; j < region.size(); ++j) {
				const auto edge = subtract(region[(j + 1) % region.size()], region[j]);
				result.push_back({{-sign * edge.y, sign * edge.x}, region[j]});
			}
			return result;
		}
		std::vector<Vector2> distinct;
		for (const auto &v : region)
			if (std::find(distinct.begin(), distinct.end(), v) == distinct.end()) distinct.push_back(v);
		if (distinct.size() == 1) {
			for (Point n : {Point{1, 0}, Point{-1, 0}, Point{0, 1}, Point{0, -1}}) result.push_back({n, distinct[0]});
			return result;
		}
		const Vector2 &a = distinct[0], &b = distinct[1];
		const auto d = subtract(b, a);
		result.push_back({{-d.y, d.x}, a});
		result.push_back({{d.y, -d.x}, a});
		result.push_back({d, a});
		result.push_back({{-d.x, -d.y}, b});
		return result;
	}

	bool clip(const Vector2 &from, const Vector2 &to,
		const std::vector<HalfPlane> &planes, Real tolerance,
		Real &entry, Real &exit) {
		entry = 0;
		exit = 1;
		const auto direction = subtract(to, from);
		for (const auto &h : planes) {
			const Real value = dot(h.normal, subtract(from, h.point)) + tolerance * norm(h.normal);
			const Real slope = dot(h.normal, direction);
			if (slope == 0) {
				if (value < 0) return false;
			} else if (slope > 0) {
				entry = std::max(entry, -value / slope);
			} else {
				exit = std::min(exit, -value / slope);
			}
			if (entry > exit) return false;
		}
		return true;
	}
}

namespace tpp {
	OrderedPathValidation validate_ordered_path(
		const Vector2 &start, const Vector2 &target,
		const std::vector<std::vector<Vector2>> &polygons,
		const std::vector<Vector2> &path
	) {
		OrderedPathValidation result;
		Real magnitude = 0, extent = 0;
		auto inspect = [&](const Vector2 &p) {
			if (!p.is_finite()) return false;
			magnitude = std::max({magnitude, std::abs(Real(p.x)), std::abs(Real(p.y))});
			extent = std::max(extent, norm(subtract(p, start)));
			return true;
		};
		if (path.empty() || !inspect(start) || !inspect(target)) return result;
		// A bad returned path must not enlarge the instance's tolerance by
		// introducing distant points of its own.
		for (const auto &p : path) if (!p.is_finite()) return result;
		// Polygons with nonempty interior, points (one distinct vertex) and
		// segments (two distinct vertices).
		std::vector<Real> signs;
		for (const auto &polygon : polygons) {
			if (polygon.empty()) return result;
			Real area = 0;
			size_t distinct = 0;
			for (size_t j = 0; j < polygon.size(); ++j) {
				if (!inspect(polygon[j])) return result;
				if (std::find(polygon.begin(), polygon.begin() + j, polygon[j]) == polygon.begin() + j) ++distinct;
				area += cross(subtract(polygon[j], polygon[0]),
					subtract(polygon[(j + 1) % polygon.size()], polygon[0]));
			}
			if (distinct >= 3 && area == 0) return result;
			signs.push_back(distinct < 3 ? 0 : area > 0 ? 1 : -1);
		}
		// Distance units throughout: no fixed epsilon on cross products or on
		// segment parameters. The first term accounts for input double rounding.
		const Real tolerance = 64 * std::numeric_limits<double>::epsilon() * magnitude
			+ 1e-12L * extent;
		result.coordinate_tolerance = double(tolerance);
		if (norm(subtract(path.front(), start)) > tolerance
			|| norm(subtract(path.back(), target)) > tolerance) return result;
		Real length = 0;
		for (size_t j = 1; j < path.size(); ++j) length += norm(subtract(path[j], path[j - 1]));
		result.length = double(length);
		if (!std::isfinite(result.length)) return result;
		std::vector<std::vector<HalfPlane>> planes;
		for (size_t i = 0; i < polygons.size(); ++i) planes.push_back(half_planes(polygons[i], signs[i]));
		for (size_t j = 0; j < std::max(size_t(1), path.size() - 1); ++j) {
			Real minimum = 0;
			while (result.visited_polygons < polygons.size()) {
				Real entry, exit;
				const auto i = result.visited_polygons;
				if (!clip(path[j], path[std::min(j + 1, path.size() - 1)],
					planes[i], tolerance, entry, exit)) break;
				minimum = std::max(minimum, entry);
				if (minimum > exit) break;
				++result.visited_polygons;
			}
		}
		result.valid = result.visited_polygons == polygons.size();
		return result;
	}
}
