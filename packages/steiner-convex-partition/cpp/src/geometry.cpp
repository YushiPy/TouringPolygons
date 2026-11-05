#include "internal.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace steiner_convex_partition::detail {

	using exact::closed_segments_meet;
	using exact::dot_sign;
	using exact::on_closed_collinear;
	using exact::open_segment_meets_closed;
	using exact::orient;

	PolygonData prepare_polygon(const Polygon &polygon) {
		std::vector<Vec2> points;
		std::vector<std::size_t> indices;
		points.reserve(polygon.size());
		for (std::size_t i = 0; i < polygon.size(); ++i) {
			const Point &p = polygon[i];
			if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
				throw std::invalid_argument("polygon has a non-finite coordinate");
			}
			const Vec2 v {p.x, p.y};
			if (!points.empty() && points.back() == v) continue;
			points.push_back(v);
			indices.push_back(i);
		}
		while (points.size() > 1 && points.front() == points.back()) {
			points.pop_back();
			indices.pop_back();
		}
		if (points.size() < 3) {
			throw std::invalid_argument("polygon needs at least three distinct vertices");
		}

		// The lowest, then leftmost, vertex is strictly convex; its turn gives the
		// orientation.
		std::size_t lowest = 0;
		for (std::size_t i = 1; i < points.size(); ++i) {
			if (points[i].y < points[lowest].y || (points[i].y == points[lowest].y && points[i].x < points[lowest].x)) {
				lowest = i;
			}
		}
		const std::size_t count = points.size();
		const int turn = orient(points[(lowest + count - 1) % count], points[lowest], points[(lowest + 1) % count]);
		if (turn == 0) {
			throw std::invalid_argument("polygon is degenerate");
		}
		if (turn < 0) {
			std::reverse(points.begin(), points.end());
			std::reverse(indices.begin(), indices.end());
		}

		PolygonData data;
		data.pts = std::move(points);
		data.original_index = std::move(indices);
		data.n = static_cast<int>(data.pts.size());

		const int n = data.n;
		for (int e = 0; e < n; ++e) {
			const Vec2 &a = data.at(e);
			const Vec2 &b = data.at(e + 1);
			// The next edge must not fold back over this one.
			const Vec2 &c = data.at(e + 2);
			if (orient(a, b, c) == 0 && dot_sign(b, a, b, c) > 0) {
				throw std::invalid_argument("polygon is not simple: consecutive edges overlap");
			}
			for (int f = e + 2; f < n; ++f) {
				if (e == 0 && f == n - 1) continue;
				if (closed_segments_meet(a, b, data.at(f), data.at(f + 1))) {
					throw std::invalid_argument("polygon is not simple: edges intersect");
				}
			}
		}

		data.vertex_notch.assign(static_cast<std::size_t>(n), -1);
		for (int v = 0; v < n; ++v) {
			if (orient(data.prev(v), data.at(v), data.next(v)) < 0) {
				data.vertex_notch[static_cast<std::size_t>(v)] = static_cast<int>(data.notch_vertex.size());
				data.notch_vertex.push_back(v);
			}
		}
		data.c = static_cast<int>(data.notch_vertex.size());
		return data;
	}

	bool strictly_inside_corner(const PolygonData &polygon, int vertex, const Vec2 &target) {
		const Vec2 &v = polygon.at(vertex);
		const Vec2 &next = polygon.next(vertex);
		const Vec2 &prev = polygon.prev(vertex);
		const int turn = orient(prev, v, next);
		if (turn > 0) {
			return orient(v, next, target) > 0 && orient(v, target, prev) > 0;
		}
		if (turn == 0) {
			return orient(v, next, target) > 0;
		}
		// Reflex: inside unless in the closed exterior wedge from prev to next.
		return !(orient(v, prev, target) >= 0 && orient(v, target, next) >= 0);
	}

	bool open_segment_clear(const PolygonData &polygon, const Vec2 &a, const Vec2 &b) {
		for (int e = 0; e < polygon.n; ++e) {
			if (open_segment_meets_closed(a, b, polygon.at(e), polygon.at(e + 1))) return false;
		}
		return true;
	}

	bool point_on_boundary(const PolygonData &polygon, const Vec2 &point) {
		for (int e = 0; e < polygon.n; ++e) {
			const Vec2 &u = polygon.at(e);
			const Vec2 &w = polygon.at(e + 1);
			if (orient(u, w, point) == 0 && on_closed_collinear(u, point, w)) return true;
		}
		return false;
	}
}
