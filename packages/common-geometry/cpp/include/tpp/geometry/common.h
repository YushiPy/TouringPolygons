#pragma once

#include "tpp/geometry/vec2.h"

#include <format>
#include <iostream>
#include <vector>

namespace std {

	template<>
	struct formatter<std::vector<Vector2>> : formatter<Vector2> {
		auto format(const std::vector<Vector2>& vec, std::format_context& ctx) const {
			auto out = std::format_to(ctx.out(), "{{");
			for (size_t i = 0; i < vec.size(); ++i) {
				if (i > 0) {
					out = std::format_to(out, ", ");
				}
				out = formatter<Vector2>::format(vec[i], ctx);
			}
			return std::format_to(out, "}}");
		}
	};

	template<>
	struct formatter<std::vector<std::vector<Vector2>>> : formatter<std::vector<Vector2>> {
		auto format(const std::vector<std::vector<Vector2>>& mat, std::format_context& ctx) const {
			auto out = std::format_to(ctx.out(), "{{");
			for (size_t i = 0; i < mat.size(); ++i) {
				if (i > 0) {
					out = std::format_to(out, ", ");
				}
				out = formatter<std::vector<Vector2>>::format(mat[i], ctx);
			}
			return std::format_to(out, "}}");
		}
	};
}

namespace tpp {

	Vector2 segment_segment_intersection(const Vector2& start1, const Vector2& end1, const Vector2& start2, const Vector2& end2);
	Vector2 segment_segment_intersection_safe(const Vector2& start1, const Vector2& end1, const Vector2& start2, const Vector2& end2);
	Vector2 reflect_segment(const Vector2& point, const Vector2& vertex1, const Vector2& vertex2);

	bool point_in_cone(const Vector2& point, const Vector2& vertex, const Vector2& ray1, const Vector2& ray2);
	bool point_in_edge(const Vector2& point, const Vector2& vertex1, const Vector2& vertex2, const Vector2& ray1, const Vector2& ray2);
	bool point_in_cone_plus(const Vector2& point, const Vector2& vertex, const Vector2& ray1, const Vector2& ray2);
	bool point_in_edge_plus(const Vector2& point, const Vector2& vertex1, const Vector2& vertex2, const Vector2& ray1, const Vector2& ray2);

	std::vector<Vector2> remove_collinear_points(const std::vector<Vector2>& points);
	std::vector<Vector2> remove_collinear_points(const std::vector<Vector2>& points, double epsilon);
	void remove_collinear_points_inplace(std::vector<Vector2>& points);
	void remove_collinear_points_inplace(std::vector<Vector2>& points, double epsilon);

	inline std::ostream& operator<<(std::ostream& os, const std::vector<Vector2>& vec) {
		return os << std::format("{}", vec);
	}

	inline std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<Vector2>>& mat) {
		return os << std::format("{}", mat);
	}
}
