#include "certified_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tpp::certified_detail {
	double path_length(const Polygon &path) {
		double value = 0;
		for (size_t i = 1; i < path.size(); ++i) value += path[i - 1].distance_to(path[i]);
		return value;
	}

	double dual_bound(const Polygon &q, const std::vector<Polygon> &polygons, double smoothing) {
		using Real = long double;
		std::vector<Vector2> directions;
		for (size_t i = 1; i < q.size(); ++i) {
			auto d = q[i] - q[i - 1];
			const double norm = std::hypot(d.length(), smoothing);
			directions.push_back(norm == 0 ? Vector2{} : d / norm);
		}
		auto evaluate = [&](const std::vector<Vector2> &u) {
			Real value = (q.back() - q.front()).dot(u.back());
			for (size_t i = 0; i < polygons.size(); ++i) {
				double support = std::numeric_limits<double>::infinity();
				for (auto v : polygons[i]) support = std::min(support, (v - q.front()).dot(u[i] - u[i + 1]));
				value += support;
			}
			return double(value);
		};
		double best = evaluate(directions);
		for (bool reverse : {false, true}) {
			auto filled = directions;
			Vector2 previous;
			for (size_t k = 0; k < directions.size(); ++k) {
				const size_t i = reverse ? directions.size() - 1 - k : k;
				if (filled[i].length_squared() == 0) filled[i] = previous;
				else previous = filled[i];
			}
			for (size_t k = 0; k < directions.size(); ++k) {
				const size_t i = reverse ? k : directions.size() - 1 - k;
				if (filled[i].length_squared() == 0) filled[i] = previous;
				else previous = filled[i];
			}
			best = std::max(best, evaluate(filled));
		}
		return best;
	}

	bool recover_contacts(const Polygon &path, const std::vector<Polygon> &polygons, Polygon &q) {
		if (path.size() < 2) return false;
		q = {path.front()};
		size_t segment = 1;
		double rate = 0;
		for (const auto &polygon : polygons) {
			bool found = false;
			while (segment < path.size()) {
				double lo = rate, hi = 1;
				const auto a = path[segment - 1], d = path[segment] - a;
				for (size_t j = 0; j < polygon.size(); ++j) {
					const auto edge = polygon[(j + 1) % polygon.size()] - polygon[j];
					const double c = edge.cross(a - polygon[j]), slope = edge.cross(d);
					if (slope > 0) lo = std::max(lo, -c / slope);
					else if (slope < 0) hi = std::min(hi, -c / slope);
					else if (c < -1e-12 * edge.length()) hi = -1;
				}
				if (lo <= hi + 1e-12 && hi >= rate && lo <= 1) {
					rate = std::clamp(lo, rate, 1.0);
					q.push_back(a + rate * d);
					found = true;
					break;
				}
				++segment;
				rate = 0;
			}
			if (!found) return false;
		}
		q.push_back(path.back());
		return true;
	}

	bool repair_contacts(const Polygon &path, const std::vector<Polygon> &polygons, Polygon &q) {
		if (path.size() < 2) return false;
		double scale = 1;
		for (auto v : path) scale = std::max({scale, std::abs(v.x), std::abs(v.y)});
		for (const auto &p : polygons) for (auto v : p)
			scale = std::max({scale, std::abs(v.x), std::abs(v.y)});
		const double tolerance = 64 * std::numeric_limits<double>::epsilon() * scale;
		q = {path.front()};
		size_t segment = 1;
		double rate = 0;
		for (const auto &polygon : polygons) {
			bool found = false;
			while (segment < path.size()) {
				const auto a = path[segment - 1], d = path[segment] - a;
				double lo = rate, hi = 1;
				for (size_t j = 0; j < polygon.size(); ++j) {
					const auto edge = polygon[(j + 1) % polygon.size()] - polygon[j];
					const double c = edge.cross(a - polygon[j]) + tolerance * edge.length();
					const double slope = edge.cross(d);
					if (slope > 0) lo = std::max(lo, -c / slope);
					else if (slope < 0) hi = std::min(hi, -c / slope);
					else if (c < 0) hi = -1;
				}
				if (lo <= hi && hi >= rate && lo <= 1) {
					rate = std::clamp(lo, rate, 1.0);
					const auto point = a + rate * d;
					bool inside = true;
					Vector2 closest;
					double distance = std::numeric_limits<double>::infinity();
					for (size_t j = 0; j < polygon.size(); ++j) {
						const auto v = polygon[j], w = polygon[(j + 1) % polygon.size()], edge = w - v;
						inside &= edge.cross(point - v) >= 0;
						const double t = edge.length_squared() == 0 ? 0
							: std::clamp((point - v).dot(edge) / edge.length_squared(), 0.0, 1.0);
						const auto candidate = t == 0 ? v : t == 1 ? w : v + t * edge;
						const double squared = (candidate - point).length_squared();
						if (squared < distance) { distance = squared; closest = candidate; }
					}
					// Expanded halfplanes only locate a candidate. Project it back onto
					// the original polygon before using it as a primal upper bound.
					// Contacts need not remain on the original geometric trajectory;
					// the independent dual certificate decides whether repair suffices.
					q.push_back(inside ? point : closest);
					found = true;
					break;
				}
				++segment;
				rate = 0;
			}
			if (!found) return false;
		}
		q.push_back(path.back());
		return true;
	}
}
