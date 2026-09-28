#include "unordered_bounds.h"
#include "tpp/convex/rational.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tpp::unordered_detail {
	std::vector<double> insertion_lower_bounds(const Polygon &contacts,
		const std::vector<const Polygon *> &regions, const Polygon &inserted, bool cycle) {
		const size_t n = regions.size();
		if (cycle) {
			if (!n || contacts.size() != n + 1 || inserted.empty())
				throw std::invalid_argument("Invalid cycle insertion-bound reference.");
			using R = ConvexRational;
			using P = ConvexRationalPoint;
			const P origin(contacts.front());
			auto direction = [](Vector2 a, Vector2 b) {
				const P d = P(b) - P(a); const R squared = d.dot(d);
				if (squared == 0) return P{};
				double norm = std::hypot(b.x-a.x, b.y-a.y);
				if (!std::isfinite(norm) || norm == 0) return P{};
				while (R(norm)*R(norm) < squared) norm = std::nextafter(norm, INFINITY);
				return d*(R(1)/R(norm));
			};
			auto support = [&](const Polygon &polygon, const P &normal) {
				R value = (P(polygon.front())-origin).dot(normal);
				for (auto v : polygon) value = std::min(value, (P(v)-origin).dot(normal));
				return value;
			};
			std::vector<P> u; std::vector<R> terms(n); R parent = 0;
			for (size_t i=0;i<n;++i) u.push_back(direction(contacts[i],contacts[i+1]));
			for (size_t i=0;i<n;++i) parent += terms[i] = support(*regions[i],u[(i+n-1)%n]-u[i]);
			std::vector<double> bounds(n);
			for (size_t i=0;i<n;++i) {
				const size_t next=(i+1)%n;
				const auto point=best_contact(contacts[i],contacts[i+1],inserted,inserted.front());
				const P left=direction(contacts[i],point),right=direction(point,contacts[i+1]);
				R value=support(inserted,left-right);
				if (n==1) value+=support(*regions[0],right-left);
				else value+=parent-terms[i]-terms[next]+support(*regions[i],u[(i+n-1)%n]-left)
					+support(*regions[next],right-u[next]);
				value=std::max(R(0),value); double rounded=value.convert_to<double>();
				while (R(rounded)>value) rounded=std::nextafter(rounded,-INFINITY);
				bounds[i]=rounded;
			}
			return bounds;
		}
		if (contacts.size() != n + 2 || inserted.empty())
			throw std::invalid_argument("Invalid insertion-bound reference path.");
		const auto start = contacts.front(), target = contacts.back();
		auto direction = [](Vector2 delta) {
			const double length = delta.length();
			return length == 0 ? Vector2{} : delta / length;
		};
		auto support = [&](const Polygon &polygon, Vector2 normal) {
			double value = std::numeric_limits<double>::infinity();
			for (auto vertex : polygon) value = std::min(value, (vertex - start).dot(normal));
			return value;
		};
		std::vector<Vector2> raw;
		for (size_t i = 0; i <= n; ++i) raw.push_back(direction(contacts[i + 1] - contacts[i]));
		std::vector<Vector2> directions = raw;
		std::vector<double> supports(n);
		long double value = -std::numeric_limits<long double>::infinity();
		// A zero-length segment admits any unit-ball dual vector. Try the raw
		// vector and both neighboring extensions once for the entire sibling set.
		for (int fill = 0; fill < 3; ++fill) {
			auto candidate = raw;
			if (fill) {
				Vector2 previous;
				for (size_t k = 0; k <= n; ++k) {
					const size_t i = fill == 1 ? k : n - k;
					if (candidate[i].length_squared() == 0) candidate[i] = previous;
					else previous = candidate[i];
				}
				for (size_t k = 0; k <= n; ++k) {
					const size_t i = fill == 1 ? n - k : k;
					if (candidate[i].length_squared() == 0) candidate[i] = previous;
					else previous = candidate[i];
				}
			}
			std::vector<double> terms(n);
			long double bound = (target - start).dot(candidate.back());
			for (size_t i = 0; i < n; ++i) {
				terms[i] = support(*regions[i], candidate[i] - candidate[i + 1]);
				bound += terms[i];
			}
			if (bound > value) { value = bound; directions = std::move(candidate); supports = std::move(terms); }
		}
		double scale = std::max(1.0, start.distance_to(target));
		for (auto region : regions) for (auto v : *region) scale = std::max(scale, start.distance_to(v));
		for (auto v : inserted) scale = std::max(scale, start.distance_to(v));
		const double safety = 1e-12 * scale * (n + 2);
		std::vector<double> bounds(n + 1);
		for (size_t j = 0; j <= n; ++j) {
			const auto point = best_contact(contacts[j], contacts[j + 1], inserted, inserted.front());
			const auto left = direction(point - contacts[j]), right = direction(contacts[j + 1] - point);
			long double bound = value + support(inserted, left - right);
			// Inserting one region changes just its own support term and those
			// of its two neighbors. All other terms are reused from the parent.
			if (j) bound += support(*regions[j - 1], directions[j - 1] - left) - supports[j - 1];
			if (j < n) bound += support(*regions[j], right - directions[j + 1]) - supports[j];
			else bound += (target - start).dot(right - directions.back());
			bounds[j] = std::max(start.distance_to(target), double(bound) - safety);
		}
		return bounds;
	}
}
