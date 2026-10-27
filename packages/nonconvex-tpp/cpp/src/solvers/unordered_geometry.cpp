#include "unordered_geometry.h"
#include "tpp/geometry/common.h"
#include "tpp/convex/rational.h"
#include <bit>
#include <stdexcept>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <tuple>

namespace {
	using tpp::unordered_detail::Polygon;
	using tpp::unordered_detail::ContactEdge;
	using tpp::unordered_detail::ContactSegment;
	using tpp::unordered_detail::PreparedContactPolygon;
	using tpp::unordered_detail::PreparedContactPath;

	Vector2 project(Vector2 point, const ContactEdge &edge) {
		return edge.start + edge.direction * (edge.squared == 0 ? 0
			: std::clamp((point - edge.start).dot(edge.direction) / edge.squared, 0.0, 1.0));
	}

	template<class EdgeAt>
	bool inside_edges(Vector2 point, size_t count, const EdgeAt &edge_at, double tolerance) {
		bool result = false;
		for (size_t i = 0, j = count - 1; i < count; j = i++) {
			const auto edge = edge_at(j);
			const auto a = edge.start, b = edge.end;
			if ((point - project(point, edge)).length_squared() <= tolerance * tolerance) return true;
			if ((a.y > point.y) != (b.y > point.y)
				&& point.x < a.x + (b.x - a.x) * (point.y - a.y) / (b.y - a.y)) result = !result;
		}
		return result;
	}
	bool inside(Vector2 point, const Polygon &polygon, double tolerance) {
		return inside_edges(point, polygon.size(), [&](size_t i) {
			return ContactEdge(polygon[i], polygon[(i+1)%polygon.size()]);
		}, tolerance);
	}
}

namespace tpp::unordered_detail {
	ContactEdge::ContactEdge(Vector2 a, Vector2 b)
		:start(a),end(b),direction(b-a),squared(direction.length_squared()) {}
	ContactSegment::ContactSegment(Vector2 a, Vector2 b):ContactEdge(a,b),
		minimum{std::min(a.x,b.x),std::min(a.y,b.y)},
		maximum{std::max(a.x,b.x),std::max(a.y,b.y)} {}
	PreparedContactPolygon::PreparedContactPolygon(const Polygon &polygon)
		:minimum(polygon.front()),maximum(polygon.front()) {
		edges.reserve(polygon.size());
		for(size_t i=0;i<polygon.size();++i) {
			const auto p=polygon[i];
			minimum.x=std::min(minimum.x,p.x);minimum.y=std::min(minimum.y,p.y);
			maximum.x=std::max(maximum.x,p.x);maximum.y=std::max(maximum.y,p.y);
			edges.emplace_back(p,polygon[(i+1)%polygon.size()]);
		}
		const size_t m=polygon.size();
		if(m>=64) {
			int sign=0;convex=true;
			for(size_t i=0;i<m&&convex;++i) {
				const int s=turn_sign(polygon[i],polygon[(i+1)%m],polygon[(i+2)%m]);
				if(s==0||(sign!=0&&s!=sign))convex=false;
				sign=s;
			}
			reversed=convex&&sign<0;
		}
	}
	PreparedContactPath::PreparedContactPath(const Polygon &path) {
		prepare(path);
	}
	void PreparedContactPath::prepare(const Polygon &path) {
		segments.clear();
		segments.reserve(path.empty()?0:path.size()-1);
		for(size_t i=1;i<path.size();++i)segments.emplace_back(path[i-1],path[i]);
	}
	namespace {
		double polygon_perimeter(const Polygon &polygon) {
			double perimeter = 0;
			for (size_t i = 0; i < polygon.size(); ++i)
				perimeter += polygon[i].distance_to(polygon[(i + 1) % polygon.size()]);
			return perimeter;
		}
	}

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

	double perimeter_sampling_work_budget(double log2_complexity) {
		constexpr double base_budget = 1'000'000.0;
		constexpr double maximum_adaptive_factor = 8.0;
		const double complexity = std::isfinite(log2_complexity) ? std::max(0.0, log2_complexity) : 0.0;
		double budget = base_budget;
		if (const char *raw_budget = std::getenv("TPP_APPROX_WORK_BUDGET")) {
			const double parsed_budget = std::atof(raw_budget);
			if (parsed_budget > 0) budget = parsed_budget;
		}
		const char *mode = std::getenv("TPP_APPROX_BUDGET_MODE");
		if (mode != nullptr && std::string_view(mode) == "adaptive")
			budget *= std::min(maximum_adaptive_factor, std::exp2(complexity / 32.0));
		return budget;
	}

	std::vector<size_t> choose_perimeter_sample_point_counts(
		const std::vector<Polygon> &polygons, double work_budget,
		PerimeterSamplingWorkModel model) {
		std::vector<double> perimeters;
		perimeters.reserve(polygons.size());
		for (const auto &polygon : polygons) perimeters.push_back(polygon_perimeter(polygon));

		double weighted_work = 0;
		if (model == PerimeterSamplingWorkModel::AdjacentPairs) {
			for (size_t i = 0; i + 1 < perimeters.size(); ++i)
				weighted_work += perimeters[i] * perimeters[i + 1];
		} else {
			for (size_t i = 0; i < perimeters.size(); ++i)
				for (size_t j = i + 1; j < perimeters.size(); ++j)
					weighted_work += perimeters[i] * perimeters[j];
		}

		double scale = 0;
		if (weighted_work > 0 && std::isfinite(work_budget) && work_budget > 0) {
			scale = std::sqrt(work_budget / weighted_work);
		} else if (model == PerimeterSamplingWorkModel::AdjacentPairs
			&& !perimeters.empty() && perimeters.front() > 0 && std::isfinite(work_budget) && work_budget > 0) {
			scale = std::sqrt(work_budget) / perimeters.front();
		}

		std::vector<size_t> counts;
		counts.reserve(polygons.size());
		for (size_t i = 0; i < polygons.size(); ++i) {
			const double requested = std::ceil(perimeters[i] * scale);
			const size_t requested_count = std::isfinite(requested) && requested > 0
				? static_cast<size_t>(requested) : 0;
			counts.push_back(std::max(polygons[i].size(), requested_count));
		}
		return counts;
	}

	Polygon evenly_spaced_perimeter_points(const Polygon &polygon, size_t point_count) {
		if (polygon.empty() || point_count == 0) return {};
		const double perimeter = polygon_perimeter(polygon);
		if (perimeter == 0) return Polygon(point_count, polygon.front());

		Polygon sampled;
		sampled.reserve(std::max(point_count, polygon.size()));
		sampled.insert(sampled.end(), polygon.begin(), polygon.end());
		if (point_count <= polygon.size()) return sampled;

		const size_t extra_point_count = point_count - polygon.size();
		const double spacing = perimeter / static_cast<double>(extra_point_count);
		size_t edge_index = 0;
		double edge_start_distance = 0;
		double edge_length = polygon[0].distance_to(polygon[1 % polygon.size()]);
		for (size_t sample_index = 0; sample_index < extra_point_count; ++sample_index) {
			const double target_distance = spacing * (static_cast<double>(sample_index) + 0.5);
			while (edge_index + 1 < polygon.size() && edge_start_distance + edge_length < target_distance) {
				edge_start_distance += edge_length;
				++edge_index;
				edge_length = polygon[edge_index].distance_to(polygon[(edge_index + 1) % polygon.size()]);
			}
			const auto &a = polygon[edge_index];
			const auto &b = polygon[(edge_index + 1) % polygon.size()];
			const double weight = edge_length == 0 ? 0 : (target_distance - edge_start_distance) / edge_length;
			sampled.push_back(a.lerp(b, weight));
		}
		return sampled;
	}

	int turn_sign(Vector2 a, Vector2 b, Vector2 c) {
		const double l = (b.x - a.x) * (c.y - b.y), r = (b.y - a.y) * (c.x - b.x), d = l - r;
		const double bound = 0x1p-50 * (std::abs(l) + std::abs(r)) + 0x1p-1069;
		if (std::isfinite(d) && std::isfinite(bound)) {
			if (d > bound) return 1;
			if (d < -bound) return -1;
		}
		using R = ConvexRational;
		const R exact = (R(b.x) - R(a.x)) * (R(c.y) - R(b.y)) - (R(b.y) - R(a.y)) * (R(c.x) - R(b.x));
		return (exact > 0) - (exact < 0);
	}
	Polygon convex_hull(Polygon polygon) {
		std::sort(polygon.begin(), polygon.end(), [](auto a, auto b) { return std::tie(a.x, a.y) < std::tie(b.x, b.y); });
		polygon.erase(std::unique(polygon.begin(), polygon.end(), [](auto a, auto b) { return a.x == b.x && a.y == b.y; }), polygon.end());
		if (polygon.size() <= 2) return polygon;
		// Exact turn signs: the hull is strictly convex in exact arithmetic,
		// which the logarithmic supports of the insertion bounds require.
		Polygon hull;
		for (auto vertex : polygon) {
			while (hull.size() > 1 && turn_sign(hull[hull.size() - 2], hull.back(), vertex) <= 0) hull.pop_back();
			hull.push_back(vertex);
		}
		const size_t lower = hull.size();
		for (size_t i = polygon.size() - 1; i-- > 0;) {
			while (hull.size() > lower && turn_sign(hull[hull.size() - 2], hull.back(), polygon[i]) <= 0) hull.pop_back();
			hull.push_back(polygon[i]);
		}
		hull.pop_back();
		return hull;
	}

	template<class SegmentAt, class EdgeAt>
	Contact contact_impl(size_t segment_count, const SegmentAt &segment_at,
		size_t edge_count, const EdgeAt &edge_at, Vector2 minimum, Vector2 maximum, double tolerance) {
		Contact best{std::numeric_limits<double>::infinity(), 0, std::numeric_limits<double>::infinity()};
		double best_squared = std::numeric_limits<double>::infinity();
		const double tolerance_squared = tolerance * tolerance;
		for (size_t i = 0; i < segment_count; ++i) {
			const auto segment=segment_at(i);
			const auto a=segment.start,b=segment.end,direction=segment.direction;
			const double dx = std::max({0.0, minimum.x - segment.maximum.x, segment.minimum.x - maximum.x});
			const double dy = std::max({0.0, minimum.y - segment.maximum.y, segment.minimum.y - maximum.y});
			// A segment whose bounding box is farther away than the best actual
			// contact cannot change either coverage or the closest distance.
			if (dx * dx + dy * dy > best_squared) continue;
			if (inside_edges(a, edge_count, edge_at, tolerance)) return {0, double(i)};
			double first = std::numeric_limits<double>::infinity();
			for (size_t j = 0; j < edge_count; ++j) {
				const auto polygon_edge=edge_at(j);
				const auto c=polygon_edge.start,e=polygon_edge.end,edge=polygon_edge.direction;
				const double denominator = direction.cross(edge);
				if (denominator != 0) {
					const double t = (c - a).cross(edge) / denominator;
					const double u = (c - a).cross(direction) / denominator;
					if (t >= 0 && t <= 1 && u >= 0 && u <= 1) first = std::min(first, t);
				}
				for (const auto &[p, q] : {std::pair{a, project(a, polygon_edge)}, std::pair{b, project(b, polygon_edge)},
					std::pair{project(c, segment), c}, std::pair{project(e, segment), e}}) {
					const double squared = (p - q).length_squared();
					if (squared <= tolerance_squared || squared < best_squared) {
						const double t = segment.squared == 0 ? 0 : (p - a).dot(direction) / segment.squared;
						if (squared <= tolerance_squared) first = std::min(first, t);
						if (squared < best_squared) { best_squared = squared; best.position = double(i) + t; best.polygon_point = q; }
					}
				}
			}
			if (std::isfinite(first)) return {0, double(i) + first};
			if (inside_edges(b, edge_count, edge_at, tolerance)) return {0, double(i+1)};
		}
		best.squared_distance = best_squared;
		best.distance = std::sqrt(best_squared);
		return best;
	}
	std::optional<Polygon> counterclockwise_convex(const Polygon &polygon) {
		const size_t m = polygon.size();
		if (m < 3) return {};
		int sign = 0;
		for (size_t i = 0; i < m; ++i) {
			const Vector2 a = polygon[i], b = polygon[(i + 1) % m], c = polygon[(i + 2) % m];
			const double turn = (b - a).cross(c - b);
			const int s = (turn > 0) - (turn < 0);
			if (s == 0 || (sign != 0 && s != sign)) return {};
			sign = s;
		}
		Polygon result = polygon;
		if (sign < 0) std::reverse(result.begin(), result.end());
		return result;
	}
	namespace {
		// Index of a vertex maximizing d.v (KACTL's extrVertex, then a local walk).
		template<class VertexAt> size_t extreme_index(size_t count, const VertexAt &vertex, Vector2 d) {
			const long n = long(count);
			auto value = [&](long i) { return d.dot(vertex(size_t(((i % n) + n) % n))); };
			auto cmp = [&](long i, long j) { const double a = value(i), b = value(j); return (b > a) - (b < a); };
			auto extr = [&](long i) { return cmp(i + 1, i) >= 0 && cmp(i, i - 1 + n) < 0; };
			long best = 0;
			if (!extr(0)) {
				long lo = 0, hi = n;
				while (lo + 1 < hi) {
					const long mid = (lo + hi) / 2;
					if (extr(mid)) { lo = mid; break; }
					const int ls = cmp(lo + 1, lo), ms = cmp(mid + 1, mid);
					if (ls < ms || (ls == ms && ls == cmp(lo, mid))) hi = mid; else lo = mid;
				}
				best = lo;
			}
			for (int step = 0; step < 64; ++step) {
				if (value(best - 1) > value(best)) --best;
				else if (value(best + 1) > value(best)) ++best;
				else break;
			}
			return size_t(((best % n) + n) % n);
		}
		size_t support_index(const Polygon &p, Vector2 d) {
			return extreme_index(p.size(), [&](size_t i) { return p[i]; }, d);
		}
	}
	double convex_distance(const Polygon &a, const Polygon &b) {
		// GJK on the Minkowski difference a - b: the closest point to the origin.
		auto support = [&](Vector2 d) { return a[support_index(a, d)] - b[support_index(b, -d)]; };
		std::vector<Vector2> simplex{a[0] - b[0]};
		Vector2 v = simplex[0];
		for (int iteration = 0; iteration < 64; ++iteration) {
			const double vv = v.dot(v);
			if (vv == 0) return 0;
			const Vector2 w = support(-v);
			if (vv - v.dot(w) <= 1e-12 * vv) break;
			simplex.push_back(w);
			// Closest point of the simplex (point, segment or triangle) to the origin.
			if (simplex.size() == 2) {
				const Vector2 p = simplex[0], q = simplex[1], e = q - p;
				const double t = std::clamp(-p.dot(e) / e.dot(e), 0.0, 1.0);
				if (t == 0) simplex = {p}; else if (t == 1) simplex = {q};
				v = p + e * t;
			} else {
				const Vector2 p = simplex[0], q = simplex[1], r = simplex[2];
				const double s1 = (q - p).cross(-p), s2 = (r - q).cross(-q), s3 = (p - r).cross(-r);
				if ((s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0)) return 0;
				double best = INFINITY; std::vector<Vector2> kept; Vector2 point;
				for (const auto &[x, y] : {std::pair{p, q}, std::pair{q, r}, std::pair{r, p}}) {
					const Vector2 e = y - x; const double t = std::clamp(-x.dot(e) / e.dot(e), 0.0, 1.0);
					const Vector2 c = x + e * t;
					if (c.dot(c) < best) { best = c.dot(c); point = c; kept = t == 0 ? std::vector{x} : t == 1 ? std::vector{y} : std::vector{x, y}; }
				}
				simplex = kept; v = point;
			}
		}
		return std::sqrt(v.dot(v));
	}
	Vector2 best_contact_convex(Vector2 left, Vector2 right, const Polygon &polygon, Vector2 preferred) {
		const size_t m = polygon.size();
		if (m < 64) return best_contact(left, right, polygon, preferred);
		auto loop = [&] { return best_contact(left, right, polygon, preferred); };
		const double turn = (polygon[m / 3] - polygon[0]).cross(polygon[2 * m / 3] - polygon[0]);
		if (!std::isfinite(turn) || turn == 0) return loop();
		// A counterclockwise view: view vertex j is input vertex at(j).
		const bool reversed = turn < 0;
		auto at = [&](size_t j) { j %= m; return reversed ? m - 1 - j : j; };
		auto vertex = [&](size_t j) { return polygon[at(j)]; };
		const Vector2 o = vertex(0);
		// Closed membership: the fan wedge (o, v[j], v[j+1]) by binary search.
		auto in = [&](Vector2 q) {
			if ((vertex(1) - o).cross(q - o) < 0 || (vertex(m - 1) - o).cross(q - o) > 0) return false;
			size_t lo = 1, hi = m - 1;
			while (hi - lo > 1) {
				const size_t mid = lo + (hi - lo) / 2;
				if ((vertex(mid) - o).cross(q - o) >= 0) lo = mid; else hi = mid;
			}
			return (vertex(lo + 1) - vertex(lo)).cross(q - vertex(lo)) >= 0;
		};
		if (in(left)) return left;
		if (in(right)) return right;
		// View edges around the candidates: the crossings of the line through
		// left and right, else the feature of Dror et al.'s last-step map of
		// `left` holding `right` (cones built on the fly).
		std::vector<size_t> edges;
		const Vector2 d = right - left, normal{-d.y, d.x};
		auto extreme = [&](Vector2 direction) {
			const size_t i = support_index(polygon, direction);
			return reversed ? m - 1 - i : i;
		};
		bool crosses = false;
		if (normal.x != 0 || normal.y != 0) {
			const size_t top = extreme(normal), bottom = extreme(-normal);
			if (normal.dot(vertex(top) - left) >= 0 && normal.dot(vertex(bottom) - left) <= 0) {
				auto crossing = [&](size_t from, size_t to, bool falling) {
					size_t lo = 0, hi = (to + m - from) % m;
					while (hi - lo > 1) {
						const size_t mid = (lo + hi) / 2;
						const double side = normal.dot(vertex(from + mid) - left);
						if (falling ? side >= 0 : side <= 0) lo = mid; else hi = mid;
					}
					return (from + lo) % m;
				};
				// The chord of the line meets the segment: a crossing is optimal.
				double lo = 0, hi = 1;
				const size_t first = crossing(top, bottom, true), second = crossing(bottom, top, false);
				for (const size_t edge : {first, second})
					for (const size_t j : {(edge + m - 1) % m, edge, (edge + 1) % m}) {
						const Vector2 e = vertex(j + 1) - vertex(j);
						const double offset = e.cross(left - vertex(j)), slope = e.cross(d);
						if (slope > 0) lo = std::max(lo, -offset / slope);
						else if (slope < 0) hi = std::min(hi, -offset / slope);
					}
				if (lo <= hi) {
					crosses = true;
					for (const size_t edge : {first, second})
						for (const size_t j : {(edge + m - 1) % m, edge, (edge + 1) % m}) edges.push_back(j);
				}
			}
		}
		if (!crosses) {
			auto cone = [&](size_t j) {
				const Vector2 before = vertex(j + m - 1), here = vertex(j), after = vertex(j + 1), diff = (here - left).normalized();
				return std::pair{diff.cross(here - before) < 0 ? diff.reflect(here - before) : diff,
					diff.cross(here - after) > 0 ? diff.reflect(here - after) : diff};
			};
			size_t location;
			if (const auto c = cone(0); point_in_cone_plus(right, vertex(0), c.first, c.second)) location = 0;
			else {
				size_t lo = 0, hi = m - 1;
				bool found = false;
				while (lo != hi) {
					const size_t mid = lo + (hi - lo) / 2, next = mid + 1;
					const auto c = cone(next);
					if (point_in_cone_plus(right, vertex(next), c.first, c.second)) { location = 2 * next; found = true; break; }
					if (point_in_edge_plus(right, vertex(lo), vertex(next), cone(lo).second, c.first)) hi = mid;
					else lo = mid + 1;
				}
				if (!found) location = 2 * lo + 1;
			}
			const size_t j = location / 2;
			for (const size_t k : {j + m - 2, j + m - 1, j, j + 1}) edges.push_back(k % m);
		}
		// The loop's candidate on each input edge, then the certificate: q is
		// optimal iff the cost's descent direction g there is an outward normal,
		// i.e. q is extreme toward g (or g vanishes, q on the segment).
		auto cost = [&](Vector2 p) { return left.distance_to(p) + right.distance_to(p); };
		Vector2 best = preferred;
		double value = cost(best);
		for (const size_t view_edge : edges) {
			const size_t j = reversed ? at(view_edge + 1) : at(view_edge);
			const auto a = polygon[j], b = polygon[(j + 1) % m], edge = b - a;
			const double squared = edge.length_squared();
			if (squared == 0) continue;
			const double h_left = std::abs(edge.cross(left - a)), h_right = std::abs(edge.cross(right - a));
			const double weight = h_left + h_right == 0 ? 0 : h_left / (h_left + h_right);
			const double rate = std::clamp(((1 - weight) * (left - a).dot(edge)
				+ weight * (right - a).dot(edge)) / squared, 0.0, 1.0);
			const auto candidate = rate == 0 ? a : rate == 1 ? b : a + rate * edge;
			const double candidate_value = cost(candidate);
			if (candidate_value < value) { best = candidate; value = candidate_value; }
		}
		const double scale = left.distance_to(best) + right.distance_to(best);
		if (!(value <= d.length() * (1 + 1e-12) + 1e-300)) {
			const Vector2 to_left = left - best, to_right = right - best;
			const double l = to_left.length(), r = to_right.length();
			if (!(l > 0 && r > 0)) return loop();
			const Vector2 g = to_left / l + to_right / r;
			const Vector2 top = polygon[support_index(polygon, g)];
			if (!(g.dot(top - best) <= 1e-9 * g.length() * (scale + top.distance_to(best)))) return loop();
		}
		return best;
	}
	Contact contact(const Polygon &path, const Polygon &polygon, double tolerance) {
		Vector2 minimum=polygon.front(),maximum=polygon.front();
		for(auto p:polygon) {
			minimum.x=std::min(minimum.x,p.x);minimum.y=std::min(minimum.y,p.y);
			maximum.x=std::max(maximum.x,p.x);maximum.y=std::max(maximum.y,p.y);
		}
		return contact_impl(path.empty()?0:path.size()-1,
			[&](size_t i){return ContactSegment(path[i],path[i+1]);},polygon.size(),
			[&](size_t i){return ContactEdge(polygon[i],polygon[(i+1)%polygon.size()]);},minimum,maximum,tolerance);
	}
	Contact contact(const PreparedContactPath &path, const PreparedContactPolygon &polygon, double tolerance) {
		return contact_impl(path.segments.size(),[&](size_t i)->const ContactSegment&{return path.segments[i];},
			polygon.edges.size(),[&](size_t i)->const ContactEdge&{return polygon.edges[i];},
			polygon.minimum,polygon.maximum,tolerance);
	}
	double path_point_distance(const PreparedContactPath &path, Vector2 point) {
		double best = std::numeric_limits<double>::infinity();
		for (const auto &segment : path.segments) best = std::min(best, (project(point, segment) - point).length_squared());
		return std::sqrt(best);
	}
	namespace {
		// Polygon vertices of the final GJK simplex between the segment ab and a
		// counterclockwise convex polygon (the closest features), or nothing
		// when the two meet.
		template<class VertexAt>
		std::optional<std::vector<size_t>> closest_vertices(Vector2 a, Vector2 b, size_t m, const VertexAt &vertex) {
			struct Point { Vector2 w; size_t p; };
			auto support = [&](Vector2 d) {
				const size_t p = extreme_index(m, vertex, -d);
				return Point{((b - a).dot(d) > 0 ? b : a) - vertex(p), p};
			};
			std::vector<Point> simplex{{a - vertex(0), 0}};
			Vector2 v = simplex[0].w;
			for (int iteration = 0; iteration < 64; ++iteration) {
				const double vv = v.dot(v);
				if (vv == 0) return {};
				const Point w = support(-v);
				if (vv - v.dot(w.w) <= 1e-12 * vv) break;
				simplex.push_back(w);
				if (simplex.size() == 2) {
					const Vector2 p = simplex[0].w, e = simplex[1].w - p;
					const double t = std::clamp(-p.dot(e) / e.dot(e), 0.0, 1.0);
					if (t == 0) simplex = {simplex[0]}; else if (t == 1) simplex = {simplex[1]};
					v = p + e * t;
				} else {
					const Vector2 p = simplex[0].w, q = simplex[1].w, r = simplex[2].w;
					const double s1 = (q - p).cross(-p), s2 = (r - q).cross(-q), s3 = (p - r).cross(-r);
					if ((s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0)) return {};
					double best = INFINITY; std::vector<Point> kept; Vector2 point;
					for (const auto &[x, y] : {std::pair{simplex[0], simplex[1]}, std::pair{simplex[1], simplex[2]}, std::pair{simplex[2], simplex[0]}}) {
						const Vector2 e = y.w - x.w; const double t = std::clamp(-x.w.dot(e) / e.dot(e), 0.0, 1.0);
						const Vector2 c = x.w + e * t;
						if (c.dot(c) < best) { best = c.dot(c); point = c; kept = t == 0 ? std::vector{x} : t == 1 ? std::vector{y} : std::vector{x, y}; }
					}
					simplex = kept; v = point;
				}
			}
			std::vector<size_t> vertices;
			for (const auto &point : simplex) vertices.push_back(point.p);
			return vertices;
		}
		// One segment against a convex prepared polygon: contact_impl's scan
		// restricted to the edges around the point nearest a and around the
		// closest features (GJK) or, when they meet, the entry and exit of the
		// segment's line (binary search between the extremes on either side).
		// Same arithmetic, edge order and ties as the scan on those edges, which
		// hold every pair the scan's result comes from.
		Contact convex_segment_contact(const ContactSegment &segment, const PreparedContactPolygon &polygon, double tolerance) {
			const size_t m = polygon.edges.size();
			auto at = [&](size_t j) { j %= m; return polygon.reversed ? m - 1 - j : j; };
			auto vertex = [&](size_t j) { return polygon.edges[at(j)].start; };
			const Vector2 a = segment.start, b = segment.end, direction = segment.direction, o = vertex(0);
			auto in = [&](Vector2 q) {
				if ((vertex(1) - o).cross(q - o) < 0 || (vertex(m - 1) - o).cross(q - o) > 0) return false;
				size_t lo = 1, hi = m - 1;
				while (hi - lo > 1) {
					const size_t mid = lo + (hi - lo) / 2;
					if ((vertex(mid) - o).cross(q - o) >= 0) lo = mid; else hi = mid;
				}
				return (vertex(lo + 1) - vertex(lo)).cross(q - vertex(lo)) >= 0;
			};
			if (in(a)) return {0, 0};
			std::vector<size_t> edges;
			// Input edges of the view edges j-2 .. j+1 around view vertex j.
			auto around = [&](size_t j) {
				for (const size_t k : {j + m - 2, j + m - 1, j, j + 1}) edges.push_back(polygon.reversed ? at(k + 1) : at(k));
			};
			// The scan's first touching point is a itself when a is within the
			// tolerance, wherever the closest features are.
			if (const auto near = closest_vertices(a, a, m, vertex)) for (const size_t p : *near) around(p);
			if (const auto features = closest_vertices(a, b, m, vertex)) for (const size_t p : *features) around(p);
			else {
				const Vector2 normal{-direction.y, direction.x};
				if (normal.x != 0 || normal.y != 0) {
					const size_t top = extreme_index(m, vertex, normal), bottom = extreme_index(m, vertex, -normal);
					auto crossing = [&](size_t from, size_t to, bool falling) {
						size_t lo = 0, hi = (to + m - from) % m;
						while (hi - lo > 1) {
							const size_t mid = (lo + hi) / 2;
							const double side = normal.dot(vertex(from + mid) - a);
							if (falling ? side >= 0 : side <= 0) lo = mid; else hi = mid;
						}
						return (from + lo) % m;
					};
					for (const size_t edge : {crossing(top, bottom, true), crossing(bottom, top, false)}) { around(edge); around(edge + 1); }
				}
			}
			std::sort(edges.begin(), edges.end());
			edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
			// Vertices within the tolerance of the segment are touching points of
			// the scan: grow the edges along every such run (empty for a
			// tolerance below the edge lengths near the contact).
			const double tolerance_squared = tolerance * tolerance;
			auto near = [&](Vector2 v) { return (project(v, segment) - v).length_squared() <= tolerance_squared; };
			for (size_t k = 0; k < edges.size(); ++k) {
				const size_t j = edges[k], before = (j + m - 1) % m, after = (j + 1) % m;
				for (const auto &[v, neighbor] : {std::pair{polygon.edges[j].start, before}, std::pair{polygon.edges[j].end, after}})
					if (near(v) && std::find(edges.begin(), edges.end(), neighbor) == edges.end()) edges.push_back(neighbor);
			}
			std::sort(edges.begin(), edges.end());
			Contact best{std::numeric_limits<double>::infinity(), 0, std::numeric_limits<double>::infinity()};
			double best_squared = std::numeric_limits<double>::infinity(), first = std::numeric_limits<double>::infinity();
			for (const size_t j : edges) {
				const auto &polygon_edge = polygon.edges[j];
				const auto c = polygon_edge.start, e = polygon_edge.end, edge = polygon_edge.direction;
				const double denominator = direction.cross(edge);
				if (denominator != 0) {
					const double t = (c - a).cross(edge) / denominator;
					const double u = (c - a).cross(direction) / denominator;
					if (t >= 0 && t <= 1 && u >= 0 && u <= 1) first = std::min(first, t);
				}
				for (const auto &[p, q] : {std::pair{a, project(a, polygon_edge)}, std::pair{b, project(b, polygon_edge)},
					std::pair{project(c, segment), c}, std::pair{project(e, segment), e}}) {
					const double squared = (p - q).length_squared();
					if (squared <= tolerance_squared || squared < best_squared) {
						const double t = segment.squared == 0 ? 0 : (p - a).dot(direction) / segment.squared;
						if (squared <= tolerance_squared) first = std::min(first, t);
						if (squared < best_squared) { best_squared = squared; best.position = t; best.polygon_point = q; }
					}
				}
			}
			if (std::isfinite(first)) return {0, first};
			if (in(b)) return {0, 1};
			best.squared_distance = best_squared;
			best.distance = std::sqrt(best_squared);
			return best;
		}
	}
    void SegmentContactCache::invalidate() {
        for(auto &entry:table_)entry.polygon=std::numeric_limits<size_t>::max();
    }
    Contact SegmentContactCache::query(const PreparedContactPath &path,
            const PreparedContactPolygon &polygon,size_t polygon_index,size_t polygon_count,double tolerance,
            double upper_distance) {
        if(polygon_index>=polygon_count)throw std::out_of_range("Invalid segment-cache polygon");
        const auto tolerance_bits=std::bit_cast<uint64_t>(tolerance);
        if(polygons_.size()!=polygon_count||tolerance_bits_!=tolerance_bits) {
            // About 64 slots per region, 1,024 to 32,768 in all (80 B each).
            const size_t slots=std::bit_ceil(std::clamp<size_t>(64*polygon_count,1024,32768));
            if(table_.size()!=slots)table_.assign(slots,Entry{});
            else invalidate();
            polygons_.assign(polygon_count,nullptr);tolerance_bits_=tolerance_bits;
        }
        if(polygons_[polygon_index]!=&polygon) {
            invalidate();polygons_[polygon_index]=&polygon;
        }
        const size_t mask=table_.size()-1;
        Contact best{INFINITY,0,INFINITY};
        // The margin dwarfs the rounding of both distances, so the skipped
        // segments are farther than every closest or touching one.
        const double scale=std::max({std::abs(polygon.minimum.x),std::abs(polygon.minimum.y),
            std::abs(polygon.maximum.x),std::abs(polygon.maximum.y)});
        const double limit=upper_distance*(1+1e-9)+tolerance+1e-12*scale;
        const double limit_squared=std::isfinite(limit)?limit*limit:INFINITY;
        for(size_t i=0;i<path.segments.size();++i) {
            const auto &segment=path.segments[i];
            const double dx=std::max({0.0,polygon.minimum.x-segment.maximum.x,segment.minimum.x-polygon.maximum.x});
            const double dy=std::max({0.0,polygon.minimum.y-segment.maximum.y,segment.minimum.y-polygon.maximum.y});
            if(dx*dx+dy*dy>std::min(best.squared_distance,limit_squared))continue;
            const Key key{std::bit_cast<uint64_t>(segment.start.x),std::bit_cast<uint64_t>(segment.start.y),
                std::bit_cast<uint64_t>(segment.end.x),std::bit_cast<uint64_t>(segment.end.y)};
            uint64_t hash=polygon_index*0x9e3779b97f4a7c15ULL;
            for(auto bits:key) {
                hash^=bits+0x9e3779b97f4a7c15ULL+(hash<<6)+(hash>>2);
                hash*=0xff51afd7ed558ccdULL;hash^=hash>>33;
            }
            auto &entry=table_[hash&mask];
            ++queries;
            if(entry.polygon==polygon_index&&entry.segment==key)++hits;
            else {
                entry.segment=key;entry.polygon=polygon_index;
                entry.contact=polygon.convex?convex_segment_contact(segment,polygon,tolerance):
                    contact_impl(1,[&](size_t)->const ContactSegment&{return segment;},
                    polygon.edges.size(),[&](size_t j)->const ContactEdge&{return polygon.edges[j];},
                    polygon.minimum,polygon.maximum,tolerance);
            }
            auto candidate=entry.contact;candidate.position+=double(i);
            if(candidate.distance==0)return candidate;
            if(candidate.squared_distance<best.squared_distance)best=candidate;
        }
        best.distance=std::sqrt(best.squared_distance);
        return best;
    }

}
