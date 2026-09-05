#include "tpp/nonconvex/unordered.h"
#include "tpp/convex/certified.h"
#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace {
	using Polygon = std::vector<Vector2>;
	constexpr size_t none = std::numeric_limits<size_t>::max();

	double length(const Polygon &path) {
		double result = 0;
		for (size_t i = 1; i < path.size(); ++i) result += path[i - 1].distance_to(path[i]);
		return result;
	}

	Polygon hull(Polygon p) {
		std::sort(p.begin(), p.end(), [](auto a, auto b) { return std::tie(a.x, a.y) < std::tie(b.x, b.y); });
		p.erase(std::unique(p.begin(), p.end(), [](auto a, auto b) { return a.x == b.x && a.y == b.y; }), p.end());
		Polygon h;
		for (auto v : p) {
			while (h.size() > 1 && (h.back() - h[h.size() - 2]).cross(v - h.back()) <= 0) h.pop_back();
			h.push_back(v);
		}
		const size_t lower = h.size();
		for (size_t i = p.size() - 1; i-- > 0;) {
			while (h.size() > lower && (h.back() - h[h.size() - 2]).cross(p[i] - h.back()) <= 0) h.pop_back();
			h.push_back(p[i]);
		}
		h.pop_back();
		return h;
	}

	Vector2 project(Vector2 p, Vector2 a, Vector2 b) {
		const auto d = b - a;
		return a + d * (d.length_squared() == 0 ? 0 : std::clamp((p - a).dot(d) / d.length_squared(), 0.0, 1.0));
	}

	bool inside(Vector2 p, const Polygon &poly, double eps) {
		bool result = false;
		for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
			const auto a = poly[j], b = poly[i];
			if (p.distance_to(project(p, a, b)) <= eps) return true;
			if ((a.y > p.y) != (b.y > p.y) && p.x < a.x + (b.x - a.x) * (p.y - a.y) / (b.y - a.y)) result = !result;
		}
		return result;
	}

	struct Contact { double distance = std::numeric_limits<double>::infinity(); double position = 0; };

	Contact contact(const Polygon &path, const Polygon &poly, double eps) {
		Contact best;
		for (size_t i = 1; i < path.size(); ++i) {
			const auto a = path[i - 1], b = path[i], d = b - a;
			if (inside(a, poly, eps)) return {0, double(i - 1)};
			double first = std::numeric_limits<double>::infinity();
			for (size_t j = 0; j < poly.size(); ++j) {
				const auto c = poly[j], e = poly[(j + 1) % poly.size()], v = e - c;
				const double denominator = d.cross(v);
				if (denominator != 0) {
					const double t = (c - a).cross(v) / denominator, u = (c - a).cross(d) / denominator;
					if (t >= 0 && t <= 1 && u >= 0 && u <= 1) first = std::min(first, t);
				}
				for (const auto &[p, q] : {std::pair{a, project(a, c, e)}, std::pair{b, project(b, c, e)},
					std::pair{project(c, a, b), c}, std::pair{project(e, a, b), e}}) {
					const double distance = p.distance_to(q);
					const double t = d.length_squared() == 0 ? 0 : (p - a).dot(d) / d.length_squared();
					if (distance <= eps) first = std::min(first, t);
					if (distance < best.distance) best = {distance, double(i - 1) + t};
				}
			}
			if (std::isfinite(first)) return {0, double(i - 1) + first};
			if (inside(b, poly, eps)) return {0, double(i)};
		}
		return best;
	}

	struct Element { size_t polygon; size_t piece = none; };
	struct Node {
		std::vector<Element> sequence;
		Polygon path;
		double bound = 0;
		size_t serial = 0;
	};
	struct Later {
		bool operator()(const Node &a, const Node &b) const {
			return std::tie(a.bound, a.serial) > std::tie(b.bound, b.serial);
		}
	};
}

namespace tpp {
	UnorderedTppSolveResult tpp_nonconvex_unordered_solve(
		const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &input,
		const UnorderedTppSolveOptions &options
	) {
		const auto began = std::chrono::steady_clock::now();
		auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count(); };
		if (!start.is_finite() || !target.is_finite() || std::isnan(options.max_seconds) || options.max_seconds < 0
			|| !std::isfinite(options.absolute_gap) || options.absolute_gap < 0
			|| !std::isfinite(options.relative_gap) || options.relative_gap < 0
			|| !std::isfinite(options.feasibility_tolerance) || options.feasibility_tolerance <= 0)
			throw std::invalid_argument("Invalid endpoints or unordered TPP options.");
		std::vector<Polygon> polygons = input, hulls;
		double normalization_error = 0;
		for (auto &p : polygons) {
			const double duplicate_tolerance = options.feasibility_tolerance * 1e-4;
			Polygon cleaned;
			for (auto v : p) {
				if (!cleaned.empty() && cleaned.back().distance_to(v) <= duplicate_tolerance)
					normalization_error += 2 * cleaned.back().distance_to(v);
				else cleaned.push_back(v);
			}
			p = std::move(cleaned);
			if (p.size() > 1 && p.front().distance_to(p.back()) <= duplicate_tolerance) {
				normalization_error += 2 * p.front().distance_to(p.back());
				p.pop_back();
			}
			if (p.size() < 3 || !std::all_of(p.begin(), p.end(), [](auto v) { return v.is_finite(); }))
				throw std::invalid_argument("Expected finite, nondegenerate simple polygons.");
			double area = 0;
			for (size_t i = 0; i < p.size(); ++i) area += (p[i] - p[0]).cross(p[(i + 1) % p.size()] - p[0]);
			if (area == 0) throw std::invalid_argument("Zero-area polygon.");
			if (area < 0) std::reverse(p.begin(), p.end());
			hulls.push_back(hull(p));
		}
		UnorderedTppSolveResult result;
		const size_t n = polygons.size();
		std::vector<size_t> initial_order;
		const double eps = options.feasibility_tolerance;
		auto covered = [&](const Polygon &path) {
			return std::all_of(polygons.begin(), polygons.end(), [&](const auto &p) { return contact(path, p, eps).distance <= eps; });
		};
		auto improve = [&](const Polygon &path) {
			const double value = length(path);
			if (std::isfinite(value) && value < result.upper_bound && covered(path)) {
				result.path = path;
				result.upper_bound = value;
			}
		};
		result.lower_bound = start.distance_to(target);
		improve({start, target});
		if (!std::isfinite(result.upper_bound)) {
			Polygon initial{start};
			std::vector<bool> used(n);
			for (size_t k = 0; k < n; ++k) {
				double best = std::numeric_limits<double>::infinity();
				size_t selected = none;
				Vector2 point;
				for (size_t j = 0; j < n; ++j) if (!used[j]) for (auto v : polygons[j]) {
					const double distance = initial.back().distance_to(v);
					if (distance < best) { best = distance; selected = j; point = v; }
				}
				used[selected] = true;
				initial_order.push_back(selected);
				initial.push_back(point);
			}
			initial.push_back(target);
			for (size_t pass = 0; pass < 10; ++pass) {
				bool changed = false;
				for (size_t i = 1; i < n; ++i) for (size_t j = i + 1; j <= n; ++j) {
					const double delta = initial[i - 1].distance_to(initial[j]) + initial[i].distance_to(initial[j + 1])
						- initial[i - 1].distance_to(initial[i]) - initial[j].distance_to(initial[j + 1]);
					if (delta < -eps) {
						std::reverse(initial.begin() + i, initial.begin() + j + 1);
						std::reverse(initial_order.begin() + i - 1, initial_order.begin() + j);
						changed = true;
					}
				}
				if (!changed) break;
			}
			for (size_t pass = 0; pass < 8 && elapsed() < options.max_seconds; ++pass) {
				for (size_t k = 0; k < n; ++k) {
					const auto &p = polygons[initial_order[k]];
					const auto left = initial[k], right = initial[k + 2];
					auto cost = [&](Vector2 v) { return left.distance_to(v) + right.distance_to(v); };
					double best = cost(initial[k + 1]);
					for (size_t j = 0; j < p.size(); ++j) {
						const auto a = p[j], d = p[(j + 1) % p.size()] - a;
						double lo = 0, hi = 1;
						for (size_t iteration = 0; iteration < 36; ++iteration) {
							const double u = (2 * lo + hi) / 3, v = (lo + 2 * hi) / 3;
							if (cost(a + u * d) < cost(a + v * d)) hi = v; else lo = u;
						}
						const auto candidate = a + ((lo + hi) / 2) * d;
						if (cost(candidate) < best) { best = cost(candidate); initial[k + 1] = candidate; }
					}
				}
			}
			// Every heuristic contact stays on its original polygon.
			improve(initial);
		}
		auto gap = [&] { return options.absolute_gap + options.relative_gap * std::abs(result.upper_bound); };
		auto limited = [&] { return result.calls >= options.max_calls || elapsed() >= options.max_seconds; };
		std::vector<std::vector<Polygon>> pieces(n);
		DynamicConvexTppWorkspace workspace;
		auto solve = [&](Node &node) {
			std::vector<Polygon> selected;
			for (auto e : node.sequence) selected.push_back(e.piece == none ? hulls[e.polygon] : pieces[e.polygon][e.piece]);
			++result.calls;
			const auto certified = tpp_convex_solve_certified(start, target, selected, workspace, gap() * .25);
			node.path = certified.path;
			result.fallback_calls += certified.used_fallback;
			node.bound = std::max(node.bound, certified.lower_bound);
			if (node.path.size() < 2 || !std::all_of(node.path.begin(), node.path.end(), [](auto v) { return v.is_finite(); }))
				throw std::runtime_error("Convex oracle returned an invalid path.");
			
		};
		double settled_bound = result.upper_bound;
		std::priority_queue<Node, std::vector<Node>, Later> queue;
		queue.push({{}, {start, target}, result.lower_bound, 0});
		size_t serial = 1;
		std::optional<Node> dive;
		auto frontier_bound = [&] {
			return std::min(queue.empty() ? result.upper_bound : queue.top().bound, dive ? dive->bound : result.upper_bound);
		};
		while (!queue.empty() || dive) {
			result.peak_queue = std::max(result.peak_queue, queue.size() + size_t(dive.has_value()));
			result.lower_bound = std::min(result.upper_bound, frontier_bound());
			if (result.upper_bound - result.lower_bound <= gap() || limited()) break;
			const bool diving = dive.has_value() || (options.dive_interval && result.nodes % options.dive_interval == 0);
			Node node;
			if (dive) { node = std::move(*dive); dive.reset(); }
			else { node = queue.top(); queue.pop(); }
			++result.nodes;
			if (node.path.empty()) solve(node);
			if (node.bound >= result.upper_bound - gap()) { settled_bound = std::min(settled_bound, node.bound); continue; }
			improve(node.path);
			if (node.bound >= result.upper_bound - gap()) { settled_bound = std::min(settled_bound, node.bound); continue; }
			size_t chosen = none;
			double farthest = eps;
			for (size_t j = 0; j < n; ++j) {
				const double distance = contact(node.path, polygons[j], eps).distance;
				if (distance > farthest) { farthest = distance; chosen = j; }
			}
			if (chosen == none) {
				// An unresolved numerical oracle gap must remain in the global certificate.
				queue.push(std::move(node));
				break;
			}
			auto found = std::find_if(node.sequence.begin(), node.sequence.end(), [&](auto e) { return e.polygon == chosen; });
			std::vector<Node> children;
			if (found != node.sequence.end()) {
				if (found->piece != none) throw std::runtime_error("Certified oracle failed to visit an assigned piece.");
				++result.decomposition_branches;
				if (pieces[chosen].empty()) {
					for (auto piece : decompose_polygon(polygons[chosen])) {
						piece = hull(std::move(piece));
						if (piece.size() >= 3) pieces[chosen].push_back(std::move(piece));
					}
				}
				if (pieces[chosen].empty()) throw std::runtime_error("Empty convex decomposition.");
				const size_t position = found - node.sequence.begin();
				for (size_t j = 0; j < pieces[chosen].size(); ++j) {
					auto sequence = node.sequence;
					sequence[position].piece = j;
					children.push_back({std::move(sequence), {}, node.bound, serial++});
				}
			} else {
				++result.insertion_branches;
				for (size_t j = 0; j <= node.sequence.size(); ++j) {
					auto sequence = node.sequence;
					sequence.insert(sequence.begin() + j, {chosen});
					children.push_back({std::move(sequence), {}, node.bound, serial++});
				}
			}
			for (auto &child : children) {
				if (!limited()) { solve(child); improve(child.path); }
				if (child.bound < result.upper_bound - gap()) {
					if (diving && (!dive || child.bound < dive->bound)) {
						if (dive) queue.push(std::move(*dive));
						dive = std::move(child);
					} else queue.push(std::move(child));
				} else settled_bound = std::min(settled_bound, child.bound);
			}
		}
		result.lower_bound = std::min({result.upper_bound, settled_bound, frontier_bound()});
		result.lower_bound = std::max(start.distance_to(target), result.lower_bound - normalization_error);
		result.exact = result.upper_bound - result.lower_bound <= gap();
		result.termination = result.exact ? UnorderedTppTermination::Optimal
			: result.calls >= options.max_calls ? UnorderedTppTermination::CallLimit
			: elapsed() >= options.max_seconds ? UnorderedTppTermination::TimeLimit
			: UnorderedTppTermination::NumericalLimit;
		std::vector<std::pair<double, size_t>> visits;
		for (size_t j = 0; j < n; ++j) visits.emplace_back(contact(result.path, polygons[j], eps).position, j);
		std::sort(visits.begin(), visits.end());
		for (auto [position, j] : visits) result.order.push_back(j);
		result.seconds = elapsed();
		return result;
	}
}
