// Builds the pieces: inserts the patterns into the polygon, then removes every
// remaining notch with one cut (the naive decomposition), as in Lemma 4 of
// Chazelle and Dobkin. Faces are kept as counterclockwise cycles of point ids
// and every insertion is checked exactly, so the faces tile the polygon by
// construction.

#include "exact_geometry.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <utility>

namespace steiner_convex_partition::detail {

	namespace {

		using exact::BigInt;
		using exact::HPoint;

		// Counterclockwise hull with exact turns (collinear points dropped).
		std::vector<Vec2> convex_hull(std::vector<Vec2> points) {
			std::sort(points.begin(), points.end(), [](const Vec2 &a, const Vec2 &b) {
				return a.x < b.x || (a.x == b.x && a.y < b.y);
			});
			points.erase(std::unique(points.begin(), points.end()), points.end());
			if (points.size() < 3) return points;
			std::vector<Vec2> hull;
			for (int pass = 0; pass < 2; ++pass) {
				const std::size_t floor = hull.size();
				for (const Vec2 &p : points) {
					while (hull.size() >= floor + 2 && exact::orient(hull[hull.size() - 2], hull.back(), p) <= 0) hull.pop_back();
					hull.push_back(p);
				}
				hull.pop_back();
				std::reverse(points.begin(), points.end());
			}
			return hull;
		}

		class Assembler {
		public:
			Assembler(const PolygonData &polygon, const std::vector<PatternGeometry> &patterns)
				: polygon_(polygon), patterns_(patterns) {
				for (const Vec2 &p : polygon.pts) frame_.include(p);
				for (const PatternGeometry &pattern : patterns) {
					for (const PatternEdge &edge : pattern.edges) {
						frame_.include(edge.a.point);
						frame_.include(edge.b.point);
					}
				}
				for (const Vec2 &p : polygon.pts) add_double_point(p);
				for (int v = 0; v < polygon.n; ++v) boundary_edges_[{v, (v + 1) % polygon.n}] = v;
				std::vector<int> outer(static_cast<std::size_t>(polygon.n));
				for (int v = 0; v < polygon.n; ++v) outer[static_cast<std::size_t>(v)] = v;
				faces_.push_back(std::move(outer));
			}

			AssemblyOutput run(bool validate) {
				AssemblyOutput output;
				for (const PatternGeometry &pattern : patterns_) {
					if (!insert_pattern(pattern)) {
						output.failure = failure_.empty() ? "could not insert a pattern" : failure_;
						return output;
					}
				}
				for (int v : polygon_.notch_vertex) {
					if (!cut_notch(v)) {
						output.failure = failure_.empty() ? "could not cut a notch" : failure_;
						return output;
					}
				}

				round_boundary_points_outward();
				for (const std::vector<int> &face : faces_) {
					std::vector<Vec2> ring;
					ring.reserve(face.size());
					bool rounded = false;
					for (int id : face) {
						ring.push_back(approx_[static_cast<std::size_t>(id)]);
						if (!exact_double_[static_cast<std::size_t>(id)]) rounded = true;
					}
					// A rounded point can leave the piece reflex by a rounding error; its
					// hull is convex and still covers it.
					if (rounded) {
						output.exact_coordinates = false;
						ring = convex_hull(std::move(ring));
					}
					Polygon piece;
					piece.reserve(ring.size());
					for (const Vec2 &p : ring) piece.push_back({p.x, p.y});
					output.pieces.push_back(std::move(piece));
				}
				output.cut_endpoints = cut_endpoints_;

				std::vector<HPoint> boundary;
				for (int v = 0; v < polygon_.n; ++v) boundary.push_back(points_[static_cast<std::size_t>(v)]);
				std::vector<std::vector<HPoint>> pieces;
				for (const std::vector<int> &face : faces_) {
					std::vector<HPoint> ring;
					for (int id : face) ring.push_back(points_[static_cast<std::size_t>(id)]);
					std::string why;
					if (!exact::is_weakly_convex(ring, &why)) {
						output.failure = "assembled face is not convex: " + why;
						return output;
					}
					pieces.push_back(std::move(ring));
				}
				if (validate) {
					const exact::ValidationReport report = exact::validate_partition(boundary, pieces, frame_);
					if (!report.ok) {
						output.failure = "validation failed: " + report.reason;
						return output;
					}
					output.validated = true;
				}
				return output;
			}

		private:
			const PolygonData &polygon_;
			const std::vector<PatternGeometry> &patterns_;
			exact::Frame frame_;
			std::vector<HPoint> points_;
			std::vector<Vec2> approx_;
			std::vector<char> exact_double_;
			std::map<std::pair<double, double>, int> steiner_ids_;
			std::vector<std::vector<int>> faces_;
			// Face edges on the polygon boundary, counterclockwise, with the index of
			// the polygon edge that contains them.
			std::map<std::pair<int, int>, int> boundary_edges_;
			// Inexact cut endpoints on the boundary and their polygon edge.
			std::map<int, int> boundary_points_;
			std::size_t cut_endpoints_ = 0;
			std::string failure_;

			int add_double_point(const Vec2 &p) {
				points_.push_back(frame_.point(p));
				approx_.push_back(p);
				exact_double_.push_back(1);
				return static_cast<int>(points_.size()) - 1;
			}

			int add_point(const HPoint &p) {
				points_.push_back(p);
				const Vec2 approx = frame_.approximate(p);
				approx_.push_back(approx);
				exact_double_.push_back(frame_.equals(p, approx) ? 1 : 0);
				return static_cast<int>(points_.size()) - 1;
			}

			const HPoint &hp(int id) const {
				return points_[static_cast<std::size_t>(id)];
			}

			int node_id(const NodeRef &node) {
				if (node.vertex >= 0) return node.vertex;
				const auto key = std::make_pair(node.point.x, node.point.y);
				auto it = steiner_ids_.find(key);
				if (it != steiner_ids_.end()) return it->second;
				const int id = add_double_point(node.point);
				steiner_ids_.emplace(key, id);
				return id;
			}

			bool insert_pattern(const PatternGeometry &pattern) {
				std::vector<std::pair<int, int>> edges;
				std::map<int, std::vector<std::size_t>> incident;
				std::map<int, bool> placed;
				for (const PatternEdge &edge : pattern.edges) {
					const int a = node_id(edge.a);
					const int b = node_id(edge.b);
					incident[a].push_back(edges.size());
					incident[b].push_back(edges.size());
					placed[a] = edge.a.vertex >= 0;
					placed[b] = edge.b.vertex >= 0;
					edges.emplace_back(a, b);
				}
				std::vector<char> used(edges.size(), 0);
				for (;;) {
					std::size_t start = edges.size();
					int from = -1;
					for (std::size_t e = 0; e < edges.size() && start == edges.size(); ++e) {
						if (used[e]) continue;
						if (placed[edges[e].first]) {
							start = e;
							from = edges[e].first;
						} else if (placed[edges[e].second]) {
							start = e;
							from = edges[e].second;
						}
					}
					if (start == edges.size()) break;
					std::vector<int> path {from};
					std::size_t edge = start;
					int current = from;
					for (;;) {
						used[edge] = 1;
						const int other = edges[edge].first == current ? edges[edge].second : edges[edge].first;
						path.push_back(other);
						current = other;
						if (placed[current]) break;
						std::size_t next_edge = edges.size();
						for (std::size_t candidate : incident[current]) {
							if (!used[candidate]) {
								next_edge = candidate;
								break;
							}
						}
						if (next_edge == edges.size()) {
							failure_ = "pattern has a dangling Steiner point";
							return false;
						}
						edge = next_edge;
					}
					if (!insert_chord(path)) return false;
					for (int id : path) placed[id] = true;
				}
				for (char u : used) {
					if (!u) {
						failure_ = "pattern is not connected to the boundary";
						return false;
					}
				}
				return true;
			}

			static std::optional<std::size_t> position(const std::vector<int> &face, int id) {
				for (std::size_t k = 0; k < face.size(); ++k) {
					if (face[k] == id) return k;
				}
				return std::nullopt;
			}

			bool chord_fits(const std::vector<int> &face, std::size_t at, const std::vector<int> &path) const {
				const std::size_t m = face.size();
				const HPoint &prev = hp(face[(at + m - 1) % m]);
				const HPoint &here = hp(face[at]);
				const HPoint &next = hp(face[(at + 1) % m]);
				if (!exact::strictly_inside_corner(prev, here, next, hp(path[1]))) return false;
				for (std::size_t t = 1; t + 1 < path.size(); ++t) {
					for (int id : face) {
						if (exact::same_point(hp(id), hp(path[t]))) return false;
					}
				}
				for (std::size_t t = 0; t + 1 < path.size(); ++t) {
					const HPoint &a = hp(path[t]);
					const HPoint &b = hp(path[t + 1]);
					for (std::size_t k = 0; k < m; ++k) {
						if (exact::open_segment_meets_closed(a, b, hp(face[k]), hp(face[(k + 1) % m]))) return false;
					}
				}
				for (std::size_t s = 0; s + 1 < path.size(); ++s) {
					for (std::size_t t = s + 2; t + 1 < path.size(); ++t) {
						if (exact::closed_segments_meet(hp(path[s]), hp(path[s + 1]), hp(path[t]), hp(path[t + 1]))) return false;
					}
				}
				return true;
			}

			bool insert_chord(const std::vector<int> &path) {
				const int first = path.front();
				const int last = path.back();
				for (std::size_t f = 0; f < faces_.size(); ++f) {
					const std::vector<int> &face = faces_[f];
					const std::optional<std::size_t> a = position(face, first);
					const std::optional<std::size_t> b = position(face, last);
					if (!a || !b || *a == *b) continue;
					if (!chord_fits(face, *a, path)) continue;
					const std::size_t m = face.size();
					std::vector<int> one;
					std::vector<int> two;
					for (std::size_t k = *a;; k = (k + 1) % m) {
						one.push_back(face[k]);
						if (k == *b) break;
					}
					for (std::size_t t = path.size() - 1; t-- > 1;) one.push_back(path[t]);
					for (std::size_t k = *b;; k = (k + 1) % m) {
						two.push_back(face[k]);
						if (k == *a) break;
					}
					for (std::size_t t = 1; t + 1 < path.size(); ++t) two.push_back(path[t]);
					faces_[f] = std::move(one);
					faces_.push_back(std::move(two));
					return true;
				}
				failure_ = "no face accepts a chord";
				return false;
			}

			// A cut endpoint that is not a double is rounded in the output. Every face
			// that has it uses the same rounded point, so the rounded pieces still
			// tile their union; on the polygon boundary the point is rounded to a
			// double on or outside its edge, so that union covers the polygon (a
			// piece may turn reflex by a rounding error; its convex hull covers it).
			void round_boundary_points_outward() {
				for (const auto &[id, edge] : boundary_points_) {
					const std::size_t k = static_cast<std::size_t>(id);
					if (exact_double_[k]) continue;
					const Vec2 &a = polygon_.at(edge);
					const Vec2 &b = polygon_.at(edge + 1);
					const Vec2 near = approx_[k];
					Vec2 best = near;
					long double best_distance = INFINITY;
					for (int dx = -2; dx <= 2; ++dx) {
						for (int dy = -2; dy <= 2; ++dy) {
							Vec2 q = near;
							for (int t = 0; t < std::abs(dx); ++t) q.x = std::nextafter(q.x, dx < 0 ? -INFINITY : INFINITY);
							for (int t = 0; t < std::abs(dy); ++t) q.y = std::nextafter(q.y, dy < 0 ? -INFINITY : INFINITY);
							if (exact::orient(a, b, q) > 0) continue;
							const long double ex = static_cast<long double>(q.x) - near.x;
							const long double ey = static_cast<long double>(q.y) - near.y;
							const long double distance = ex * ex + ey * ey;
							if (distance < best_distance) {
								best_distance = distance;
								best = q;
							}
						}
					}
					approx_[k] = best;
				}
			}

			// Removes the reflex angle at polygon vertex v, if it is still there.
			bool cut_notch(int v) {
				for (std::size_t f = 0; f < faces_.size(); ++f) {
					const std::optional<std::size_t> at = position(faces_[f], v);
					if (!at) continue;
					std::vector<int> &face = faces_[f];
					const std::size_t m = face.size();
					const int prev = face[(*at + m - 1) % m];
					const int next = face[(*at + 1) % m];
					if (exact::orient(hp(prev), hp(v), hp(next)) >= 0) continue;
					return cut_reflex_corner(f, *at);
				}
				return true;
			}

			bool cut_reflex_corner(std::size_t f, std::size_t at) {
				std::vector<int> &face = faces_[f];
				const std::size_t m = face.size();
				const int v = face[at];
				const int prev = face[(at + m - 1) % m];
				const int next = face[(at + 1) % m];
				const HPoint &pv = hp(v);

				// A diagonal to an existing vertex of the face, nearest the bisector of
				// the cone of cuts that remove the angle.
				const Vec2 av = approx_[static_cast<std::size_t>(v)];
				const Vec2 ap = approx_[static_cast<std::size_t>(prev)];
				const Vec2 an = approx_[static_cast<std::size_t>(next)];
				const long double e1x = av.x - static_cast<long double>(ap.x);
				const long double e1y = av.y - static_cast<long double>(ap.y);
				const long double e2x = av.x - static_cast<long double>(an.x);
				const long double e2y = av.y - static_cast<long double>(an.y);
				const long double l1 = std::hypot(e1x, e1y);
				const long double l2 = std::hypot(e2x, e2y);
				const long double bx = e1x / l1 + e2x / l2;
				const long double by = e1y / l1 + e2y / l2;
				int best_vertex = -1;
				long double best_score = -2.0L;
				for (std::size_t k = 0; k < m; ++k) {
					const int u = face[k];
					if (u == v || u == prev || u == next) continue;
					const HPoint &pu = hp(u);
					if (!exact::gap_at_most_half_turn(pv, hp(next), pu) || !exact::gap_at_most_half_turn(pv, pu, hp(prev))) continue;
					bool clear = true;
					for (std::size_t e = 0; e < m && clear; ++e) {
						if (exact::open_segment_meets_closed(pv, pu, hp(face[e]), hp(face[(e + 1) % m]))) clear = false;
					}
					if (!clear) continue;
					const Vec2 au = approx_[static_cast<std::size_t>(u)];
					const long double dx = au.x - static_cast<long double>(av.x);
					const long double dy = au.y - static_cast<long double>(av.y);
					const long double length = std::hypot(dx, dy) * std::hypot(bx, by);
					const long double score = length > 0.0L ? (dx * bx + dy * by) / length : -1.0L;
					if (score > best_score) {
						best_score = score;
						best_vertex = u;
					}
				}
				if (best_vertex >= 0) return insert_chord({v, best_vertex});

				// Otherwise extend an edge at v to the first point of the face boundary.
				std::optional<std::pair<HPoint, std::size_t>> best_hit;
				int best_rank = -1;
				for (const int from : {prev, next}) {
					std::optional<std::pair<HPoint, std::size_t>> hit = ray_hit(face, at, from);
					if (!hit) continue;
					const std::size_t e = hit->second;
					const bool at_vertex = exact::same_point(hit->first, hp(face[e])) || exact::same_point(hit->first, hp(face[(e + 1) % m]));
					const Vec2 approx = frame_.approximate(hit->first);
					const int rank = at_vertex ? 2 : (frame_.equals(hit->first, approx) ? 1 : 0);
					if (rank > best_rank) {
						best_rank = rank;
						best_hit = hit;
					}
				}
				if (!best_hit) {
					failure_ = "no cut removes a reflex angle";
					return false;
				}
				const std::size_t e = best_hit->second;
				const int u = face[e];
				const int w = face[(e + 1) % m];
				int target;
				if (exact::same_point(best_hit->first, hp(u))) {
					target = u;
				} else if (exact::same_point(best_hit->first, hp(w))) {
					target = w;
				} else {
					target = add_point(best_hit->first);
					++cut_endpoints_;
					face.insert(face.begin() + static_cast<std::ptrdiff_t>(e + 1), target);
					const auto on_boundary = boundary_edges_.find({u, w});
					if (on_boundary != boundary_edges_.end()) {
						const int edge = on_boundary->second;
						boundary_edges_.erase(on_boundary);
						boundary_edges_[{u, target}] = edge;
						boundary_edges_[{target, w}] = edge;
						boundary_points_[target] = edge;
					} else {
						insert_into_twin(f, u, w, target);
					}
				}
				return insert_chord({v, target});
			}

			// Keeps the neighbor across edge (u, w) consistent by adding the new point
			// on its edge (w, u) too.
			void insert_into_twin(std::size_t f, int u, int w, int point) {
				for (std::size_t g = 0; g < faces_.size(); ++g) {
					if (g == f) continue;
					std::vector<int> &face = faces_[g];
					const std::size_t m = face.size();
					for (std::size_t k = 0; k < m; ++k) {
						if (face[k] == w && face[(k + 1) % m] == u) {
							face.insert(face.begin() + static_cast<std::ptrdiff_t>(k + 1), point);
							return;
						}
					}
				}
			}

			// First point where the ray from face[at] along the extension of the edge
			// from `from` meets the face boundary, with the edge index it lies on.
			std::optional<std::pair<HPoint, std::size_t>> ray_hit(const std::vector<int> &face, std::size_t at, int from) const {
				const std::size_t m = face.size();
				const HPoint &origin = hp(face[at]);
				const HPoint &back = hp(from);
				const BigInt dx = origin.x * back.w - back.x * origin.w;
				const BigInt dy = origin.y * back.w - back.y * origin.w;
				std::optional<std::pair<HPoint, std::size_t>> best;
				BigInt best_num;
				BigInt best_den;
				auto consider = [&](const HPoint &x, std::size_t e) {
					// Distance along the ray: num / x.w up to positive factors.
					const BigInt num = (x.x * origin.w - origin.x * x.w) * dx + (x.y * origin.w - origin.y * x.w) * dy;
					if (num.sign() <= 0) return;
					if (!best || compare(num * best_den, best_num * x.w) < 0) {
						best = std::make_pair(x, e);
						best_num = num;
						best_den = x.w;
					}
				};
				for (std::size_t e = 0; e < m; ++e) {
					const std::size_t e2 = (e + 1) % m;
					if (e == at || e2 == at) continue;
					const HPoint &u = hp(face[e]);
					const HPoint &w = hp(face[e2]);
					const HPoint x = exact::line_intersection(back, origin, u, w);
					if (x.w.is_zero()) {
						if (exact::orient(back, origin, u) == 0) {
							consider(u, e);
							consider(w, e);
						}
						continue;
					}
					if (!exact::on_closed(u, x, w)) continue;
					consider(x, e);
				}
				return best;
			}
		};
	}

	AssemblyOutput assemble(const PolygonData &polygon, const std::vector<PatternGeometry> &patterns, bool validate) {
		Assembler assembler(polygon, patterns);
		return assembler.run(validate);
	}
}
