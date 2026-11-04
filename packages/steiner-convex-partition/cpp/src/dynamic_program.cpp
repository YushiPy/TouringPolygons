// The dynamic program of Chazelle and Dobkin over circular intervals of notches.
//
// Notches are numbered 0..c-1 counterclockwise. An interval [i..j] uses
// absolute indices i <= j <= i + c - 1, reduced modulo c on access.
//
//   S(i, j)  the maximum number of compatible patterns whose notches all lie in
//            [i..j]. Two patterns are compatible when their notch sequences do
//            not interleave around the boundary (Chazelle and Dobkin, p. 85), so
//            the problem is a non-crossing interval DP.
//   B(i, j)  the best forward Y-subtree: a subtree of a Y-pattern attached to
//            notch i by one arm, with its other notches in (i..j], to be
//            completed by a second arm at i from outside the interval. Among the
//            subtrees whose gaps hold S(i, j) patterns (relation (2) of the
//            paper), only the one whose arm leaves the most room for the second
//            arm is kept.
//   F(i, j)  the mirror image: a backward subtree attached to notch j.
//
// At a notch with two arms, ordered counterclockwise from the edge to the next
// vertex, the arm toward the later notches (the forward subtree) comes first.
// The best forward arm is the most counterclockwise one; the best backward arm
// the most clockwise one.

#include "internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

namespace steiner_convex_partition::detail {

	namespace {

		using exact::dot_sign;
		using exact::gap_at_most_half_turn;
		using exact::orient;

		// ---------------------------------------------------------------- junctions

		enum class Role : std::uint8_t {
			alone,              // single arm: must remove the reflex angle alone
			root_forward,       // root of a forward subtree: only the first gap
			root_backward,      // root of a backward subtree: only the last gap
			with_backward_arm,  // the junction arm comes before an existing arm
			with_forward_arm,   // the junction arm comes after an existing arm
		};

		struct RoleSpec {
			Role role = Role::alone;
			Vec2 arm;
		};

		enum class Objective : std::uint8_t {
			none,
			first_forward,  // maximize the counterclockwise angle at the first notch
			last_backward,  // maximize the clockwise angle at the last notch
		};

		struct HalfPlane {
			Vec2 p;
			Vec2 q;
			bool strict = false;  // orient(p, q, s) > 0, else >= 0
		};

		struct ExcludedRay {
			Vec2 origin;
			Vec2 through;
		};

		struct LongPoint {
			long double x = 0.0L;
			long double y = 0.0L;
		};

		long double cross_long(const LongPoint &o, const LongPoint &a, const LongPoint &b) {
			return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
		}

		LongPoint to_long(const Vec2 &p) {
			return {static_cast<long double>(p.x), static_cast<long double>(p.y)};
		}

		Vec2 to_double(const LongPoint &p) {
			return {static_cast<double>(p.x), static_cast<double>(p.y)};
		}

		bool in_closed_triangle(const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &p) {
			return orient(a, b, p) >= 0 && orient(b, c, p) >= 0 && orient(c, a, p) >= 0;
		}

		// Does the polygon edge (u, w) meet the closed side [corner, z] anywhere but
		// at `corner`? `incident` says the edge starts or ends at `corner`.
		bool edge_meets_side(const Vec2 &u, const Vec2 &w, const Vec2 &corner, const Vec2 &z, bool incident) {
			if (incident) {
				const Vec2 &other = (u == corner) ? w : u;
				return orient(corner, z, other) == 0 && dot_sign(corner, other, corner, z) > 0;
			}
			return exact::closed_segments_meet(u, w, corner, z);
		}

		class JunctionSolver {
		public:
			explicit JunctionSolver(const PolygonData &polygon)
				: polygon_(polygon) {}

			std::size_t calls = 0;
			std::size_t successes = 0;

			// Steiner point s joining the notch vertices va, vb, vc (counterclockwise
			// on the boundary) with the given roles, or nothing.
			std::optional<Vec2> solve(int va, int vb, int vc, const std::array<RoleSpec, 3> &roles, Objective objective) {
				++calls;
				const int vertices[3] = {va, vb, vc};
				const Vec2 corners[3] = {polygon_.at(va), polygon_.at(vb), polygon_.at(vc)};
				if (orient(corners[0], corners[1], corners[2]) <= 0) return std::nullopt;

				planes_.clear();
				rays_.clear();
				for (int t = 0; t < 3; ++t) {
					planes_.push_back({corners[t], corners[(t + 1) % 3], false});
				}
				for (int t = 0; t < 3; ++t) {
					add_role(vertices[t], roles[static_cast<std::size_t>(t)]);
				}
				// Cheap rejection before the boundary scan: the angular constraints
				// alone, slightly relaxed, already leave nothing in most failing cases.
				tolerance_ = relaxation(corners);
				clip(corners, tolerance_, loose_);
				if (loose_.empty()) return std::nullopt;
				// The boundary constraints depend on the triangle only; they are
				// cached while the outer notches stay the same.
				ScanCache &cache = scans_[static_cast<std::size_t>(vb)];
				if (cache.stamp != stamp_ || cache.first != va || cache.last != vc) {
					cache.stamp = stamp_;
					cache.first = va;
					cache.last = vc;
					const std::size_t before = planes_.size();
					cache.feasible = true;
					for (int t = 0; t < 3 && cache.feasible; ++t) {
						cache.feasible = scan_arc(vertices[t], vertices[(t + 1) % 3], corners[t], corners[(t + 1) % 3], corners[(t + 2) % 3]);
					}
					cache.planes.assign(planes_.begin() + static_cast<std::ptrdiff_t>(before), planes_.end());
					planes_.resize(before);
				}
				if (!cache.feasible) return std::nullopt;
				planes_.insert(planes_.end(), cache.planes.begin(), cache.planes.end());

				std::optional<Vec2> best = search(corners, objective);
				if (best) ++successes;
				return best;
			}

			// Starts a new outer pair of notches: cached boundary scans expire.
			void begin_interval() {
				++stamp_;
			}

		private:
			struct ScanCache {
				std::size_t stamp = 0;
				int first = -1;
				int last = -1;
				bool feasible = false;
				std::vector<HalfPlane> planes;
			};

			const PolygonData &polygon_;
			std::vector<ScanCache> scans_ = std::vector<ScanCache>(static_cast<std::size_t>(polygon_.n));
			std::size_t stamp_ = 1;
			std::vector<HalfPlane> planes_;
			std::vector<ExcludedRay> rays_;
			std::vector<LongPoint> loose_;
			std::vector<LongPoint> tight_;
			std::vector<LongPoint> scratch_;
			std::vector<Vec2> candidates_;
			std::vector<Vec2> valid_;
			long double tolerance_ = 0.0L;

			void add_role(int vertex, const RoleSpec &spec) {
				const Vec2 &v = polygon_.at(vertex);
				const Vec2 &next = polygon_.next(vertex);
				const Vec2 &prev = polygon_.prev(vertex);
				switch (spec.role) {
					case Role::alone:
						planes_.push_back({v, next, false});
						planes_.push_back({prev, v, false});
						rays_.push_back({v, next});
						rays_.push_back({v, prev});
						break;
					case Role::root_forward:
						planes_.push_back({v, next, false});
						rays_.push_back({v, next});
						break;
					case Role::root_backward:
						planes_.push_back({prev, v, false});
						rays_.push_back({v, prev});
						break;
					case Role::with_backward_arm:
						planes_.push_back({v, next, false});
						planes_.push_back({spec.arm, v, false});
						rays_.push_back({v, next});
						rays_.push_back({v, spec.arm});
						break;
					case Role::with_forward_arm:
						planes_.push_back({v, spec.arm, false});
						planes_.push_back({prev, v, false});
						rays_.push_back({v, spec.arm});
						rays_.push_back({v, prev});
						break;
				}
			}

			// The boundary arc from vertex vx to vertex vy (counterclockwise) must lie,
			// inside the triangle (x, y, z), in the sub-triangle (x, y, s) and off the
			// arms x-s and y-s. This holds for every s iff no arc edge meets the sides
			// [x, z] or [y, z] (other than at x and y), s stays strictly left of x->y
			// when an arc edge touches the open side (x, y), and the arc vertices in
			// the triangle give two binding strict constraints, one through x and one
			// through y.
			bool scan_arc(int vx, int vy, const Vec2 &x, const Vec2 &y, const Vec2 &z) {
				bool touches_xy = false;
				bool have_x = false;
				bool have_y = false;
				Vec2 binding_x;
				Vec2 binding_y;
				const int n = polygon_.n;
				// Every test below involves the triangle; an edge outside its bounding
				// box passes them all.
				const double min_x = std::min({x.x, y.x, z.x});
				const double max_x = std::max({x.x, y.x, z.x});
				const double min_y = std::min({x.y, y.y, z.y});
				const double max_y = std::max({x.y, y.y, z.y});
				for (int e = vx; e != vy; e = (e + 1) % n) {
					const Vec2 &u = polygon_.at(e);
					const Vec2 &w = polygon_.at(e + 1);
					if (std::max(u.x, w.x) < min_x || std::min(u.x, w.x) > max_x || std::max(u.y, w.y) < min_y || std::min(u.y, w.y) > max_y) continue;
					const bool starts_at_x = e == vx;
					const bool ends_at_y = (e + 1) % n == vy;
					if (edge_meets_side(u, w, x, z, starts_at_x)) return false;
					if (edge_meets_side(u, w, y, z, ends_at_y)) return false;
					if (!touches_xy && exact::open_segment_meets_closed(x, y, u, w)) touches_xy = true;
					if (!ends_at_y && in_closed_triangle(x, y, z, w)) {
						// From y the binding vertex is the most clockwise one; from x the most
						// counterclockwise one.
						if (!have_y || orient(y, w, binding_y) > 0) binding_y = w;
						if (!have_x || orient(x, binding_x, w) > 0) binding_x = w;
						have_x = have_y = true;
					}
				}
				if (touches_xy) planes_.push_back({x, y, true});
				if (have_x) planes_.push_back({x, binding_x, true});
				if (have_y) planes_.push_back({binding_y, y, true});
				return true;
			}

			bool satisfies(const Vec2 &s, const Vec2 (&corners)[3]) const {
				for (const Vec2 &corner : corners) {
					if (s == corner) return false;
				}
				for (const HalfPlane &h : planes_) {
					const int o = orient(h.p, h.q, s);
					if (h.strict ? o <= 0 : o < 0) return false;
				}
				for (const ExcludedRay &r : rays_) {
					if (orient(r.origin, r.through, s) == 0 && dot_sign(r.origin, r.through, r.origin, s) > 0) return false;
				}
				return true;
			}

			// Direct check of the three arms against the whole boundary: a safety net
			// for the constraint derivation above.
			bool arms_clear(const Vec2 &s, const Vec2 (&corners)[3]) const {
				if (point_on_boundary(polygon_, s)) return false;
				for (const Vec2 &corner : corners) {
					if (!open_segment_clear(polygon_, corner, s)) return false;
				}
				return true;
			}

			bool better(const Vec2 &a, const Vec2 &b, const Vec2 (&corners)[3], Objective objective) const {
				if (objective == Objective::first_forward) return orient(corners[0], b, a) > 0;
				if (objective == Objective::last_backward) return orient(corners[2], a, b) > 0;
				return false;
			}

			// A distance, relative to the triangle's size, by which the relaxed clip
			// widens every constraint.
			static long double relaxation(const Vec2 (&corners)[3]) {
				long double scale = 0.0L;
				for (int t = 0; t < 3; ++t) {
					const long double dx = static_cast<long double>(corners[(t + 1) % 3].x) - corners[t].x;
					const long double dy = static_cast<long double>(corners[(t + 1) % 3].y) - corners[t].y;
					scale = std::max(scale, dx * dx + dy * dy);
				}
				return std::sqrt(scale) * 1.0e-9L;
			}

			// Clips the triangle by every constraint, widened by the distance `slack`, in long
			// double; the result only steers the search, every answer is checked
			// exactly.
			void clip(const Vec2 (&corners)[3], long double slack, std::vector<LongPoint> &region) {
				region.assign({to_long(corners[0]), to_long(corners[1]), to_long(corners[2])});
				for (const HalfPlane &h : planes_) {
					const LongPoint p = to_long(h.p);
					const LongPoint q = to_long(h.q);
					const long double length = std::hypot(q.x - p.x, q.y - p.y);
					const long double bound = -slack * length;
					scratch_.clear();
					const std::size_t m = region.size();
					for (std::size_t k = 0; k < m; ++k) {
						const LongPoint &a = region[k];
						const LongPoint &b = region[(k + 1) % m];
						const long double fa = cross_long(p, q, a);
						const long double fb = cross_long(p, q, b);
						const bool ka = fa >= bound;
						const bool kb = fb >= bound;
						if (ka) scratch_.push_back(a);
						if (ka != kb) {
							const long double t = (fa - bound) / (fa - fb);
							scratch_.push_back({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)});
						}
					}
					region.swap(scratch_);
					if (region.empty()) return;
				}
			}

			std::optional<Vec2> search(const Vec2 (&corners)[3], Objective objective) {
				clip(corners, tolerance_, loose_);
				if (loose_.empty()) return std::nullopt;
				clip(corners, 0.0L, tight_);
				const std::vector<LongPoint> &region = tight_.empty() ? loose_ : tight_;

				candidates_.clear();
				LongPoint centroid {0.0L, 0.0L};
				for (const LongPoint &p : region) {
					centroid.x += p.x;
					centroid.y += p.y;
				}
				centroid.x /= static_cast<long double>(region.size());
				centroid.y /= static_cast<long double>(region.size());
				LongPoint extreme = centroid;
				if (objective != Objective::none) {
					const LongPoint root = to_long(objective == Objective::first_forward ? corners[0] : corners[2]);
					extreme = region.front();
					for (const LongPoint &p : region) {
						const long double turn = cross_long(root, extreme, p);
						if (objective == Objective::first_forward ? turn > 0.0L : turn < 0.0L) extreme = p;
					}
					add_with_neighbors(candidates_, to_double(extreme));
				}
				static constexpr long double steps[] = {
					0.0L, 1.0e-12L, 1.0e-9L, 1.0e-6L, 1.0e-4L, 1.0e-2L, 0.125L, 0.5L, 1.0L,
				};
				for (const long double t : steps) {
					candidates_.push_back(to_double({
						extreme.x + t * (centroid.x - extreme.x),
						extreme.y + t * (centroid.y - extreme.y),
					}));
				}
				std::optional<Vec2> found = pick(candidates_, corners, objective);
				if (found) return found;

				// The region may be thin or a single point that rounding lost: try the
				// vertices of the arrangement of constraint lines near the relaxed
				// region, computed exactly.
				long double min_x = loose_.front().x;
				long double max_x = min_x;
				long double min_y = loose_.front().y;
				long double max_y = min_y;
				for (const LongPoint &p : loose_) {
					min_x = std::min(min_x, p.x);
					max_x = std::max(max_x, p.x);
					min_y = std::min(min_y, p.y);
					max_y = std::max(max_y, p.y);
				}
				const long double margin = std::max({max_x - min_x, max_y - min_y, tolerance_});
				candidates_.clear();
				for (std::size_t a = 0; a < planes_.size(); ++a) {
					for (std::size_t b = a + 1; b < planes_.size(); ++b) {
						const std::optional<LongPoint> near = approximate_intersection(planes_[a], planes_[b]);
						if (!near) continue;
						if (near->x < min_x - margin || near->x > max_x + margin || near->y < min_y - margin || near->y > max_y + margin) continue;
						std::optional<Vec2> point = exact_intersection(planes_[a], planes_[b]);
						if (point) add_with_neighbors(candidates_, *point);
					}
				}
				return pick(candidates_, corners, objective);
			}

			static std::optional<LongPoint> approximate_intersection(const HalfPlane &a, const HalfPlane &b) {
				const LongPoint p = to_long(a.p);
				const LongPoint r {static_cast<long double>(a.q.x) - a.p.x, static_cast<long double>(a.q.y) - a.p.y};
				const LongPoint q = to_long(b.p);
				const LongPoint s {static_cast<long double>(b.q.x) - b.p.x, static_cast<long double>(b.q.y) - b.p.y};
				const long double det = r.x * s.y - r.y * s.x;
				if (det == 0.0L) return std::nullopt;
				const long double t = ((q.x - p.x) * s.y - (q.y - p.y) * s.x) / det;
				return LongPoint {p.x + t * r.x, p.y + t * r.y};
			}

			static void add_with_neighbors(std::vector<Vec2> &out, const Vec2 &p) {
				for (int dx = -1; dx <= 1; ++dx) {
					for (int dy = -1; dy <= 1; ++dy) {
						Vec2 q = p;
						if (dx < 0) q.x = std::nextafter(q.x, -INFINITY);
						if (dx > 0) q.x = std::nextafter(q.x, INFINITY);
						if (dy < 0) q.y = std::nextafter(q.y, -INFINITY);
						if (dy > 0) q.y = std::nextafter(q.y, INFINITY);
						out.push_back(q);
					}
				}
			}

			static std::optional<Vec2> exact_intersection(const HalfPlane &a, const HalfPlane &b) {
				exact::Frame frame;
				frame.include(a.p);
				frame.include(a.q);
				frame.include(b.p);
				frame.include(b.q);
				const exact::HPoint point = exact::line_intersection(frame.point(a.p), frame.point(a.q), frame.point(b.p), frame.point(b.q));
				if (point.w.is_zero()) return std::nullopt;
				const Vec2 approx = frame.approximate(point);
				if (!std::isfinite(approx.x) || !std::isfinite(approx.y)) return std::nullopt;
				return approx;
			}

			std::optional<Vec2> pick(const std::vector<Vec2> &candidates, const Vec2 (&corners)[3], Objective objective) {
				std::vector<Vec2> &valid = valid_;
				valid.clear();
				for (const Vec2 &s : candidates) {
					if (!std::isfinite(s.x) || !std::isfinite(s.y)) continue;
					if (satisfies(s, corners)) valid.push_back(s);
				}
				if (objective != Objective::none) {
					std::stable_sort(valid.begin(), valid.end(), [&](const Vec2 &a, const Vec2 &b) {
						return better(a, b, corners, objective);
					});
				}
				for (const Vec2 &s : valid) {
					if (arms_clear(s, corners)) return s;
				}
				return std::nullopt;
			}
		};

		// ---------------------------------------------------------------- X4

		struct X4Points {
			bool ok = false;
			Vec2 a;  // joins the first two notches
			Vec2 b;  // joins the last two notches
		};

		class X4Solver {
		public:
			explicit X4Solver(const PolygonData &polygon)
				: polygon_(polygon) {
				// Unit directions of the two edge extensions at every notch, and their
				// bisector.
				directions_.resize(static_cast<std::size_t>(polygon.n));
				for (int v : polygon.notch_vertex) {
					const LongPoint p = to_long(polygon.at(v));
					const LongPoint prev = to_long(polygon.prev(v));
					const LongPoint next = to_long(polygon.next(v));
					Directions &d = directions_[static_cast<std::size_t>(v)];
					d.extension[0] = normalized({p.x - prev.x, p.y - prev.y});
					d.extension[1] = normalized({p.x - next.x, p.y - next.y});
					d.bisector = normalized({d.extension[0].x + d.extension[1].x, d.extension[0].y + d.extension[1].y});
				}
			}

			std::size_t calls = 0;
			std::size_t successes = 0;

			// Loose X4-pattern on the notch vertices vp, vq, vr, vs (counterclockwise):
			// a Steiner point a joined to vp and vq, b joined to vr and vs, and the
			// middle edge a-b. Chazelle and Dobkin (Lemma 16) show that an X4-pattern
			// not reducible to a Y-pattern can be moved so that each notch's edge may
			// lie along either extension of its polygon edges; candidates come from
			// those extensions.
			// Necessary condition on one pair (p, q) of a loose X4-pattern: the four
			// edge extensions at p and q meet ahead of both notches, and some of
			// those meeting points (or their average) is joined to p and q by clear
			// segments that remove both reflex angles.
			bool pair_possible(int vp, int vq) {
				return pair(vp, vq).possible;
			}

			X4Points solve(int vp, int vq, int vr, int vs) {
				++calls;
				X4Points result;
				const int ids[4] = {vp, vq, vr, vs};
				const PairInfo &first = pair(vp, vq);
				const PairInfo &second = pair(vr, vs);
				if (!first.possible || !second.possible) return result;
				const std::vector<LongPoint> corners_a(first.corners, first.corners + 4);
				const std::vector<LongPoint> corners_b(second.corners, second.corners + 4);
				std::vector<LongPoint> centers_a = {first.center};
				std::vector<LongPoint> centers_b = {second.center};
				if (first.has_bisector) centers_a.push_back(first.bisector);
				if (second.has_bisector) centers_b.push_back(second.bisector);

				std::vector<std::pair<Vec2, Vec2>> candidates;
				for (const LongPoint &a : centers_a) {
					for (const LongPoint &b : centers_b) candidates.emplace_back(to_double(a), to_double(b));
				}
				for (const LongPoint &a : corners_a) {
					for (const LongPoint &b : corners_b) candidates.emplace_back(to_double(a), to_double(b));
				}
				for (const auto &[a, b] : candidates) {
					if (valid(ids, a, b)) {
						result.ok = true;
						result.a = a;
						result.b = b;
						++successes;
						return result;
					}
				}
				return result;
			}

			bool valid(const int (&ids)[4], const Vec2 &a, const Vec2 &b) const {
				const Vec2 p = polygon_.at(ids[0]);
				const Vec2 q = polygon_.at(ids[1]);
				const Vec2 r = polygon_.at(ids[2]);
				const Vec2 s = polygon_.at(ids[3]);
				if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(b.x) || !std::isfinite(b.y)) return false;
				if (a == b) return false;
				for (const Vec2 &v : {p, q, r, s}) {
					if (a == v || b == v) return false;
				}
				if (orient(p, q, b) <= 0 || !in_closed_triangle(p, q, b, a)) return false;
				if (orient(r, s, a) <= 0 || !in_closed_triangle(r, s, a, b)) return false;
				if (!resolves_alone(ids[0], a) || !resolves_alone(ids[1], a)) return false;
				if (!resolves_alone(ids[2], b) || !resolves_alone(ids[3], b)) return false;
				if (point_on_boundary(polygon_, a) || point_on_boundary(polygon_, b)) return false;
				for (const auto &[u, w] : {std::pair {p, a}, std::pair {q, a}, std::pair {r, b}, std::pair {s, b}, std::pair {a, b}}) {
					if (!open_segment_clear(polygon_, u, w)) return false;
				}
				for (const Vec2 &u : {p, q}) {
					for (const Vec2 &w : {r, s}) {
						if (exact::closed_segments_meet(u, a, w, b)) return false;
					}
				}
				return true;
			}

			bool resolves_alone(int vertex, const Vec2 &target) const {
				const Vec2 &v = polygon_.at(vertex);
				return gap_at_most_half_turn(v, polygon_.next(vertex), target)
					&& gap_at_most_half_turn(v, target, polygon_.prev(vertex));
			}

		private:
			struct Directions {
				LongPoint extension[2];
				LongPoint bisector;
			};

			struct PairInfo {
				bool known = false;
				bool possible = false;
				LongPoint corners[4];
				LongPoint center;
				bool has_bisector = false;
				LongPoint bisector;
			};

			const PolygonData &polygon_;
			std::vector<Directions> directions_;
			std::vector<PairInfo> pairs_ = std::vector<PairInfo>(static_cast<std::size_t>(polygon_.c) * static_cast<std::size_t>(polygon_.c));

			const PairInfo &pair(int vp, int vq) {
				const std::size_t c = static_cast<std::size_t>(polygon_.c);
				PairInfo &info = pairs_[static_cast<std::size_t>(polygon_.vertex_notch[static_cast<std::size_t>(vp)]) * c
					+ static_cast<std::size_t>(polygon_.vertex_notch[static_cast<std::size_t>(vq)])];
				if (info.known) return info;
				info.known = true;
				const Directions &dp = directions_[static_cast<std::size_t>(vp)];
				const Directions &dq = directions_[static_cast<std::size_t>(vq)];
				for (int x = 0; x < 2; ++x) {
					for (int y = 0; y < 2; ++y) {
						const std::optional<LongPoint> meet = ray_meet(polygon_.at(vp), dp.extension[x], polygon_.at(vq), dq.extension[y]);
						if (!meet) return info;
						info.corners[2 * x + y] = *meet;
					}
				}
				info.center = average(std::vector<LongPoint>(info.corners, info.corners + 4));
				if (const std::optional<LongPoint> meet = ray_meet(polygon_.at(vp), dp.bisector, polygon_.at(vq), dq.bisector)) {
					info.has_bisector = true;
					info.bisector = *meet;
				}
				const Vec2 &p = polygon_.at(vp);
				const Vec2 &q = polygon_.at(vq);
				auto usable = [&](const LongPoint &point) {
					const Vec2 a = to_double(point);
					if (!std::isfinite(a.x) || !std::isfinite(a.y) || a == p || a == q) return false;
					return resolves_alone(vp, a) && resolves_alone(vq, a) && !point_on_boundary(polygon_, a)
						&& open_segment_clear(polygon_, p, a) && open_segment_clear(polygon_, q, a);
				};
				info.possible = usable(info.center);
				for (int t = 0; t < 4 && !info.possible; ++t) info.possible = usable(info.corners[t]);
				return info;
			}

			static LongPoint normalized(const LongPoint &d) {
				const long double length = std::hypot(d.x, d.y);
				if (length == 0.0L) return d;
				return {d.x / length, d.y / length};
			}

			static LongPoint average(const std::vector<LongPoint> &points) {
				LongPoint result {0.0L, 0.0L};
				for (const LongPoint &p : points) {
					result.x += p.x;
					result.y += p.y;
				}
				result.x /= static_cast<long double>(points.size());
				result.y /= static_cast<long double>(points.size());
				return result;
			}

			static std::optional<LongPoint> ray_meet(const Vec2 &p0, const LongPoint &dp, const Vec2 &q0, const LongPoint &dq) {
				const long double det = dp.x * dq.y - dp.y * dq.x;
				if (std::fabs(det) < 1.0e-18L) return std::nullopt;
				const long double rx = static_cast<long double>(q0.x) - p0.x;
				const long double ry = static_cast<long double>(q0.y) - p0.y;
				const long double t = (rx * dq.y - ry * dq.x) / det;
				const long double u = (rx * dp.y - ry * dp.x) / det;
				if (!(t > 0.0L) || !(u > 0.0L)) return std::nullopt;
				return LongPoint {p0.x + t * dp.x, p0.y + t * dp.y};
			}
		};

		// ---------------------------------------------------------------- tables

		struct Half {
			enum class Kind : std::uint8_t {
				none,
				extension,        // B(i, k) or F(k, j) with a wider gap
				diagonal_forward,  // diagonal to a notch that carries a forward subtree
				diagonal_backward, // diagonal to a notch that carries a backward subtree
				diagonal_plain,   // diagonal to a notch with a single arm
				junction,         // arm to a Steiner point
			};
			Kind kind = Kind::none;
			std::uint8_t flags = 0;
			int k = -1;
			Vec2 arm;
			Vec2 steiner;

			bool exists() const {
				return kind != Kind::none;
			}
		};

		struct Choice {
			enum class Kind : std::uint8_t { none, split, x4, patch, diagonal, junction_forward, junction_backward };
			Kind kind = Kind::none;
			std::uint8_t flags = 0;
			int k = -1;
			int a = -1;
			int b = -1;
			Vec2 p1;
			Vec2 p2;
		};

		class DynamicProgram {
		public:
			DynamicProgram(const PolygonData &polygon, const Options &options)
				: polygon_(polygon), options_(options), c_(polygon.c), junctions_(polygon), x4_(polygon) {}

			DpOutput run() {
				DpOutput output;
				if (c_ == 0) return output;
				const std::size_t cs = static_cast<std::size_t>(c_);
				s_.assign(cs, std::vector<int>(cs + 1, 0));
				choice_.assign(cs, std::vector<Choice>(cs + 1));
				forward_.assign(cs, std::vector<Half>(cs + 1));
				backward_.assign(cs, std::vector<Half>(cs + 1));
				compute_visibility();

				for (int len = 2; len <= c_; ++len) {
					for (int i = 0; i < c_; ++i) {
						const int j = i + len - 1;
						junctions_.begin_interval();
						compute_s(i, j);
						if (len < c_) {
							compute_forward(i, j);
							compute_backward(i, j);
						}
					}
				}

				int best_start = 0;
				output.rotation_min = output.rotation_max = s_[0][cs];
				for (int i = 0; i < c_; ++i) {
					const int value = s_[static_cast<std::size_t>(i)][cs];
					output.rotation_min = std::min(output.rotation_min, value);
					if (value > output.rotation_max) {
						output.rotation_max = value;
						best_start = i;
					}
				}
				if (c_ >= 2) reconstruct_s(best_start, best_start + c_ - 1, output.patterns);
				output.junction_calls = junctions_.calls;
				output.junction_successes = junctions_.successes;
				output.x4_calls = x4_.calls;
				output.x4_successes = x4_.successes;
				for (PatternGeometry &pattern : output.patterns) finish_pattern(pattern);
				return output;
			}

		private:
			const PolygonData &polygon_;
			Options options_;
			int c_;
			JunctionSolver junctions_;
			X4Solver x4_;
			std::vector<std::vector<int>> s_;
			std::vector<std::vector<Choice>> choice_;
			std::vector<std::vector<Half>> forward_;
			std::vector<std::vector<Half>> backward_;
			std::vector<std::vector<char>> visible_;
			std::unordered_map<std::uint64_t, X4Points> x4_cache_;

			int wrap(int t) const {
				const int r = t % c_;
				return r < 0 ? r + c_ : r;
			}

			int sv(int i, int j) const {
				if (j < i) return 0;
				return s_[static_cast<std::size_t>(wrap(i))][static_cast<std::size_t>(j - i + 1)];
			}

			const Half &fwd(int i, int j) const {
				return forward_[static_cast<std::size_t>(wrap(i))][static_cast<std::size_t>(j - i + 1)];
			}

			const Half &bwd(int i, int j) const {
				return backward_[static_cast<std::size_t>(wrap(j))][static_cast<std::size_t>(j - i + 1)];
			}

			int vtx(int t) const {
				return polygon_.notch_vertex[static_cast<std::size_t>(wrap(t))];
			}

			const Vec2 &pt(int t) const {
				return polygon_.at(vtx(t));
			}

			bool visible(int a, int b) const {
				return visible_[static_cast<std::size_t>(wrap(a))][static_cast<std::size_t>(wrap(b))] != 0;
			}

			void compute_visibility() {
				const std::size_t cs = static_cast<std::size_t>(c_);
				visible_.assign(cs, std::vector<char>(cs, 0));
				for (int a = 0; a < c_; ++a) {
					for (int b = a + 1; b < c_; ++b) {
						const int va = vtx(a);
						const int vb = vtx(b);
						const bool ok = strictly_inside_corner(polygon_, va, polygon_.at(vb))
							&& strictly_inside_corner(polygon_, vb, polygon_.at(va))
							&& open_segment_clear(polygon_, polygon_.at(va), polygon_.at(vb));
						visible_[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = ok;
						visible_[static_cast<std::size_t>(b)][static_cast<std::size_t>(a)] = ok;
					}
				}
			}

			// Angular tests at notch t for an arm toward x.
			bool gap(int t, const Vec2 &a, const Vec2 &b) const {
				return gap_at_most_half_turn(pt(t), a, b);
			}

			const Vec2 &next_of(int t) const {
				return polygon_.next(vtx(t));
			}

			const Vec2 &prev_of(int t) const {
				return polygon_.prev(vtx(t));
			}

			bool alone(int t, const Vec2 &x) const {
				return gap(t, next_of(t), x) && gap(t, x, prev_of(t));
			}

			// Order: next edge, forward arm, x, previous edge.
			bool after_forward(int t, const Vec2 &forward_arm, const Vec2 &x) const {
				return gap(t, forward_arm, x) && gap(t, x, prev_of(t));
			}

			// Order: next edge, x, backward arm, previous edge.
			bool before_backward(int t, const Vec2 &x, const Vec2 &backward_arm) const {
				return gap(t, next_of(t), x) && gap(t, x, backward_arm);
			}

			bool face_forward(int t, const Vec2 &x) const {
				return gap(t, next_of(t), x);
			}

			bool face_backward(int t, const Vec2 &x) const {
				return gap(t, x, prev_of(t));
			}

			static RoleSpec role_alone() {
				return {Role::alone, {}};
			}

			// ---------------------------------------------------------- S(i, j)

			void compute_s(int i, int j) {
				const int len = j - i + 1;
				int best = 0;
				Choice choice;
				auto offer = [&](int value, const Choice &candidate) {
					if (value > best) {
						best = value;
						choice = candidate;
					}
				};

				for (int k = i; k < j; ++k) {
					Choice split;
					split.kind = Choice::Kind::split;
					split.k = k;
					offer(sv(i, k) + sv(k + 1, j), split);
				}

				// A notch k with two arms on the path from i to j.
				for (int k = i + 1; k < j; ++k) {
					const Half &left = bwd(i, k);
					const Half &right = fwd(k, j);
					if (!left.exists() || !right.exists()) continue;
					const int value = 1 + sv(i, k) + sv(k, j);
					if (value <= best) continue;
					if (gap(k, right.arm, left.arm)) {
						Choice patch;
						patch.kind = Choice::Kind::patch;
						patch.k = k;
						offer(value, patch);
					}
				}

				// The diagonal i-j, with an optional forward subtree at i and backward
				// subtree at j.
				if (visible(i, j)) {
					const Vec2 &pi = pt(i);
					const Vec2 &pj = pt(j);
					auto diagonal = [&](std::uint8_t variant, int k) {
						Choice d;
						d.kind = Choice::Kind::diagonal;
						d.flags = variant;
						d.k = k;
						return d;
					};
					{
						const int value = 1 + sv(i + 1, j - 1);
						if (value > best && alone(i, pj) && alone(j, pi)) offer(value, diagonal(0, -1));
					}
					if (len >= 3 && fwd(i, j - 1).exists()) {
						const int value = 1 + sv(i, j - 1);
						if (value > best && after_forward(i, fwd(i, j - 1).arm, pj) && alone(j, pi)) offer(value, diagonal(1, -1));
					}
					if (len >= 3 && bwd(i + 1, j).exists()) {
						const int value = 1 + sv(i + 1, j);
						if (value > best && alone(i, pj) && before_backward(j, pi, bwd(i + 1, j).arm)) offer(value, diagonal(2, -1));
					}
					for (int k = i + 2; k <= j - 1; ++k) {
						const Half &x = fwd(i, k - 1);
						const Half &y = bwd(k, j);
						if (!x.exists() || !y.exists()) continue;
						const int value = 1 + sv(i, k - 1) + sv(k, j);
						if (value <= best) continue;
						if (after_forward(i, x.arm, pj) && before_backward(j, pi, y.arm)) offer(value, diagonal(3, k));
					}
				}

				if (options_.steiner_points) {
					// A Steiner point on the path from i to j, joined to a third notch k.
					for (int k = i + 1; k < j; ++k) {
						// Forward subtrees at i and k, j alone.
						try_junction_combos(i, k, j, best, choice, true);
						// Backward subtrees at k and j, i alone.
						try_junction_combos(i, k, j, best, choice, false);
					}
				}

				// The gaps of an X4-pattern lie in (i..j) without its two inner notches.
				if (options_.x4_patterns && len >= 4 && 1 + sv(i + 1, j - 1) > best) {
					for (int a = i + 1; a < j; ++a) {
						if (1 + sv(i + 1, a - 1) + sv(a + 1, j - 1) <= best) continue;
						for (int b = a + 1; b < j; ++b) {
							const int value = 1 + sv(i + 1, a - 1) + sv(a + 1, b - 1) + sv(b + 1, j - 1);
							if (value <= best) continue;
							for (int pairing = 0; pairing < 2; ++pairing) {
								// pairing 0: {i, a} and {b, j}; pairing 1: {a, b} and {j, i}.
								const int order[4] = {
									pairing == 0 ? i : a,
									pairing == 0 ? a : b,
									pairing == 0 ? b : j,
									pairing == 0 ? j : i,
								};
								if (!x4_.pair_possible(vtx(order[0]), vtx(order[1])) || !x4_.pair_possible(vtx(order[2]), vtx(order[3]))) continue;
								const X4Points &points = x4_lookup(order);
								if (!points.ok) continue;
								Choice x4;
								x4.kind = Choice::Kind::x4;
								x4.flags = static_cast<std::uint8_t>(pairing);
								x4.a = a;
								x4.b = b;
								x4.p1 = points.a;
								x4.p2 = points.b;
								offer(value, x4);
								break;
							}
						}
					}
				}

				s_[static_cast<std::size_t>(wrap(i))][static_cast<std::size_t>(len)] = best;
				choice_[static_cast<std::size_t>(wrap(i))][static_cast<std::size_t>(len)] = choice;
			}

			// Junction joining i, k, j inside S(i, j). With `forward_side`, i and k may
			// carry forward subtrees B(i, k-1) and B(k, j-1) and j is alone; otherwise i
			// is alone and k, j may carry backward subtrees F(i+1, k) and F(k+1, j).
			//
			// These are the only shapes left by condition 2 of Y-patterns: on a path
			// N2-N3-N2 the other arms of the two notches point to opposite sides. The
			// path i-s-j has every subtree inside [i..j] on its k side, so i and j are
			// not both of degree 2. The path i-s-k separates (i..k) from (k..j], so a
			// forward subtree at i (in (i..k)) needs k's subtree forward (in (k..j));
			// symmetrically on the path k-s-j a backward subtree at j needs k's
			// subtree backward.
			void try_junction_combos(int i, int k, int j, int &best, Choice &choice, bool forward_side) {
				struct Combo {
					int value;
					std::uint8_t flags;
				};
				std::array<Combo, 4> combos {};
				int count = 0;
				for (std::uint8_t flags = 0; flags < 4; ++flags) {
					const bool first = (flags & 1) != 0;
					const bool second = (flags & 2) != 0;
					int value = 1;
					if (forward_side) {
						if (first && !(k - 1 > i && fwd(i, k - 1).exists())) continue;
						if (second && !(j - 1 > k && fwd(k, j - 1).exists())) continue;
						value += first ? sv(i, k - 1) : sv(i + 1, k - 1);
						value += second ? sv(k, j - 1) : sv(k + 1, j - 1);
					} else {
						if (first && !(k > i + 1 && bwd(i + 1, k).exists())) continue;
						if (second && !(j > k + 1 && bwd(k + 1, j).exists())) continue;
						value += first ? sv(i + 1, k) : sv(i + 1, k - 1);
						value += second ? sv(k + 1, j) : sv(k + 1, j - 1);
					}
					combos[static_cast<std::size_t>(count++)] = {value, flags};
				}
				for (int a = 1; a < count; ++a) {
					for (int b = a; b > 0 && combos[static_cast<std::size_t>(b)].value > combos[static_cast<std::size_t>(b - 1)].value; --b) {
						std::swap(combos[static_cast<std::size_t>(b)], combos[static_cast<std::size_t>(b - 1)]);
					}
				}
				for (int t = 0; t < count; ++t) {
					const Combo &combo = combos[static_cast<std::size_t>(t)];
					if (combo.value <= best) break;
					const bool first = (combo.flags & 1) != 0;
					const bool second = (combo.flags & 2) != 0;
					std::array<RoleSpec, 3> roles {role_alone(), role_alone(), role_alone()};
					if (forward_side) {
						if (first) roles[0] = {Role::with_forward_arm, fwd(i, k - 1).arm};
						if (second) roles[1] = {Role::with_forward_arm, fwd(k, j - 1).arm};
					} else {
						if (first) roles[1] = {Role::with_backward_arm, bwd(i + 1, k).arm};
						if (second) roles[2] = {Role::with_backward_arm, bwd(k + 1, j).arm};
					}
					std::optional<Vec2> s = junctions_.solve(vtx(i), vtx(k), vtx(j), roles, Objective::none);
					if (!s) continue;
					best = combo.value;
					choice = {};
					choice.kind = forward_side ? Choice::Kind::junction_forward : Choice::Kind::junction_backward;
					choice.flags = combo.flags;
					choice.k = k;
					choice.p1 = *s;
					break;
				}
			}

			const X4Points &x4_lookup(const int (&order)[4]) {
				int keys[4];
				for (int t = 0; t < 4; ++t) keys[t] = wrap(order[t]);
				// Canonical form: rotate so that the smallest notch comes first; the
				// pairing is then either (0,1)(2,3) or (1,2)(3,0).
				int start = 0;
				for (int t = 1; t < 4; ++t) {
					if (keys[t] < keys[start]) start = t;
				}
				const bool shifted = start % 2 == 1;
				int canonical[4];
				for (int t = 0; t < 4; ++t) canonical[t] = keys[(start + t) % 4];
				std::uint64_t key = shifted ? 1 : 0;
				for (int t = 0; t < 4; ++t) key = key * 1000003ULL + static_cast<std::uint64_t>(canonical[t]);
				auto it = x4_cache_.find(key);
				if (it != x4_cache_.end()) return it->second;
				const int first = shifted ? canonical[1] : canonical[0];
				const int second = shifted ? canonical[2] : canonical[1];
				const int third = shifted ? canonical[3] : canonical[2];
				const int fourth = shifted ? canonical[0] : canonical[3];
				X4Points points = x4_.solve(
					polygon_.notch_vertex[static_cast<std::size_t>(first)],
					polygon_.notch_vertex[static_cast<std::size_t>(second)],
					polygon_.notch_vertex[static_cast<std::size_t>(third)],
					polygon_.notch_vertex[static_cast<std::size_t>(fourth)]
				);
				// Stored for the order (first, second | third, fourth); a lookup with the
				// pairs swapped reads the same points under the other names.
				return x4_cache_.emplace(key, points).first->second;
			}

			// ---------------------------------------------------------- B(i, j)

			void compute_forward(int i, int j) {
				const int target = sv(i, j);
				const Vec2 &pi = pt(i);
				const Vec2 &pj = pt(j);
				Half best;
				auto offer = [&](const Half &candidate) {
					if (!best.exists() || orient(pi, best.arm, candidate.arm) > 0) best = candidate;
				};

				for (int k = i + 1; k < j; ++k) {
					const Half &inner = fwd(i, k);
					if (inner.exists() && sv(i, k) + sv(k + 1, j) == target) {
						Half h = inner;
						h.kind = Half::Kind::extension;
						h.k = k;
						h.flags = 0;
						offer(h);
					}
				}
				for (int k = i + 1; k < j; ++k) {
					const Half &sub = fwd(k, j);
					if (!sub.exists() || !visible(i, k)) continue;
					if (sv(i + 1, k - 1) + sv(k, j) != target) continue;
					const Vec2 &pk = pt(k);
					if (face_forward(i, pk) && after_forward(k, sub.arm, pi)) {
						Half h;
						h.kind = Half::Kind::diagonal_forward;
						h.k = k;
						h.arm = pk;
						offer(h);
					}
				}
				if (visible(i, j)) {
					if (j >= i + 2 && bwd(i + 1, j).exists() && sv(i + 1, j) == target
						&& face_forward(i, pj) && before_backward(j, pi, bwd(i + 1, j).arm)) {
						Half h;
						h.kind = Half::Kind::diagonal_backward;
						h.k = j;
						h.arm = pj;
						offer(h);
					}
					if (sv(i + 1, j - 1) == target && face_forward(i, pj) && alone(j, pi)) {
						Half h;
						h.kind = Half::Kind::diagonal_plain;
						h.k = j;
						h.arm = pj;
						offer(h);
					}
				}
				// A Steiner point s joined to k and j. Root i is of degree 2 with its
				// other arm outside [i..j], beyond the path i-s-k and the path i-s-j, so
				// condition 2 leaves k and j only backward subtrees, inside (i..k) and
				// (k..j).
				if (options_.steiner_points) {
					for (int k = i + 1; k < j; ++k) {
						// No junction arm can beat the direction toward j.
						if (best.exists() && !(orient(pi, best.arm, pj) > 0)) break;
						for (std::uint8_t flags = 0; flags < 4; ++flags) {
							const bool first = (flags & 1) != 0;
							const bool second = (flags & 2) != 0;
							if (first && !(k > i + 1 && bwd(i + 1, k).exists())) continue;
							if (second && !(j > k + 1 && bwd(k + 1, j).exists())) continue;
							const int value = (first ? sv(i + 1, k) : sv(i + 1, k - 1)) + (second ? sv(k + 1, j) : sv(k + 1, j - 1));
							if (value != target) continue;
							std::array<RoleSpec, 3> roles {RoleSpec {Role::root_forward, {}}, role_alone(), role_alone()};
							if (first) roles[1] = {Role::with_backward_arm, bwd(i + 1, k).arm};
							if (second) roles[2] = {Role::with_backward_arm, bwd(k + 1, j).arm};
							std::optional<Vec2> s = junctions_.solve(vtx(i), vtx(k), vtx(j), roles, Objective::first_forward);
							if (!s) continue;
							Half h;
							h.kind = Half::Kind::junction;
							h.k = k;
							h.flags = flags;
							h.arm = *s;
							h.steiner = *s;
							offer(h);
						}
					}
				}
				forward_[static_cast<std::size_t>(wrap(i))][static_cast<std::size_t>(j - i + 1)] = best;
			}

			// ---------------------------------------------------------- F(i, j)

			void compute_backward(int i, int j) {
				const int target = sv(i, j);
				const Vec2 &pi = pt(i);
				const Vec2 &pj = pt(j);
				Half best;
				auto offer = [&](const Half &candidate) {
					if (!best.exists() || orient(pj, candidate.arm, best.arm) > 0) best = candidate;
				};

				for (int k = i + 1; k < j; ++k) {
					const Half &inner = bwd(k, j);
					if (inner.exists() && sv(i, k - 1) + sv(k, j) == target) {
						Half h = inner;
						h.kind = Half::Kind::extension;
						h.k = k;
						h.flags = 0;
						offer(h);
					}
				}
				for (int k = i + 1; k < j; ++k) {
					const Half &sub = bwd(i, k);
					if (!sub.exists() || !visible(j, k)) continue;
					if (sv(i, k) + sv(k + 1, j - 1) != target) continue;
					const Vec2 &pk = pt(k);
					if (face_backward(j, pk) && before_backward(k, pj, sub.arm)) {
						Half h;
						h.kind = Half::Kind::diagonal_backward;
						h.k = k;
						h.arm = pk;
						offer(h);
					}
				}
				if (visible(i, j)) {
					if (j - 1 > i && fwd(i, j - 1).exists() && sv(i, j - 1) == target
						&& face_backward(j, pi) && after_forward(i, fwd(i, j - 1).arm, pj)) {
						Half h;
						h.kind = Half::Kind::diagonal_forward;
						h.k = i;
						h.arm = pi;
						offer(h);
					}
					if (sv(i + 1, j - 1) == target && face_backward(j, pi) && alone(i, pj)) {
						Half h;
						h.kind = Half::Kind::diagonal_plain;
						h.k = i;
						h.arm = pi;
						offer(h);
					}
				}
				if (options_.steiner_points) {
					for (int k = j - 1; k > i; --k) {
						if (best.exists() && !(orient(pj, pi, best.arm) > 0)) break;
						for (std::uint8_t flags = 0; flags < 4; ++flags) {
							const bool first = (flags & 1) != 0;
							const bool second = (flags & 2) != 0;
							if (first && !(k - 1 > i && fwd(i, k - 1).exists())) continue;
							if (second && !(j - 1 > k && fwd(k, j - 1).exists())) continue;
							const int value = (first ? sv(i, k - 1) : sv(i + 1, k - 1)) + (second ? sv(k, j - 1) : sv(k + 1, j - 1));
							if (value != target) continue;
							std::array<RoleSpec, 3> roles {role_alone(), role_alone(), RoleSpec {Role::root_backward, {}}};
							if (first) roles[0] = {Role::with_forward_arm, fwd(i, k - 1).arm};
							if (second) roles[1] = {Role::with_forward_arm, fwd(k, j - 1).arm};
							std::optional<Vec2> s = junctions_.solve(vtx(i), vtx(k), vtx(j), roles, Objective::last_backward);
							if (!s) continue;
							Half h;
							h.kind = Half::Kind::junction;
							h.k = k;
							h.flags = flags;
							h.arm = *s;
							h.steiner = *s;
							offer(h);
						}
					}
				}
				backward_[static_cast<std::size_t>(wrap(j))][static_cast<std::size_t>(j - i + 1)] = best;
			}

			// ---------------------------------------------------------- reconstruction

			NodeRef notch_node(int t) const {
				NodeRef node;
				node.vertex = vtx(t);
				node.point = pt(t);
				return node;
			}

			static NodeRef steiner_node(const Vec2 &p) {
				NodeRef node;
				node.vertex = -1;
				node.point = p;
				return node;
			}

			static void add_edge(PatternGeometry &pattern, const NodeRef &a, const NodeRef &b) {
				pattern.edges.push_back({a, b});
			}

			void reconstruct_s(int i, int j, std::vector<PatternGeometry> &out) const {
				if (j - i + 1 < 2) return;
				const Choice &choice = choice_[static_cast<std::size_t>(wrap(i))][static_cast<std::size_t>(j - i + 1)];
				switch (choice.kind) {
					case Choice::Kind::none:
						return;
					case Choice::Kind::split:
						reconstruct_s(i, choice.k, out);
						reconstruct_s(choice.k + 1, j, out);
						return;
					case Choice::Kind::x4: {
						PatternGeometry pattern;
						pattern.kind = PatternKind::x4;
						const int a = choice.a;
						const int b = choice.b;
						const bool first_pairing = choice.flags == 0;
						const int order[4] = {
							first_pairing ? i : a,
							first_pairing ? a : b,
							first_pairing ? b : j,
							first_pairing ? j : i,
						};
						// The cached points follow the canonical order; match them by
						// checking which Steiner point serves which pair.
						const auto [pa, pb] = x4_points_for(order, choice.p1, choice.p2);
						const NodeRef sa = steiner_node(pa);
						const NodeRef sb = steiner_node(pb);
						add_edge(pattern, notch_node(order[0]), sa);
						add_edge(pattern, notch_node(order[1]), sa);
						add_edge(pattern, sa, sb);
						add_edge(pattern, notch_node(order[2]), sb);
						add_edge(pattern, notch_node(order[3]), sb);
						out.push_back(std::move(pattern));
						reconstruct_s(i + 1, a - 1, out);
						reconstruct_s(a + 1, b - 1, out);
						reconstruct_s(b + 1, j - 1, out);
						return;
					}
					case Choice::Kind::patch: {
						PatternGeometry pattern;
						reconstruct_backward(i, choice.k, pattern, out);
						reconstruct_forward(choice.k, j, pattern, out);
						out.push_back(std::move(pattern));
						return;
					}
					case Choice::Kind::diagonal: {
						PatternGeometry pattern;
						add_edge(pattern, notch_node(i), notch_node(j));
						switch (choice.flags) {
							case 0:
								reconstruct_s(i + 1, j - 1, out);
								break;
							case 1:
								reconstruct_forward(i, j - 1, pattern, out);
								break;
							case 2:
								reconstruct_backward(i + 1, j, pattern, out);
								break;
							default:
								reconstruct_forward(i, choice.k - 1, pattern, out);
								reconstruct_backward(choice.k, j, pattern, out);
								break;
						}
						out.push_back(std::move(pattern));
						return;
					}
					case Choice::Kind::junction_forward:
					case Choice::Kind::junction_backward: {
						PatternGeometry pattern;
						const int k = choice.k;
						const NodeRef s = steiner_node(choice.p1);
						add_edge(pattern, notch_node(i), s);
						add_edge(pattern, notch_node(k), s);
						add_edge(pattern, notch_node(j), s);
						const bool first = (choice.flags & 1) != 0;
						const bool second = (choice.flags & 2) != 0;
						if (choice.kind == Choice::Kind::junction_forward) {
							if (first) reconstruct_forward(i, k - 1, pattern, out);
							else reconstruct_s(i + 1, k - 1, out);
							if (second) reconstruct_forward(k, j - 1, pattern, out);
							else reconstruct_s(k + 1, j - 1, out);
						} else {
							if (first) reconstruct_backward(i + 1, k, pattern, out);
							else reconstruct_s(i + 1, k - 1, out);
							if (second) reconstruct_backward(k + 1, j, pattern, out);
							else reconstruct_s(k + 1, j - 1, out);
						}
						out.push_back(std::move(pattern));
						return;
					}
				}
			}

			std::pair<Vec2, Vec2> x4_points_for(const int (&order)[4], const Vec2 &p1, const Vec2 &p2) const {
				// p1 and p2 are the cached points (first pair, second pair) of the
				// canonical order; the first point here must serve order[0], order[1].
				const X4Solver &solver = x4_;
				const int ids[4] = {vtx(order[0]), vtx(order[1]), vtx(order[2]), vtx(order[3])};
				if (solver.valid(ids, p1, p2)) return {p1, p2};
				return {p2, p1};
			}

			void reconstruct_forward(int i, int j, PatternGeometry &pattern, std::vector<PatternGeometry> &out) const {
				const Half &h = fwd(i, j);
				switch (h.kind) {
					case Half::Kind::none:
						return;
					case Half::Kind::extension:
						reconstruct_forward(i, h.k, pattern, out);
						reconstruct_s(h.k + 1, j, out);
						return;
					case Half::Kind::diagonal_forward:
						add_edge(pattern, notch_node(i), notch_node(h.k));
						reconstruct_s(i + 1, h.k - 1, out);
						reconstruct_forward(h.k, j, pattern, out);
						return;
					case Half::Kind::diagonal_backward:
						add_edge(pattern, notch_node(i), notch_node(j));
						reconstruct_backward(i + 1, j, pattern, out);
						return;
					case Half::Kind::diagonal_plain:
						add_edge(pattern, notch_node(i), notch_node(j));
						reconstruct_s(i + 1, j - 1, out);
						return;
					case Half::Kind::junction: {
						const int k = h.k;
						const NodeRef s = steiner_node(h.steiner);
						add_edge(pattern, notch_node(i), s);
						add_edge(pattern, notch_node(k), s);
						add_edge(pattern, notch_node(j), s);
						if (h.flags & 1) reconstruct_backward(i + 1, k, pattern, out);
						else reconstruct_s(i + 1, k - 1, out);
						if (h.flags & 2) reconstruct_backward(k + 1, j, pattern, out);
						else reconstruct_s(k + 1, j - 1, out);
						return;
					}
				}
			}

			void reconstruct_backward(int i, int j, PatternGeometry &pattern, std::vector<PatternGeometry> &out) const {
				const Half &h = bwd(i, j);
				switch (h.kind) {
					case Half::Kind::none:
						return;
					case Half::Kind::extension:
						reconstruct_s(i, h.k - 1, out);
						reconstruct_backward(h.k, j, pattern, out);
						return;
					case Half::Kind::diagonal_backward:
						add_edge(pattern, notch_node(j), notch_node(h.k));
						reconstruct_backward(i, h.k, pattern, out);
						reconstruct_s(h.k + 1, j - 1, out);
						return;
					case Half::Kind::diagonal_forward:
						add_edge(pattern, notch_node(j), notch_node(i));
						reconstruct_forward(i, j - 1, pattern, out);
						return;
					case Half::Kind::diagonal_plain:
						add_edge(pattern, notch_node(j), notch_node(i));
						reconstruct_s(i + 1, j - 1, out);
						return;
					case Half::Kind::junction: {
						const int k = h.k;
						const NodeRef s = steiner_node(h.steiner);
						add_edge(pattern, notch_node(i), s);
						add_edge(pattern, notch_node(k), s);
						add_edge(pattern, notch_node(j), s);
						if (h.flags & 1) reconstruct_forward(i, k - 1, pattern, out);
						else reconstruct_s(i + 1, k - 1, out);
						if (h.flags & 2) reconstruct_forward(k, j - 1, pattern, out);
						else reconstruct_s(k + 1, j - 1, out);
						return;
					}
				}
			}

			static void finish_pattern(PatternGeometry &pattern) {
				std::vector<int> notches;
				std::vector<Vec2> steiner;
				for (const PatternEdge &edge : pattern.edges) {
					for (const NodeRef *node : {&edge.a, &edge.b}) {
						if (node->vertex >= 0) {
							if (std::find(notches.begin(), notches.end(), node->vertex) == notches.end()) notches.push_back(node->vertex);
						} else if (std::find(steiner.begin(), steiner.end(), node->point) == steiner.end()) {
							steiner.push_back(node->point);
						}
					}
				}
				std::sort(notches.begin(), notches.end());
				pattern.notch_vertices = std::move(notches);
				pattern.steiner_points = steiner.size();
				if (pattern.kind != PatternKind::x4) {
					if (!steiner.empty()) pattern.kind = PatternKind::steiner;
					else if (pattern.edges.size() == 1) pattern.kind = PatternKind::x2;
					else pattern.kind = PatternKind::diagonals;
				}
			}
		};
	}

	DpOutput run_dynamic_program(const PolygonData &polygon, const Options &options) {
		DynamicProgram program(polygon, options);
		return program.run();
	}
}
