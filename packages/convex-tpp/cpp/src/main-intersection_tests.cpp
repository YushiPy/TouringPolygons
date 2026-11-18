#include "tests.h"
#include "tpp/convex/solution.h"
#include "tpp_convex.h"
#include "tpp/convex/detail/convex_contacts.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <print>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using std::vector;

namespace {

	// The binary64 test that convex_contacts.h replaces, with every pair of edges.
	bool reference_intersect_or_touch(const vector<Vector2> &a, const vector<Vector2> &b) {
		constexpr double epsilon = 1e-8;
		auto bounds = [](const vector<Vector2> &p) {
			double x0 = p.front().x, x1 = p.front().x, y0 = p.front().y, y1 = p.front().y;
			for (const auto &q : p) {
				x0 = std::min(x0, q.x); x1 = std::max(x1, q.x);
				y0 = std::min(y0, q.y); y1 = std::max(y1, q.y);
			}
			return std::array<double, 4>{x0, x1, y0, y1};
		};
		const auto ab = bounds(a), bb = bounds(b);
		if (ab[1] < bb[0] - epsilon || bb[1] < ab[0] - epsilon || ab[3] < bb[2] - epsilon || bb[3] < ab[2] - epsilon) {
			return false;
		}
		for (size_t i = 0; i < a.size(); i++) {
			for (size_t j = 0; j < b.size(); j++) {
				if (tpp::segment_segment_intersection_safe(a[i], a[(i + 1) % a.size()], b[j], b[(j + 1) % b.size()]).is_finite()) {
					return true;
				}
			}
		}
		return tpp::detail::point_in_convex_polygon_closed_linear(a.front(), b) ||
			tpp::detail::point_in_convex_polygon_closed_linear(b.front(), a);
	}

	// A convex polygon with integer vertices, so that every cross product below is
	// exact in binary64: the hull of random grid points, sometimes with midpoints
	// of edges kept as collinear vertices, sometimes clockwise, sometimes an
	// axis-aligned rectangle (whose sides touch and overlap often).
	vector<Vector2> random_convex_polygon(std::mt19937_64 &random) {
		std::uniform_int_distribution<int> coordinate(-12, 12), size(1, 8), kind(0, 3);
		vector<Vector2> polygon;
		if (kind(random) == 0) {
			const double x = coordinate(random), y = coordinate(random);
			const double w = 2 * size(random), h = 2 * size(random);
			polygon = {{x, y}, {x + w, y}, {x + w, y + h}, {x, y + h}};
		} else {
			vector<Vector2> points;
			const int count = 3 + size(random);
			for (int i = 0; i < count; i++) {
				points.emplace_back(2 * coordinate(random), 2 * coordinate(random));
			}
			std::sort(points.begin(), points.end(), [](const Vector2 &p, const Vector2 &q) {
				return p.x < q.x || (p.x == q.x && p.y < q.y);
			});
			points.erase(std::unique(points.begin(), points.end(), [](const Vector2 &p, const Vector2 &q) {
				return p.x == q.x && p.y == q.y;
			}), points.end());
			// Andrew's monotone chain, counterclockwise, without collinear points.
			vector<Vector2> hull;
			for (int pass = 0; pass < 2; pass++) {
				const size_t base = hull.size();
				for (const auto &p : points) {
					while (hull.size() >= base + 2 && (hull[hull.size() - 1] - hull[hull.size() - 2]).cross(p - hull[hull.size() - 2]) <= 0) {
						hull.pop_back();
					}
					hull.push_back(p);
				}
				hull.pop_back();
				std::reverse(points.begin(), points.end());
			}
			if (hull.size() < 3) {
				return random_convex_polygon(random);
			}
			polygon = hull;
		}
		if (kind(random) == 0) {
			vector<Vector2> refined;
			for (size_t i = 0; i < polygon.size(); i++) {
				refined.push_back(polygon[i]);
				refined.push_back((polygon[i] + polygon[(i + 1) % polygon.size()]) / 2);
			}
			polygon = refined;
		}
		if (kind(random) == 0) {
			std::reverse(polygon.begin(), polygon.end());
		}
		if (kind(random) == 0) {
			std::rotate(polygon.begin(), polygon.begin() + 1, polygon.end());
		}
		return polygon;
	}

	void test_convex_contacts() {
		std::mt19937_64 random(20261117);
		size_t meeting = 0, touching_only = 0, disjoint = 0;
		for (int trial = 0; trial < 20000; trial++) {
			const auto a = random_convex_polygon(random), b = random_convex_polygon(random);
			const auto ca = tpp::detail::lexicographic_chains(a), cb = tpp::detail::lexicographic_chains(b);
			if (!ca.valid || !cb.valid) {
				throw std::runtime_error("convex contacts: a convex polygon must split into two chains");
			}
			std::set<std::pair<size_t, size_t>> reported;
			size_t calls = 0;
			tpp::detail::for_each_edge_pair_that_may_touch(a, ca, b, cb, [&](size_t j, size_t k) {
				++calls;
				if (!reported.insert({j, k}).second) {
					throw std::runtime_error("convex contacts: a pair of edges was reported twice");
				}
				return false;
			});
			if (calls > 4 * (a.size() + b.size()) + 4) {
				throw std::runtime_error(std::format("convex contacts: {} pairs for {} and {} edges", calls, a.size(), b.size()));
			}
			bool meet = false;
			for (size_t j = 0; j < a.size(); j++) {
				for (size_t k = 0; k < b.size(); k++) {
					if (tpp::detail::closed_segments_meet(a[j], a[(j + 1) % a.size()], b[k], b[(k + 1) % b.size()])) {
						meet = true;
						if (!reported.contains({j, k})) {
							throw std::runtime_error("convex contacts: a pair of edges that meet was not reported");
						}
					}
				}
			}
			if (meet != tpp::detail::convex_boundaries_meet(a, ca, b, cb)) {
				throw std::runtime_error("convex contacts: boundaries meet differs from every pair of edges");
			}
			const bool reference = reference_intersect_or_touch(a, b);
			if (reference != tpp::detail::convex_polygons_intersect_or_touch_double(a, ca, b, cb)) {
				throw std::runtime_error("convex contacts: the binary64 test differs from the one it replaces");
			}
			if (reference != !tpp::detail::pairwise_disjoint_double({a, b})) {
				throw std::runtime_error("convex contacts: pairwise_disjoint_double differs from the test it replaces");
			}
			meeting += meet;
			touching_only += reference && !meet;
			disjoint += !reference;
			// Membership: vertices, midpoints of edges and grid points, inside and out.
			std::uniform_int_distribution<int> coordinate(-30, 30);
			vector<Vector2> points(a.begin(), a.end());
			for (size_t j = 0; j < a.size(); j++) {
				points.push_back((a[j] + a[(j + 1) % a.size()]) / 2);
			}
			for (int i = 0; i < 16; i++) {
				points.emplace_back(coordinate(random), coordinate(random));
			}
			for (const auto &point : points) {
				if (tpp::detail::point_in_convex_polygon_closed_linear(point, a) != tpp::detail::point_in_convex_polygon_closed_logarithmic(point, a)) {
					throw std::runtime_error("convex contacts: logarithmic membership differs from the linear test");
				}
			}
		}
		if (meeting == 0 || disjoint == 0 || touching_only == 0) {
			throw std::runtime_error(std::format("convex contacts: the random cases miss a configuration ({} meeting, {} nested, {} disjoint)", meeting, touching_only, disjoint));
		}

		// Crossing rectangles: no vertex of either lies in the other, but they intersect.
		const vector<Vector2> wide{{-3, -1}, {3, -1}, {3, 1}, {-3, 1}}, tall{{-1, -3}, {1, -3}, {1, 3}, {-1, 3}};
		if (tpp::detail::pairwise_disjoint_double({wide, tall})) {
			throw std::runtime_error("convex contacts: crossing rectangles must intersect");
		}
	}

	double path_length(const vector<Vector2> &path) {
		double result = 0.0;

		for (size_t i = 1; i < path.size(); i++) {
			result += path[i - 1].distance_to(path[i]);
		}

		return result;
	}

	void expect_close(double actual, double expected, const std::string &name) {
		if (std::fabs(actual - expected) > 1e-8) {
			throw std::runtime_error(
				std::format("{}: expected length {}, got {}", name, expected, actual)
			);
		}
	}

	void expect_ordered_path(
		const Vector2 &start, const Vector2 &target,
		const vector<Vector2> &path, const vector<vector<Vector2>> &polygons,
		const std::string &name
	) {
		const auto validation = tpp::validate_ordered_path(start, target, polygons, path);
		if (!validation.valid) {
			throw std::runtime_error(std::format("{}: ordered visitation fails at polygon {}",
				name, validation.visited_polygons + 1));
		}
	}

	void verify_case(
		const std::string &name,
		const Vector2 &start,
		const Vector2 &target,
		const vector<vector<Vector2>> &polygons,
		double expected_length
	) {
		const auto lazy = tpp::tpp_convex_solve_binary_search_lazy(start, target, polygons);
		const auto eager = tpp::tpp_convex_solve_binary_search_eager(start, target, polygons);
		const auto linear_lazy = tpp::tpp_convex_solve_linear_search_lazy(start, target, polygons);
		const auto linear_eager = tpp::tpp_convex_solve_linear_search_eager(start, target, polygons);
		const auto lazy_length = tpp::tpp_convex_solve_length_binary_search_lazy(start, target, polygons);
		const auto eager_length = tpp::tpp_convex_solve_length_binary_search_eager(start, target, polygons);
		const auto linear_lazy_length = tpp::tpp_convex_solve_length_linear_search_lazy(start, target, polygons);
		const auto linear_eager_length = tpp::tpp_convex_solve_length_linear_search_eager(start, target, polygons);

		if (!tpp::is_valid_solution(start, target, polygons, lazy)) {
			throw std::runtime_error(name + ": lazy solver returned an invalid path");
		}

		if (!tpp::is_valid_solution(start, target, polygons, eager)) {
			throw std::runtime_error(name + ": eager solver returned an invalid path");
		}

		if (!tpp::is_valid_solution(start, target, polygons, linear_lazy)) {
			throw std::runtime_error(name + ": linear lazy solver returned an invalid path");
		}

		if (!tpp::is_valid_solution(start, target, polygons, linear_eager)) {
			throw std::runtime_error(name + ": linear eager solver returned an invalid path");
		}

		expect_close(path_length(lazy), expected_length, name + " lazy materialized");
		expect_close(path_length(eager), expected_length, name + " eager materialized");
		expect_close(path_length(linear_lazy), expected_length, name + " linear lazy materialized");
		expect_close(path_length(linear_eager), expected_length, name + " linear eager materialized");
		expect_close(lazy_length, expected_length, name + " lazy length");
		expect_close(eager_length, expected_length, name + " eager length");
		expect_close(linear_lazy_length, expected_length, name + " linear lazy length");
		expect_close(linear_eager_length, expected_length, name + " linear eager length");
	}

	void verify_touching_case(
		const std::string &name,
		const Vector2 &start,
		const Vector2 &target,
		const vector<vector<Vector2>> &polygons,
		double expected_length,
		size_t expected_path_size = 0
	) {
		const auto lazy = tpp::tpp_convex_solve_binary_search_lazy(start, target, polygons);
		const auto eager = tpp::tpp_convex_solve_binary_search_eager(start, target, polygons);

		expect_ordered_path(start, target, lazy, polygons, name + " lazy");
		expect_ordered_path(start, target, eager, polygons, name + " eager");

		if (expected_path_size != 0 && lazy.size() != expected_path_size) {
			throw std::runtime_error(std::format("{} lazy: expected {} path points, got {}", name, expected_path_size, lazy.size()));
		}

		if (expected_path_size != 0 && eager.size() != expected_path_size) {
			throw std::runtime_error(std::format("{} eager: expected {} path points, got {}", name, expected_path_size, eager.size()));
		}

		expect_close(path_length(lazy), expected_length, name + " lazy materialized");
		expect_close(path_length(eager), expected_length, name + " eager materialized");
		expect_close(tpp::tpp_convex_solve_length_binary_search_lazy(start, target, polygons), expected_length, name + " lazy length");
		expect_close(tpp::tpp_convex_solve_length_binary_search_eager(start, target, polygons), expected_length, name + " eager length");
	}
}

int main() {
	test_convex_contacts();

	class RepeatedQuery final : public tpp::Solution {
	public:
		using tpp::Solution::Solution;
		size_t calls = 0;
		int64_t locate_point(const Vector2& point, size_t i) override {
			++calls;
			query(point, i - 1);
			return -1;
		}
	};
	const Vector2 cache_start(0, 0), cache_target(1, 1);
	const vector<vector<Vector2>> cache_polygons(10, {{2, 2}, {3, 2}, {2, 3}});
	RepeatedQuery repeated(cache_start, cache_target, cache_polygons);
	repeated.initialize_storage();
	repeated.query({4, 4}, cache_polygons.size());
	if (repeated.calls != cache_polygons.size()) {
		throw std::runtime_error("Repeated recursive queries must reuse their previous result.");
	}

	verify_case(
		"straight line visits separated convex polygons",
		Vector2(5.6599944320712705, 41.90941119153675),
		Vector2(27.21288557906459, -0.2783727728285078),
		{
			{
				Vector2(4.0033866926503325, 29.702672605790646),
				Vector2(17.283386692650346, 34.76267260579065),
				Vector2(13.983386692650333, 42.422672605790645),
				Vector2(1.0833866926503326, 36.80267260579065),
			},
			{
				Vector2(7.9916286191536745, 18.635407850779508),
				Vector2(21.181628619153678, 23.79540785077951),
				Vector2(18.251628619153678, 31.54540785077951),
				Vector2(5.071628619153675, 26.02540785077951),
			},
			{
				Vector2(11.88102171492205, 9.621930679287306),
				Vector2(24.231021714922047, 14.571930679287306),
				Vector2(21.86102171492205, 22.151930679287307),
				Vector2(8.671021714922048, 16.991930679287307),
			},
			{
				Vector2(16.895345211581294, 0.6087778396436541),
				Vector2(29.525345211581293, 5.8387778396436545),
				Vector2(26.225345211581292, 13.128777839643654),
				Vector2(13.875345211581294, 8.178777839643654),
			},
		},
		47.37442593444375
	);

	verify_case(
		"straight line crosses overlapping squares",
		Vector2(-2, 0),
		Vector2(3, 0),
		{
			{Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)},
			{Vector2(0, -1), Vector2(2, -1), Vector2(2, 1), Vector2(0, 1)},
		},
		5.0
	);

	verify_case(
		"target inside last polygon",
		Vector2(-2, 0),
		Vector2(0.5, 0),
		{
			{Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)},
			{Vector2(0, -1), Vector2(2, -1), Vector2(2, 1), Vector2(0, 1)},
		},
		2.5
	);

	verify_case(
		"target inside skew last polygon after reflection",
		Vector2(0.36700299616392595, 0.08906283997628592),
		Vector2(0.6052234643909907, 0.07032639865505626),
		{
			{
				Vector2(0.7425033774263062, 0.39474078408116847),
				Vector2(0.613098070483441, 0.5122740445155141),
				Vector2(0.24981344732273675, 0.3650606678098691),
			},
			{
				Vector2(0.5103829186955261, 0.22484050006586248),
				Vector2(0.49494925823445046, 0.008769253610802816),
				Vector2(0.5780535837940887, -0.06246302544031562),
				Vector2(0.6552218860994671, 0.15954424426900377),
			},
		},
		0.6432371109761946
	);

	verify_touching_case(
		"intersecting convex polygons require second contact",
		Vector2(0.36700299616392595, 0.08906283997628592),
		Vector2(0.6052234643909907, 0.07032639865505626),
		{
			{
				Vector2(0.813665848719522, 0.35969091612613247),
				Vector2(0.6842605417766566, 0.47722417656047805),
				Vector2(0.3209759186159518, 0.3300107998548331),
			},
			{
				Vector2(0.1261945059352388, 0.28356747517486985),
				Vector2(0.385880929252367, 0.30976093782165826),
				Vector2(0.4556637421496295, 0.4172455669175958),
				Vector2(0.3182102641538071, 0.5970644633278386),
			},
		},
		0.581842291484105
	);

	verify_touching_case(
		"problem instance 1 intersecting triangles",
		Vector2(0.31545199525507134, 0.09935234956713011),
		Vector2(0.6052234643909907, 0.07032639865505626),
		{
			{
				Vector2(0.3209759186159518, 0.3300107998548331),
				Vector2(0.6842605417766566, 0.47722417656047805),
				Vector2(0.813665848719522, 0.35969091612613247),
			},
			{
				Vector2(0.332652898650887, 0.5965839842936889),
				Vector2(0.4701063766467094, 0.41676508788344635),
				Vector2(0.4003235637494469, 0.3092804587875088),
				Vector2(0.14063714043231867, 0.2830869961407204),
			},
		},
		0.5827093381286692
	);

	verify_touching_case(
		"problem instance 2 intersecting triangles",
		Vector2(0.3368950507369469, 0.10262852539848555),
		Vector2(0.6052234643909907, 0.07032639865505626),
		{
			{
				Vector2(0.28431890403510046, 0.3268764921507503),
				Vector2(0.6476035271958053, 0.4740898688563962),
				Vector2(0.7770088341386706, 0.35655660842204967),
			},
			{
				Vector2(0.39797599762989433, 0.5645356676562588),
				Vector2(0.486273041837264, 0.35630691090949673),
				Vector2(0.20596023941132602, 0.2510386795032903),
			},
		},
		0.56879423031879461
	);

	verify_touching_case(
		"problem instance 3 intersecting triangles",
		Vector2(0.3368950507369469, 0.10262852539848555),
		Vector2(0.6052234643909907, 0.07032639865505626),
		{
			{
				Vector2(0.28431890403510046, 0.3268764921507503),
				Vector2(0.6476035271958053, 0.4740898688563962),
				Vector2(0.764982282478466, 0.3584256331225459),
			},
			{
				Vector2(0.34132700957925033, 0.21668258376715543),
				Vector2(0.47561163938781914, 0.381393663885594),
				Vector2(0.1802507510455204, 0.2563984718528892),
			},
		},
		0.5699301241460529
	);

	verify_touching_case(
		"intersecting polygon can be touched by outgoing segment",
		Vector2(0.2, 0.1),
		Vector2(0.6000000000000001, 0.1),
		{
			{
				Vector2(0.2, 0.35000000000000003),
				Vector2(0.7000000000000001, 0.55),
				Vector2(0.7000000000000001, 0.35000000000000003),
			},
			{
				Vector2(0.2, 0.30000000000000004),
				Vector2(0.2, 0.15000000000000002),
				Vector2(0.7000000000000001, 0.15000000000000002),
				Vector2(0.5903471022310567, 0.3563539238069492),
			},
		},
		0.6403124237432849,
		3
	);

	verify_case(
		"overlap skips middle visit",
		Vector2(-3, 0),
		Vector2(4, 0),
		{
			{Vector2(-2, -1), Vector2(0, -1), Vector2(0, 1), Vector2(-2, 1)},
			{Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)},
			{Vector2(0, -1), Vector2(2, -1), Vector2(2, 1), Vector2(0, 1)},
		},
		7.0
	);

	std::println("All intersection tests passed.");
}
