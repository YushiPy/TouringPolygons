// Fixed-order non-convex TPP (tpp_nonconvex_solve) against brute force: every
// choice of one convex piece per polygon solved by the convex solver. The
// first cases are routes of the CVRPN solutions of Birgin and Matos
// (P-n55-k15 route 2, tai385 route 13) where, with fused multiply-add, a hull
// with a repeated vertex gave a bound above the optimum and the
// branch-and-bound returned a longer path marked exact.
#include "tpp/convex/solver.h"
#include "tpp/nonconvex/decomposition.h"
#include "tpp/nonconvex/solver.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tpp;
using Polygon = std::vector<Vector2>;

namespace {

	double path_length(const std::vector<Vector2> &path) {
		double length = 0.0;
		for (size_t i = 1; i < path.size(); i++) length += path[i - 1].distance_to(path[i]);
		return length;
	}

	double brute_force(const Vector2 &start, const Vector2 &target, const std::vector<std::vector<Polygon>> &pieces) {
		std::vector<size_t> choice(pieces.size(), 0);
		double best = std::numeric_limits<double>::infinity();
		while (true) {
			std::vector<Polygon> input;
			for (size_t i = 0; i < pieces.size(); i++) input.push_back(pieces[i][choice[i]]);
			best = std::min(best, path_length(tpp_convex_solve_binary_search_lazy(start, target, input)));
			size_t i = 0;
			for (; i < pieces.size(); i++) {
				if (++choice[i] < pieces[i].size()) break;
				choice[i] = 0;
			}
			if (i == pieces.size()) return best;
		}
	}

	void check(const std::string &name, const Vector2 &start, const Vector2 &target, const std::vector<Polygon> &polygons) {
		std::vector<std::vector<Polygon>> pieces;
		for (const auto &polygon : polygons) pieces.push_back(decompose_polygon(polygon));
		const double expected = brute_force(start, target, pieces);
		NonconvexTppSolveOptions options;
		options.max_calls = 1000000;
		options.max_seconds = 60.0;
		const auto result = tpp_nonconvex_solve(start, target, polygons, options);
		const double length = path_length(result.path);
		if (!result.exact) throw std::runtime_error(name + ": the branch-and-bound did not finish.");
		if (std::abs(length - expected) > 1e-9 * expected) {
			throw std::runtime_error(name + ": length " + std::to_string(length) + ", brute force " + std::to_string(expected) + ".");
		}
	}

	// Star-shaped polygons are simple and, with random radii, usually non-convex.
	Polygon star(std::mt19937_64 &rng, const Vector2 &center, double radius, size_t vertices) {
		std::uniform_real_distribution<double> unit(0.0, 1.0);
		Polygon polygon;
		for (size_t i = 0; i < vertices; i++) {
			const double angle = 2 * std::numbers::pi * (i + 0.8 * unit(rng)) / double(vertices);
			const double r = radius * (0.3 + 0.7 * unit(rng));
			polygon.push_back({center.x + r * std::cos(angle), center.y + r * std::sin(angle)});
		}
		return polygon;
	}

}

int main() {
	check("P-n55-k15 route 2", {40.0, 40.0}, {40.0, 40.0}, {
		{{55.5218530142774, 50.96434164802779}, {54.68059344155014, 50.686183359677564}, {54.06620755844719, 50.68226779599322},
			{54.156044416993666, 50.17941993840763}, {54.34472967442704, 49.392592292787924}, {55.77933152641724, 50.21034399702303}},
		{{62.442144802803895, 48.5615216876931}, {61.693843295547765, 48.499293597735466}, {61.37873151046181, 47.738661660060515},
			{62.334664264557695, 47.043326199974935}},
		{{66.1604639397451, 41.82926847980339}, {66.1708575960372, 41.39766085947442}, {66.29512778902885, 40.53317665661676},
			{66.54410281570718, 40.13928978977931}, {67.5709972492286, 40.101950843790064}, {67.92611073429886, 40.559646969285524}},
		{{62.56587516038205, 35.57389806070358}, {61.63727557770428, 35.9736496259933}, {61.08706466199712, 35.992029316593765},
			{61.30842688172734, 35.17384174014012}, {62.306033697480494, 34.42330486758769}, {62.77950593479586, 34.1616758682283},
			{62.65252220993553, 35.18925205889671}},
		{{54.61607582984116, 34.874989755842655}, {54.22545250772963, 34.59592471035072}, {54.00542648730572, 33.99191553874004},
			{54.33457055154871, 33.78605324986681}, {54.107453464135965, 33.400973643364395}, {54.44275192582651, 33.153375801910386},
			{55.19306425615144, 33.18193434941835}, {55.6305884666447, 33.69418027985673}},
	});

	check("tai385 route 13", {5332.0, 1525.0}, {5332.0, 1525.0}, {
		{{5189.036105913982, 1624.8243118839139}, {5174.655400307236, 1619.4457426893362}, {5175.008242417518, 1605.1560730997016},
			{5179.826360854472, 1602.4007347021125}, {5188.502734878754, 1603.2902476368715}, {5196.476207038407, 1601.3049583446464}},
		{{5153.004227781975, 1670.1539167542624}, {5150.799006967515, 1681.95934736957}, {5139.811136541699, 1679.6246185290897},
			{5134.376338666851, 1668.5911257679038}, {5130.294362149544, 1659.294088740599}, {5141.4727634951405, 1656.9768849958205},
			{5146.666535665345, 1658.3132354768945}, {5148.711379869985, 1662.587640185656}},
		{{5126.768582575569, 1668.649562243896}, {5117.838264618095, 1675.5978325920123}, {5109.961189098777, 1681.232705717844},
			{5102.5143332350635, 1669.4423536233683}, {5105.572894620987, 1659.9123488235234}, {5115.118050993502, 1661.8832377495576},
			{5120.677505072278, 1656.149214911325}},
		{{5067.721600899271, 1601.918035702877}, {5058.111853760382, 1609.4109026173783}, {5052.1658098442795, 1609.7569297694727},
			{5048.162923553621, 1600.0369729404333}, {5051.64659218063, 1591.4469888752626}, {5050.70314574336, 1586.961429937249},
			{5071.380043770587, 1584.5776663530971}, {5066.039274160784, 1589.5332723543393}},
		{{5161.118916584094, 1580.0886044105512}, {5155.476724616013, 1577.0840791785065}, {5145.520820987947, 1580.8698536118716},
			{5146.386550784361, 1556.7193903776067}, {5155.664873987621, 1558.4778656616852}, {5163.1703648593675, 1564.9956282626142}},
	});

	// General validation; on their own these do not reproduce the repeated
	// hull vertex, which needs a shared vertex at an extreme of the sort.
	std::mt19937_64 rng(20261028);
	std::uniform_real_distribution<double> coordinate(0.0, 100.0);
	const int cases = 400;
	for (int c = 0; c < cases; c++) {
		std::vector<Polygon> polygons;
		for (int i = 0; i < 4; i++) polygons.push_back(star(rng, {coordinate(rng), coordinate(rng)}, 4.0 + 0.1 * coordinate(rng), 7));
		const Vector2 start{coordinate(rng), coordinate(rng)};
		const Vector2 target = c % 2 ? start : Vector2{coordinate(rng), coordinate(rng)};
		check("random case " + std::to_string(c), start, target, polygons);
	}

	std::cout << "Fixed-order non-convex TPP: " << cases + 2 << " cases equal to brute force.\n";
}
