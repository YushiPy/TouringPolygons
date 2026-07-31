
#include "common.h"
#include "tpp_convex.h"

#include "gurobi_c++.h"

#include <vector>
#include <array>
#include <stdexcept>
#include <string>

using Point = std::array<double, 2>;
using Polygon = std::vector<Point>;

struct Halfplanes {
	std::vector<std::array<double, 2>> A; // row per edge
	std::vector<double> b;
};

Halfplanes vertices_to_halfplanes(const Polygon& verts) {
	int n = verts.size();
	Halfplanes hp;
	hp.A.resize(n);
	hp.b.resize(n);
	for (int i = 0; i < n; i++) {
		auto [x0, y0] = verts[i];
		auto [x1, y1] = verts[(i + 1) % n];
		double dx = x1 - x0, dy = y1 - y0;
		hp.A[i] = {dy, -dx};
		hp.b[i] = dy * x0 - dx * y0;
	}
	return hp;
}

std::vector<Point> shortest_path(
	const Point& start,
	const Point& end,
	const std::vector<Polygon>& polygons
) {
	int k = polygons.size();
	std::vector<Halfplanes> halfplanes;
	for (const auto& poly : polygons)
		halfplanes.push_back(vertices_to_halfplanes(poly));

	GRBEnv env(true);
	env.set(GRB_IntParam_OutputFlag, 0);
	env.start();
	GRBModel m(env);

	// p[i] = 2D point in polygon i
	std::vector<std::array<GRBVar, 2>> p(k);
	for (int i = 0; i < k; i++) {
		p[i][0] = m.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "p_" + std::to_string(i) + "_x");
		p[i][1] = m.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "p_" + std::to_string(i) + "_y");
	}

	// t[i] = length of segment i (k+1 segments total)
	std::vector<GRBVar> t(k + 1);
	for (int i = 0; i <= k; i++)
		t[i] = m.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "t_" + std::to_string(i));

	// polygon membership: A @ p[i] <= b
	for (int i = 0; i < k; i++) {
		const auto& hp = halfplanes[i];
		for (int j = 0; j < (int)hp.b.size(); j++) {
			m.addConstr(
				hp.A[j][0] * p[i][0] + hp.A[j][1] * p[i][1] <= hp.b[j]
			);
		}
	}

	// helper: add norm constraint t[seg] >= ||q - r||
	// using addGenConstrNorm on a 2-element array of aux vars
	auto add_norm_constr = [&](GRBVar& t_var, GRBLinExpr dx_expr, GRBLinExpr dy_expr, const std::string& name) {
		GRBVar d0 = m.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, name + "_dx");
		GRBVar d1 = m.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, name + "_dy");
		m.addConstr(d0 == dx_expr);
		m.addConstr(d1 == dy_expr);
		GRBVar dvars[2] = {d0, d1};
		m.addGenConstrNorm(t_var, dvars, 2, 2.0);
	};

	// start -> p[0]
	add_norm_constr(t[0],
		p[0][0] - start[0],
		p[0][1] - start[1],
		"d_start"
	);

	// p[i] -> p[i+1]
	for (int i = 0; i < k - 1; i++) {
		add_norm_constr(t[i + 1],
			p[i + 1][0] - p[i][0],
			p[i + 1][1] - p[i][1],
			"d_" + std::to_string(i)
		);
	}

	// p[k-1] -> end
	add_norm_constr(t[k],
		end[0] - p[k - 1][0],
		end[1] - p[k - 1][1],
		"d_end"
	);

	// minimize sum of t
	GRBLinExpr obj = 0;
	for (int i = 0; i <= k; i++) obj += t[i];
	m.setObjective(obj, GRB_MINIMIZE);

	m.optimize();

	if (m.get(GRB_IntAttr_Status) != GRB_OPTIMAL)
		throw std::runtime_error("Model status: " + std::to_string(m.get(GRB_IntAttr_Status)));

	std::vector<Point> result(k);
	for (int i = 0; i < k; i++)
		result[i] = {p[i][0].get(GRB_DoubleAttr_X), p[i][1].get(GRB_DoubleAttr_X)};

	return result;
}

namespace tpp {

	std::vector<Vector2> tpp_convex_solve_gurobi(const Vector2& start, const Vector2& target, const std::vector<std::vector<Vector2>>& polygons) {
		
		std::vector<Polygon> polys;
		
		for (const auto& poly : polygons) {

			Polygon p;

			for (const auto& v : poly) {
				p.push_back({v.x, v.y});
			}

			polys.push_back(std::move(p));
		}
		
		auto path = shortest_path({start.x, start.y}, {target.x, target.y}, polys);
		
		std::vector<Vector2> result;

		result.reserve(path.size() + 2);
		result.push_back(start);

		for (const auto& p : path) {
			result.push_back({p[0], p[1]});
		}

		result.push_back(target);
		
		return tpp::remove_collinear_points(result, 1e-6);
	}
}
