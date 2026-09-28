#include <gurobi_c++.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Point { double x, y; };
using Polygon = std::vector<Point>;
using Polygons = std::vector<Polygon>;
struct Instance { std::string name; Polygons polygons; };

double signed_area(const Polygon &polygon) {
  double area = 0;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const auto &a = polygon[i];
    const auto &b = polygon[(i + 1) % polygon.size()];
    area += a.x * b.y - a.y * b.x;
  }
  return area / 2;
}

void normalize_ccw(Polygons &polygons) {
  for (auto &polygon : polygons) {
    if (polygon.size() < 3 || signed_area(polygon) == 0)
      throw std::invalid_argument("Each input must be a nondegenerate polygon");
    if (signed_area(polygon) < 0) std::reverse(polygon.begin(), polygon.end());
  }
}

std::vector<Instance> instances() {
  return {
    {"two_oblique_triangles", {
      {{0,0},{2,0},{0,1}},
      {{4,3},{6,3},{5,5}}
    }},
    {"three_asymmetric_boxes", {
      {{0,0},{1.4,0},{1.4,1},{0,1}},
      {{4,0.7},{5,0.7},{5,2.2},{4,2.2}},
      {{3.5,5},{4.8,5},{4.8,6},{3.5,6}}
    }},
    {"four_asymmetric_boxes", {
      {{0,0},{1.4,0},{1.4,1},{0,1}},
      {{4,0.7},{5,0.7},{5,2.2},{4,2.2}},
      {{5.7,5},{7,5},{7,6},{5.7,6}},
      {{-1,4.3},{0.1,4.3},{0.1,5.5},{-1,5.5}}
    }},
    {"five_scattered_quadrilaterals", {
      {{0,0},{1.2,0},{1.2,0.8},{0,0.8}},
      {{4,0.3},{5.1,0.3},{5.1,1.4},{4,1.4}},
      {{6.2,4},{7.4,4},{7.4,5.1},{6.2,5.1}},
      {{2.1,6.3},{3.3,6.3},{3.3,7.2},{2.1,7.2}},
      {{-1.4,3.8},{-0.2,3.8},{-0.2,4.9},{-1.4,4.9}}
    }}
  };
}

void add_polygon_constraints(GRBModel &model, const std::array<GRBVar, 2> &q, const Polygon &polygon) {
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const Point a = polygon[i];
    const Point b = polygon[(i + 1) % polygon.size()];
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    // For a CCW edge, the polygon is to its left: dy*x - dx*y <= dy*a.x - dx*a.y.
    model.addConstr(dy * q[0] - dx * q[1] <= dy * a.x - dx * a.y);
  }
}

struct Solution {
  int status = 0;
  double objective = std::numeric_limits<double>::quiet_NaN();
  double bound = std::numeric_limits<double>::quiet_NaN();
  double seconds = 0;
  double recomputed_length = std::numeric_limits<double>::quiet_NaN();
  std::vector<Point> contacts;
  double feasible_recomputed_length = std::numeric_limits<double>::quiet_NaN();
  std::vector<Point> feasible_contacts;
};

Solution solve(const Instance &instance, GRBEnv &environment) {
  const std::size_t k = instance.polygons.size();
  if (k == 0) throw std::invalid_argument("Empty cycle");

  GRBModel model(environment);
  model.set(GRB_IntParam_Threads, 1);
  model.set(GRB_IntParam_Method, 2);
  model.set(GRB_IntParam_NumericFocus, 2);
  model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);
  model.set(GRB_DoubleParam_OptimalityTol, 1e-9);
  model.set(GRB_DoubleParam_BarConvTol, 1e-10);
  model.set(GRB_DoubleParam_TimeLimit, 30);

  std::vector<std::array<GRBVar, 2>> q(k);
  for (std::size_t i = 0; i < k; ++i) {
    double min_x = instance.polygons[i].front().x;
    double max_x = min_x;
    double min_y = instance.polygons[i].front().y;
    double max_y = min_y;
    for (const Point p : instance.polygons[i]) {
      min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
    }
    q[i][0] = model.addVar(min_x, max_x, 0, GRB_CONTINUOUS, "q_" + std::to_string(i) + "_x");
    q[i][1] = model.addVar(min_y, max_y, 0, GRB_CONTINUOUS, "q_" + std::to_string(i) + "_y");
    add_polygon_constraints(model, q[i], instance.polygons[i]);
  }

  GRBLinExpr objective = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const std::size_t next = (i + 1) % k;
    GRBVar dx = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "dx_" + std::to_string(i));
    GRBVar dy = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "dy_" + std::to_string(i));
    model.addConstr(dx == q[next][0] - q[i][0]);
    model.addConstr(dy == q[next][1] - q[i][1]);
    GRBVar link_length = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "length_" + std::to_string(i));
    GRBVar delta[2] = {dx, dy};
    model.addGenConstrNorm(link_length, delta, 2, 2.0, "soc_" + std::to_string(i));
    objective += link_length;
  }
  model.setObjective(objective, GRB_MINIMIZE);

  model.optimize();

  Solution solution;
  solution.status = model.get(GRB_IntAttr_Status);
  if (model.get(GRB_IntAttr_SolCount) == 0) return solution;
  solution.objective = model.get(GRB_DoubleAttr_ObjVal);
  try { solution.bound = model.get(GRB_DoubleAttr_ObjBound); }
  catch (const GRBException &) { solution.bound = solution.objective; }
  solution.seconds = model.get(GRB_DoubleAttr_Runtime);
  for (std::size_t i = 0; i < k; ++i)
    solution.contacts.push_back({q[i][0].get(GRB_DoubleAttr_X), q[i][1].get(GRB_DoubleAttr_X)});
  double length = 0;
  double feasible_length = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const Point a = solution.contacts[i];
    const Point b = solution.contacts[(i + 1) % k];
    length += std::hypot(b.x - a.x, b.y - a.y);

    Point center{0, 0};
    for (const Point vertex : instance.polygons[i]) {
      center.x += vertex.x / instance.polygons[i].size();
      center.y += vertex.y / instance.polygons[i].size();
    }
    constexpr double interior_contraction = 1e-5;
    solution.feasible_contacts.push_back({
      (1 - interior_contraction) * a.x + interior_contraction * center.x,
      (1 - interior_contraction) * a.y + interior_contraction * center.y
    });
  }
  for (std::size_t i = 0; i < k; ++i) {
    const Point a = solution.feasible_contacts[i];
    const Point b = solution.feasible_contacts[(i + 1) % k];
    feasible_length += std::hypot(b.x - a.x, b.y - a.y);
  }
  solution.recomputed_length = length;
  solution.feasible_recomputed_length = feasible_length;
  return solution;
}

int main() {
  try {
    GRBEnv environment(true);
    environment.set(GRB_IntParam_OutputFlag, 0);
    environment.start();
    std::cout << std::setprecision(17) << "{\"gurobi_version\":\"13.0.3\",\"instances\":[";
    bool first = true;
    for (auto instance : instances()) {
      normalize_ccw(instance.polygons);
      const Solution solution = solve(instance, environment);
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"name\":\"" << instance.name << "\",\"status\":" << solution.status;
      if (!solution.contacts.empty()) {
        std::cout << ",\"objective\":" << solution.objective
                  << ",\"objective_bound\":" << solution.bound
                  << ",\"recomputed_length\":" << solution.recomputed_length
                  << ",\"feasible_recomputed_length\":" << solution.feasible_recomputed_length
                  << ",\"runtime_seconds\":" << solution.seconds
                  << ",\"contacts\":[";
        for (std::size_t i = 0; i < solution.contacts.size(); ++i) {
          if (i) std::cout << ',';
          std::cout << '[' << solution.contacts[i].x << ',' << solution.contacts[i].y << ']';
        }
        std::cout << "],\"feasible_contacts\":[";
        for (std::size_t i = 0; i < solution.feasible_contacts.size(); ++i) {
          if (i) std::cout << ',';
          std::cout << '[' << solution.feasible_contacts[i].x << ',' << solution.feasible_contacts[i].y << ']';
        }
        std::cout << ']';
      }
      std::cout << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const GRBException &error) {
    std::cerr << "Gurobi error " << error.getErrorCode() << ": " << error.getMessage() << '\n';
    return 2;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
