
#include "vector2.h"
#include "common.h"
#include "tests.h"

// Math for generating random test cases and verifying solutions.
#include <vector>
#include <cmath>
#include <tuple>

// Random number generation
#include <algorithm>
#include <random>

// For plotting solutions
#include <cstdlib>
#include <string>
#include <format>
#include <chrono>

using std::vector;
using std::tuple;

vector<Vector2> regular_polygon(size_t n, const Vector2 &center, double radius, double rotation = 0.0) {

	if (n < 3) {
		throw std::invalid_argument("A polygon must have at least 3 vertices.");
	}

	vector<Vector2> vertices;

	for (size_t i = 0; i < n; i++) {
		double angle = 2.0 * M_PI * i / n + rotation;
		auto vertex = Vector2::from_angle(angle) * radius + center;
		vertices.push_back(vertex);
	}

	return vertices;
}

bool segment_polygon_intersection(const Vector2 &p1, const Vector2 &p2, const vector<Vector2> &polygon, Vector2 *intersection) {

	for (size_t i = 0; i < polygon.size(); i++) {

		const auto &v1 = polygon[i];
		const auto &v2 = polygon[(i + 1) % polygon.size()];

		auto current_intersection = tpp::segment_segment_intersection_safe(p1, p2, v1, v2);

		if (current_intersection.is_finite()) {
			
			if (intersection != nullptr) {
				*intersection = current_intersection;
			}

			return true;
		}
	}

	return false;
}


namespace tpp {

	auto rng = std::default_random_engine(std::chrono::steady_clock::now().time_since_epoch().count());

	void set_rng_seed(unsigned int seed) {
		rng.seed(seed);
	}

	tuple<Vector2, Vector2, vector<vector<Vector2>>> generate_test(const vector<size_t> &polygon_sizes) {

		const size_t height = std::ceil(std::sqrt(polygon_sizes.size()));
		const size_t width = std::ceil((double) polygon_sizes.size() / height);

		vector<size_t> indices(height * width);
		
		for (size_t i = 0; i < indices.size(); i++) {
			indices[i] = i;
		}

		// Shuffle the indices to randomize the polygon positions
		std::ranges::shuffle(indices, rng);

		vector<vector<Vector2>> polygons;

		for (size_t i = 0; i < polygon_sizes.size(); i++) {
			
			size_t index = indices[i];
			size_t row = index / width;
			size_t col = index % width;

			Vector2 center(col, row);

			std::uniform_real_distribution<double> dist(0.1, 0.45);
			double radius = dist(rng);

			polygons.push_back(regular_polygon(polygon_sizes[i], center, radius));
		}

		Vector2 start(-0.5, -0.5);
		Vector2 target(width - 0.5, height - 0.5);

		return {start, target, polygons};
	}

	tuple<Vector2, Vector2, vector<vector<Vector2>>> generate_test_bad(const vector<size_t> &polygon_sizes, bool shuffle) {

		if (shuffle) {

			vector<size_t> copy = polygon_sizes;
			
			std::ranges::shuffle(copy, rng);
			auto [start, target, polygons] = generate_test_bad(copy, false);
			
			for (auto &polygon : polygons) {
				size_t index = rng() % polygon.size();
				std::ranges::rotate(polygon, polygon.begin() + index);
			}

			return {start, target, polygons};
		}

		vector<vector<Vector2>> polygons;

		for (size_t i = 0; i < polygon_sizes.size(); i++) {

			// std::uniform_real_distribution<double> dist(0.25, 0.5);
			// double radius = dist(rng);
			double radius = 0.45;

			Vector2 center(i, 0);
			polygons.push_back(regular_polygon(polygon_sizes[i], center, radius));
		}

		Vector2 start(-1.0, 0);
		Vector2 target(polygon_sizes.size(), 0);

		return {start, target, polygons};
	}

	std::tuple<Vector2, Vector2, std::vector<std::vector<Vector2>>> generate_test_good(const std::vector<size_t> &polygon_sizes, bool shuffle) {

		if (polygon_sizes.empty()) {
			Vector2 start(0, 0);
			Vector2 target(1, 0);
			return {start, target, {}};
		}

		if (shuffle) {
			
			vector<size_t> copy = polygon_sizes;
			std::ranges::shuffle(copy, rng);
			auto [start, target, polygons] = generate_test_good(copy, false);
			
			for (auto &polygon : polygons) {
				size_t index = rng() % polygon.size();
				std::ranges::rotate(polygon, polygon.begin() + index);
			}

			return {start, target, polygons};
		}

		vector<vector<Vector2>> polygons;
		
		Vector2 start(0, 0);

		const double max_r = 0.45;
		const double min_x = 1.0;
		
		double x = min_x;
		const auto first = regular_polygon(polygon_sizes[0], Vector2(x, 0), max_r, M_PI);

		polygons.push_back(first);

		for (size_t i = 1; i < polygon_sizes.size(); i++) {

			auto n = polygon_sizes[i];

			if (n <= 4) {
				x = min_x;
			} else {
				double h = max_r / std::sin(2 * M_PI / n);
				x = std::max(min_x, h - x + max_r);
			}

			double poly_x = x;
			double angle = M_PI;

			if (i % 2 == 1) {
				poly_x *= -1;
				angle = 0;
			}

			const auto polygon = regular_polygon(n, Vector2(poly_x, 0), max_r, angle);
			polygons.push_back(polygon);
		}

		Vector2 target(-min_x / 3, 0);
		
		return {start, target, polygons};
	}

	bool is_valid_solution(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons, const vector<Vector2> &solution) {

		// A correct solution must start at `start`, end at `target`, have no consecutive collinear points, and visit the polygons in order without skipping any.
		if (solution.size() < 2) {
			return false;
		}

		if (!solution.front().is_equal_approx(start) || !solution.back().is_equal_approx(target)) {
			return false;
		}

		if (solution.size() > 2) {

			for (size_t i = 1; i < solution.size() - 1; i++) {

				const auto &p1 = solution[i - 1];
				const auto &p2 = solution[i];
				const auto &p3 = solution[i + 1];

				const auto &d1 = p2 - p1;
				const auto &d2 = p3 - p2;

				if (d1.is_same_direction(d2)) {
					return false;
				}
			}
		}

		size_t polygon_index = 0;
		size_t path_index = 1;

		Vector2 segment_start = start;

		// Tracks whether the current segment has visited at least one polygon. 
		// Every segment must visit at least one polygon, but it can visit more than one.
		bool segment_visits_a_polygon = false;

		while (polygon_index < polygons.size() && path_index < solution.size()) {
			
			const auto &polygon = polygons[polygon_index];

			const auto &point = solution[path_index];

			bool is_on_vertex = false;

			// Verify if the point is on a vertex and if it is a valid bend.
			for (size_t i = 0; i < polygon.size(); i++) {
				
				const auto &vertex = polygon[i];

				if (!point.is_equal_approx(vertex)) {
					continue;
				}

				is_on_vertex = true;

				// If the target point is on a vertex, it is valid.
				if (path_index == solution.size() - 1) {
					return true;
				}

				const auto &before = polygon[(i + polygon.size() - 1) % polygon.size()];
				const auto &after = polygon[(i + 1) % polygon.size()];

				const auto last = segment_start;
				const auto diff = vertex - last;

				auto ray1 = diff.reflect(vertex - before);
				auto ray2 = diff.reflect(vertex - after);

				if (diff.cross(vertex - before) >= 0) {
					ray1 = diff;
				}

				if (diff.cross(vertex - after) <= 0) {
					ray2 = diff;
				}

				const auto &next_point = solution[path_index + 1];

				if (!tpp::point_in_cone_plus(next_point, vertex, ray1, ray2)) {
					return false;
				}

				break;
			}

			if (is_on_vertex) {
				segment_visits_a_polygon = false;
				polygon_index++;
				path_index++;
				segment_start = solution[path_index - 1];
				continue;
			}

			bool is_on_edge = false;

			// Verify if the point is on an edge and if it follows the reflection rule.
			for (size_t i = 0; i < polygon.size(); i++) {
				
				const auto &v1 = polygon[i];
				const auto &v2 = polygon[(i + 1) % polygon.size()];
				
				// Verify if the point is on the edge
				if (!(point - v1).is_same_direction(v2 - v1) || (point - v2).dot(v1 - v2) <= 0) {
					continue;
				}

				is_on_edge = true;

				// If the target point is on an edge, it is valid if it follows the reflection rule.
				if (path_index == solution.size() - 1) {
					return true;
				}

				const auto &next_point = solution[path_index + 1];
				const auto reflected = next_point.reflect_line(v1, v2);

				const auto d1 = reflected - point;
				const auto d2 = point - segment_start;

				// We verify the reflection rule
				if (!d1.is_same_direction(d2)) {
					return false;
				}

				break;
			}

			if (is_on_edge) {
				segment_visits_a_polygon = false;
				polygon_index++;
				path_index++;
				segment_start = solution[path_index - 1];
				continue;
			}

			bool crosses_polygon = false;

			for (size_t i = 0; i < polygon.size(); i++) {
				
				const auto &v1 = polygon[i];
				const auto &v2 = polygon[(i + 1) % polygon.size()];

				const auto intersection = tpp::segment_segment_intersection_safe(segment_start, point, v1, v2);

				if (intersection.is_finite()) {
					crosses_polygon = true;
					segment_start = intersection;
					break;
				}
			}

			if (crosses_polygon) {
				segment_visits_a_polygon = true;
				polygon_index++;
				continue;
			}

			if (!segment_visits_a_polygon) {
				return false;
			}

			path_index++;
			segment_visits_a_polygon = false;
		}

		return true;
	}

	bool solutions_equal(const vector<Vector2> &sol1, const vector<Vector2> &sol2) {

		if (sol1.size() != sol2.size()) {
			return false;
		}

		for (size_t i = 0; i < sol1.size(); i++) {
			if (!sol1[i].is_equal_approx(sol2[i])) {
				return false;
			}
		}

		return true;
	}

	std::string vectors_to_string(const vector<Vector2> &vectors) {
		
		std::string result = "[";

		for (size_t i = 0; i < vectors.size(); i++) {
			
			result += std::format("({}, {})", vectors[i].x, vectors[i].y);

			if (i < vectors.size() - 1) {
				result += ", ";
			}
		}

		result += "]";
		
		return result;
	}

	std::string polygons_to_string(const vector<vector<Vector2>> &polygons) {
		
		std::string result = "[";

		for (size_t i = 0; i < polygons.size(); i++) {
			result += vectors_to_string(polygons[i]);

			if (i < polygons.size() - 1) {
				result += ", ";
			}
		}

		result += "]";
		
		return result;
	}

	void plot_solution(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons, const vector<Vector2> &solution) {

		std::string code;

		// std::println("{}", std::format("start = ({}, {})\ntarget = ({}, {})\npolygons = {}\nsolution = {}\n", start.x, start.y, target.x, target.y, polygons_to_string(polygons), vectors_to_string(solution)));

		code += std::format("start = ({}, {})\ntarget = ({}, {})\npolygons = {}\nsolution = {}\n", start.x, start.y, target.x, target.y, polygons_to_string(polygons), vectors_to_string(solution));
		code += R""""(

SQUARE = True
SCALE = 1.2

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

minx = min(start[0], target[0], min(vertex[0] for polygon in polygons for vertex in polygon))
maxx = max(start[0], target[0], max(vertex[0] for polygon in polygons for vertex in polygon))
miny = min(start[1], target[1], min(vertex[1] for polygon in polygons for vertex in polygon))
maxy = max(start[1], target[1], max(vertex[1] for polygon in polygons for vertex in polygon))

width = maxx - minx
height = maxy - miny

centerx = (minx + maxx) / 2
centery = (miny + maxy) / 2

if SQUARE:
	width = max(width, height)
	height = width

width *= SCALE
height *= SCALE

bbox = (centerx - width / 2, centerx + width / 2, centery - height / 2, centery + height / 2)

ax.set_xlim(bbox[0], bbox[1])
ax.set_ylim(bbox[2], bbox[3])

ax.set_aspect('equal', adjustable='box')

ax.fill([bbox[0], bbox[1], bbox[1], bbox[0]], [bbox[2], bbox[2], bbox[3], bbox[3]], color='lightgray')

ax.scatter(*start, color='green', label='Start', zorder=5)
ax.scatter(*target, color='red', label='Target', zorder=5)

ax.text(start[0] + 0.05, start[1] + 0.05, 's', ha='center', va='center', backgroundcolor='white')
ax.text(target[0] + 0.05, target[1] + 0.05, 't', ha='center', va='center', backgroundcolor='white')

for i, polygon in enumerate(polygons):
	ax.fill(*zip(*polygon), alpha=0.5, edgecolor='black')
	centerx = sum(vertex[0] for vertex in polygon) / len(polygon)
	centery = sum(vertex[1] for vertex in polygon) / len(polygon)
	ax.text(centerx, centery, f'{i + 1}', ha='center', va='center')

ax.plot(*zip(*solution), color='blue', label='Solution')
ax.legend()

plt.tight_layout()

try:
	plt.show()
except KeyboardInterrupt:
	pass
		)"""";

		std::string cmd = std::format("python3 -c \"{}\"", code);
		std::system(cmd.c_str());
	}

	std::vector<std::byte> encode_test(const Vector2 &start, const Vector2 &target, const vector<vector<Vector2>> &polygons, const vector<Vector2> &solution) {

		std::vector<std::byte> data;

		auto append_vector2 = [&data](const Vector2 &v) {
			const auto x_bytes = std::bit_cast<std::array<std::byte, sizeof(double)>>(v.x);
			const auto y_bytes = std::bit_cast<std::array<std::byte, sizeof(double)>>(v.y);
			data.insert(data.end(), x_bytes.begin(), x_bytes.end());
			data.insert(data.end(), y_bytes.begin(), y_bytes.end());
		};

		auto append_size_t = [&data](size_t value) {
			const auto bytes = std::bit_cast<std::array<std::byte, sizeof(size_t)>>(value);
			data.insert(data.end(), bytes.begin(), bytes.end());
		};

		append_vector2(start);
		append_vector2(target);
		append_size_t(polygons.size());

		for (const auto &polygon : polygons) {
			append_size_t(polygon.size());
			for (const auto &vertex : polygon) {
				append_vector2(vertex);
			}
		}

		append_size_t(solution.size());

		for (const auto &point : solution) {
			append_vector2(point);
		}

		return data;
	}

	TestCase decode_test(const std::byte *data, size_t &offset) {

		Vector2 start, target;
		std::vector<std::vector<Vector2>> polygons;
		std::vector<Vector2> solution;

		auto read_vector2 = [&data](size_t &offset) -> Vector2 {
			double x = std::bit_cast<double>(std::array<std::byte, sizeof(double)>{data[offset], data[offset + 1], data[offset + 2], data[offset + 3], data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]});
			double y = std::bit_cast<double>(std::array<std::byte, sizeof(double)>{data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11], data[offset + 12], data[offset + 13], data[offset + 14], data[offset + 15]});
			offset += sizeof(double) * 2;
			return Vector2(x, y);
		};

		auto read_size_t = [&data](size_t &offset) -> size_t {
			size_t value = std::bit_cast<size_t>(std::array<std::byte, sizeof(size_t)>{data[offset], data[offset + 1], data[offset + 2], data[offset + 3], data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]});
			offset += sizeof(size_t);
			return value;
		};

		start = read_vector2(offset);
		target = read_vector2(offset);

		size_t num_polygons = read_size_t(offset);
		polygons.reserve(num_polygons);

		for (size_t i = 0; i < num_polygons; i++) {

			size_t num_vertices = read_size_t(offset);

			std::vector<Vector2> polygon;
			polygon.reserve(num_vertices);

			for (size_t j = 0; j < num_vertices; j++) {
				polygon.push_back(read_vector2(offset));
			}

			polygons.push_back(polygon);
		}

		size_t num_solution_points = read_size_t(offset);
		solution.reserve(num_solution_points);

		for (size_t i = 0; i < num_solution_points; i++) {
			solution.push_back(read_vector2(offset));
		}
		
		return {start, target, polygons, solution};
	}

	TestCase decode_test(std::istream &ifs) {

		auto read_vector2 = [&ifs]() -> Vector2 {
			double x, y;
			ifs.read(reinterpret_cast<char*>(&x), sizeof(double));
			ifs.read(reinterpret_cast<char*>(&y), sizeof(double));
			return Vector2(x, y);
		};

		auto read_size_t = [&ifs]() -> size_t {
			size_t value;
			ifs.read(reinterpret_cast<char*>(&value), sizeof(size_t));
			return value;
		};

		const auto start = read_vector2();
		const auto target = read_vector2();

		const size_t num_polygons = read_size_t();
		std::vector<std::vector<Vector2>> polygons(num_polygons);
		for (auto &polygon : polygons) {
			polygon.resize(read_size_t());
			for (auto &v : polygon) v = read_vector2();
		}

		const size_t num_solution = read_size_t();
		std::vector<Vector2> solution(num_solution);
		for (auto &v : solution) v = read_vector2();

		return {start, target, polygons, solution};
	}
}
