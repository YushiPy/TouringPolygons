#include "optimal_convex_partition/optimal_convex_partition.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <stack>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace optimal_convex_partition {

	namespace {

		enum class Orientation {
			right,
			collinear,
			left,
		};

		enum class EdgeValidity {
			not_valid,
			start_valid,
			end_valid,
			both_valid,
		};

		using Diagonal = std::pair<unsigned int, unsigned int>;
		using DiagonalList = std::list<Diagonal>;
		using BigInt = __int128_t;

		struct Edge {
			bool done = false;
			EdgeValidity validity = EdgeValidity::not_valid;
			bool visible = false;
			int value = 0;
			DiagonalList solution;

			bool is_valid() const {
				return validity != EdgeValidity::not_valid;
			}
		};

		struct StackRecord {
			unsigned int vertex = 0;
			int value = 0;
			DiagonalList solution;
		};

		struct VertexState {
			unsigned int vertex = 0;
			std::list<StackRecord> stack;
			StackRecord best_so_far;

			explicit VertexState(unsigned int vertex_num)
				: vertex(vertex_num) {}

			bool stack_empty() const {
				return stack.empty();
			}

			StackRecord stack_top() const {
				return stack.back();
			}

			void stack_push(unsigned int old_vertex, int record_value, const DiagonalList &diag_list) {
				best_so_far = {old_vertex, record_value, diag_list};
				stack.push_back(best_so_far);
			}

			void stack_pop() {
				best_so_far = stack.back();
				stack.pop_back();
			}
		};

			template <class T>
			using Matrix = std::vector<std::vector<T>>;

			bool point_equal(const Point &a, const Point &b) {
				return a.x == b.x && a.y == b.y;
			}

			struct ExactDouble {
				BigInt mantissa;
				int exponent = 0;
			};

			ExactDouble exact_double(double value) {
				const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
				const bool negative = (bits >> 63U) != 0;
				const auto exponent_bits = static_cast<int>((bits >> 52U) & 0x7ffU);
				const std::uint64_t fraction = bits & ((std::uint64_t{1} << 52U) - 1U);

				if (exponent_bits == 0 && fraction == 0) {
					return {};
				}

				BigInt mantissa = 0;
				int exponent = 0;
				if (exponent_bits == 0) {
					mantissa = static_cast<BigInt>(fraction);
					exponent = -1074;
				} else {
					mantissa = static_cast<BigInt>((std::uint64_t{1} << 52U) | fraction);
					exponent = exponent_bits - 1023 - 52;
				}

				if (negative) {
					mantissa = -mantissa;
				}

				return {mantissa, exponent};
			}

			bool multiply_by_power_of_two(BigInt &value, int exponent) {
				if (exponent < 0 || exponent >= 120) {
					return false;
				}

				value *= (static_cast<BigInt>(1) << static_cast<unsigned int>(exponent));
				return true;
			}

			ExactDouble exact_subtract(const ExactDouble &a, const ExactDouble &b) {
				if (a.mantissa == 0) {
					return {-b.mantissa, b.exponent};
				}

				if (b.mantissa == 0) {
					return a;
				}

				const int exponent = std::min(a.exponent, b.exponent);
				BigInt mantissa = a.mantissa;
				if (!multiply_by_power_of_two(mantissa, a.exponent - exponent)) {
					return {};
				}
				BigInt other = b.mantissa;
				if (!multiply_by_power_of_two(other, b.exponent - exponent)) {
					return {};
				}
				return {mantissa - other, exponent};
			}

			int exact_orientation_sign(const Point &a, const Point &b, const Point &c) {
				const ExactDouble ax = exact_double(a.x);
				const ExactDouble ay = exact_double(a.y);
				const ExactDouble bx = exact_double(b.x);
				const ExactDouble by = exact_double(b.y);
				const ExactDouble cx = exact_double(c.x);
				const ExactDouble cy = exact_double(c.y);

				const ExactDouble bax = exact_subtract(bx, ax);
				const ExactDouble bay = exact_subtract(by, ay);
				const ExactDouble cax = exact_subtract(cx, ax);
				const ExactDouble cay = exact_subtract(cy, ay);

				BigInt left = bax.mantissa * cay.mantissa;
				const int left_exponent = bax.exponent + cay.exponent;
				BigInt right = bay.mantissa * cax.mantissa;
				const int right_exponent = bay.exponent + cax.exponent;
				const int exponent = std::min(left_exponent, right_exponent);
				if (!multiply_by_power_of_two(left, left_exponent - exponent)
					|| !multiply_by_power_of_two(right, right_exponent - exponent)) {
					return 0;
				}

				const BigInt determinant = left - right;
				if (determinant > 0) {
					return 1;
				}

				if (determinant < 0) {
					return -1;
				}

				return 0;
			}

			Orientation orientation(const Point &a, const Point &b, const Point &c) {
				if (point_equal(a, b) || point_equal(a, c) || point_equal(b, c)) {
					return Orientation::collinear;
				}

				const double bax = b.x - a.x;
				const double bay = b.y - a.y;
				const double cax = c.x - a.x;
				const double cay = c.y - a.y;
				const double left = bax * cay;
				const double right = bay * cax;
				const double cross = left - right;
				const double error_bound = std::numeric_limits<double>::epsilon() * 16.0
					* (std::fabs(left) + std::fabs(right));

				int sign = 0;
				if (std::isfinite(cross) && std::fabs(cross) > error_bound) {
					sign = cross > 0 ? 1 : -1;
				} else {
					sign = exact_orientation_sign(a, b, c);
				}

				if (sign > 0) {
					return Orientation::left;
				}

				return sign < 0 ? Orientation::right : Orientation::collinear;
			}

			Orientation orientation_strict(const Point &a, const Point &b, const Point &c) {
				return orientation(a, b, c);
			}

		bool left_turn(const Point &a, const Point &b, const Point &c) {
			return orientation(a, b, c) == Orientation::left;
		}

		bool right_turn(const Point &a, const Point &b, const Point &c) {
			return left_turn(b, a, c);
		}

		bool on_segment(const Point &a, const Point &b, const Point &p) {
			if (orientation(a, b, p) != Orientation::collinear) {
				return false;
			}

			return std::min(a.x, b.x) <= p.x && p.x <= std::max(a.x, b.x)
				&& std::min(a.y, b.y) <= p.y && p.y <= std::max(a.y, b.y);
		}

		bool segments_intersect(const Point &a, const Point &b, const Point &c, const Point &d) {
			const Orientation o1 = orientation(a, b, c);
			const Orientation o2 = orientation(a, b, d);
			const Orientation o3 = orientation(c, d, a);
			const Orientation o4 = orientation(c, d, b);

			if (o1 != o2 && o3 != o4) {
				return true;
			}

			return (o1 == Orientation::collinear && on_segment(a, b, c))
				|| (o2 == Orientation::collinear && on_segment(a, b, d))
				|| (o3 == Orientation::collinear && on_segment(c, d, a))
				|| (o4 == Orientation::collinear && on_segment(c, d, b));
		}

		void set_valid_edge(Edge &edge,
		                    const Point &p1, const Point &p2, const Point &p3,
		                    const Point &p4, const Point &p5, const Point &p6) {
			edge.validity = EdgeValidity::not_valid;

			if (right_turn(p1, p2, p3)) {
				edge.validity = EdgeValidity::start_valid;
			}

			if (right_turn(p4, p5, p6)) {
				edge.validity = edge.validity == EdgeValidity::start_valid
					? EdgeValidity::both_valid
					: EdgeValidity::end_valid;
			}
		}

		bool segment_leaves_interior_at_endpoint(const Polygon &polygon, unsigned int endpoint, unsigned int other) {
			const size_t n = polygon.size();
			const size_t prev = endpoint == 0 ? n - 1 : endpoint - 1;
			const size_t next = (endpoint + 1) % n;

			if (right_turn(polygon[prev], polygon[endpoint], polygon[next])) {
				return right_turn(polygon[prev], polygon[endpoint], polygon[other])
					&& right_turn(polygon[other], polygon[endpoint], polygon[next]);
			}

			return right_turn(polygon[prev], polygon[endpoint], polygon[other])
				|| right_turn(polygon[other], polygon[endpoint], polygon[next]);
		}

		bool is_visible_n3(const Polygon &polygon, unsigned int i, unsigned int j) {
			const size_t n = polygon.size();
			if ((i + 1) % n == j || (j + 1) % n == i) {
				return true;
			}

			if (segment_leaves_interior_at_endpoint(polygon, i, j)
				|| segment_leaves_interior_at_endpoint(polygon, j, i)) {
				return false;
			}

			const size_t prev_i = i == 0 ? n - 1 : i - 1;
			const size_t prev_j = j == 0 ? n - 1 : j - 1;

			for (size_t e = 0; e < n; ++e) {
				if (e == i || e == prev_i || e == j || e == prev_j) {
					continue;
				}

				const size_t next_e = e == n - 1 ? 0 : e + 1;
				if (segments_intersect(polygon[i], polygon[j], polygon[e], polygon[next_e])) {
					return false;
				}
			}

			return true;
		}

		bool collinearly_visible(unsigned int edge_num1,
		                         unsigned int e_num,
		                         unsigned int edge_num2,
		                         const Matrix<Edge> &edges,
		                         const Polygon &polygon) {
			return (e_num == edge_num1 + 1 || e_num + 1 == edge_num2)
				&& edges[edge_num1][edge_num2].visible
				&& orientation(polygon[edge_num1], polygon[e_num], polygon[edge_num2]) == Orientation::collinear;
		}

		void make_collinear_vertices_visible(const Polygon &polygon, Matrix<Edge> &edges) {
			const size_t n = polygon.size();
			size_t i = n - 1;
			size_t prev_j = 0;
			size_t j = 1;
			size_t start_i = 0;

			while (i > 0 && orientation(polygon[i], polygon[prev_j], polygon[j]) == Orientation::collinear) {
				prev_j = i;
				start_i = i;
				--i;
			}

			i = 0;
			prev_j = 1;
			j = 2;
			while (j < n && orientation(polygon[i], polygon[prev_j], polygon[j]) == Orientation::collinear) {
				++i;
				++prev_j;
				++j;
			}

			for (size_t k = start_i; k != prev_j;) {
				size_t next_k = k;
				do {
					next_k = next_k == n - 1 ? 0 : next_k + 1;
					if (k < next_k) {
						edges[k][next_k].visible = true;
					} else {
						edges[next_k][k].visible = true;
					}
				} while (next_k != prev_j);
				k = k == n - 1 ? 0 : k + 1;
			}

			i = prev_j;
			while (i < n) {
				prev_j = i + 1;
				j = i + 2;
				while (j < n && orientation(polygon[i], polygon[prev_j], polygon[j]) == Orientation::collinear) {
					++j;
					++prev_j;
				}

				if (prev_j < n) {
					for (size_t k = i; k != prev_j; ++k) {
						size_t next_k = k;
						do {
							++next_k;
							edges[k][next_k].visible = true;
						} while (next_k != prev_j);
					}
				}

				i = prev_j;
			}
		}

			int decompose(unsigned int edge_num1,
		              unsigned int edge_num2,
		              const Polygon &polygon,
		              Matrix<Edge> &edges,
		              DiagonalList &diag_list);

		int best_so_far(VertexState &pivot_vertex,
		                unsigned int extension,
		                const Polygon &polygon,
		                DiagonalList &diag_list) {
			StackRecord best = pivot_vertex.best_so_far;
			if (std::getenv("OCP_DEBUG_DP") != nullptr) {
				std::cerr << "best pivot=" << pivot_vertex.vertex << " extension=" << extension
					<< " initial=(" << best.vertex << ", " << best.value << ")\n";
			}

			while (!pivot_vertex.stack_empty()) {
				StackRecord old = pivot_vertex.stack_top();
				if (std::getenv("OCP_DEBUG_DP") != nullptr) {
					std::cerr << "  old=(" << old.vertex << ", " << old.value << ") right_turn="
						<< right_turn(polygon[old.vertex], polygon[pivot_vertex.vertex], polygon[extension]) << '\n';
				}
				if (right_turn(polygon[old.vertex], polygon[pivot_vertex.vertex], polygon[extension])) {
					diag_list = best.solution;
					return best.value;
				}

				if (old.value < best.value) {
					best = old;
				}

				pivot_vertex.stack_pop();
			}

			diag_list = best.solution;
			return best.value;
		}

		void load(int current,
		          std::vector<VertexState> &v_list,
		          const Polygon &polygon,
		          Matrix<Edge> &edges) {
			DiagonalList diag_list1;
			DiagonalList diag_list2;

			for (int previous = current - 1; previous >= 0; --previous) {
				const unsigned int previous_vertex = v_list[static_cast<size_t>(previous)].vertex;
				const unsigned int current_vertex = v_list[static_cast<size_t>(current)].vertex;

				if (!edges[previous_vertex][current_vertex].is_valid()
					&& !(edges[previous_vertex][current_vertex].visible
						&& !v_list[static_cast<size_t>(previous)].stack_empty())) {
					continue;
				}

				const int num_polygons =
					decompose(previous_vertex, current_vertex, polygon, edges, diag_list1)
					+ best_so_far(v_list[static_cast<size_t>(previous)], current_vertex, polygon, diag_list2);

				diag_list1.splice(diag_list1.end(), diag_list2);
				v_list[static_cast<size_t>(current)].stack_push(previous_vertex, num_polygons, diag_list1);
				if (std::getenv("OCP_DEBUG_DP") != nullptr) {
					std::cerr << "load current=" << current_vertex << " previous=" << previous_vertex
						<< " num=" << num_polygons << " diagonals:";
					for (const auto &[a, b] : diag_list1) {
						std::cerr << " (" << a << ", " << b << ')';
					}
					std::cerr << '\n';
				}
			}
		}

		int decompose(unsigned int edge_num1,
		              unsigned int edge_num2,
		              const Polygon &polygon,
		              Matrix<Edge> &edges,
		              DiagonalList &diag_list) {
			Edge &edge = edges[edge_num1][edge_num2];
			if (edge.done) {
				diag_list = edge.solution;
				return edge.value;
			}

			const EdgeValidity old_validity = edge.validity;
			edge.validity = EdgeValidity::not_valid;

			std::vector<VertexState> v_list;
			for (unsigned int e_num = edge_num1; e_num <= edge_num2; ++e_num) {
				if ((edges[edge_num1][e_num].visible && edges[e_num][edge_num2].visible)
					|| collinearly_visible(edge_num1, e_num, edge_num2, edges, polygon)) {
					v_list.emplace_back(e_num);
				}
			}
			if (std::getenv("OCP_DEBUG_DP") != nullptr) {
				std::cerr << "decompose " << edge_num1 << '-' << edge_num2 << " v_list:";
				for (const auto &vertex : v_list) {
					std::cerr << ' ' << vertex.vertex;
				}
				std::cerr << '\n';
			}

			for (size_t v = 0; v < v_list.size(); ++v) {
				load(static_cast<int>(v), v_list, polygon, edges);
			}

			if (v_list.empty()) {
				throw std::runtime_error(
					"optimal partition DP found no visible candidate vertices for edge "
					+ std::to_string(edge_num1) + "-" + std::to_string(edge_num2)
				);
			}

			const int num_pieces = best_so_far(v_list.back(), edge_num1, polygon, diag_list) + 1;
			diag_list.push_back({edge_num1, edge_num2});
			if (std::getenv("OCP_DEBUG_DP") != nullptr) {
				std::cerr << "decompose " << edge_num1 << '-' << edge_num2 << " result=" << num_pieces
					<< " diagonals:";
				for (const auto &[a, b] : diag_list) {
					std::cerr << " (" << a << ", " << b << ')';
				}
				std::cerr << '\n';
			}

			edge.value = num_pieces;
			edge.solution = diag_list;
			edge.done = true;
			edge.validity = old_validity;
			return num_pieces;
		}

		struct PartitionGraph {
			const Polygon &polygon;
			std::vector<std::list<unsigned int>> diagonals;

			explicit PartitionGraph(const Polygon &poly)
				: polygon(poly), diagonals(poly.size()) {}

			void insert_diagonal(unsigned int a, unsigned int b) {
				if (a >= diagonals.size() || b >= diagonals.size()) {
					throw std::runtime_error("diagonal endpoint is outside polygon");
				}

				diagonals[a].push_back(b);
				diagonals[b].push_back(a);
			}

			void dump_diagonals(const char *label) const {
				if (std::getenv("OCP_DEBUG") == nullptr) {
					return;
				}

				std::cerr << label << '\n';
				for (unsigned int vertex = 0; vertex < diagonals.size(); ++vertex) {
					std::cerr << "  " << vertex << ':';
					for (const unsigned int endpoint : diagonals[vertex]) {
						std::cerr << ' ' << endpoint;
					}
					std::cerr << '\n';
				}
			}

			bool diagonal_less(unsigned int vertex, unsigned int d1, unsigned int d2) const {
				const size_t n = polygon.size();
				const unsigned int prev = vertex == 0 ? static_cast<unsigned int>(n - 1) : vertex - 1;
				const unsigned int next = (vertex + 1) % n;
				const Orientation vertex_orientation = orientation(polygon[prev], polygon[vertex], polygon[next]);
				const Orientation d1_orientation = orientation(polygon[prev], polygon[vertex], polygon[d1]);
				const Orientation d2_orientation = orientation(polygon[prev], polygon[vertex], polygon[d2]);
				const Orientation d1_to_d2 = orientation(polygon[d1], polygon[vertex], polygon[d2]);

				if (d1_orientation == d2_orientation) {
					return d1_to_d2 == Orientation::left;
				}

				if (d1_orientation == Orientation::collinear) {
					return d2_orientation == vertex_orientation;
				}

				return d1_orientation != vertex_orientation;
			}

			void sort_diagonals() {
				for (unsigned int vertex = 0; vertex < diagonals.size(); ++vertex) {
					diagonals[vertex].sort([&](unsigned int a, unsigned int b) {
						return diagonal_less(vertex, a, b);
					});
					diagonals[vertex].unique();
				}
			}

			bool cuts_reflex_angle(unsigned int vertex, unsigned int endpoint) const {
				const size_t n = polygon.size();
				unsigned int prev = vertex == 0 ? static_cast<unsigned int>(n - 1) : vertex - 1;
				unsigned int next = (vertex + 1) % n;

				auto it = diagonals[vertex].begin();
				for (; it != diagonals[vertex].end() && *it != endpoint; ++it) {
					prev = *it;
				}

				auto next_it = it;
				++next_it;
				if (next_it != diagonals[vertex].end()) {
					next = *next_it;
				}

				return left_turn(polygon[vertex], polygon[prev], polygon[next]);
			}

			bool diagonal_is_necessary(unsigned int a, unsigned int b) const {
				return cuts_reflex_angle(a, b) || cuts_reflex_angle(b, a);
			}

			void prune_diagonals() {
				for (unsigned int vertex = 0; vertex < diagonals.size(); ++vertex) {
					for (auto it = diagonals[vertex].begin(); it != diagonals[vertex].end();) {
						const unsigned int endpoint = *it;
						if (!diagonal_is_necessary(vertex, endpoint)) {
							diagonals[endpoint].remove(vertex);
							it = diagonals[vertex].erase(it);
						} else {
							++it;
						}
					}
				}
			}

			unsigned int make_polygon(unsigned int start,
			                          std::vector<size_t> &current_diag,
			                          Partition &result,
			                          size_t &recursion_budget) const {
				if (std::getenv("OCP_DEBUG_PARTITION") != nullptr) {
					std::cerr << "make_polygon start=" << start << '\n';
				}
				if (recursion_budget-- == 0) {
					throw std::runtime_error("partition traversal recursion budget exceeded");
				}

				Polygon new_polygon;
				unsigned int next = start;
				size_t steps = 0;

				do {
					if (++steps > polygon.size() + diagonals.size() + 8) {
						throw std::runtime_error("partition traversal did not close");
					}

					new_polygon.push_back(polygon[next]);
					if (std::getenv("OCP_DEBUG_PARTITION") != nullptr) {
						std::cerr << "  add " << next << '\n';
					}

					if (current_diag[next] < diagonals[next].size()) {
						auto it = diagonals[next].begin();
						std::advance(it, static_cast<long>(current_diag[next]));
						const unsigned int diag = *it;
						if (diag >= polygon.size()) {
							throw std::runtime_error("partition traversal reached invalid diagonal endpoint");
						}
						++current_diag[next];
						if (std::getenv("OCP_DEBUG_PARTITION") != nullptr) {
							std::cerr << "  diag " << next << " -> " << diag << '\n';
						}

						if (diag == start) {
							if (std::getenv("OCP_DEBUG_PARTITION") != nullptr) {
								std::cerr << "  close at " << next << '\n';
							}
							result.push_back(std::move(new_polygon));
							return next;
						}

						next = make_polygon(next, current_diag, result, recursion_budget);
					} else {
						next = (next + 1) % polygon.size();
					}
				} while (next != start);

				result.push_back(std::move(new_polygon));
				return next;
			}

			Partition partition(bool prune) {
				sort_diagonals();
				dump_diagonals("diagonals after sort");
				if (prune) {
					prune_diagonals();
					dump_diagonals("diagonals after prune");
				}

				Partition result;
				std::vector<size_t> current_diag(polygon.size(), 0);
				size_t recursion_budget = polygon.size() * polygon.size() + 16;
				make_polygon(0, current_diag, result, recursion_budget);
				return result;
			}
		};

		bool less_xy(const Point &a, const Point &b) {
			return a.x < b.x || (a.x == b.x && a.y < b.y);
		}

		int compare_x(const Point &a, const Point &b) {
			return (a.x > b.x) - (a.x < b.x);
		}

		int compare_y(const Point &a, const Point &b) {
			return (a.y > b.y) - (a.y < b.y);
		}

		bool collinear_ordered(const Point &a, const Point &b, const Point &c) {
			if (orientation(a, b, c) != Orientation::collinear) {
				return false;
			}

			return std::min(a.x, c.x) <= b.x && b.x <= std::max(a.x, c.x)
				&& std::min(a.y, c.y) <= b.y && b.y <= std::max(a.y, c.y);
		}

		bool strictly_ordered_along_line(const Point &a, const Point &b, const Point &c) {
			return collinear_ordered(a, b, c) && !point_equal(a, b) && !point_equal(b, c);
		}

		struct PointLess {
			bool operator()(const Point &a, const Point &b) const {
				return less_xy(a, b);
			}
		};

		struct PointPairLess {
			bool operator()(const std::pair<Point, Point> &a, const std::pair<Point, Point> &b) const {
				if (less_xy(a.first, b.first)) {
					return true;
				}

				if (less_xy(b.first, a.first)) {
					return false;
				}

				return less_xy(a.second, b.second);
			}
		};

		struct SegmentLessYx {
			bool operator()(const std::pair<Point, Point> &p, const std::pair<Point, Point> &q) const {
				Point p_smaller = p.first;
				Point p_larger = p.second;
				Point q_smaller = q.first;
				Point q_larger = q.second;

				if (!less_xy(p.first, p.second)) {
					std::swap(p_smaller, p_larger);
				}

				if (!less_xy(q.first, q.second)) {
					std::swap(q_smaller, q_larger);
				}

				if (compare_x(p_larger, q_smaller) < 0) {
					return true;
				}

				if (compare_x(p_larger, q_smaller) == 0) {
					const int y_comp = compare_y(p_larger, q_smaller);
					if (y_comp < 0) {
						return true;
					}
					if (y_comp > 0) {
						return false;
					}
					return true;
				}

				if (compare_x(q_larger, p_smaller) < 0) {
					return false;
				}

				if (compare_x(q_larger, p_smaller) == 0) {
					const int y_comp = compare_y(p_smaller, q_larger);
					if (y_comp < 0) {
						return true;
					}
					if (y_comp > 0) {
						return false;
					}
					return false;
				}

				if (compare_x(p_smaller, q_smaller) < 0 && compare_x(q_smaller, p_larger) < 0) {
					return left_turn(p_smaller, p_larger, q_smaller);
				}

				if (compare_x(p_smaller, q_larger) < 0 && compare_x(q_larger, p_larger) < 0) {
					return left_turn(p_smaller, p_larger, q_larger);
				}

				if (compare_x(q_smaller, p_smaller) < 0 && compare_x(p_smaller, q_larger) < 0) {
					return right_turn(q_smaller, q_larger, p_smaller);
				}

				if (compare_x(q_smaller, p_larger) < 0 && compare_x(p_larger, q_larger) < 0) {
					return right_turn(q_smaller, q_larger, p_larger);
				}

				int y_comp = compare_y(p_smaller, q_smaller);
				if (y_comp < 0) {
					return true;
				}
				if (y_comp > 0) {
					return false;
				}

				y_comp = compare_y(p_larger, q_larger);
				return y_comp < 0;
			}
		};

		struct RotationNode {
			Point point;
			int parent = -1;
			int left_sibling = -1;
			int right_sibling = -1;
			int rightmost_child = -1;
		};

		struct RotationTree {
			std::vector<RotationNode> nodes;
			int p_inf = -1;
			int p_minus_inf = -1;

			explicit RotationTree(const Polygon &polygon) {
				for (const Point &point : polygon) {
					nodes.push_back({point});
				}

				std::stable_sort(nodes.begin(), nodes.end(), [](const RotationNode &a, const RotationNode &b) {
					return less_xy(b.point, a.point);
				});

				nodes.erase(std::unique(nodes.begin(), nodes.end(), [](const RotationNode &a, const RotationNode &b) {
					return point_equal(a.point, b.point);
				}), nodes.end());

				nodes.push_back(nodes.back());
				nodes.push_back(nodes.back());
				p_inf = static_cast<int>(nodes.size()) - 1;
				p_minus_inf = p_inf - 1;

				set_rightmost_child(p_minus_inf, p_inf);
				for (int child = 0; child != p_minus_inf; ++child) {
					set_rightmost_child(child, p_minus_inf);
				}
			}

			int end() const {
				return -1;
			}

			int rightmost_point_ref() const {
				return 0;
			}

			int right_sibling(int p) const {
				return p == end() ? end() : nodes[p].right_sibling;
			}

			int left_sibling(int p) const {
				return p == end() ? end() : nodes[p].left_sibling;
			}

			int rightmost_child(int p) const {
				return p == end() ? end() : nodes[p].rightmost_child;
			}

			int parent(int p) const {
				return p == end() ? end() : nodes[p].parent;
			}

			bool parent_is_p_infinity(int p) const {
				return parent(p) == p_inf;
			}

			bool parent_is_p_minus_infinity(int p) const {
				return parent(p) == p_minus_inf;
			}

			void set_parent(int p, int q) {
				if (q == end()) {
					return;
				}
				nodes[q].parent = p;
			}

			void set_rightmost_child(int p, int q) {
				if (q == end()) {
					return;
				}

				if (p != end()) {
					nodes[p].right_sibling = end();
					if (rightmost_child(q) != end()) {
						nodes[p].left_sibling = rightmost_child(q);
						nodes[rightmost_child(q)].right_sibling = p;
					} else {
						nodes[p].left_sibling = end();
					}
					nodes[p].parent = q;
					nodes[q].rightmost_child = p;
				} else {
					nodes[q].rightmost_child = end();
				}
			}

			void set_left_sibling(int p, int q) {
				if (q == end()) {
					return;
				}

				if (p != end()) {
					if (left_sibling(q) != end()) {
						nodes[left_sibling(q)].right_sibling = p;
						nodes[p].left_sibling = left_sibling(q);
					} else {
						nodes[p].left_sibling = end();
					}
					nodes[q].left_sibling = p;
					nodes[p].right_sibling = q;
					set_parent(parent(q), p);
				} else {
					if (left_sibling(q) != end()) {
						nodes[left_sibling(q)].right_sibling = end();
					}
					nodes[q].left_sibling = end();
				}
			}

			void set_right_sibling(int p, int q) {
				if (q == end()) {
					return;
				}

				if (p != end()) {
					if (right_sibling(q) != end()) {
						nodes[right_sibling(q)].left_sibling = p;
						nodes[p].right_sibling = right_sibling(q);
					} else {
						nodes[p].right_sibling = end();
					}
					nodes[q].right_sibling = p;
					nodes[p].left_sibling = q;
					set_parent(parent(q), p);
				} else {
					if (right_sibling(q) != end()) {
						nodes[right_sibling(q)].left_sibling = end();
					}
					nodes[q].right_sibling = end();
				}
			}

			void erase(int p) {
				int sibling = right_sibling(p);
				if (sibling != end()) {
					set_left_sibling(left_sibling(p), sibling);
				}

				sibling = left_sibling(p);
				if (sibling != end()) {
					set_right_sibling(right_sibling(p), sibling);
				}

				const int p_parent = parent(p);
				if (rightmost_child(p_parent) == p) {
					set_rightmost_child(left_sibling(p), p_parent);
				}
			}
		};

		struct VisibilityGraph {
			static constexpr size_t none = std::numeric_limits<size_t>::max();

			const Polygon &polygon;
			std::set<std::pair<Point, Point>, PointPairLess> edges;
			std::map<Point, std::pair<size_t, size_t>, PointLess> vertex_map;

			explicit VisibilityGraph(const Polygon &poly)
				: polygon(poly) {
				build();
			}

			bool is_edge(size_t a, size_t b) const {
				return is_edge(polygon[a], polygon[b]);
			}

			bool is_edge(Point a, Point b) const {
				if (!less_xy(a, b)) {
					std::swap(a, b);
				}
				return edges.find({a, b}) != edges.end();
			}

			void insert_edge(Point a, Point b) {
				if (!less_xy(a, b)) {
					std::swap(a, b);
				}
				if (std::getenv("OCP_DEBUG_VIS") != nullptr) {
					std::cerr << "insert edge (" << a.x << ", " << a.y << ") ("
						<< b.x << ", " << b.y << ")\n";
				}
				edges.insert({a, b});
			}

			size_t next(size_t i) const {
				return (i + 1) % polygon.size();
			}

			size_t prev(size_t i) const {
				return i == 0 ? polygon.size() - 1 : i - 1;
			}

			bool is_next_to(size_t p, size_t q) const {
				return next(p) == q;
			}

			bool are_adjacent(size_t p, size_t q) const {
				return next(p) == q || next(q) == p;
			}

			void initialize_vertex_map() {
				std::vector<size_t> iterator_list(polygon.size());
				for (size_t i = 0; i < polygon.size(); ++i) {
					iterator_list[i] = i;
					vertex_map.insert({polygon[i], {i, none}});
				}

				std::stable_sort(iterator_list.begin(), iterator_list.end(), [&](size_t a, size_t b) {
					return less_xy(polygon[a], polygon[b]);
				});

				std::set<std::pair<Point, Point>, SegmentLessYx> ordered_edges;

				for (const size_t event : iterator_list) {
					const size_t next_endpt = next(event);
					auto edge_it = ordered_edges.lower_bound({polygon[event], polygon[next_endpt]});

					if (edge_it != ordered_edges.begin()) {
						--edge_it;
						auto vm_it = vertex_map.find(polygon[event]);
						auto vis_it = vertex_map.find(edge_it->first);

						if (!is_next_to(vis_it->second.first, event)) {
							if (less_xy(vis_it->first, vm_it->first)) {
								vm_it->second.second = next(vis_it->second.first);
							} else {
								vm_it->second.second = vis_it->second.first;
							}
						} else if (edge_it != ordered_edges.begin()) {
							--edge_it;
							if (edge_it != ordered_edges.begin()) {
								vis_it = vertex_map.find(edge_it->first);
								if (less_xy(vis_it->first, vm_it->first)) {
									vm_it->second.second = next(vis_it->second.first);
								} else {
									vm_it->second.second = vis_it->second.first;
								}
							}
						}
					}

					const size_t prev_endpt = prev(event);
					if (less_xy(polygon[event], polygon[next_endpt])) {
						ordered_edges.insert({polygon[event], polygon[next_endpt]});
					} else {
						ordered_edges.erase({polygon[event], polygon[next_endpt]});
					}

					if (less_xy(polygon[event], polygon[prev_endpt])) {
						ordered_edges.insert({polygon[prev_endpt], polygon[event]});
					} else {
						ordered_edges.erase({polygon[prev_endpt], polygon[event]});
					}
				}
			}

			bool left_turn_to_parent(int p, int q, const RotationTree &tree) const {
				if (tree.parent_is_p_infinity(q)) {
					return less_xy(tree.nodes[p].point, tree.nodes[q].point);
				}

				const Point &parent_point = tree.nodes[tree.parent(q)].point;
				if (orientation(tree.nodes[p].point, tree.nodes[q].point, parent_point) == Orientation::collinear
					&& collinear_ordered(tree.nodes[p].point, tree.nodes[q].point, parent_point)) {
					return true;
				}

				return left_turn(tree.nodes[p].point, tree.nodes[q].point, parent_point);
			}

			bool diagonal_in_interior(size_t p, size_t q) const {
				const size_t before = prev(p);
				const size_t after = next(p);

				if (right_turn(polygon[before], polygon[p], polygon[after])) {
					if (right_turn(polygon[before], polygon[p], polygon[q])
						&& right_turn(polygon[q], polygon[p], polygon[after])) {
						return false;
					}
				} else if (right_turn(polygon[before], polygon[p], polygon[q])
					|| right_turn(polygon[q], polygon[p], polygon[after])) {
					return false;
				}

				return true;
			}

			bool point_is_visible(size_t point_to_see, std::map<Point, std::pair<size_t, size_t>, PointLess>::iterator looker) const {
				const size_t vis = looker->second.second;
				const size_t next_vis = next(vis);
				const size_t prev_vis = prev(vis);
				const size_t looker_idx = looker->second.first;

				if (vis == point_to_see) {
					return true;
				}

				if ((looker_idx == prev_vis && point_to_see == next_vis)
					|| (looker_idx == next_vis && point_to_see == prev_vis)) {
					if (orientation(polygon[prev_vis], polygon[vis], polygon[next_vis]) == Orientation::collinear
						&& (collinear_ordered(looker->first, polygon[vis], polygon[point_to_see])
							|| collinear_ordered(polygon[point_to_see], polygon[vis], looker->first))) {
						return false;
					}
					return true;
				}

				if (looker_idx == prev_vis || point_to_see == prev_vis) {
					return !(orientation(polygon[vis], polygon[next_vis], looker->first)
							!= orientation(polygon[vis], polygon[next_vis], polygon[point_to_see])
						&& orientation(looker->first, polygon[point_to_see], polygon[vis])
							!= orientation(looker->first, polygon[point_to_see], polygon[next_vis]));
				}

				if (looker_idx == next_vis || point_to_see == next_vis) {
					return !(orientation(polygon[vis], polygon[prev_vis], looker->first)
							!= orientation(polygon[vis], polygon[prev_vis], polygon[point_to_see])
						&& orientation(looker->first, polygon[point_to_see], polygon[vis])
							!= orientation(looker->first, polygon[point_to_see], polygon[prev_vis]));
				}

				if (orientation(polygon[vis], polygon[next_vis], looker->first)
						!= orientation(polygon[vis], polygon[next_vis], polygon[point_to_see])
					&& orientation(looker->first, polygon[point_to_see], polygon[vis])
						!= orientation(looker->first, polygon[point_to_see], polygon[next_vis])) {
					return false;
				}

				if (orientation(polygon[vis], polygon[prev_vis], looker->first)
						!= orientation(polygon[vis], polygon[prev_vis], polygon[point_to_see])
					&& orientation(looker->first, polygon[point_to_see], polygon[vis])
						!= orientation(looker->first, polygon[point_to_see], polygon[prev_vis])) {
					return false;
				}

				return true;
			}

			void update_visibility(std::map<Point, std::pair<size_t, size_t>, PointLess>::iterator p_it,
			                       std::map<Point, std::pair<size_t, size_t>, PointLess>::iterator q_it,
			                       bool adjacent) {
				size_t prev_q = prev(q_it->second.first);
				size_t turn_q = prev_q == p_it->second.first ? next(q_it->second.first) : prev_q;

				if (adjacent) {
					if (orientation(p_it->first, q_it->first, polygon[turn_q]) == Orientation::right) {
						p_it->second.second = q_it->second.second;
					} else {
						p_it->second.second = q_it->second.first;
					}
				} else if (q_it->second.first == p_it->second.second || prev_q == p_it->second.second) {
					turn_q = next(q_it->second.first);
					if (q_it->second.second == none
						|| orientation(p_it->first, q_it->first, polygon[turn_q]) != Orientation::right) {
						p_it->second.second = q_it->second.first;
					} else {
						p_it->second.second = q_it->second.second;
					}
				} else if (p_it->second.second != none) {
					const size_t next_v_p = next(p_it->second.second);
					const Point a = polygon[p_it->second.second];
					const Point b = polygon[next_v_p];
					const Point p = p_it->first;
					const Point q = q_it->first;
					const Orientation pqa = orientation(p, q, a);
					const Orientation pqb = orientation(p, q, b);
					const Orientation abp = orientation(a, b, p);
					const Orientation abq = orientation(a, b, q);
					bool change = false;

					if (pqa == Orientation::collinear && pqb == Orientation::collinear) {
						change = collinear_ordered(p, q, a) && collinear_ordered(p, q, b);
					} else if (pqa == Orientation::collinear) {
						change = collinear_ordered(p, q, a);
					} else if (pqb == Orientation::collinear) {
						change = collinear_ordered(p, q, b);
					} else if (pqa == pqb) {
						change = true;
					} else if (abp == Orientation::collinear || abq == Orientation::collinear) {
						change = false;
					} else if (abp != abq) {
						change = false;
					} else if (pqb == Orientation::right) {
						change = abp == Orientation::right;
					} else {
						change = abp == Orientation::left;
					}

					if (change) {
						p_it->second.second = q_it->second.first;
					}
				} else {
					p_it->second.second = q_it->second.first;
				}
			}

			void update_collinear_visibility(std::map<Point, std::pair<size_t, size_t>, PointLess>::iterator p_it,
			                                 std::map<Point, std::pair<size_t, size_t>, PointLess>::iterator q_it) {
				const size_t prev_q = prev(q_it->second.first);
				const size_t next_q = next(q_it->second.first);

				if (left_turn(p_it->first, q_it->first, polygon[prev_q]) && point_is_visible(prev_q, p_it)) {
					p_it->second.second = prev_q;
				}

				if (left_turn(p_it->first, q_it->first, polygon[next_q]) && point_is_visible(next_q, p_it)) {
					p_it->second.second = next_q;
				}
			}

			void handle(int p_node, int q_node, const RotationTree &tree) {
				auto p_it = vertex_map.find(tree.nodes[p_node].point);
				auto q_it = vertex_map.find(tree.nodes[q_node].point);
				if (p_it == vertex_map.end() || q_it == vertex_map.end()) {
					return;
				}
				if (std::getenv("OCP_DEBUG_VIS") != nullptr) {
					std::cerr << "handle p_node=" << p_node << " q_node=" << q_node
						<< " p_idx=" << p_it->second.first << " q_idx=" << q_it->second.first
						<< " p=(" << p_it->first.x << ", " << p_it->first.y << ")"
						<< " q=(" << q_it->first.x << ", " << q_it->first.y << ")"
						<< " p_vis=" << (p_it->second.second == none ? -1 : static_cast<int>(p_it->second.second))
						<< " q_vis=" << (q_it->second.second == none ? -1 : static_cast<int>(q_it->second.second))
						<< '\n';
				}

				if (are_adjacent(p_it->second.first, q_it->second.first)) {
					insert_edge(tree.nodes[p_node].point, tree.nodes[q_node].point);
					update_visibility(p_it, q_it, true);
					return;
				}

				const bool interior_at_p = diagonal_in_interior(p_it->second.first, q_it->second.first);
				const bool interior_at_q = diagonal_in_interior(q_it->second.first, p_it->second.first);

				if (interior_at_p && interior_at_q) {
					if (p_it->second.second != none
						&& strictly_ordered_along_line(p_it->first, polygon[p_it->second.second], q_it->first)) {
						update_collinear_visibility(p_it, q_it);
					} else if (p_it->second.second == none || point_is_visible(q_it->second.first, p_it)) {
						insert_edge(tree.nodes[p_node].point, tree.nodes[q_node].point);
						update_visibility(p_it, q_it, false);
					}
				} else if (!interior_at_p && !interior_at_q) {
					if (p_it->second.second == none || point_is_visible(q_it->second.first, p_it)) {
						p_it->second.second = q_it->second.first;
					}
				}
			}

			void build() {
				RotationTree tree(polygon);
				initialize_vertex_map();

				std::stack<int, std::list<int>> stack;
				stack.push(tree.rightmost_point_ref());

				while (!stack.empty()) {
					const int p = stack.top();
					stack.pop();
					const int p_r = tree.right_sibling(p);
					const int q = tree.parent(p);

					if (!tree.parent_is_p_minus_infinity(p)) {
						handle(p, q, tree);
					}

					int z = tree.left_sibling(q);
					tree.erase(p);
					if (z == tree.end() || !left_turn_to_parent(p, z, tree)) {
						tree.set_left_sibling(p, q);
					} else {
						while (tree.rightmost_child(z) != tree.end()
							&& !right_turn(tree.nodes[p].point, tree.nodes[tree.rightmost_child(z)].point, tree.nodes[z].point)) {
							z = tree.rightmost_child(z);
						}

						tree.set_rightmost_child(p, z);
						if (!stack.empty() && z == stack.top()) {
							stack.pop();
						}
					}

					if (tree.left_sibling(p) == tree.end() && !tree.parent_is_p_infinity(p)) {
						stack.push(p);
					}

					if (p_r != tree.end()) {
						stack.push(p_r);
					}
				}
			}
			};

			void preprocessing(const Polygon &polygon, Matrix<Edge> &edges) {
				const size_t n = polygon.size();
				const VisibilityGraph graph(polygon);

				for (size_t i = 0; i < n; ++i) {
					const size_t prev_i = i == 0 ? n - 1 : i - 1;
					const size_t next_i = (i + 1) % n;
					const size_t next_next_i = (next_i + 1) % n;
					edges[i][i].visible = true;

					if (next_i != 0) {
						edges[i][next_i].visible = true;
						edges[i][next_i].done = true;
					}

					set_valid_edge(edges[i][next_i],
					               polygon[prev_i], polygon[i], polygon[next_i],
					               polygon[i], polygon[next_i], polygon[next_next_i]);

					for (size_t j = i + 2; j < n; ++j) {
						if (!graph.is_edge(i, j)) {
							continue;
						}

						const size_t prev_j = j - 1;
						const size_t next_j = (j + 1) % n;
						edges[i][j].visible = true;
						set_valid_edge(edges[i][j],
						               polygon[prev_i], polygon[i], polygon[next_i],
						               polygon[prev_j], polygon[j], polygon[next_j]);

						if (j == i + 2) {
							edges[i][j].value = 1;
							edges[i][j].solution.push_back({static_cast<unsigned int>(i), static_cast<unsigned int>(j)});
							edges[i][j].done = true;
						}
					}
				}

				make_collinear_vertices_visible(polygon, edges);

				if (std::getenv("OCP_DEBUG") != nullptr) {
					std::cerr << "visible edges:";
					for (size_t a = 0; a < polygon.size(); ++a) {
						for (size_t b = a + 1; b < polygon.size(); ++b) {
							if (edges[a][b].visible) {
								std::cerr << " (" << a << ", " << b << ')';
							}
						}
					}
					std::cerr << '\n';
				}
			}

		Orientation polygon_orientation_like_cgal(const Polygon &polygon) {
			size_t left = 0;
			for (size_t i = 1; i < polygon.size(); ++i) {
				if (less_xy(polygon[i], polygon[left])) {
					left = i;
				}
			}

			const size_t prev = left == 0 ? polygon.size() - 1 : left - 1;
			const size_t next = (left + 1) % polygon.size();
			const Orientation result = orientation_strict(polygon[prev], polygon[left], polygon[next]);
			if (std::getenv("OCP_DEBUG_ORIENTATION") != nullptr) {
				std::cerr << "orientation left=" << left << " prev=" << prev << " next=" << next
					<< " result=" << static_cast<int>(result)
					<< " prev_pt=(" << polygon[prev].x << ", " << polygon[prev].y << ")"
					<< " left_pt=(" << polygon[left].x << ", " << polygon[left].y << ")"
					<< " next_pt=(" << polygon[next].x << ", " << polygon[next].y << ")\n";
			}
			return result;
		}
	}

	Partition decompose_polygon(const Polygon &polygon) {
		if (polygon.empty()) {
			return {};
		}

		Polygon normalized_polygon = polygon;
		if (polygon_orientation_like_cgal(normalized_polygon) == Orientation::right) {
			std::reverse(normalized_polygon.begin() + 1, normalized_polygon.end());
		}

		if (normalized_polygon.size() < 4) {
			return {normalized_polygon};
		}

		Matrix<Edge> edges(normalized_polygon.size(), std::vector<Edge>(normalized_polygon.size()));
		preprocessing(normalized_polygon, edges);

		DiagonalList diag_list;
		decompose(0, static_cast<unsigned int>(normalized_polygon.size() - 1), normalized_polygon, edges, diag_list);

		if (!diag_list.empty()) {
			diag_list.pop_back();
		}

		PartitionGraph graph(normalized_polygon);
		for (const auto &[a, b] : diag_list) {
			graph.insert_diagonal(a, b);
		}

		if (std::getenv("OCP_DEBUG") != nullptr) {
			std::cerr << "raw diagonal list:";
			for (const auto &[a, b] : diag_list) {
				std::cerr << " (" << a << ", " << b << ')';
			}
			std::cerr << '\n';
		}

		return graph.partition(true);
	}
}
