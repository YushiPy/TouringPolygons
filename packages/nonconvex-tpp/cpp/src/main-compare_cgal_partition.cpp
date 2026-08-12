#include "common.h"
#include "tests.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>
#include <optimal_convex_partition/optimal_convex_partition.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <list>
#include <string>
#include <vector>

namespace {

	using Partition = std::vector<std::vector<Vector2>>;
	using K = CGAL::Exact_predicates_inexact_constructions_kernel;
	using Traits = CGAL::Partition_traits_2<K>;
	using CgalPoint = Traits::Point_2;
	using CgalPolygon = Traits::Polygon_2;
	using CgalPolygonList = std::list<CgalPolygon>;

	struct Mismatch {
		std::string reason;
		size_t piece_index = 0;
		size_t vertex_index = 0;
	};

	bool partitions_equal(const Partition &expected, const Partition &actual, Mismatch &mismatch) {
		if (expected.size() != actual.size()) {
			mismatch.reason = "piece count differs";
			return false;
		}

		for (size_t piece_index = 0; piece_index < expected.size(); ++piece_index) {
			const auto &expected_piece = expected[piece_index];
			const auto &actual_piece = actual[piece_index];

			if (expected_piece.size() != actual_piece.size()) {
				mismatch.reason = "piece vertex count differs";
				mismatch.piece_index = piece_index;
				return false;
			}

			for (size_t vertex_index = 0; vertex_index < expected_piece.size(); ++vertex_index) {
				if (expected_piece[vertex_index] != actual_piece[vertex_index]) {
					mismatch.reason = "vertex coordinate differs";
					mismatch.piece_index = piece_index;
					mismatch.vertex_index = vertex_index;
					return false;
				}
			}
		}

		return true;
	}

	void print_partition_summary(const char *label, const Partition &partition) {
		std::cerr << label << " pieces=" << partition.size();
		for (const auto &piece : partition) {
			std::cerr << ' ' << piece.size();
		}
		std::cerr << '\n';
	}

	void print_partition(const char *label, const Partition &partition) {
		std::cerr << label << ":\n";
		for (size_t piece_index = 0; piece_index < partition.size(); ++piece_index) {
			std::cerr << "  piece " << piece_index << ':';
			for (const auto &point : partition[piece_index]) {
				std::cerr << " (" << point.x << ", " << point.y << ')';
			}
			std::cerr << '\n';
		}
	}

	void print_polygon_vertices(const std::vector<Vector2> &polygon) {
		std::cerr << "input polygon:\n";
		for (size_t i = 0; i < polygon.size(); ++i) {
			std::cerr << "  " << i << ": (" << polygon[i].x << ", " << polygon[i].y << ")\n";
		}
	}

	Partition cgal_decompose_polygon(const std::vector<Vector2> &polygon) {
		std::vector<CgalPoint> points;
		points.reserve(polygon.size());

		for (const auto &point : polygon) {
			points.emplace_back(point.x, point.y);
		}

		CgalPolygon input(points.begin(), points.end());

		if (input.orientation() == CGAL::CLOCKWISE) {
			input.reverse_orientation();
		}

		assert(input.is_simple());

		CgalPolygonList pieces;
		CGAL::optimal_convex_partition_2(input.vertices_begin(), input.vertices_end(), std::back_inserter(pieces));

		assert(CGAL::convex_partition_is_valid_2(
			input.vertices_begin(), input.vertices_end(),
			pieces.begin(), pieces.end()
		));

		Partition result;
		result.reserve(pieces.size());

		for (const auto &piece : pieces) {
			auto &converted_piece = result.emplace_back();
			converted_piece.reserve(piece.size());

			for (auto vertex = piece.vertices_begin(); vertex != piece.vertices_end(); ++vertex) {
				converted_piece.emplace_back(vertex->x(), vertex->y());
			}
		}

		return result;
	}

	optimal_convex_partition::Polygon to_standalone_polygon(const std::vector<Vector2> &polygon) {
		optimal_convex_partition::Polygon result;
		result.reserve(polygon.size());

		for (const auto &point : polygon) {
			result.push_back({point.x, point.y});
		}

		return result;
	}

	Partition from_standalone_partition(const optimal_convex_partition::Partition &partition) {
		Partition result;
		result.reserve(partition.size());

		for (const auto &piece : partition) {
			auto &converted_piece = result.emplace_back();
			converted_piece.reserve(piece.size());

			for (const auto &point : piece) {
				converted_piece.emplace_back(point.x, point.y);
			}
		}

		return result;
	}
}

int main(int argc, char **argv) {
	const std::string filename = argc > 1
		? argv[1]
		: "benchmarks/suites/canonical-v1.bin";

	const auto test_cases = tpp::load_test_cases(filename);

	size_t polygon_count = 0;
	size_t vertex_count = 0;

	std::chrono::nanoseconds cgal_time(0);
	std::chrono::nanoseconds standalone_time(0);

	for (size_t case_index = 0; case_index < test_cases.size(); ++case_index) {
		const auto &test_case = test_cases[case_index];

		for (size_t polygon_index = 0; polygon_index < test_case.polygons.size(); ++polygon_index) {
			const auto &polygon = test_case.polygons[polygon_index];
			++polygon_count;
			vertex_count += polygon.size();

			const auto cgal_start = std::chrono::steady_clock::now();
			const auto cgal_partition = cgal_decompose_polygon(polygon);
			cgal_time += std::chrono::steady_clock::now() - cgal_start;

			Partition standalone_partition;
			try {
				const auto standalone_start = std::chrono::steady_clock::now();
				standalone_partition = from_standalone_partition(
					optimal_convex_partition::decompose_polygon(to_standalone_polygon(polygon))
				);
				standalone_time += std::chrono::steady_clock::now() - standalone_start;
			} catch (const std::exception &error) {
				std::cerr
					<< "Standalone implementation threw in case " << case_index
					<< ", polygon " << polygon_index
					<< ": " << error.what() << '\n';
				return EXIT_FAILURE;
			}

			Mismatch mismatch;
			if (!partitions_equal(cgal_partition, standalone_partition, mismatch)) {
				std::cerr
					<< "Partition mismatch in case " << case_index
					<< ", polygon " << polygon_index
					<< ": " << mismatch.reason
					<< " (piece " << mismatch.piece_index
					<< ", vertex " << mismatch.vertex_index << ")\n";

				print_partition_summary("CGAL", cgal_partition);
				print_partition_summary("standalone", standalone_partition);
				print_polygon_vertices(polygon);
				print_partition("CGAL", cgal_partition);
				print_partition("standalone", standalone_partition);
				return EXIT_FAILURE;
			}
		}
	}

	std::cout
		<< "Matched CGAL output for " << polygon_count
		<< " polygons across " << test_cases.size()
		<< " test cases (" << vertex_count << " input vertices).\n";

	std::cout
		<< "CGAL time: " << cgal_time.count() << " ns\n"
		<< "Standalone time: " << standalone_time.count() << " ns\n";

	return EXIT_SUCCESS;
}
