#include <optimal_convex_partition/optimal_convex_partition.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>

namespace {

	struct FixturePoint {
		double x = 0.0;
		double y = 0.0;
	};

	using FixturePolygon = std::vector<FixturePoint>;
	using FixturePartition = std::vector<FixturePolygon>;

	template <class T>
	T read_value(std::ifstream &input) {
		T value {};
		input.read(reinterpret_cast<char *>(&value), sizeof(T));
		return value;
	}

	bool write_all(int fd, const void *data, size_t size) {
		const auto *bytes = static_cast<const char *>(data);
		while (size > 0) {
			const ssize_t written = write(fd, bytes, size);
			if (written <= 0) {
				return false;
			}
			bytes += written;
			size -= static_cast<size_t>(written);
		}
		return true;
	}

	bool read_all(int fd, void *data, size_t size) {
		auto *bytes = static_cast<char *>(data);
		while (size > 0) {
			const ssize_t read_count = read(fd, bytes, size);
			if (read_count <= 0) {
				return false;
			}
			bytes += read_count;
			size -= static_cast<size_t>(read_count);
		}
		return true;
	}

	template <class T>
	bool write_pipe_value(int fd, const T &value) {
		return write_all(fd, &value, sizeof(T));
	}

	template <class T>
	bool read_pipe_value(int fd, T &value) {
		return read_all(fd, &value, sizeof(T));
	}

	optimal_convex_partition::Polygon to_standalone_polygon(const FixturePolygon &polygon) {
		optimal_convex_partition::Polygon result;
		result.reserve(polygon.size());
		for (const FixturePoint &point : polygon) {
			result.push_back({point.x, point.y});
		}
		return result;
	}

	FixturePartition to_fixture_partition(const optimal_convex_partition::Partition &partition) {
		FixturePartition result;
		result.reserve(partition.size());
		for (const auto &piece : partition) {
			auto &converted_piece = result.emplace_back();
			converted_piece.reserve(piece.size());
			for (const auto &point : piece) {
				converted_piece.push_back({point.x, point.y});
			}
		}
		return result;
	}

	bool partitions_equal(const FixturePartition &expected, const FixturePartition &actual) {
		if (expected.size() != actual.size()) {
			return false;
		}

		for (size_t piece_index = 0; piece_index < expected.size(); ++piece_index) {
			if (expected[piece_index].size() != actual[piece_index].size()) {
				return false;
			}

			for (size_t vertex_index = 0; vertex_index < expected[piece_index].size(); ++vertex_index) {
				const FixturePoint &a = expected[piece_index][vertex_index];
				const FixturePoint &b = actual[piece_index][vertex_index];
				if (a.x != b.x || a.y != b.y) {
					return false;
				}
			}
		}

		return true;
	}

	void print_partition_summary(const char *label, const FixturePartition &partition) {
		std::cerr << label << " pieces=" << partition.size();
		for (const auto &piece : partition) {
			std::cerr << ' ' << piece.size();
		}
		std::cerr << '\n';
	}

	bool read_cgal_partition_from_child(int fd, FixturePartition &partition) {
		size_t piece_count = 0;
		if (!read_pipe_value(fd, piece_count)) {
			return false;
		}

		partition.resize(piece_count);
		for (FixturePolygon &piece : partition) {
			size_t vertex_count = 0;
			if (!read_pipe_value(fd, vertex_count)) {
				return false;
			}

			piece.resize(vertex_count);
			for (FixturePoint &point : piece) {
				if (!read_pipe_value(fd, point)) {
					return false;
				}
			}
		}

		return true;
	}

	bool compute_cgal_partition(const FixturePolygon &polygon,
	                            FixturePartition &partition,
	                            bool &crashed) {
		int fds[2];
		if (pipe(fds) != 0) {
			std::cerr << "pipe failed: " << std::strerror(errno) << '\n';
			return false;
		}

		const pid_t pid = fork();
		if (pid < 0) {
			std::cerr << "fork failed: " << std::strerror(errno) << '\n';
			close(fds[0]);
			close(fds[1]);
			return false;
		}

		if (pid == 0) {
			close(fds[0]);

			using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
			using Traits = CGAL::Partition_traits_2<Kernel>;
			using CgalPoint = Traits::Point_2;
			using CgalPolygon = Traits::Polygon_2;

			std::vector<CgalPoint> points;
			points.reserve(polygon.size());
			for (const FixturePoint &point : polygon) {
				points.emplace_back(point.x, point.y);
			}

			CgalPolygon cgal_polygon(points.begin(), points.end());
			if (cgal_polygon.orientation() == CGAL::CLOCKWISE) {
				cgal_polygon.reverse_orientation();
			}

			std::list<CgalPolygon> pieces;
			CGAL::optimal_convex_partition_2(
				cgal_polygon.vertices_begin(),
				cgal_polygon.vertices_end(),
				std::back_inserter(pieces)
			);

			if (!write_pipe_value(fds[1], pieces.size())) {
				_exit(3);
			}

			for (const CgalPolygon &piece : pieces) {
				if (!write_pipe_value(fds[1], piece.size())) {
					_exit(3);
				}

				for (auto it = piece.vertices_begin(); it != piece.vertices_end(); ++it) {
					const FixturePoint point {it->x(), it->y()};
					if (!write_pipe_value(fds[1], point)) {
						_exit(3);
					}
				}
			}

			close(fds[1]);
			_exit(0);
		}

		close(fds[1]);
		const bool read_ok = read_cgal_partition_from_child(fds[0], partition);
		close(fds[0]);

		int status = 0;
		waitpid(pid, &status, 0);

		crashed = WIFSIGNALED(status);
		if (crashed) {
			return true;
		}

		if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || !read_ok) {
			std::cerr << "CGAL child failed without a signal\n";
			return false;
		}

		return true;
	}
}

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cerr << "usage: " << argv[0] << " <TouringPolygons test_cases.bin>\n";
		return 2;
	}

	std::ifstream input(argv[1], std::ios::binary);
	if (!input) {
		std::cerr << "failed to open fixture file: " << argv[1] << '\n';
		return 2;
	}

	size_t checked = 0;
	size_t skipped_cgal_crashes = 0;
	size_t case_index = 0;

	while (input.peek() != EOF) {
		(void) read_value<FixturePoint>(input);
		(void) read_value<FixturePoint>(input);

		const size_t polygon_count = read_value<size_t>(input);
		std::vector<FixturePolygon> polygons(polygon_count);
		for (FixturePolygon &polygon : polygons) {
			polygon.resize(read_value<size_t>(input));
			for (FixturePoint &point : polygon) {
				point = read_value<FixturePoint>(input);
			}
		}

		const size_t solution_count = read_value<size_t>(input);
		for (size_t i = 0; i < solution_count; ++i) {
			(void) read_value<FixturePoint>(input);
		}

		for (size_t polygon_index = 0; polygon_index < polygons.size(); ++polygon_index) {
			FixturePartition cgal_partition;
			bool cgal_crashed = false;
			if (!compute_cgal_partition(polygons[polygon_index], cgal_partition, cgal_crashed)) {
				return 2;
			}

			if (cgal_crashed) {
				std::cerr
					<< "skipping CGAL crash case " << case_index
					<< ", polygon " << polygon_index << '\n';
				++skipped_cgal_crashes;
				continue;
			}

			FixturePartition standalone_partition;
			try {
				standalone_partition = to_fixture_partition(
					optimal_convex_partition::decompose_polygon(
						to_standalone_polygon(polygons[polygon_index])
					)
				);
			} catch (const std::exception &error) {
				std::cerr
					<< "standalone threw in case " << case_index
					<< ", polygon " << polygon_index
					<< ": " << error.what() << '\n';
				return 1;
			}

			if (!partitions_equal(cgal_partition, standalone_partition)) {
				std::cerr
					<< "partition mismatch in case " << case_index
					<< ", polygon " << polygon_index << '\n';
				print_partition_summary("CGAL", cgal_partition);
				print_partition_summary("standalone", standalone_partition);
				return 1;
			}

			++checked;
		}

		++case_index;
	}

	std::cout
		<< "matched " << checked
		<< " polygons, skipped " << skipped_cgal_crashes
		<< " CGAL crash polygon(s)\n";

	return 0;
}
