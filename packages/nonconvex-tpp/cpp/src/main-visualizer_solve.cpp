#include "tpp/nonconvex/solver.h"
#include "vector2.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

	Vector2 read_point() {
		double x = 0.0;
		double y = 0.0;
		std::cin >> x >> y;
		return {x, y};
	}

	void write_response(const tpp::NonconvexTppSolveResult &result) {
		std::cout << std::setprecision(17);
		std::cout << "OK "
			<< (result.exact ? 1 : 0) << ' '
			<< result.calls << ' '
			<< result.seconds << ' '
			<< result.path.size() << '\n';

		for (const auto &point : result.path) {
			std::cout << point.x << ' ' << point.y << '\n';
		}
	}

}

int main() {
	try {
		const Vector2 start = read_point();
		const Vector2 target = read_point();

		size_t polygon_count = 0;
		tpp::NonconvexTppSolveOptions options;
		std::cin >> polygon_count >> options.max_calls >> options.max_seconds;

		std::vector<std::vector<Vector2>> polygons(polygon_count);
		for (auto &polygon : polygons) {
			size_t vertex_count = 0;
			std::cin >> vertex_count;
			polygon.reserve(vertex_count);

			for (size_t i = 0; i < vertex_count; i++) {
				polygon.push_back(read_point());
			}
		}

		if (!std::cin) {
			throw std::runtime_error("Invalid input.");
		}

		write_response(tpp::tpp_nonconvex_solve(start, target, polygons, options));
		return 0;
	} catch (const std::exception &error) {
		std::cout << "ERR " << error.what() << '\n';
		return 1;
	}
}
