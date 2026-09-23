#include <optimal_convex_partition/optimal_convex_partition.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>

// Build-time adapter to the same C++ partition library used by the solver.
// One polygon per input line; one JSON array of convex pieces per output line.
int main() {
	std::cout << std::setprecision(17);
	std::size_t count = 0;
	while (std::cin >> count) {
		optimal_convex_partition::Polygon polygon(count);
		for (auto &point : polygon) {
			if (!(std::cin >> point.x >> point.y)) return 1;
		}
		try {
			const auto pieces = optimal_convex_partition::decompose_polygon(polygon);
			std::cout << '[';
			for (std::size_t index = 0; index < pieces.size(); ++index) {
				if (index) std::cout << ',';
				std::cout << '[';
				for (std::size_t vertex = 0; vertex < pieces[index].size(); ++vertex) {
					if (vertex) std::cout << ',';
					std::cout << '[' << pieces[index][vertex].x << ',' << pieces[index][vertex].y << ']';
				}
				std::cout << ']';
			}
			std::cout << "]\n";
		} catch (const std::exception &error) {
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
