#include <optimal_convex_partition/optimal_convex_partition.h>
#include <iostream>
#include <iomanip>

int main() {
	std::cout << std::setprecision(17);
	std::size_t count;
	while (std::cin >> count) {
		optimal_convex_partition::Polygon polygon(count);
		for (auto &point : polygon) if (!(std::cin >> point.x >> point.y)) return 1;
		try {
			const auto pieces = optimal_convex_partition::decompose_polygon(polygon);
			std::cout << '[';
			for (std::size_t i = 0; i < pieces.size(); ++i) {
				if (i) std::cout << ',';
				std::cout << '[';
				for (std::size_t j = 0; j < pieces[i].size(); ++j) {
					if (j) std::cout << ',';
					std::cout << '[' << pieces[i][j].x << ',' << pieces[i][j].y << ']';
				}
				std::cout << ']';
			}
			std::cout << "]\n";
		} catch (const std::exception &error) { std::cerr << error.what(); return 1; }
	}
}
