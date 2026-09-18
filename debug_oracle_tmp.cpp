#include "tpp/convex/certified.h"
#include <iomanip>
#include <iostream>

int main() {
	Vector2 start{0.0, 0.0};
	Vector2 target{140.95, 151.64};
	std::vector<Vector2> polygon{
		{52.96, 4.6}, {73.7, 3.21}, {87.5, 20.55}, {88.89, 40.86},
		{79.72, 59.63}, {57.92, 61.1}, {54.3, 24.35}
	};
	tpp::DynamicConvexTppWorkspace workspace;
	auto result = tpp::tpp_convex_solve_certified(
		start, target, {polygon}, workspace, 1e-9,
		std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
	std::cout << std::setprecision(17)
		<< "lower " << result.lower_bound << " upper " << result.upper_bound << "\n";
	for (auto point : result.path) std::cout << point.x << " " << point.y << "\n";
}
