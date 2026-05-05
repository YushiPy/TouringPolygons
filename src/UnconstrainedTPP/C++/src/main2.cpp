
#include "vector2.h"
#include "tpp_convex_naive.h"
#include "tests.h"

int main() {

	Vector2 start(-1.0, 0.5);
	Vector2 target(3.0, 0.5);

	std::vector<std::vector<Vector2>> polygons = {
		{
			Vector2(-1.0, 2.0),
			Vector2(1.0, 1.0),
			Vector2(3.0, 2.0),
			Vector2(1.0, 3.0),
		},
		{
			Vector2(2.0, -2.5),
			Vector2(0.0, -2.0),
			Vector2(0.0, -4.0),
		},
	};

	auto path = tpp::tpp_convex_solve(start, target, polygons);
	bool is_valid = tpp::is_valid_solution(start, target, polygons, path);
}