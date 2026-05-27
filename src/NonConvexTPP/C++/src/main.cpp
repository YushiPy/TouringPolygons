
#include "common.h"

int main() {

	std::vector<Vector2> polygon = {
		Vector2(0, 0),
		Vector2(1, 0),
		Vector2(0.5, 0.25),
		Vector2(0, 1)
	};

	const auto decomposition = tpp::decompose_polygon(polygon);

	for (const auto &piece : decomposition) {
		std::println("{}", piece);
	}
}