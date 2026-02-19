
#include "vector2.h"

#include <print>
#define print(...) std::println(__VA_ARGS__)

int main() {
	auto v = Vector2(3.0, 4.0);
	print("Length {}", v.length());
	print("Normalized {}", v.normalized());
	print("Angle {}", v.angle());
	print("{}", v);
}
