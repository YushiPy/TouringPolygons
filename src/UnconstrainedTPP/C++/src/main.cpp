
#include <print>
#include <chrono>
#include "vector2.h"

int main() {
	std::print("Hello, World!\n");
	
	
	auto start = std::chrono::high_resolution_clock::now();
	
	Vector2 x;
	
	for (size_t i = 0; i < 100'000'000; i++) {

		auto before(Vector2(i, i));
		auto vertex = before + Vector2(2 * i, - 3 * i);
		auto after = vertex * Vector2(0.5, 0.5 * i);

		auto last = after;
		auto diff = vertex - last;

		// ray1 = vector_reflect(diff, vector_perpendicular(vector_sub(vertex, before)))
		// ray2 = vector_reflect(diff, vector_perpendicular(vector_sub(vertex, after)))
		auto ray1 = diff.reflect((vertex - before).orthogonal());
		auto ray2 = diff.reflect((vertex - after).orthogonal());

		x += ray1 + ray2;
	}

	std::println("Final x: {}", x);

	auto end = std::chrono::high_resolution_clock::now();
	auto ellaped = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::println("Time taken: {} ms", ellaped);
	
	return 0;
}