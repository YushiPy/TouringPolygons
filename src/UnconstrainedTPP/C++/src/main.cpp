
#include <print>
#include <chrono>

#include "vector2.h"
#include "tpp_convex_naive.h"
#include "tests.h"

using Polygon = std::vector<Vector2>;

int main() {

	std::vector<std::tuple<Vector2, Vector2, std::vector<std::vector<Vector2>>, std::vector<Vector2>>> tests = {
	{
		Vector2(0.0, 1.0),
		Vector2(2.0, 1.0),
		{
			{
				Vector2(2.0, 3.0),
				Vector2(-1.0, 2.0),
				Vector2(2.0, 2.0),
			},
		},
		{
			Vector2(0.0, 1.0),
			Vector2(1.0, 2.0),
			Vector2(2.0, 1.0),
		}
	},
	{
		Vector2(0.0, 1.0),
		Vector2(3.0, 2.0),
		{
			{
				Vector2(2.0, 3.0),
				Vector2(-1.0, 2.0),
				Vector2(2.0, 2.0),
			},
		},
		{
			Vector2(0.0, 1.0),
			Vector2(2.0, 2.0),
			Vector2(3.0, 2.0),
		}
	},
	{
		Vector2(0.0, 1.0),
		Vector2(0.0, 3.0),
		{
			{
				Vector2(2.0, 3.0),
				Vector2(-1.0, 2.0),
				Vector2(2.0, 2.0),
			},
		},
		{
			Vector2(0.0, 1.0),
			Vector2(0.0, 3.0),
		}
	},
	{
		Vector2(0.0, 0.0),
		Vector2(-2.0, 1.0),
		{
			{
				Vector2(-1.0, 2.0),
				Vector2(0.7745461995542708, 1.0969540776602265),
				Vector2(2.0, 2.0),
				Vector2(2.0, 3.0),
				Vector2(0.01042765904450249, 3.997485680411619),
			},
		},
		{
			Vector2(0.0, 0.0),
			Vector2(-0.3877793180272373, 1.688448015291771),
			Vector2(-2.0, 1.0),
		}
	},
	{
		Vector2(-1.0, 0.5),
		Vector2(3.0, 0.5),
		{
			{
				Vector2(-1.0, 2.0),
				Vector2(1.0, 1.0),
				Vector2(3.0, 2.0),
				Vector2(1.0, 3.0),
			},
		},
		{
			Vector2(-1.0, 0.5),
			Vector2(1.0, 1.0),
			Vector2(3.0, 0.5),
		}
	},
	{
		Vector2(-1.0, 0.5),
		Vector2(3.0, 0.5),
		{
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
		},
		{
			Vector2(-1.0, 0.5),
			Vector2(0.44179104477611975, 1.2791044776119402),
			Vector2(0.9670014347202298, -2.241750358680058),
			Vector2(3.0, 0.5),
		}
	},
	{
		Vector2(4.0, 1.5),
		Vector2(-1.5, -1.0),
		{
			{
				Vector2(-1.0, 2.0),
				Vector2(1.0, 1.0),
				Vector2(3.0, 2.0),
				Vector2(1.0, 3.0),
			},
			{
				Vector2(2.0, -2.0),
				Vector2(0.0, -2.0),
				Vector2(1.0, -4.0),
			},
		},
		{
			Vector2(4.0, 1.5),
			Vector2(2.2857142857142856, 1.6428571428571428),
			Vector2(0.0, -2.0),
			Vector2(-1.5, -1.0),
		}
	},
	{
		Vector2(-2.0, 0.5),
		Vector2(-1.0, 5.5),
		{
			{
				Vector2(-1.0, 2.0),
				Vector2(1.0, 1.0),
				Vector2(3.0, 2.0),
				Vector2(1.0, 3.0),
			},
			{
				Vector2(2.0, -2.0),
				Vector2(0.0, -2.0),
				Vector2(1.0, -4.0),
			},
			{
				Vector2(-5.5892251770067904, -2.0635401226162013),
				Vector2(-4.0, -2.5),
				Vector2(-3.0, -1.5),
				Vector2(-4.0, 1.0),
				Vector2(-5.5, 1.5),
			},
			{
				Vector2(-0.5, 3.5),
				Vector2(-3.5, 5.5),
				Vector2(-3.0, 2.5),
			},
		},
		{
			Vector2(-2.0, 0.5),
			Vector2(-0.25454545454545463, 1.6272727272727272),
			Vector2(0.0, -2.0),
			Vector2(-3.417422867513612, -0.45644283121597096),
			Vector2(-1.0, 5.5),
		}
	},
	{
		Vector2(-2.0, 0.0),
		Vector2(2.0, 0.0),
		{
			{
				Vector2(-1.0, 1.0),
				Vector2(1.0, 1.0),
				Vector2(0.0, 2.0),
			},
		},
		{
			Vector2(-2.0, 0.0),
			Vector2(0.0, 1.0),
			Vector2(2.0, 0.0),
		}
	},
	{
		Vector2(-1.0, 5.0),
		Vector2(-0.5, 4.5),
		{
			{
				Vector2(0.0, 4.0),
				Vector2(0.0, 2.0),
				Vector2(2.0, 2.0),
			},
		},
		{
			Vector2(-1.0, 5.0),
			Vector2(0.0, 4.0),
			Vector2(-0.5, 4.5),
		}
	},
	{
		Vector2(-0.6, 5.2),
		Vector2(-0.6, 4.8),
		{
			{
				Vector2(0.0, 5.0),
				Vector2(0.0, 2.0),
				Vector2(2.8, 2.0),
			},
		},
		{
			Vector2(-0.6, 5.2),
			Vector2(0.0, 5.0),
			Vector2(-0.6, 4.8),
		}
	},
	{
		Vector2(4.5, 2.0),
		Vector2(3.5, 2.0),
		{
			{
				Vector2(0.0, 5.0),
				Vector2(0.0, 2.0),
				Vector2(2.8, 2.0),
			},
		},
		{
			Vector2(4.5, 2.0),
			Vector2(2.8, 2.0),
			Vector2(3.5, 2.0),
		}
	},
	{
		Vector2(0.0, 4.0),
		Vector2(4.0, 0.0),
		{
			{
				Vector2(3.0, 3.0),
				Vector2(1.0, 3.0),
				Vector2(1.0, 1.0),
				Vector2(3.0, 1.0),
			},
		},
		{
			Vector2(0.0, 4.0),
			Vector2(4.0, 0.0),
		}
	},
	{
		Vector2(-1.7926052592116688, 2.0041878360341565),
		Vector2(-1.2727245469438635, -2.1926283895019223),
		{
			{
				Vector2(-2.8043080390114095, -3.654186702016027),
				Vector2(-1.7541858929206575, -5.033836318424196),
				Vector2(-1.0844353498351025, -3.4345790546288875),
			},
			{
				Vector2(2.0975681598044416, 2.0395046385241002),
				Vector2(3.094175815858267, 3.1866802982459888),
				Vector2(1.9470001561363786, 4.183287954299814),
				Vector2(0.950392500082553, 3.0361122945779258),
			},
		},
		{
			Vector2(-1.7926052592116688, 2.0041878360341565),
			Vector2(-1.0844353498351025, -3.4345790546288875),
			Vector2(2.0975681598044416, 2.0395046385241002),
			Vector2(-1.2727245469438635, -2.1926283895019223),
		}
	},
	{
		Vector2(-1.0, 0.0),
		Vector2(-1.0, 2.8000000000000003),
		{
			{
				Vector2(-1.0, 3.0),
				Vector2(-1.0, 2.0),
				Vector2(2.0, 2.0),
				Vector2(0.2, 3.0),
			},
		},
		{
			Vector2(-1.0, 0.0),
			Vector2(-1.0, 2.8000000000000003),
		}
	},
	{
		Vector2(-1.0, 1.0),
		Vector2(-1.0, 2.6),
		{
			{
				Vector2(-1.0, 3.0),
				Vector2(-1.0, 2.0),
				Vector2(2.0, 2.0),
				Vector2(2.0, 3.0),
			},
		},
		{
			Vector2(-1.0, 1.0),
			Vector2(-1.0, 2.6),
		}
	},
	{
		Vector2(-2.0, 2.0),
		Vector2(16.0, 2.0),
		{
			{
				Vector2(2.0, 2.0),
				Vector2(0.0, 4.0),
				Vector2(0.0, 2.0),
			},
			{
				Vector2(5.0, 2.0),
				Vector2(3.0, 4.0),
				Vector2(3.0, 2.0),
			},
			{
				Vector2(8.0, 2.0),
				Vector2(6.0, 4.0),
				Vector2(6.0, 2.0),
			},
			{
				Vector2(11.0, 2.0),
				Vector2(9.0, 4.0),
				Vector2(9.0, 2.0),
			},
			{
				Vector2(14.0, 2.0),
				Vector2(12.0, 4.0),
				Vector2(12.0, 2.0),
			},
		},
		{
			Vector2(-2.0, 2.0),
			Vector2(16.0, 2.0),
		}
	},
	{
		Vector2(0.0, 0.0),
		Vector2(0.0, 1.2000000000000002),
		{
			{
				Vector2(0.0, 3.0),
				Vector2(-1.0, 2.0),
				Vector2(1.0, 2.0),
			},
		},
		{
			Vector2(0.0, 0.0),
			Vector2(0.0, 2.0),
			Vector2(0.0, 1.2000000000000002),
		}
	},
};

	for (size_t i = 0; i < tests.size(); i++) {
		
		const auto &[start, target, polygons, expected] = tests[i];

		auto start_time = std::chrono::high_resolution_clock::now();
		auto result = tpp::tpp_convex_solve(start, target, polygons);
		auto end_time = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> elapsed = end_time - start_time;

		if (!tpp::is_valid_solution(start, target, polygons, result)) {
			std::println("Test {} failed: invalid solution", i);
			//continue;
			break;
		}

		bool all_approx_equal = true;

		for (size_t j = 0; j < expected.size(); j++) {

			const auto &res = result[j];
			const auto &exp = expected[j];

			if (!res.is_equal_approx(exp)) {
				std::println("Test {} failed at index {}: expected {}, got {}", i, j, exp, res);
				all_approx_equal = false;
				break;
			}
		}

		if (!all_approx_equal) {
			
			std::println("start={}; target={}", start, target);
			for (const auto &polygon : polygons) {
				for (const auto &vertex : polygon) {
					std::print("{}, ", vertex);
				}
				std::println();
			}
			for (const auto &point : result) {
				std::print("{}, ", point);
			}
			std::println();
			for (const auto &point : expected) {
				std::print("{}, ", point);
			}
			std::println();
			break;
		}

		std::println("Test {} passed in {} seconds", i, elapsed.count());
	}
}