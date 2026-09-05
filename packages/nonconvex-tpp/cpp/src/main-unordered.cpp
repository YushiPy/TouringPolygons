#include "tpp/nonconvex/unordered.h"
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
	try {
		Vector2 start, target;
		size_t n;
		tpp::UnorderedTppSolveOptions options;
		if (!(std::cin >> start.x >> start.y >> target.x >> target.y >> n >> options.max_calls >> options.max_seconds))
			throw std::invalid_argument("Expected sx sy tx ty polygon_count max_calls max_seconds.");
		std::vector<std::vector<Vector2>> polygons(n);
		for (auto &p : polygons) {
			size_t m;
			if (!(std::cin >> m)) throw std::invalid_argument("Expected vertex count.");
			p.resize(m);
			for (auto &v : p) if (!(std::cin >> v.x >> v.y)) throw std::invalid_argument("Expected vertex coordinates.");
		}
		const auto r = tpp::tpp_nonconvex_unordered_solve(start, target, polygons, options);
		const char *termination[] = {"optimal", "call_limit", "time_limit", "numerical_limit"};
		std::cout << std::setprecision(17) << "{\"exact\":" << (r.exact ? "true" : "false")
			<< ",\"termination\":\"" << termination[static_cast<size_t>(r.termination)] << "\""
			<< ",\"lower_bound\":" << r.lower_bound << ",\"upper_bound\":" << r.upper_bound
			<< ",\"seconds\":" << r.seconds << ",\"calls\":" << r.calls << ",\"nodes\":" << r.nodes
			<< ",\"fallback_calls\":" << r.fallback_calls
			<< ",\"insertion_branches\":" << r.insertion_branches << ",\"decomposition_branches\":" << r.decomposition_branches
			<< ",\"peak_queue\":" << r.peak_queue << ",\"order\":[";
		for (size_t i = 0; i < r.order.size(); ++i) std::cout << (i ? "," : "") << r.order[i];
		std::cout << "],\"path\":[";
		for (size_t i = 0; i < r.path.size(); ++i) std::cout << (i ? "," : "") << '[' << r.path[i].x << ',' << r.path[i].y << ']';
		std::cout << "]}\n";
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
