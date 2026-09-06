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
			<< ",\"fallback_geometric_path_invalid_calls\":" << r.fallback_geometric_path_invalid_calls
			<< ",\"fallback_certificate_gap_calls\":" << r.fallback_certificate_gap_calls
			<< ",\"extended_precision_calls\":" << r.extended_precision_calls
			<< ",\"repaired_geometric_path_calls\":" << r.repaired_geometric_path_calls
			<< ",\"insertion_branches\":" << r.insertion_branches << ",\"decomposition_branches\":" << r.decomposition_branches
			<< ",\"peak_queue\":" << r.peak_queue
			<< ",\"profile\":{\"timing_semantics\":\"preprocessing, initial_heuristic, search, and finalization are disjoint top-level phases; initial_heuristic includes heuristic_visit_check; search includes convex_oracle, decomposition, search_visit_check, and exclusive search_maintenance; convex_oracle includes its geometric, certificate, and fallback phases; fallback includes its long_double and extended_precision phases; visit_check is the sum across top-level phases and overlaps them\""
			<< ",\"preprocessing_seconds\":" << r.preprocessing_seconds
			<< ",\"initial_heuristic_seconds\":" << r.initial_heuristic_seconds
			<< ",\"search_seconds\":" << r.search_seconds
			<< ",\"finalization_seconds\":" << r.finalization_seconds
			<< ",\"convex_oracle_seconds\":" << r.convex_oracle_seconds
			<< ",\"convex_geometric_solver_seconds\":" << r.convex_geometric_solver_seconds
			<< ",\"convex_certificate_verification_seconds\":" << r.convex_certificate_verification_seconds
			<< ",\"convex_fallback_seconds\":" << r.convex_fallback_seconds
			<< ",\"convex_fallback_long_double_seconds\":" << r.convex_fallback_long_double_seconds
			<< ",\"convex_fallback_extended_precision_seconds\":" << r.convex_fallback_extended_precision_seconds
			<< ",\"decomposition_seconds\":" << r.decomposition_seconds
			<< ",\"visit_check_seconds\":" << r.visit_check_seconds
			<< ",\"heuristic_visit_check_seconds\":" << r.heuristic_visit_check_seconds
			<< ",\"search_visit_check_seconds\":" << r.search_visit_check_seconds
			<< ",\"finalization_visit_check_seconds\":" << r.finalization_visit_check_seconds
			<< ",\"search_maintenance_seconds\":" << r.search_maintenance_seconds << "}"
			<< ",\"order\":[";
		for (size_t i = 0; i < r.order.size(); ++i) std::cout << (i ? "," : "") << r.order[i];
		std::cout << "],\"path\":[";
		for (size_t i = 0; i < r.path.size(); ++i) std::cout << (i ? "," : "") << '[' << r.path[i].x << ',' << r.path[i].y << ']';
		std::cout << "]}\n";
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
