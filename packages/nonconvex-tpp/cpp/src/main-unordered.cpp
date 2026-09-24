#include "tpp/nonconvex/unordered.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
	void json_string(const std::string &value) {
		std::cout << '"';
		for (const char character : value) {
			switch (character) {
				case '"': std::cout << "\\\""; break;
				case '\\': std::cout << "\\\\"; break;
				case '\n': std::cout << "\\n"; break;
				case '\r': std::cout << "\\r"; break;
				case '\t': std::cout << "\\t"; break;
				default: std::cout << character; break;
			}
		}
		std::cout << '"';
	}

	void json_size(size_t value) {
		if (value == std::numeric_limits<size_t>::max()) std::cout << "null";
		else std::cout << value;
	}

	void json_double(double value) {
		if (std::isfinite(value)) std::cout << value;
		else std::cout << "null";
	}

	void json_sizes(const std::vector<size_t> &values) {
		std::cout << '[';
		for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
		std::cout << ']';
	}

	void json_path(const std::vector<Vector2> &path) {
		std::cout << '[';
		for (size_t i = 0; i < path.size(); ++i)
			std::cout << (i ? "," : "") << '[' << path[i].x << ',' << path[i].y << ']';
		std::cout << ']';
	}

	void json_trace_event(const tpp::UnorderedTppTraceEvent &event) {
		std::cout << "{\"kind\":";
		json_string(event.kind);
		std::cout << ",\"node\":"; json_size(event.node);
		std::cout << ",\"parent\":"; json_size(event.parent);
		std::cout << ",\"polygon\":"; json_size(event.polygon);
		std::cout << ",\"piece\":"; json_size(event.piece);
		std::cout << ",\"position\":"; json_size(event.position);
		std::cout << ",\"pass\":"; json_size(event.pass);
		std::cout << ",\"sequence\":"; json_sizes(event.sequence);
		std::cout << ",\"order\":"; json_sizes(event.order);
		std::cout << ",\"path\":"; json_path(event.path);
		std::cout << ",\"lower_bound\":"; json_double(event.lower_bound);
		std::cout << ",\"upper_bound\":"; json_double(event.upper_bound);
		std::cout << ",\"length\":"; json_double(event.length);
		std::cout << ",\"pruned\":" << (event.pruned ? "true" : "false");
		std::cout << ",\"source\":"; json_string(event.source);
		std::cout << ",\"reason\":"; json_string(event.reason);
		std::cout << '}';
	}
}

int main(int argc, char **argv) {
	try {
		Vector2 start, target;
		size_t n;
		tpp::UnorderedTppSolveOptions options;
		bool read_initial_path = false;
		for (int i = 1; i < argc; ++i) {
			const std::string flag = argv[i];
			if (flag == "--help") {
				std::cout << "Usage: tpp-unordered [--absolute-gap N] [--relative-gap N] [--oracle-relative-gap N] [--dive-interval N] [--endpoint-sum-root] [--detour-root] [--bidirectional-initial] [--initial-path] [--trace]\n"
					<< "stdin: sx sy tx ty polygon_count max_calls max_seconds, then each polygon's vertex count and coordinates. With --initial-path, append path point count and coordinates, including endpoints.\n";
				return 0;
			}
			if (flag == "--bidirectional-initial") {
				options.bidirectional_initial_heuristic = true;
				continue;
			}
			if (flag == "--endpoint-sum-root") {
				options.endpoint_sum_root = true;
				continue;
			}
			if (flag == "--detour-root") {
				options.detour_root = true;
				continue;
			}
			if (flag == "--trace") {
				options.trace = true;
				continue;
			}
			if (flag == "--initial-path") {
				if (read_initial_path) throw std::invalid_argument("Repeated --initial-path.");
				read_initial_path = true;
				continue;
			}
			if (flag == "--dive-interval") {
				if (++i >= argc) throw std::invalid_argument("Expected a value after --dive-interval.");
				const std::string value = argv[i];
				if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
					throw std::invalid_argument("Invalid dive interval: " + value);
				size_t parsed = 0;
				options.dive_interval = std::stoull(value, &parsed);
				if (parsed != value.size()) throw std::invalid_argument("Invalid dive interval: " + value);
				continue;
			}
			if (++i >= argc) throw std::invalid_argument("Expected a value after " + flag);
			size_t parsed = 0;
			const std::string text = argv[i];
			const double value = std::stod(text, &parsed);
			if (parsed != text.size()) throw std::invalid_argument("Invalid numeric option: " + text);
			if (flag == "--absolute-gap") options.absolute_gap = value;
			else if (flag == "--relative-gap") options.relative_gap = value;
			else if (flag == "--oracle-relative-gap") options.oracle_relative_gap = value;
			else throw std::invalid_argument("Unknown option: " + flag);
		}
		if (!(std::cin >> start.x >> start.y >> target.x >> target.y >> n >> options.max_calls >> options.max_seconds))
			throw std::invalid_argument("Expected sx sy tx ty polygon_count max_calls max_seconds.");
		std::vector<std::vector<Vector2>> polygons(n);
		for (auto &p : polygons) {
			size_t m;
			if (!(std::cin >> m)) throw std::invalid_argument("Expected vertex count.");
			p.resize(m);
			for (auto &v : p) if (!(std::cin >> v.x >> v.y)) throw std::invalid_argument("Expected vertex coordinates.");
		}
		if (read_initial_path) {
			size_t count;
			if (!(std::cin >> count)) throw std::invalid_argument("Expected initial path point count.");
			options.initial_path.emplace(count);
			for (auto &point : *options.initial_path)
				if (!(std::cin >> point.x >> point.y)) throw std::invalid_argument("Expected initial path coordinates.");
		}
		const auto r = tpp::tpp_nonconvex_unordered_solve(start, target, polygons, options);
		const char *termination[] = {"optimal", "call_limit", "time_limit", "numerical_limit"};
		std::cout << std::setprecision(17) << "{\"schema_version\":\"free_order_v1\",\"exact\":" << (r.exact ? "true" : "false")
			<< ",\"termination\":\"" << termination[static_cast<size_t>(r.termination)] << "\""
			<< ",\"lower_bound\":" << r.lower_bound << ",\"upper_bound\":" << r.upper_bound
			<< ",\"initial_lower_bound\":"; json_double(r.initial_lower_bound);
		std::cout << ",\"initial_upper_bound\":"; json_double(r.initial_upper_bound);
		std::cout << ",\"initial_length\":"; json_double(r.initial_length);
		std::cout << ",\"incumbent_length\":"; json_double(r.incumbent_length);
		std::cout << ",\"first_best_update_length\":"; json_double(r.first_best_update_length);
		std::cout << ",\"final_length\":"; json_double(r.final_length);
		std::cout << ",\"first_incumbent_seconds\":"; json_double(r.first_incumbent_seconds);
		std::cout << ",\"initial_gap_percent\":"; json_double(r.initial_gap_percent);
		std::cout << ",\"final_absolute_gap\":"; json_double(r.final_absolute_gap);
		std::cout << ",\"final_relative_gap\":"; json_double(r.final_relative_gap);
		std::cout
			<< ",\"seconds\":" << r.seconds << ",\"calls\":" << r.calls << ",\"nodes\":" << r.nodes
			<< ",\"relaxation_calls\":" << r.relaxation_calls
			<< ",\"refinement_calls\":" << r.refinement_calls
			<< ",\"complete_order_oracle_calls\":" << r.complete_order_oracle_calls
			<< ",\"complete_piece_oracle_calls\":" << r.complete_piece_oracle_calls
			<< ",\"oracle_cutoff_calls\":" << r.oracle_cutoff_calls
			<< ",\"oracle_dual_cutoff_prunes\":" << r.oracle_dual_cutoff_prunes
			<< ",\"screened_nodes\":" << r.screened_nodes
			<< ",\"sibling_bound_prunes\":" << r.sibling_bound_prunes
			<< ",\"partial_states_created\":" << r.partial_states_created
			<< ",\"children_generated\":" << r.children_generated
			<< ",\"children_queued\":" << r.children_queued
			<< ",\"pruned_nodes\":" << r.pruned_nodes
			<< ",\"pruned_states\":" << r.pruned_states
			<< ",\"bound_prunes\":" << r.bound_prunes
			<< ",\"incumbent_prunes\":" << r.incumbent_prunes
			<< ",\"insertion_positions_considered\":" << r.insertion_positions_considered
			<< ",\"insertion_positions_pruned\":" << r.insertion_positions_pruned
			<< ",\"branch_events\":" << r.branch_events
			<< ",\"total_branching\":" << r.total_branching
			<< ",\"max_observed_branching\":" << r.max_observed_branching
			<< ",\"max_sequence_depth\":" << r.max_sequence_depth
			<< ",\"sequence_depth_sum\":" << r.sequence_depth_sum
			<< ",\"sequence_depth_samples\":" << r.sequence_depth_samples
			<< ",\"incumbent_updates\":" << r.incumbent_updates
			<< ",\"best_updates\":" << r.best_updates
			<< ",\"decomposed_polygons\":" << r.decomposed_polygons
			<< ",\"convex_pieces_generated\":" << r.convex_pieces_generated
			<< ",\"convex_pieces_min\":"; json_size(r.convex_pieces_min);
		std::cout << ",\"convex_pieces_max\":" << r.convex_pieces_max
			<< ",\"polygon_vertices_total\":" << r.polygon_vertices_total;
		std::cout << ",\"polygon_vertices_min\":"; json_size(r.polygon_vertices_min);
		std::cout << ",\"polygon_vertices_max\":" << r.polygon_vertices_max
			<< ",\"order_space_log2\":" << r.order_space_log2
			<< ",\"mean_branching_factor\":" << (r.branch_events ? static_cast<double>(r.total_branching) / r.branch_events : 0.0)
			<< ",\"mean_sequence_depth\":" << (r.sequence_depth_samples ? static_cast<double>(r.sequence_depth_sum) / r.sequence_depth_samples : 0.0)
			<< ",\"calls_per_expanded_node\":" << (r.nodes ? static_cast<double>(r.calls) / r.nodes : 0.0)
			<< ",\"seconds_per_call\":" << (r.calls ? r.seconds / r.calls : 0.0)
			<< ",\"decomposition_percent\":" << (r.seconds ? 100.0 * r.decomposition_seconds / r.seconds : 0.0)
			<< ",\"search_percent\":" << (r.seconds ? 100.0 * r.search_seconds / r.seconds : 0.0)
			<< ",\"solver_seconds\":" << r.seconds
			<< ",\"bnb_seconds\":" << r.search_seconds
			<< ",\"fallback_calls\":" << r.fallback_calls
			<< ",\"fallback_geometric_path_invalid_calls\":" << r.fallback_geometric_path_invalid_calls
			<< ",\"fallback_certificate_gap_calls\":" << r.fallback_certificate_gap_calls
			<< ",\"fallback_locator_exception_calls\":" << r.fallback_locator_exception_calls
			<< ",\"fallback_nonfinite_calls\":" << r.fallback_nonfinite_calls
			<< ",\"fallback_contact_construction_calls\":" << r.fallback_contact_construction_calls
			<< ",\"fallback_membership_ordering_calls\":" << r.fallback_membership_ordering_calls
			<< ",\"fallback_local_optimality_calls\":" << r.fallback_local_optimality_calls
			<< ",\"fallback_coincident_contact_calls\":" << r.fallback_coincident_contact_calls
			<< ",\"predicate_exact_evaluations\":" << r.predicate_exact_evaluations
			<< ",\"extended_precision_calls\":" << r.extended_precision_calls
			<< ",\"oracle_time_limit_calls\":" << r.oracle_time_limit_calls
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
			<< ",\"convex_contact_materialization_seconds\":" << r.convex_contact_materialization_seconds
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
		std::cout << ']';
		if (options.trace) {
			std::cout << ",\"trace\":[";
			for (size_t i = 0; i < r.trace.size(); ++i) {
				if (i) std::cout << ',';
				json_trace_event(r.trace[i]);
			}
			std::cout << ']';
		}
		std::cout << "}\n";
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
