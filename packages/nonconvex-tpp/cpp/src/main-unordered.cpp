#include "tpp/nonconvex/unordered.h"
#include <atomic>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
	std::atomic_flag interrupted = ATOMIC_FLAG_INIT;

	extern "C" void handle_interrupt(int) {
		interrupted.test_and_set(std::memory_order_relaxed);
	}

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
	std::signal(SIGINT, handle_interrupt);
	try {
		Vector2 start, target;
		size_t n;
		tpp::UnorderedTppSolveOptions options;
		bool read_initial_path = false;
		bool cycle = false;
        bool explicit_strategy = false;
		for (int i = 1; i < argc; ++i) {
			const std::string flag = argv[i];
			if (flag == "--help") {
				std::cout << "Usage: tpp-unordered [--cycle] [--cycle-optimization cache|dual|features|lazy|root|branch|one-tree|learn|memo|bound-first|dual-screen|interval|share-bounds|proposal-bound|primal-starts] [--portfolio | --portfolio-no-sharing | --search-strategy best-bound|dfs-bfs] [--threads N] [--absolute-gap N] [--relative-gap N] [--feasibility-tolerance N] [--oracle-relative-gap N] [--dive-interval N] [--endpoint-sum-root] [--detour-root] [--bidirectional-initial] [--sampled-perimeter-initial] [--convex-initial-refinement] [--initial-path] [--trace] [--oracle-capture FILE]\n"
					<< "stdin: sx sy tx ty polygon_count max_calls max_seconds, then each polygon's vertex count and coordinates. With --initial-path, append path point count and coordinates, including endpoints.\n";
				std::cout << "--cycle solves TSPN: input endpoints are ignored; output and any initial path must be closed.\n";
				return 0;
			}
			if(flag=="--oracle-capture") {
                if(++i>=argc)throw std::invalid_argument("Expected an oracle capture path.");
                options.oracle_capture_file=argv[i];continue;
            }
			if (flag == "--cycle") {cycle = true; continue;}
            if(flag=="--cycle-optimization") {
                if(++i>=argc)throw std::invalid_argument("Expected a cycle optimization.");
                const std::string mode=argv[i];
                if(mode=="cache")options.cycle_cache=true;
                else if(mode=="dual")options.cycle_dual_reuse=true;
                else if(mode=="features")options.cycle_active_features=true;
                else if(mode=="lazy")options.cycle_lazy=true;
                else if(mode=="root")options.cycle_separated_root=true;
                else if(mode=="branch")options.cycle_strong_branching=true;
                else if(mode=="one-tree")options.cycle_one_tree=true;
                else if(mode=="learn")options.cycle_learned_branching=true;
                else if(mode=="memo")options.cycle_memo=true;
                else if(mode=="bound-first")options.cycle_bound_first=true;
                else if(mode=="dual-screen")options.cycle_dual_screen=true;
                else if(mode=="interval")options.cycle_interval_certificate=true;
                else if(mode=="proposal-bound")options.cycle_proposal_bound=true;
                else if(mode=="primal-starts")options.cycle_primal_starts=true;
                else if(mode=="share-bounds")options.cycle_share_bounds=true;
                else throw std::invalid_argument("Unknown cycle optimization: "+mode);
                continue;
            }
            if(flag=="--portfolio" || flag=="--portfolio-no-sharing") {
                options.portfolio=true;
                if(flag=="--portfolio-no-sharing") options.portfolio_share_incumbents=false;
                continue;
            }
            if(flag=="--search-strategy") {
                if(++i>=argc) throw std::invalid_argument("Expected a search strategy.");
                const std::string strategy=argv[i];
                if(strategy=="best-bound") options.search_strategy=tpp::UnorderedSearchStrategy::BestBoundDive;
                else if(strategy=="dfs-bfs") options.search_strategy=tpp::UnorderedSearchStrategy::DfsBfs;
                else throw std::invalid_argument("Expected best-bound or dfs-bfs.");
                explicit_strategy=true;continue;
            }
			if (flag == "--bidirectional-initial") {
				options.bidirectional_initial_heuristic = true;
				continue;
			}
			if (flag == "--sampled-perimeter-initial") {
				options.sampled_perimeter_initial_heuristic = true;
				continue;
			}
			if (flag == "--convex-initial-refinement") {
				options.convex_initial_refinement = true;
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
			if (flag == "--threads") {
				if (++i >= argc) throw std::invalid_argument("Expected a value after --threads.");
				const std::string value = argv[i];
				if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
					throw std::invalid_argument("Invalid thread count: " + value);
				size_t parsed = 0;
				options.threads = std::stoull(value, &parsed);
				if (parsed != value.size() || options.threads == 0
					|| options.threads > static_cast<size_t>(std::numeric_limits<int>::max()))
					throw std::invalid_argument("Invalid thread count: " + value);
				continue;
			}
			if (++i >= argc) throw std::invalid_argument("Expected a value after " + flag);
			size_t parsed = 0;
			const std::string text = argv[i];
			const double value = std::stod(text, &parsed);
			if (parsed != text.size()) throw std::invalid_argument("Invalid numeric option: " + text);
			if (flag == "--absolute-gap") options.absolute_gap = value;
			else if (flag == "--relative-gap") options.relative_gap = value;
			else if (flag == "--feasibility-tolerance") options.feasibility_tolerance = value;
			else if (flag == "--oracle-relative-gap") options.oracle_relative_gap = value;
			else throw std::invalid_argument("Unknown option: " + flag);
		}
		if(options.portfolio && explicit_strategy) throw std::invalid_argument("Portfolio selects both strategies; omit --search-strategy.");
		options.stop_requested = [] { return interrupted.test(std::memory_order_relaxed); };
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
		const auto r = cycle ? tpp::tpp_nonconvex_tspn_solve(polygons, options)
			: tpp::tpp_nonconvex_unordered_solve(start, target, polygons, options);
		const char *termination[] = {"optimal", "call_limit", "time_limit", "numerical_limit", "portfolio_stopped", "interrupted"};
		std::cout << std::setprecision(17) << "{\"schema_version\":\"free_order_v1\",\"exact\":" << (r.exact ? "true" : "false")
			<< ",\"mode\":\"" << (cycle?"cycle":"path") << "\""
			<< ",\"termination\":\"" << termination[static_cast<size_t>(r.termination)] << "\""
			<< ",\"lower_bound\":" << r.lower_bound << ",\"upper_bound\":" << r.upper_bound
			<< ",\"initial_lower_bound\":"; json_double(r.initial_lower_bound);
		std::cout << ",\"initial_upper_bound\":"; json_double(r.initial_upper_bound);
		std::cout << ",\"initial_length\":"; json_double(r.initial_length);
		std::cout << ",\"initial_sampling_work_budget\":" << r.initial_sampling_work_budget
			<< ",\"initial_sampled_extra_points\":" << r.initial_sampled_extra_points
			<< ",\"initial_convex_refinement_calls\":" << r.initial_convex_refinement_calls
			<< ",\"initial_convex_refinement_seconds\":" << r.initial_convex_refinement_seconds
			<< ",\"initial_convex_refinement_improved\":" << (r.initial_convex_refinement_improved ? "true" : "false")
			<< ",\"initial_convex_refinement_time_limited\":" << (r.initial_convex_refinement_time_limited ? "true" : "false")
			<< ",\"initial_convex_refinement_error\":";
		json_string(r.initial_convex_refinement_error);
		std::cout << ",\"incumbent_length\":"; json_double(r.incumbent_length);
		std::cout << ",\"first_best_update_length\":"; json_double(r.first_best_update_length);
		std::cout << ",\"final_length\":"; json_double(r.final_length);
		std::cout << ",\"first_incumbent_seconds\":"; json_double(r.first_incumbent_seconds);
		std::cout << ",\"initial_gap_percent\":"; json_double(r.initial_gap_percent);
		std::cout << ",\"final_absolute_gap\":"; json_double(r.final_absolute_gap);
		std::cout << ",\"final_relative_gap\":"; json_double(r.final_relative_gap);
		std::cout
			<< ",\"oracle_profiled_calls\":" << r.oracle_profiled_calls
			<< ",\"oracle_max_call_seconds\":" << r.oracle_max_call_seconds
			<< ",\"oracle_fallback_call_seconds\":" << r.oracle_fallback_call_seconds
			<< ",\"seconds\":" << r.seconds << ",\"threads_per_instance\":" << r.threads
			<< ",\"calls\":" << r.calls << ",\"nodes\":" << r.nodes
			<< ",\"parallel_oracle_calls\":" << r.parallel_oracle_calls
			<< ",\"parallel_oracle_batches\":" << r.parallel_oracle_batches
			<< ",\"relaxation_calls\":" << r.relaxation_calls
			<< ",\"refinement_calls\":" << r.refinement_calls
			<< ",\"complete_order_oracle_calls\":" << r.complete_order_oracle_calls
			<< ",\"complete_piece_oracle_calls\":" << r.complete_piece_oracle_calls
			<< ",\"oracle_cutoff_calls\":" << r.oracle_cutoff_calls
			<< ",\"oracle_dual_cutoff_prunes\":" << r.oracle_dual_cutoff_prunes
			<< ",\"screened_nodes\":" << r.screened_nodes
            << ",\"one_tree_calls\":" << r.one_tree_calls
            << ",\"one_tree_cache_hits\":" << r.one_tree_cache_hits
            << ",\"one_tree_iterations\":" << r.one_tree_iterations
            << ",\"one_tree_distance_queries\":" << r.one_tree_distance_queries
            << ",\"one_tree_improvements\":" << r.one_tree_improvements
            << ",\"one_tree_seconds\":" << r.one_tree_seconds
            << ",\"learned_branch_observations\":" << r.learned_branch_observations
            << ",\"learned_branch_decisions\":" << r.learned_branch_decisions
            << ",\"learned_branch_changes\":" << r.learned_branch_changes
            << ",\"cycle_certificate_interval_uses\":" << r.cycle_certificate_interval_uses
            << ",\"cycle_proposal_calls\":" << r.cycle_proposal_calls
            << ",\"cycle_proposal_accepts\":" << r.cycle_proposal_accepts
            << ",\"cycle_primal_start_candidates\":" << r.cycle_primal_start_candidates
            << ",\"cycle_primal_start_improvements\":" << r.cycle_primal_start_improvements
            << ",\"cycle_dual_screen_children\":" << r.cycle_dual_screen_children
            << ",\"cycle_dual_screen_prunes\":" << r.cycle_dual_screen_prunes
            << ",\"cycle_dual_screen_seconds\":" << r.cycle_dual_screen_seconds
            << ",\"cycle_shared_bound_queries\":" << r.cycle_shared_bound_queries
            << ",\"cycle_shared_bound_hits\":" << r.cycle_shared_bound_hits
            << ",\"cycle_shared_bound_improvements\":" << r.cycle_shared_bound_improvements
            << ",\"cycle_shared_bound_prunes\":" << r.cycle_shared_bound_prunes
            << ",\"cycle_shared_bound_seconds\":" << r.cycle_shared_bound_seconds
            << ",\"cycle_memo_queries\":" << r.cycle_memo_queries
            << ",\"cycle_memo_repeated\":" << r.cycle_memo_repeated
            << ",\"cycle_memo_hits\":" << r.cycle_memo_hits
            << ",\"cycle_certificate_cutoff_skips\":" << r.cycle_certificate_cutoff_skips
            << ",\"cycle_initial_contact_checks\":" << r.cycle_initial_contact_checks
            << ",\"cycle_initial_contact_accepts\":" << r.cycle_initial_contact_accepts
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
			<< ",\"calls_per_expanded_node\":" << (r.nodes ? static_cast<double>(r.calls - r.initial_convex_refinement_calls) / r.nodes : 0.0)
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
			<< ",\"profile\":{\"timing_semantics\":\"preprocessing, initial_heuristic, search, and finalization are disjoint top-level phases; initial_heuristic includes heuristic_visit_check and the optional initial convex refinement; search includes oracle batch wall time, decomposition, search_visit_check, and exclusive search_maintenance; convex_oracle_seconds sums per-call elapsed time and can exceed wall time when child evaluations run concurrently; convex_oracle_wall_seconds counts each batch once; the other convex_oracle counters sum per-call work; fallback includes its long_double and extended_precision phases; visit_check is the sum across top-level phases and overlaps them\""
			<< ",\"preprocessing_seconds\":" << r.preprocessing_seconds
			<< ",\"initial_heuristic_seconds\":" << r.initial_heuristic_seconds
			<< ",\"initial_convex_refinement_seconds\":" << r.initial_convex_refinement_seconds
			<< ",\"search_seconds\":" << r.search_seconds
			<< ",\"finalization_seconds\":" << r.finalization_seconds
			<< ",\"convex_oracle_seconds\":" << r.convex_oracle_seconds
			<< ",\"convex_oracle_wall_seconds\":" << r.convex_oracle_wall_seconds
            << ",\"cycle_construction_seconds\":" << r.cycle_construction_seconds
            << ",\"cycle_certification_seconds\":" << r.cycle_certification_seconds
            << ",\"cycle_rational_recovery_seconds\":" << r.cycle_rational_recovery_seconds
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
			<< ",\"oracle_call_histogram\":[";
		for(size_t i=0;i<r.oracle_call_histogram.size();++i)std::cout << (i?",":"") << r.oracle_call_histogram[i];
		std::cout << "],\"oracle_seconds_histogram\":[";
		for(size_t i=0;i<r.oracle_seconds_histogram.size();++i)std::cout << (i?",":"") << r.oracle_seconds_histogram[i];
		std::cout << "],\"order\":[";
		for (size_t i = 0; i < r.order.size(); ++i) std::cout << (i ? "," : "") << r.order[i];
		std::cout << "],\"path\":[";
		for (size_t i = 0; i < r.path.size(); ++i) std::cout << (i ? "," : "") << '[' << r.path[i].x << ',' << r.path[i].y << ']';
		std::cout << ']';
        std::cout << ",\"portfolio_workers\":" << r.portfolio_workers
            << ",\"portfolio_winner\":";json_size(r.portfolio_winner);
        std::cout << ",\"portfolio_incumbent_publications\":" << r.portfolio_incumbent_publications
            << ",\"portfolio_incumbent_imports\":" << r.portfolio_incumbent_imports
            << ",\"portfolio_proof_seconds\":" << r.portfolio_proof_seconds
            << ",\"portfolio_join_seconds\":" << r.portfolio_join_seconds
            << ",\"portfolio_runs\":[";
        for(size_t i=0;i<r.portfolio_runs.size();++i) {
            const auto &run=r.portfolio_runs[i];
            if(i)std::cout << ',';
            std::cout << "{\"strategy\":";json_string(run.strategy);
            std::cout << ",\"calls\":" << run.calls << ",\"nodes\":" << run.nodes
                << ",\"incumbent_imports\":" << run.incumbent_imports << ",\"seconds\":" << run.seconds
                << ",\"lower_bound\":";json_double(run.lower_bound);
            std::cout << ",\"upper_bound\":";json_double(run.upper_bound);
            std::cout << ",\"termination\":";json_string(termination[static_cast<size_t>(run.termination)]);
            std::cout << ",\"error\":";json_string(run.error);std::cout << '}';
        }
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
