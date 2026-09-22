# Benchmark scripts

`benchmarks/tpp.py` is the public entry point. The files in this directory are
lower-level commands or importable helpers used by that entry point, by the
dashboard, or by reproducibility reports.

## Maintained workflow

These scripts are part of the current benchmark workflow and should remain
tested and importable:

- `tpp.py`, `create_synthetic_campaign.py`, `run_generated.py` — campaigns;
- `benchmark_cases.py`, `unordered_runner.py`, `unordered_validation.py` —
  shared case and solver plumbing;
- `unordered_benchmark.py`, `summarize_unordered.py`,
  `summarize_unordered_siicusp.py` — free-order runs and audits;
- `free_order_campaign.py`, `free_order_ablation.py`,
  `free_order_metamorphic.py` — free-order experiments;
- `generate_free_order_canon.py`, `summarize_free_order_canon.py`,
  `generate_algorithm_suites.py`, `build_algorithm_suites.py` — reproducible
  suite construction;
- `run_algorithm_benchmark.py`, `compare_convex_solvers.py`, `bench.py` —
  native algorithm benchmark commands, including retained legacy split/group
  commands exposed through `tpp.py`;
- `convert_instances.py`, `normalize_polygon_orientation.py`,
  `convert_tspn_native_instances.py` — input conversion;
- `tspn_run_comparison.py`, `tspn_oracle_backends.py`,
  `analyze_oracle_swap.py`, `run_fekete_6h.py` — external comparisons.

## Snapshot and report generation

These commands regenerate tracked event data or support reports whose source
data is kept outside Git:

- `analyze_german_comparison.py` and `build_siicusp34_event_data.py` — the
  frozen SIICUSP34 export;
- `export_german_event.py`, `export_unordered_metrics.py`,
  `merge_unordered_runs.py`, `merge_progressive_unordered.py` — benchmark
  exports and progressive-result consolidation;
- `run_fekete_6h.py` remains a long-running external comparison command, not a
  default sanity check.

## Removal policy

Scripts without a current caller are not automatically obsolete: a report may
use them as a reproducibility recipe. Before removing one, search code and
documentation, identify the output it produced, and preserve the conclusion or
regeneration command in `docs/research/`. The following one-off tools were
removed in this refactor because their outputs and callers were already retired:

- `build_benchmark_report.py`, which only generated the deleted `output/` PDF;
- `run_german_per_case.py`;
- `compare_solver_result_dirs.py`;
- `consolidate_german_results.py`;
- `export_german_partitions.py`, superseded by the dashboard exporter.

Do not add generated campaign results, previews, or local solver binaries to
this directory. Keep shared parsing and validation in importable modules rather
than duplicating it in another command.
