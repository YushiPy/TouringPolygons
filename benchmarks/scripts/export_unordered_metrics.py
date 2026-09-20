#!/usr/bin/env python3
"""Export free-order JSONL results as a semicolon-delimited metrics CSV.

The JSONL emitted by ``tpp-unordered`` is the canonical artifact.  This
exporter provides a stable, analysis-friendly flat table without pretending
that fixed-order metrics such as piece-graph dominance exist in the free-order
solver.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROFILE_FIELDS = (
    "preprocessing_seconds",
    "initial_heuristic_seconds",
    "search_seconds",
    "finalization_seconds",
    "convex_oracle_seconds",
    "convex_geometric_solver_seconds",
    "convex_certificate_verification_seconds",
    "convex_contact_materialization_seconds",
    "convex_fallback_seconds",
    "convex_fallback_long_double_seconds",
    "convex_fallback_extended_precision_seconds",
    "decomposition_seconds",
    "visit_check_seconds",
    "heuristic_visit_check_seconds",
    "search_visit_check_seconds",
    "finalization_visit_check_seconds",
    "search_maintenance_seconds",
)

FIELDS = [
    "schema_version",
    "source",
    "case_index",
    "repeat_index",
    "polygons",
    "polygon_vertices_total",
    "polygon_vertices_min",
    "polygon_vertices_max",
    "order_space_log2",
    "calls",
    "relaxation_calls",
    "refinement_calls",
    "complete_order_oracle_calls",
    "complete_piece_oracle_calls",
    "oracle_cutoff_calls",
    "nodes",
    "partial_states_created",
    "children_generated",
    "children_queued",
    "screened_nodes",
    "pruned_nodes",
    "pruned_states",
    "bound_prunes",
    "incumbent_prunes",
    "insertion_branches",
    "decomposition_branches",
    "insertion_positions_considered",
    "insertion_positions_pruned",
    "branch_events",
    "total_branching",
    "mean_branching_factor",
    "max_observed_branching",
    "sequence_depth_sum",
    "sequence_depth_samples",
    "mean_sequence_depth",
    "max_sequence_depth",
    "decomposed_polygons",
    "convex_pieces_generated",
    "convex_pieces_min",
    "convex_pieces_max",
    "incumbent_updates",
    "best_updates",
    "initial_lower_bound",
    "initial_upper_bound",
    "initial_length",
    "incumbent_length",
    "first_best_update_length",
    "final_length",
    "first_incumbent_seconds",
    "initial_gap_percent",
    "lower_bound",
    "upper_bound",
    "final_absolute_gap",
    "final_relative_gap",
    "exact",
    "termination",
    "exhausted",
    "time_limited",
    "call_limited",
    "numerical_limited",
    "seconds",
    "solver_seconds",
    "search_seconds",
    "bnb_seconds",
    "seconds_per_call",
    "calls_per_expanded_node",
    "decomposition_percent",
    "fallback_calls",
    "fallback_geometric_path_invalid_calls",
    "fallback_certificate_gap_calls",
    "fallback_locator_exception_calls",
    "fallback_nonfinite_calls",
    "fallback_contact_construction_calls",
    "fallback_membership_ordering_calls",
    "fallback_local_optimality_calls",
    "fallback_coincident_contact_calls",
    "predicate_exact_evaluations",
    "extended_precision_calls",
    "oracle_time_limit_calls",
    "repaired_geometric_path_calls",
    "peak_queue",
    "sha256",
    *PROFILE_FIELDS,
]


def read_rows(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            case = int(row["case"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSONL result") from error
        if case in seen:
            raise ValueError(f"{path}: duplicate case {case}")
        seen.add(case)
        rows.append(row)
    return sorted(rows, key=lambda row: row["case"])


def value(row: dict, key: str):
    if key in row:
        return row[key]
    return row.get("profile", {}).get(key)


def ratio(numerator: float | int | None, denominator: float | int | None):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def make_record(row: dict, source: str, repeat_index: int) -> dict:
    seconds = row.get("seconds")
    calls = row.get("calls")
    nodes = row.get("nodes")
    branch_events = row.get("branch_events")
    sequence_samples = row.get("sequence_depth_samples")
    decomposition_seconds = value(row, "decomposition_seconds")
    record = {field: value(row, field) for field in FIELDS}
    record.update({
        "schema_version": row.get("schema_version", "unknown"),
        "source": source,
        "case_index": row.get("case"),
        "repeat_index": repeat_index,
        "exhausted": row.get("termination") == "optimal",
        "time_limited": row.get("termination") == "time_limit",
        "call_limited": row.get("termination") == "call_limit",
        "numerical_limited": row.get("termination") == "numerical_limit",
        "solver_seconds": seconds,
        "search_seconds": value(row, "search_seconds"),
        "bnb_seconds": value(row, "search_seconds"),
        "seconds_per_call": ratio(seconds, calls),
        "calls_per_expanded_node": ratio(calls, nodes),
        "decomposition_percent": ratio(
            None if decomposition_seconds is None else 100 * decomposition_seconds,
            seconds,
        ),
        "mean_branching_factor": ratio(row.get("total_branching"), branch_events),
        "mean_sequence_depth": ratio(row.get("sequence_depth_sum"), sequence_samples),
    })
    return {field: record.get(field) for field in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="unordered", help="source label stored in the CSV")
    parser.add_argument("--repeat-index", type=int, default=0)
    args = parser.parse_args()

    paths = [path.resolve() for path in args.runs]
    if len(set(paths)) != len(paths):
        raise ValueError("The same JSONL file cannot be supplied twice")
    rows = []
    for path in args.runs:
        rows.extend(make_record(row, args.source, args.repeat_index)
                    for row in read_rows(path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": len(args.runs), "rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
