#!/usr/bin/env python3
"""Compare two solver-comparison result directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


NUMERIC_FIELDS = (
	"wall_clock_seconds",
	"measured_work_seconds",
	"convex_solver_seconds",
	"mean_seconds_per_call",
	"total_convex_calls",
	"fully_solved_runs",
	"capped_by_calls_runs",
	"capped_by_time_runs",
	"checksum",
)

PER_INSTANCE_FIELDS = (
	"calls",
	"incumbent_solves",
	"bound_solves",
	"leaf_solves",
	"visited_nodes",
	"pruned_nodes",
	"best_updates",
	"initial_length",
	"incumbent_length",
	"final_length",
	"exhausted",
	"time_limited",
	"branch_limited",
	"max_observed_branching",
	"checksum",
)


def read_comparison(path: Path) -> dict[str, dict[str, str]]:
	with (path / "comparison.csv").open(newline="") as file:
		return {row["solver"]: row for row in csv.DictReader(file)}


def read_solver_rows(path: Path, solver: str) -> dict[tuple[int, int], dict[str, str]]:
	with (path / f"{solver}.csv").open(newline="") as file:
		return {
			(int(row["case_index"]), int(row["repeat_index"])): row
			for row in csv.DictReader(file, delimiter=";")
		}


def format_ratio(before: float, after: float) -> str:
	if before == 0.0:
		return "n/a"
	return f"{after / before:.4f}x"


def format_percent_change(before: float, after: float) -> str:
	if before == 0.0:
		return "n/a"
	return f"{(after - before) / before * 100.0:+.2f}%"


def print_metric_table(before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]) -> None:
	solvers = sorted(set(before) & set(after))
	print("| Solver | Calls Same | Solver Work Before | Solver Work After | Work Ratio | Mean/Call Before | Mean/Call After | Mean Ratio | Checksum Same |")
	print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
	for solver in solvers:
		before_row = before[solver]
		after_row = after[solver]
		before_work = float(before_row["convex_solver_seconds"])
		after_work = float(after_row["convex_solver_seconds"])
		before_mean = float(before_row["mean_seconds_per_call"])
		after_mean = float(after_row["mean_seconds_per_call"])
		print(
			f"| {solver} "
			f"| {before_row['total_convex_calls'] == after_row['total_convex_calls']} "
			f"| {before_work:.6f} "
			f"| {after_work:.6f} "
			f"| {format_ratio(before_work, after_work)} ({format_percent_change(before_work, after_work)}) "
			f"| {before_mean:.12f} "
			f"| {after_mean:.12f} "
			f"| {format_ratio(before_mean, after_mean)} ({format_percent_change(before_mean, after_mean)}) "
			f"| {before_row['checksum'] == after_row['checksum']} |"
		)


def print_correctness_diff(before_dir: Path, after_dir: Path, solvers: Sequence[str]) -> None:
	print()
	print("| Solver | Common Rows | Differing Rows | First Difference |")
	print("|---|---:|---:|---|")
	for solver in solvers:
		before_rows = read_solver_rows(before_dir, solver)
		after_rows = read_solver_rows(after_dir, solver)
		keys = sorted(set(before_rows) & set(after_rows))
		first_difference = ""
		differing = 0
		for key in keys:
			field = next(
				(
					field
					for field in PER_INSTANCE_FIELDS
					if before_rows[key][field] != after_rows[key][field]
				),
				None,
			)
			if field is not None:
				differing += 1
				if not first_difference:
					first_difference = (
						f"{key}: {field} "
						f"{before_rows[key][field]} -> {after_rows[key][field]}"
					)
		print(f"| {solver} | {len(keys)} | {differing} | {first_difference or '-'} |")


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Compare two benchmark solver-comparison result directories.")
	parser.add_argument("before", type=Path)
	parser.add_argument("after", type=Path)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	before_dir = args.before.resolve()
	after_dir = args.after.resolve()
	before = read_comparison(before_dir)
	after = read_comparison(after_dir)
	solvers = sorted(set(before) & set(after))

	print(f"Before: `{before_dir}`")
	print(f"After: `{after_dir}`")
	print()
	print_metric_table(before, after)
	print_correctness_diff(before_dir, after_dir, solvers)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
