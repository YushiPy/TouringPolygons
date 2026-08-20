#!/usr/bin/env python3
"""Run the B&B benchmark once per convex solver and compare the summaries."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import bench


DEFAULT_SUITE = bench.REPO_ROOT / "benchmarks/suites/canonical-v1.bin"
DEFAULT_OUTPUT = bench.REPO_ROOT / "benchmarks/results/solver-comparison"
TARGET = "main-bnb_workload_benchmark"


@dataclass(frozen=True)
class SolverConfig:
	name: str
	enable_gurobi: bool = False


SOLVERS = [
	SolverConfig("linear_search_lazy"),
	SolverConfig("binary_search_lazy"),
	SolverConfig("binary_search_eager"),
	SolverConfig("tan_jiang"),
	SolverConfig("gurobi", enable_gurobi=True),
]


SUMMARY_ROW_PATTERN = re.compile(r"^\| (?P<name>.+?) \| (?P<value>.+?) \|$")


def solver_names() -> list[str]:
	return [solver.name for solver in SOLVERS]


def selected_solvers(names: Sequence[str]) -> list[SolverConfig]:
	if not names:
		return list(SOLVERS)

	by_name = {solver.name: solver for solver in SOLVERS}
	missing = [name for name in names if name not in by_name]
	if missing:
		raise SystemExit(f"Unknown solver(s): {', '.join(missing)}\nChoices: {', '.join(solver_names())}")

	return [by_name[name] for name in names]


def cmake_configure_command(enable_gurobi: bool) -> list[str]:
	return [
		"cmake",
		"--preset",
		bench.BUILD_PRESET,
		f"-DTARGET={TARGET}",
		f"-DTPP_ENABLE_GUROBI={'ON' if enable_gurobi else 'OFF'}",
	]


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> int:
	print("+", " ".join(command), flush=True)
	return subprocess.run(command, cwd=bench.REPO_ROOT, env=env, check=False).returncode


def parse_summary(path: Path) -> dict[str, str]:
	result: dict[str, str] = {}
	for line in path.read_text().splitlines():
		match = SUMMARY_ROW_PATTERN.match(line.strip())
		if match:
			result[match.group("name")] = match.group("value")
	return result


def parse_leading_float(value: str) -> float:
	match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value.replace("_", ""))
	return float(match.group(0)) if match else math.nan


def parse_count(value: str) -> int:
	match = re.search(r"\d[\d_]*", value)
	return int(match.group(0).replace("_", "")) if match else 0


def build_benchmark_command(args: argparse.Namespace, suite: Path, csv_output: Path, summary_output: Path) -> list[str]:
	command = [
		str(bench.TARGET_BINARY),
		str(suite),
		str(args.max_polygons),
		str(args.max_instances),
		str(args.max_calls),
		str(args.max_branching),
	]
	if args.max_seconds is not None:
		command.append(format_optional_seconds(args.max_seconds))
	command.extend([str(args.repeat_count), str(csv_output), str(summary_output)])
	return command


def format_optional_seconds(value: str) -> str:
	return "-1" if value == "-1" else str(float(value))


def write_comparison(run_dir: Path, rows: list[dict[str, str]]) -> None:
	csv_path = run_dir / "comparison.csv"
	md_path = run_dir / "comparison.md"
	fieldnames = [
		"solver",
		"status",
		"wall_clock_seconds",
		"measured_work_seconds",
		"convex_solver_seconds",
		"mean_seconds_per_call",
		"total_convex_calls",
		"fully_solved_runs",
		"capped_by_calls_runs",
		"capped_by_time_runs",
		"checksum",
		"csv_output",
		"summary_output",
	]

	with csv_path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	lines = [
		"# Convex Solver Comparison",
		"",
		f"Run directory: `{run_dir}`",
		"",
		"| Solver | Status | Wall clock | Solver work | Mean/call | Calls | Fully solved | Call capped | Time capped |",
		"|---|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for row in rows:
		lines.append(
			"| "
			f"{row['solver']} | "
			f"{row['status']} | "
			f"{row['wall_clock_seconds']} | "
			f"{row['convex_solver_seconds']} | "
			f"{row['mean_seconds_per_call']} | "
			f"{row['total_convex_calls']} | "
			f"{row['fully_solved_runs']} | "
			f"{row['capped_by_calls_runs']} | "
			f"{row['capped_by_time_runs']} |"
		)
	lines.extend([
		"",
		f"CSV: `{csv_path}`",
	])
	md_path.write_text("\n".join(lines) + "\n")

	print(f"\nComparison: {md_path}", flush=True)
	print(f"Comparison CSV: {csv_path}", flush=True)


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Compare B&B performance across convex solvers.")
	parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--solver", action="append", choices=solver_names(), help="Run only this solver. May be repeated.")
	parser.add_argument("--threads", type=int, help="Worker threads. Defaults to the benchmark binary default.")
	parser.add_argument("--max-polygons", default="-1")
	parser.add_argument("--max-instances", default="-1")
	parser.add_argument("--max-calls", default="1000000")
	parser.add_argument("--max-branching", default="-1")
	parser.add_argument("--max-seconds", help="Per-instance B&B and Gurobi solve cap. Strongly recommended when including gurobi.")
	parser.add_argument("--repeat-count", default="1")
	parser.add_argument("--no-build", action="store_true", help="Skip configure/build and reuse the current binary. Gurobi must already be compiled in if selected.")
	parser.add_argument("--keep-going", action="store_true", help="Continue after a solver command fails.")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	suite = args.suite.resolve()
	if not suite.exists():
		raise SystemExit(f"Suite does not exist: {suite}\nRun python3 benchmarks/tpp.py generate-suites first.")
	if args.threads is not None and args.threads < 1:
		raise SystemExit("--threads must be at least 1")

	solvers = selected_solvers(args.solver or [])
	if any(solver.enable_gurobi for solver in solvers) and args.max_seconds is None:
		print("Warning: gurobi selected without --max-seconds; a single SOCP call can run for a long time.", flush=True)
	enable_gurobi = any(solver.enable_gurobi for solver in solvers)

	if not args.no_build:
		configure_status = run_command(cmake_configure_command(enable_gurobi))
		if configure_status != 0:
			return configure_status
		build_status = run_command(["cmake", "--build", "--preset", bench.BUILD_PRESET])
		if build_status != 0:
			return build_status

	timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
	run_dir = args.output.resolve() / timestamp
	run_dir.mkdir(parents=True, exist_ok=True)
	rows: list[dict[str, str]] = []
	env = os.environ.copy()
	if args.threads is not None:
		env["TPP_BENCH_THREADS"] = str(args.threads)
	if args.max_seconds is not None:
		env["TPP_BENCH_MAX_SECONDS"] = format_optional_seconds(args.max_seconds)

	for solver in solvers:
		print(f"\n## {solver.name}", flush=True)
		status = "completed"
		csv_output = run_dir / f"{solver.name}.csv"
		summary_output = run_dir / f"{solver.name}.md"
		if status == "completed":
			solver_env = env.copy()
			solver_env["TPP_BENCH_SOLVER"] = solver.name
			command = build_benchmark_command(args, suite, csv_output, summary_output)
			run_status = run_command(command, env=solver_env)
			if run_status != 0:
				status = f"benchmark failed ({run_status})"

		summary = parse_summary(summary_output) if summary_output.exists() else {}
		row = {
			"solver": solver.name,
			"status": status,
			"wall_clock_seconds": f"{parse_leading_float(summary.get('Wall-clock total', 'nan')):.6f}",
			"measured_work_seconds": f"{parse_leading_float(summary.get('Measured work', 'nan')):.6f}",
			"convex_solver_seconds": f"{parse_leading_float(summary.get('Convex solver', 'nan')):.6f}",
			"mean_seconds_per_call": f"{parse_leading_float(summary.get('Mean seconds per call', 'nan')):.12f}",
			"total_convex_calls": str(parse_count(summary.get('Total convex calls', '0'))),
			"fully_solved_runs": str(parse_count(summary.get('Fully solved runs', '0'))),
			"capped_by_calls_runs": str(parse_count(summary.get('Capped by calls runs', '0'))),
			"capped_by_time_runs": str(parse_count(summary.get('Capped by time runs', '0'))),
			"checksum": summary.get("Checksum", ""),
			"csv_output": str(csv_output),
			"summary_output": str(summary_output),
		}
		rows.append(row)

		if status != "completed" and not args.keep_going:
			write_comparison(run_dir, rows)
			return 1

	write_comparison(run_dir, rows)
	return 0 if all(row["status"] == "completed" for row in rows) else 1


if __name__ == "__main__":
	raise SystemExit(main())
