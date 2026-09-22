#!/usr/bin/env python3
"""Run the canonical algorithm suite and print its benchmark summary."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Sequence

import bench


DEFAULT_SUITE = bench.REPO_ROOT / "benchmarks/suites/canonical-v1.bin"
DEFAULT_OUTPUT = bench.REPO_ROOT / "benchmarks/results/suite-results"


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run the canonical TPP algorithm benchmark.")
	parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--threads", type=int, help="Worker threads. Defaults to all hardware threads.")
	parser.add_argument("--solver", help="Convex solver selected via TPP_BENCH_SOLVER.")
	parser.add_argument("--max-calls", default="1000000")
	parser.add_argument("--max-seconds", help="Per-instance B&B elapsed-time cap. Defaults to unlimited.")
	parser.add_argument("--repeat-count", default="1")
	parser.add_argument("--no-build", action="store_true")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	suite = args.suite.resolve()
	if not suite.exists():
		raise SystemExit(f"Canonical suite does not exist: {suite}\nRun python3 benchmarks/tpp.py generate-suites first.")
	if args.threads is not None and args.threads < 1:
		raise SystemExit("--threads must be at least 1")

	bench.ensure_target("main-bnb_workload_benchmark", no_build=args.no_build, enable_gurobi=args.solver == "gurobi")
	timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
	run_dir = args.output.resolve() / timestamp
	run_dir.mkdir(parents=True, exist_ok=True)
	basename = suite.stem
	csv_output = run_dir / f"{basename}.csv"
	summary_output = run_dir / f"{basename}.md"
	env = os.environ.copy()
	if args.threads is not None:
		env["TPP_BENCH_THREADS"] = str(args.threads)
	else:
		env.pop("TPP_BENCH_THREADS", None)
	if args.solver is not None:
		env["TPP_BENCH_SOLVER"] = args.solver
	command = [
		str(bench.TARGET_BINARY), str(suite), "-1", "-1", str(args.max_calls), "-1",
	]
	if args.max_seconds is not None:
		command.append(str(float(args.max_seconds)))
	command.extend([str(args.repeat_count), str(csv_output), str(summary_output)])
	print("+", " ".join(command), flush=True)
	completed = subprocess.run(command, cwd=bench.REPO_ROOT, env=env, check=False)
	if completed.returncode != 0:
		return completed.returncode

	print(f"\nResults: {run_dir}", flush=True)
	print(f"Summary: {summary_output}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
