#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import bench


DEFAULT_INPUT = bench.REPO_ROOT / "benchmarks/campaigns"
DEFAULT_OUTPUT = bench.REPO_ROOT / "benchmarks/results/generated-runs"


@dataclass
class Result:
	input_file: Path
	status: str
	action: str
	elapsed_seconds: float
	csv_output: Path
	summary_output: Path
	log_output: Path
	completion_marker: Path


def positive_int(value: str) -> int:
	parsed = int(value)
	if parsed < 1:
		raise argparse.ArgumentTypeError("must be at least 1")
	return parsed


def discover_inputs(input_path: Path, pattern: str) -> tuple[Path, list[Path]]:
	if input_path.is_file():
		if input_path.suffix != ".bin":
			raise SystemExit(f"Input file must end in .bin: {input_path}")
		return input_path.parent, [input_path]

	if not input_path.is_dir():
		raise SystemExit(f"Input path does not exist: {input_path}")

	files = sorted(path for path in input_path.rglob(pattern) if path.is_file())
	if not files:
		raise SystemExit(f"No files matching {pattern!r} under {input_path}")
	return input_path, files


def output_paths(input_file: Path, input_root: Path, output_root: Path) -> tuple[Path, Path, Path, Path]:
	relative = input_file.relative_to(input_root).with_suffix("")
	base = output_root / relative
	return (
		base.with_suffix(".csv"),
		base.with_suffix(".md"),
		base.with_suffix(".log"),
		base.with_suffix(".done"),
	)


def completion_signature(args: argparse.Namespace, input_file: Path) -> dict:
	input_stat = input_file.stat()
	binary_stat = bench.TARGET_BINARY.stat()
	return {
		"input_size": input_stat.st_size,
		"input_mtime_ns": input_stat.st_mtime_ns,
		"benchmark_binary_mtime_ns": binary_stat.st_mtime_ns,
		"threads": args.threads,
		"solver": args.solver,
		"max_polygons": str(args.max_polygons),
		"max_instances": str(args.max_instances),
		"max_calls": str(args.max_calls),
		"max_branching": str(args.max_branching),
		"max_seconds": str(args.max_seconds),
		"repeat_count": str(args.repeat_count),
	}


def marker_matches(path: Path, signature: dict) -> bool:
	if not path.exists():
		return False
	try:
		return json.loads(path.read_text()) == signature
	except (json.JSONDecodeError, OSError):
		return False


def write_index(path: Path, results: list[Result]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as file:
		writer = csv.writer(file)
		writer.writerow([
			"input_file",
			"status",
			"action",
			"elapsed_seconds",
			"csv_output",
			"summary_output",
			"log_output",
			"completion_marker",
		])
		for result in results:
			writer.writerow([
				result.input_file,
				result.status,
				result.action,
				f"{result.elapsed_seconds:.6f}",
				result.csv_output,
				result.summary_output,
				result.log_output,
				result.completion_marker,
			])


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run the non-convex TPP benchmark over a directory of generated .bin files."
	)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="A .bin file or directory searched recursively.")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for CSV, Markdown, and log outputs.")
	parser.add_argument("--pattern", default="*.bin", help="Filename pattern used when searching the input directory.")
	parser.add_argument("--threads", type=positive_int, help="Worker threads used inside each benchmark process.")
	parser.add_argument("--solver", help="Convex solver selected via TPP_BENCH_SOLVER.")
	parser.add_argument("--max-polygons", default="-1")
	parser.add_argument("--max-instances", default="-1")
	parser.add_argument("--max-calls", default="1000000")
	parser.add_argument("--max-branching", default="-1")
	parser.add_argument("--max-seconds", help="Per-instance B&B elapsed-time cap. Defaults to unlimited.")
	parser.add_argument("--repeat-count", default="1")
	parser.add_argument(
		"--timeout",
		type=positive_int,
		help="Maximum wall-clock seconds per .bin file. By default there is no file-level timeout.",
	)
	parser.add_argument("--no-build", action="store_true", help="Use the existing benchmark binary without building it.")
	parser.add_argument("--force", action="store_true", help="Rerun inputs whose CSV and Markdown outputs already exist.")
	parser.add_argument("--campaign-file", type=Path, help=argparse.SUPPRESS)
	parser.add_argument("--dry-run", action="store_true", help="List the selected inputs without building or running them.")
	return parser


def record_campaign_run(args: argparse.Namespace, results: Sequence[Result], started_utc: str) -> None:
	if args.campaign_file is None:
		return

	path = args.campaign_file.resolve()
	data = json.loads(path.read_text())
	try:
		index_value = str((args.output.resolve() / "run-index.csv").relative_to(path.parent))
	except ValueError:
		index_value = str(args.output.resolve() / "run-index.csv")

	data.setdefault("benchmark_runs", []).append({
		"started_utc": started_utc,
		"finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
		"parameters": {
			"threads": args.threads,
			"solver": args.solver,
			"max_polygons": args.max_polygons,
			"max_instances": args.max_instances,
			"max_calls": args.max_calls,
			"max_branching": args.max_branching,
			"max_seconds": args.max_seconds,
			"repeat_count": args.repeat_count,
			"timeout_seconds": args.timeout,
		},
		"status_counts": {
			status: sum(result.status == status for result in results)
			for status in sorted({result.status for result in results})
		},
		"index": index_value,
	})
	path.write_text(json.dumps(data, indent=2) + "\n")


def run_batch(args: argparse.Namespace) -> int:
	input_root, input_files = discover_inputs(args.input.resolve(), args.pattern)
	output_root = args.output.resolve()
	started_utc = dt.datetime.now(dt.timezone.utc).isoformat()

	print(f"Found {len(input_files)} benchmark input(s) under {input_root}", flush=True)
	if args.dry_run:
		for input_file in input_files:
			print(input_file.relative_to(input_root), flush=True)
		return 0

	bench.ensure_target("main-bnb_workload_benchmark", no_build=args.no_build)
	env = os.environ.copy()
	if args.threads is not None:
		env["TPP_BENCH_THREADS"] = str(args.threads)
	if args.solver is not None:
		env["TPP_BENCH_SOLVER"] = args.solver

	results: list[Result] = []
	for input_file in input_files:
		csv_output, summary_output, log_output, completion_marker = output_paths(input_file, input_root, output_root)
		for path in (csv_output, summary_output, log_output):
			path.parent.mkdir(parents=True, exist_ok=True)
		results.append(Result(input_file, "pending", "pending", 0.0, csv_output, summary_output, log_output, completion_marker))

	index_path = output_root / "run-index.csv"
	write_index(index_path, results)

	for number, result in enumerate(results, start=1):
		input_file = result.input_file
		csv_output = result.csv_output
		summary_output = result.summary_output
		log_output = result.log_output
		completion_marker = result.completion_marker
		signature = completion_signature(args, input_file)

		if not args.force and marker_matches(completion_marker, signature) and csv_output.exists() and summary_output.exists():
			print(f"[{number}/{len(input_files)}] skip {input_file.name} (already complete)", flush=True)
			result.status = "completed"
			result.action = "skipped"
			write_index(index_path, results)
			continue

		# A previous interrupted run may have written only one output. Remove both
		# before retrying so partial or stale files cannot look complete later.
		csv_output.unlink(missing_ok=True)
		summary_output.unlink(missing_ok=True)
		completion_marker.unlink(missing_ok=True)

		command = [
			str(bench.TARGET_BINARY),
			str(input_file),
			str(args.max_polygons),
			str(args.max_instances),
			str(args.max_calls),
			str(args.max_branching),
		]
		if args.max_seconds is not None:
			command.append(str(float(args.max_seconds)))
		command.extend([str(args.repeat_count), str(csv_output), str(summary_output)])
		print(f"[{number}/{len(input_files)}] run  {input_file.name}", flush=True)
		result.action = "running"
		write_index(index_path, results)
		started = time.monotonic()
		status = "completed"
		with log_output.open("w") as log_file:
			log_file.write("+ " + " ".join(command) + "\n")
			log_file.flush()
			try:
				completed = subprocess.run(
					command,
					cwd=bench.REPO_ROOT,
					env=env,
					stdout=log_file,
					stderr=subprocess.STDOUT,
					timeout=args.timeout,
					check=False,
				)
				if completed.returncode != 0:
					status = f"failed ({completed.returncode})"
			except subprocess.TimeoutExpired:
				status = "timed out"
			except KeyboardInterrupt:
				result.status = "interrupted"
				result.action = "interrupted"
				result.elapsed_seconds = time.monotonic() - started
				write_index(index_path, results)
				record_campaign_run(args, results, started_utc)
				raise

		elapsed = time.monotonic() - started
		print(f"             {status} in {elapsed:.1f}s", flush=True)
		result.status = status
		result.action = "ran"
		result.elapsed_seconds = elapsed
		if status == "completed":
			completion_marker.write_text(json.dumps(signature, sort_keys=True) + "\n")
		write_index(index_path, results)

	completed_count = sum(result.status == "completed" for result in results)
	skipped_count = sum(result.action == "skipped" for result in results)
	failed_count = len(results) - completed_count
	record_campaign_run(args, results, started_utc)
	print(
		f"Finished: {completed_count} completed ({skipped_count} already existed), "
		f"{failed_count} failed/timed out. Index: {index_path}",
		flush=True,
	)
	return 1 if failed_count else 0


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	return run_batch(args)


if __name__ == "__main__":
	raise SystemExit(main())
