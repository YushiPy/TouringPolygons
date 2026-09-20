#!/usr/bin/env python3
"""Run the fixed-order benchmark one encoded case per subprocess.

The native workload benchmark keeps all completed case results in memory until
the end of a batch.  This wrapper makes progressive campaigns resumable and
isolates a pathological case from the rest of the batch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from benchmark_cases import read_encoded_cases, write_encoded_cases


def read_existing(path: Path) -> dict[int, dict[str, str]]:
	if not path.exists() or path.stat().st_size == 0:
		return {}
	with path.open(newline="") as file:
		return {int(row["case_index"]): row for row in csv.DictReader(file, delimiter=";")}


def write_rows(path: Path, rows: dict[int, dict[str, str]]) -> None:
	if not rows:
		return
	fieldnames = list(next(iter(rows.values())).keys())
	temporary = path.with_suffix(path.suffix + ".tmp")
	with temporary.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
		writer.writeheader()
		writer.writerows(rows[index] for index in sorted(rows))
	temporary.replace(path)


def sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as file:
		for chunk in iter(lambda: file.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def write_metadata(path: Path, metadata: dict) -> None:
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(metadata, indent=2) + "\n")
	temporary.replace(path)


def case_file(case_dir: Path, index: int, data: bytes) -> Path:
	path = case_dir / f"case-{index:04d}.bin"
	if not path.exists():
		path.write_bytes(data)
	return path


def run_case(
	index: int,
	case_dir: Path,
	case_data: bytes,
	solver: Path,
	seconds: float,
	max_calls: int,
	max_branching: int,
	optimality_tolerance_percent: float,
) -> tuple[int, dict[str, str]]:
	input_path = case_file(case_dir, index, case_data)
	output_path = case_dir / f"case-{index:04d}.csv"
	command = [
		str(solver), str(input_path), "-1", "-1", str(max_calls), str(max_branching), "1", str(output_path)
	]
	environment = os.environ.copy()
	environment["TPP_BENCH_THREADS"] = "1"
	environment["TPP_BENCH_MAX_SECONDS"] = str(seconds)
	environment["TPP_BENCH_OPTIMALITY_TOLERANCE_PERCENT"] = str(optimality_tolerance_percent)
	try:
		subprocess.run(
			command,
			env=environment,
			check=True,
			capture_output=True,
			text=True,
			timeout=max(120.0, seconds + 60.0),
		)
	except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
		detail = getattr(error, "stderr", None) or str(error)
		raise RuntimeError(f"case {index}: fixed-order benchmark failed: {detail[-1000:]}") from error
	with output_path.open(newline="") as file:
		rows = list(csv.DictReader(file, delimiter=";"))
	if len(rows) != 1:
		raise RuntimeError(f"case {index}: expected one CSV row, got {len(rows)}")
	row = rows[0]
	row["case_index"] = str(index)
	return index, row


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--suite", type=Path, required=True)
	parser.add_argument("--solver", type=Path, required=True)
	parser.add_argument("--seconds", type=float, required=True)
	parser.add_argument("--max-calls", type=int, default=10_000_000)
	parser.add_argument(
		"--optimality-tolerance-percent",
		type=float,
		default=0.0,
		help="stop when the global UB/LB gap is at most this percent; 0 disables it",
	)
	parser.add_argument(
		"--max-branching",
		type=int,
		default=-1,
		help="maximum children explored per branch; use -1 for no limit (default: -1)",
	)
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--case-list", type=Path)
	parser.add_argument("--resume", action="store_true")
	args = parser.parse_args()
	if args.seconds <= 0 or args.max_calls < 0 or args.max_branching < -1 or args.workers < 1 or args.optimality_tolerance_percent < 0:
		parser.error("seconds must be positive; calls must be nonnegative; branching must be -1 or nonnegative; workers must be positive; tolerance must be nonnegative")

	cases = read_encoded_cases(args.suite)
	if args.case_list:
		indices = [int(line) for line in args.case_list.read_text().splitlines() if line.strip()]
	else:
		indices = list(range(len(cases)))
	if len(set(indices)) != len(indices) or any(index < 0 or index >= len(cases) for index in indices):
		parser.error("case list must contain unique valid case indices")

	args.output.parent.mkdir(parents=True, exist_ok=True)
	metadata_path = args.output.with_suffix(".meta.json")
	metadata = {
		"status": "running",
		"started_at_utc": datetime.now(timezone.utc).isoformat(),
		"command": sys.argv,
		"suite": str(args.suite),
		"suite_sha256": sha256(args.suite),
		"solver": str(args.solver),
		"solver_sha256": sha256(args.solver),
		"seconds": args.seconds,
		"max_calls": args.max_calls,
		"max_branching": args.max_branching,
		"optimality_tolerance_percent": args.optimality_tolerance_percent,
		"workers": args.workers,
		"python_version": platform.python_version(),
		"platform": platform.platform(),
		"machine": platform.machine(),
	}
	write_metadata(metadata_path, metadata)
	case_dir = args.output.parent / f"{args.output.stem}-cases"
	case_dir.mkdir(parents=True, exist_ok=True)
	rows = read_existing(args.output) if args.resume else {}
	pending = [index for index in indices if index not in rows]
	print(f"Running {len(pending)} fixed-order cases with {args.workers} wrapper workers", flush=True)

	with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as executor:
		futures = {
				executor.submit(
					run_case,
				index,
				case_dir,
				cases[index].data,
				args.solver,
				args.seconds,
				args.max_calls,
				args.max_branching,
				args.optimality_tolerance_percent,
			): index
			for index in pending
		}
		for completed, future in enumerate(as_completed(futures), 1):
			index, row = future.result()
			rows[index] = row
			write_rows(args.output, rows)
			print(f"cases | {completed}/{len(pending)} | case {index} complete", flush=True)

	if set(rows) != set(indices):
		raise RuntimeError(f"incomplete output: expected {len(indices)} rows, got {len(rows)}")
	write_rows(args.output, rows)
	metadata["status"] = "complete"
	metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
	metadata["rows"] = len(rows)
	metadata["output_sha256"] = sha256(args.output)
	write_metadata(metadata_path, metadata)
	print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
	main()
