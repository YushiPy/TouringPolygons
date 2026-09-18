#!/usr/bin/env python3
"""Export convex decompositions for the adapted German event corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/scripts"))

from benchmark_cases import read_encoded_cases  # noqa: E402


def ensure_binary(binary: Path) -> None:
	if binary.exists():
		return
	lib = ROOT / "packages/optimal-convex-partition/cpp"
	source = ROOT / "apps/benchmark-dashboard/scripts/partition_cli.cpp"
	binary.parent.mkdir(parents=True, exist_ok=True)
	subprocess.run(
		[
			"c++", "-std=c++20", "-O3", "-I", str(lib / "include"),
			str(source), str(lib / "src/optimal_convex_partition.cpp"), "-o", str(binary),
		],
		check=True,
		timeout=120,
	)


def export_partitions(suite: Path, binary: Path, output: Path) -> None:
	cases = read_encoded_cases(suite)
	ensure_binary(binary)
	polygons = [polygon for case in cases for polygon in case.polygons]
	payload = "\n".join(
		str(len(polygon)) + " " + " ".join(str(value) for point in polygon for value in point)
		for polygon in polygons
	) + "\n"
	completed = subprocess.run(
		[str(binary)], input=payload, text=True, capture_output=True, check=True, timeout=300,
	)
	partitions = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
	if len(partitions) != len(polygons):
		raise ValueError(f"Expected {len(polygons)} partitions, got {len(partitions)}")

	cursor = 0
	by_case = {}
	for case in cases:
		count = len(case.polygons)
		by_case[str(case.case_index)] = partitions[cursor:cursor + count]
		cursor += count
	data = {
		"schema_version": 1,
		"algorithm": "optimal_convex_partition::decompose_polygon",
		"suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
		"cases": by_case,
	}
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("x") as file:
		json.dump(data, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
		file.write("\n")
	print(f"Exported {len(cases)} cases and {len(polygons)} polygon partitions to {output}")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--suite", type=Path, required=True)
	parser.add_argument("--binary", type=Path, default=ROOT / ".build/dashboard-partition/partition")
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args()
	export_partitions(args.suite, args.binary, args.output)


if __name__ == "__main__":
	main()
