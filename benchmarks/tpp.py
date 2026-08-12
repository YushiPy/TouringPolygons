#!/usr/bin/env python3
"""Unified command-line entry point for TPP generation and benchmarking."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS_ROOT = REPO_ROOT / "benchmarks/campaigns"
GENERATOR_SOURCE = REPO_ROOT / "packages/instance-generation/source"


def load_generation_modules():
	if str(GENERATOR_SOURCE) not in sys.path:
		sys.path.insert(0, str(GENERATOR_SOURCE))

	import gen_instances
	import generate_benchmark_matrix

	return gen_instances, generate_benchmark_matrix


def resolve_campaign(value: str) -> Path:
	path = Path(value)
	if path.is_absolute() or path.parent != Path("."):
		return path.resolve()
	return (CAMPAIGNS_ROOT / path).resolve()


def print_help() -> None:
	print(
		"""usage: python3 benchmarks/tpp.py COMMAND [arguments]

Commands:
  generate ARGS...                   Generate one binary using gen_instances.py options.
  generate-matrix NAME PBF ARGS...   Create a reproducible benchmark campaign.
  run NAME ARGS...                   Benchmark all campaign inputs, resumably.
  status NAME                        Show generation and benchmark progress.
  build-suites ARGS...               Select fixed development and canonical suites.
  benchmark ARGS...                  Run the canonical algorithm benchmark.
  split ARGS...                      Split a benchmarked binary by difficulty.
  list-groups ARGS...                List groups from a difficulty split.
  run-groups ARGS...                 Benchmark selected difficulty groups.

Campaigns created from a simple NAME live under benchmarks/campaigns/NAME.
Pass a path instead of NAME to use another location. Add --help after a command
to see that command's detailed options.
"""
	)


def command_generate(argv: Sequence[str]) -> int:
	gen_instances, _ = load_generation_modules()
	return gen_instances.main(argv)


def command_generate_matrix(argv: Sequence[str]) -> int:
	if not argv or argv[0] in {"-h", "--help"}:
		print("usage: python3 benchmarks/tpp.py generate-matrix NAME INPUT.osm.pbf [generation options]\n")
		_, matrix = load_generation_modules()
		try:
			matrix.main(["--help"])
		except SystemExit as error:
			return int(error.code or 0)
		return 0
	if len(argv) < 2:
		raise SystemExit("generate-matrix requires a campaign NAME and input .osm.pbf")

	campaign = resolve_campaign(argv[0])
	input_pbf = argv[1]
	forwarded = list(argv[2:])
	campaign_file = campaign / "campaign.json"

	if campaign_file.exists() and "--dry-run" not in forwarded:
		raise SystemExit(
			f"Campaign already exists: {campaign}\n"
			"Choose another campaign name so existing inputs and results remain reproducible."
		)

	_, matrix = load_generation_modules()
	return matrix.main([
		input_pbf,
		"--output-dir", str(campaign / "inputs"),
		"--campaign-file", str(campaign_file),
		*forwarded,
	])


def command_run(argv: Sequence[str]) -> int:
	if not argv or argv[0] in {"-h", "--help"}:
		print("usage: python3 benchmarks/tpp.py run NAME [benchmark options]\n")
		import run_generated
		try:
			run_generated.main(["--help"])
		except SystemExit as error:
			return int(error.code or 0)
		return 0

	campaign = resolve_campaign(argv[0])
	campaign_file = campaign / "campaign.json"
	if not campaign_file.exists():
		raise SystemExit(f"Not a campaign (missing campaign.json): {campaign}")

	import run_generated
	return run_generated.main([
		"--input", str(campaign / "inputs"),
		"--output", str(campaign / "results"),
		"--campaign-file", str(campaign_file),
		*argv[1:],
	])


def command_status(argv: Sequence[str]) -> int:
	if len(argv) != 1 or argv[0] in {"-h", "--help"}:
		print("usage: python3 benchmarks/tpp.py status NAME")
		return 0 if argv and argv[0] in {"-h", "--help"} else 2

	campaign = resolve_campaign(argv[0])
	campaign_file = campaign / "campaign.json"
	if not campaign_file.exists():
		raise SystemExit(f"Not a campaign (missing campaign.json): {campaign}")

	import json
	data = json.loads(campaign_file.read_text())
	inputs = data.get("inputs", [])
	existing = sum((campaign / record["file"]).exists() for record in inputs)
	print(f"Campaign: {data.get('name', campaign.name)}")
	print(f"Location: {campaign}")
	print(f"Inputs:   {existing}/{len(inputs)} generated")
	print(f"Source:   {data.get('source', {}).get('pbf', '-')}")

	index_path = campaign / "results/run-index.csv"
	if not index_path.exists():
		print("Benchmark: not started")
		return 0

	with index_path.open(newline="") as file:
		rows = list(csv.DictReader(file))
	counts = Counter(row["status"] for row in rows)
	actions = Counter(row.get("action", "") for row in rows)
	print(f"Benchmark: {len(rows)} input files indexed")
	for status, count in sorted(counts.items()):
		print(f"  {status}: {count}")
	if actions.get("skipped"):
		print(f"  resumed/skipped this run: {actions['skipped']}")
	return 0


def command_legacy(command: str, argv: Sequence[str]) -> int:
	import bench
	mapping = {
		"split": "split",
		"list-groups": "list",
		"run-groups": "run",
	}
	return bench.main([mapping[command], *argv])


def command_build_suites(argv: Sequence[str]) -> int:
	import build_algorithm_suites
	return build_algorithm_suites.main(argv)


def command_benchmark(argv: Sequence[str]) -> int:
	import run_algorithm_benchmark
	return run_algorithm_benchmark.main(argv)


def main(argv: Sequence[str] | None = None) -> int:
	arguments = list(sys.argv[1:] if argv is None else argv)
	if not arguments or arguments[0] in {"-h", "--help"}:
		print_help()
		return 0

	command, rest = arguments[0], arguments[1:]
	if command == "generate":
		return command_generate(rest)
	if command == "generate-matrix":
		return command_generate_matrix(rest)
	if command == "run":
		return command_run(rest)
	if command == "status":
		return command_status(rest)
	if command == "build-suites":
		return command_build_suites(rest)
	if command == "benchmark":
		return command_benchmark(rest)
	if command in {"split", "list-groups", "run-groups"}:
		return command_legacy(command, rest)

	print(f"Unknown command: {command}\n", file=sys.stderr)
	print_help()
	return 2


if __name__ == "__main__":
	raise SystemExit(main())
