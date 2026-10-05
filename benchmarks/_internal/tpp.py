#!/usr/bin/env python3
"""Unified command-line entry point for TPP generation and benchmarking."""

from __future__ import annotations

import csv
import shlex
import shutil
import sys
import textwrap
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import workspace


def load_generation_modules():
	import gen_instances
	import generate_benchmark_matrix

	return gen_instances, generate_benchmark_matrix


def command_create(argv: Sequence[str]) -> int:
	if not argv or argv[0] in {"-h", "--help"}:
		print(
			textwrap.dedent(
				"""\
				usage: python3 benchmarks/tpp.py create NAME [options]

				Creates benchmarks/workspace/campaigns/NAME with a synthetic .bin input, preview image,
				and campaign metadata. Add --help after NAME to see generation options.

				Examples:
				  python3 benchmarks/tpp.py create smoke --vertices 8 --polygons 20 --instances 100
				  python3 benchmarks/tpp.py create varied --vertices 4,5,6,7 --polygons 4 --shape convex
				"""
			)
		)
		return 0

	name, rest = argv[0], list(argv[1:])
	campaign = workspace.campaign_path(name)
	if rest and rest[0] in {"-h", "--help"}:
		import create_synthetic_campaign
		try:
			create_synthetic_campaign.main(["--campaign", str(campaign), "--help"])
		except SystemExit as error:
			return int(error.code or 0)
		return 0

	import create_synthetic_campaign
	return create_synthetic_campaign.main(["--campaign", str(campaign), *rest])


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

	campaign = workspace.campaign_path(argv[0])
	input_pbf = argv[1]
	forwarded = list(argv[2:])
	campaign_file = campaign / "campaign.json"
	overwrite = "--overwrite" in forwarded
	if overwrite:
		forwarded = [argument for argument in forwarded if argument != "--overwrite"]

	if campaign_file.exists() and overwrite and "--dry-run" not in forwarded:
		shutil.rmtree(campaign)
	elif campaign_file.exists() and "--dry-run" not in forwarded:
		raise SystemExit(
			f"Campaign already exists: {campaign}\n"
			"Choose another campaign name or pass --overwrite."
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

	campaign = workspace.campaign_path(argv[0])
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

	campaign = workspace.campaign_path(argv[0])
	campaign_file = campaign / "campaign.json"
	if not campaign_file.exists():
		raise SystemExit(f"Not a campaign (missing campaign.json): {campaign}")

	import json
	data = json.loads(campaign_file.read_text())
	inputs = data.get("inputs", [])
	existing = sum((campaign / record["file"]).exists() for record in inputs)
	print(f"Campaign: {data.get('name', campaign.name)}")
	print(f"Location: {campaign}")
	if data.get("type"):
		print(f"Type:     {data['type']}")
	generation = data.get("generation", {})
	if generation:
		for label, key in (
			("Instances", "instances"),
			("Polygons", "polygons"),
			("Shape", "shape"),
			("Seed", "seed"),
		):
			if key in generation:
				print(f"{label + ':':<10} {generation[key]}")
		vertices = generation.get("vertices")
		if isinstance(vertices, list) and vertices:
			if len(set(vertices)) == 1:
				print(f"Vertices: {vertices[0]} per polygon")
			else:
				print(f"Vertices: {','.join(str(value) for value in vertices)}")
	print(f"Inputs:   {existing}/{len(inputs)} generated")
	if data.get("preview"):
		print(f"Preview:  {campaign / data['preview']}")
	source = data.get("source", {}).get("pbf")
	if source:
		print(f"Source:   {source}")

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
		width = 24
		filled = round(width * count / max(1, len(rows)))
		bar = "#" * filled + "-" * (width - filled)
		print(f"  {status:<12} [{bar}] {count}")
	if actions.get("skipped"):
		print(f"  resumed/skipped this run: {actions['skipped']}")
	return 0


def command_bench(argv: Sequence[str]) -> int:
	"""Run any benchmark from the canonical options defined in run_spec."""
	import run_spec

	values = run_spec.from_cli(list(argv))
	name, arguments = run_spec.to_legacy(values)
	print(f"+ tpp.py {name} {shlex.join(arguments)}", file=sys.stderr, flush=True)
	return COMMANDS[name].run(arguments)


def command_legacy(command: str, argv: Sequence[str]) -> int:
	import bench
	mapping = {
		"split": "split",
		"list-groups": "list",
		"run-groups": "run",
	}
	return bench.main([mapping[command], *argv])


def command_module(module_name: str, argv: Sequence[str]) -> int:
	"""Invoke one internal command while keeping tpp.py as the public CLI."""
	module = __import__(module_name)
	result = module.main(list(argv))
	return int(result or 0)


def module(name: str) -> Callable[[Sequence[str]], int]:
	return lambda argv: command_module(name, argv)


@dataclass(frozen=True)
class Command:
	usage: str
	summary: str
	run: Callable[[Sequence[str]], int]


GROUPS: dict[str, dict[str, Command]] = {
	"Run a benchmark": {
		"bench": Command("--problem P [options]", "Run fixed-order TPP, free-order TPP or TSPN from one set of options "
			"(build the command with scripts/benchmark.sh).", lambda argv: command_bench(argv)),
		"tui": Command("", "Build a bench command interactively, then print, copy or run it (scripts/benchmark.sh).",
			module("benchmark_tui")),
	},
	"Setup and build": {
		"setup": Command("ARGS...", "Prepare the locked benchmark Python environment.", module("benchmark_environment")),
		"doctor": Command("", "Check compiler, Eigen/Boost, Gurobi and built tools on this machine.",
			lambda argv: command_module("native_build", ["--doctor", *argv])),
		"build": Command("[TOOL...]", "Build native tools into .build/tools (--list, --fetch-deps, --gurobi).",
			module("native_build")),
	},
	"Workspace": {
		"ls": Command("[campaigns|runs|experiments]", "List local campaigns, runs and experiments.",
			lambda argv: command_module("workspace", ["list", *argv])),
		"workspace": Command("list|migrate|path", "Manage the local workspace (TPP_WORKSPACE).", module("workspace")),
	},
	"Campaigns (instance sets + resumable runs)": {
		"create": Command("NAME ARGS...", "Create a synthetic benchmark campaign.", lambda argv: command_create(argv)),
		"generate-matrix": Command("NAME PBF ARGS...", "Create a reproducible OpenStreetMap campaign.",
			lambda argv: command_generate_matrix(argv)),
		"convert-paula": Command("NAME ARGS...", "Import 235 Paula cases with a bbox-center depot.", module("convert_paula")),
		"status": Command("NAME", "Show generation and benchmark progress.", lambda argv: command_status(argv)),
		"run": Command("NAME ARGS...", "Fixed-order B&B over all campaign inputs, resumably.", lambda argv: command_run(argv)),
		"free-order": Command("NAME ARGS...", "Free-order campaign with our and/or Fekete's solver.", module("free_order_campaign")),
	},
	"Suites and direct runs": {
		"generate": Command("ARGS...", "Generate one binary from an OSM extract.", lambda argv: command_generate(argv)),
		"generate-suites": Command("ARGS...", "Generate dev/canonical suites from the tracked corpus.", module("generate_algorithm_suites")),
		"build-suites": Command("ARGS...", "Select fixed development and canonical suites.", module("build_algorithm_suites")),
		"benchmark": Command("ARGS...", "Run the canonical fixed-order algorithm benchmark.", module("run_algorithm_benchmark")),
		"free-order-run": Command("ARGS...", "Run the free-order solver on a binary suite.", module("unordered_benchmark")),
		"free-order-sample-sizes": Command("ARGS...", "Run nested random subsets of one instance.", module("free_order_sample_sizes")),
		"free-order-metamorphic": Command("ARGS...", "Run metamorphic free-order checks.", module("free_order_metamorphic")),
		"generate-free-order-canon": Command("ARGS...", "Generate the diagnostic/canon campaign.", module("generate_free_order_canon")),
		"inspect-footprints": Command("ARGS...", "Count and compare QGIS GeoPackage footprints.", module("inspect_footprints")),
		"solve-footprints": Command("ARGS...", "Solve the polygons drawn in a GeoPackage.", module("solve_footprints")),
		"normalize": Command("ARGS...", "Normalize polygon orientation in a suite.", module("normalize_polygon_orientation")),
	},
	"Comparisons and summaries": {
		"compare-solvers": Command("ARGS...", "Compare B&B performance across convex solvers.", module("compare_convex_solvers")),
		"compare-threads": Command("ARGS...", "Compare paired 1-thread and multi-thread runs.", module("compare_thread_scaling")),
		"compare-gaps": Command("ARGS...", "Compare strict and Fekete-equivalent optimality gaps.", module("free_order_gap_comparison")),
		"free-order-ablation": Command("ARGS...", "Compare solver binaries on identical cases.", module("free_order_ablation")),
		"summarize-free-order": Command("ARGS...", "Compare completed canon runs.", module("summarize_free_order_canon")),
		"summarize-external": Command("ARGS...", "Compare our run with an external run.", module("summarize_unordered")),
		"cycle-benchmark": Command("ARGS...", "Compare certified cycle solvers with Gurobi.", module("cycle_benchmark")),
		"cycle-replay": Command("ARGS...", "Replay captured convex-cycle oracle calls.", module("cycle_replay")),
	},
	"External solver (Fekete et al.)": {
		"compare-external": Command("ARGS...", "Run the pinned external solver on a suite.", module("tspn_run_comparison")),
		"compare-oracles": Command("ARGS...", "Compare oracle backends inside the external solver.", module("tspn_oracle_backends")),
		"run-fekete": Command("ARGS...", "Run/resume the long external campaign.", module("run_fekete")),
		"tspn-compare": Command("[--campaign NAME] ...", "Run/resume the full 558-case TSPN campaign (checks, build, shards, report).",
			module("tspn_campaign")),
		"tspn-benchmark": Command("ARGS...", "Compare TSPN B&B against the Fekete SOCP B&B.", module("tspn_benchmark")),
		"convert-fekete": Command("ARGS...", "Convert the pinned Fekete instance archive.", module("convert_instances")),
		"convert-tspn": Command("ARGS...", "Convert native TSPN result instances.", module("convert_tspn_native_instances")),
		"verify-socp": Command("ARGS...", "Independently verify small endpoint TPP cases.", module("verify_socp")),
	},
	"Background jobs and other machines": {
		"jobs": Command("start|list|log|stop", "Run a command detached from the terminal; follow or stop it.", module("jobs")),
		"remote": Command("ACTION HOST ...", "Push the checkout, run, follow and pull results over SSH.", module("remote")),
	},
	"Difficulty splits (legacy)": {
		"split": Command("ARGS...", "Split a benchmarked binary by difficulty.", lambda argv: command_legacy("split", argv)),
		"list-groups": Command("ARGS...", "List groups from a difficulty split.", lambda argv: command_legacy("list-groups", argv)),
		"run-groups": Command("ARGS...", "Benchmark selected difficulty groups.", lambda argv: command_legacy("run-groups", argv)),
	},
}
COMMANDS = {name: command for group in GROUPS.values() for name, command in group.items()}
# Commands that only read or prepare state are not journaled.
UNJOURNALED = {"setup", "doctor", "ls", "workspace", "status", "jobs", "remote"}


def print_help() -> None:
	lines = [
		"usage: python3 benchmarks/tpp.py COMMAND [arguments]",
		"",
		"Typical workflow:",
		"  doctor                                   check this machine",
		"  create NAME --vertices 8 --polygons 20 --instances 100",
		"  run NAME --threads 8 --max-calls 1000000 --max-seconds 30",
		"  status NAME",
		"  ls                                       everything in the workspace",
	]
	for group, commands in GROUPS.items():
		lines += ["", f"{group}:"]
		for name, command in commands.items():
			lines.append(f"  {(name + ' ' + command.usage).rstrip():<40} {command.summary}")
	lines += [
		"",
		f"Generated data lives in the workspace ({workspace.root()});",
		"set TPP_WORKSPACE to move it. A bare campaign NAME means workspace/campaigns/NAME;",
		"pass a path instead to use another location. Add --help after a command for details.",
	]
	print("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
	arguments = list(sys.argv[1:] if argv is None else argv)
	if not arguments or arguments[0] in {"-h", "--help", "help"}:
		print_help()
		return 0

	name, rest = arguments[0], arguments[1:]
	command = COMMANDS.get(name)
	if command is None:
		print(f"Unknown command: {name}\n", file=sys.stderr)
		print_help()
		return 2
	if name in UNJOURNALED or any(flag in rest for flag in ("-h", "--help")):
		return command.run(rest)
	started = time.monotonic()
	exit_code: int | None = None
	try:
		exit_code = command.run(rest)
		return exit_code
	except SystemExit as error:
		exit_code = error.code if isinstance(error.code, int) else 1
		raise
	except KeyboardInterrupt:
		exit_code = 130
		raise
	except Exception:
		exit_code = 1
		raise
	finally:
		workspace.append_history(arguments, exit_code, time.monotonic() - started)


if __name__ == "__main__":
	raise SystemExit(main())
