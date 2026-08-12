#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PRESET = "nonconvex-release"
TARGET_BINARY = REPO_ROOT / "build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp"
BUILD_DIR = REPO_ROOT / "build/nonconvex-release"
CMAKE_CACHE = BUILD_DIR / "CMakeCache.txt"
DEFAULT_INPUT = REPO_ROOT / "packages/nonconvex-tpp/cpp/tests/test_cases_simplified2.bin"
DEFAULT_RESULTS = REPO_ROOT / "benchmarks/results/results.csv"
DEFAULT_SPLITS = REPO_ROOT / "benchmarks/results/splits"
DEFAULT_RUNS = REPO_ROOT / "benchmarks/results/runs"


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
	print("+", " ".join(command), flush=True)
	subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def configured_target() -> str | None:
	if not CMAKE_CACHE.exists():
		return None

	for line in CMAKE_CACHE.read_text().splitlines():
		if line.startswith("TARGET:STRING="):
			return line.split("=", 1)[1]

	return None


def ensure_target(target: str, *, no_build: bool = False) -> None:
	if no_build:
		if not TARGET_BINARY.exists():
			raise SystemExit(f"Binary does not exist and --no-build was passed: {TARGET_BINARY}")
		current_target = configured_target()

		if current_target is not None and current_target != target:
			raise SystemExit(f"Configured target is {current_target}, but {target} is required. Drop --no-build once to rebuild.")

		return

	if configured_target() != target or not TARGET_BINARY.exists():
		run_command(["cmake", "--preset", BUILD_PRESET, f"-DTARGET={target}"])
	else:
		print(f"+ cmake target already configured: {target}", flush=True)

	run_command(["cmake", "--build", "--preset", BUILD_PRESET])


def load_index(path: Path) -> dict:
	with path.open() as file:
		return json.load(file)


def parse_duration(value: str) -> float:
	text = value.strip().lower()

	for suffix, multiplier in (("ms", 0.001), ("s", 1.0)):
		if text.endswith(suffix):
			return float(text[:-len(suffix)]) * multiplier

	return float(text)


def format_seconds(seconds: float | None) -> str:
	if seconds is None or not math.isfinite(seconds):
		return "unbounded"

	if seconds < 0.001:
		return f"{seconds * 1_000_000:.3f}µs"

	if seconds < 1.0:
		return f"{seconds * 1000:.3f}ms"

	return f"{seconds:.3f}s"


def group_sort_key(item: tuple[str, dict]) -> tuple[float, str]:
	name, group = item
	upper = group.get("upper_seconds")
	return (math.inf if upper is None else float(upper), name)


def selected_groups(index: dict, args: argparse.Namespace) -> list[str]:
	groups = index["groups"]

	if args.group:
		missing = [name for name in args.group if name not in groups]

		if missing:
			raise SystemExit(f"Unknown group(s): {', '.join(missing)}")

		return args.group

	if args.max_time is not None:
		limit = parse_duration(args.max_time)
		selected = []

		for name, group in sorted(groups.items(), key=group_sort_key):
			upper = group.get("upper_seconds")

			if upper is None:
				if args.include_overflow:
					selected.append(name)
				continue

			if float(upper) <= limit:
				selected.append(name)

		return selected

	return list(groups.keys())


def selection_name(groups: list[str], args: argparse.Namespace) -> str:
	if args.name:
		return args.name

	if args.max_time:
		return f"under_{args.max_time.replace('.', '_')}"

	if len(groups) == 1:
		return groups[0]

	return "selected"


def combine_group_files(index: dict, groups: list[str], splits_dir: Path, output_path: Path) -> None:
	with output_path.open("wb") as output:
		for group_name in groups:
			group = index["groups"][group_name]
			input_path = splits_dir / group["file"]

			if not input_path.exists():
				raise FileNotFoundError(f"Missing group binary: {input_path}")

			with input_path.open("rb") as input_file:
				output.write(input_file.read())


def command_split(args: argparse.Namespace) -> None:
	ensure_target("main-split_benchmark_cases", no_build=args.no_build)
	args.output.mkdir(parents=True, exist_ok=True)
	run_command([
		str(TARGET_BINARY),
		str(args.input),
		str(args.csv),
		str(args.output),
	])

	if not args.no_restore:
		ensure_target("main-bnb_workload_benchmark", no_build=args.no_build)


def command_list(args: argparse.Namespace) -> None:
	index = load_index(args.index)
	groups = sorted(index["groups"].items(), key=group_sort_key)

	print("| Group | File | Count | Upper | Mean | Total | Max measured | Max calls | Flags |")
	print("|---|---|---:|---:|---:|---:|---:|---:|---|")

	for name, group in groups:
		flags = []

		if group.get("has_capped"):
			flags.append("capped")

		if group.get("has_branch_limited"):
			flags.append("branch-limited")

		print(
			"| "
			f"{name} | "
			f"{group['file']} | "
			f"{group['count']} | "
			f"{format_seconds(group.get('upper_seconds'))} | "
			f"{format_seconds(group['mean_seconds'])} | "
			f"{format_seconds(group['total_seconds'])} | "
			f"{format_seconds(group['measured_max_seconds'])} | "
			f"{group['max_calls']} | "
			f"{', '.join(flags) if flags else '-'} |"
		)


def command_run(args: argparse.Namespace) -> None:
	index = load_index(args.index)
	groups = selected_groups(index, args)

	if not groups:
		raise SystemExit("No groups selected.")

	ensure_target("main-bnb_workload_benchmark", no_build=args.no_build)
	timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
	run_dir = args.output / timestamp
	run_dir.mkdir(parents=True, exist_ok=True)
	splits_dir = args.index.parent
	env = os.environ.copy()

	if args.threads is not None:
		env["TPP_BENCH_THREADS"] = str(args.threads)

	if args.separate_groups:
		for group_name in groups:
			group = index["groups"][group_name]
			input_file = splits_dir / group["file"]
			csv_output = run_dir / f"{group_name}.csv"
			summary_output = run_dir / f"{group_name}.md"

			print(f"\n## {group_name}", flush=True)
			run_command(
				[
					str(TARGET_BINARY),
					str(input_file),
					str(args.max_polygons),
					str(args.max_instances),
					str(args.max_calls),
					str(args.max_branching),
					str(args.repeat_count),
					str(csv_output),
					str(summary_output),
				],
				env=env,
			)
	else:
		name = selection_name(groups, args)
		input_file = run_dir / f"{name}.bin"
		csv_output = run_dir / f"{name}.csv"
		summary_output = run_dir / f"{name}.md"
		combined_index = run_dir / f"{name}.groups.json"
		combine_group_files(index, groups, splits_dir, input_file)
		combined_index.write_text(json.dumps({"groups": groups}, indent=2) + "\n")

		print(f"\n## {name}", flush=True)
		run_command(
			[
				str(TARGET_BINARY),
				str(input_file),
				str(args.max_polygons),
				str(args.max_instances),
				str(args.max_calls),
				str(args.max_branching),
				str(args.repeat_count),
				str(csv_output),
				str(summary_output),
			],
			env=env,
		)

	print(f"\nWrote run outputs to {run_dir}", flush=True)


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Benchmark helper for Touring Polygons.")
	subparsers = parser.add_subparsers(dest="command", required=True)

	split_parser = subparsers.add_parser("split", help="Build and run the benchmark case splitter.")
	split_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	split_parser.add_argument("--csv", type=Path, default=DEFAULT_RESULTS)
	split_parser.add_argument("--output", type=Path, default=DEFAULT_SPLITS)
	split_parser.add_argument("--no-build", action="store_true", help="Use the existing binary without configuring or building.")
	split_parser.add_argument("--no-restore", action="store_true", help="Do not restore the build target to the benchmark runner after splitting.")
	split_parser.set_defaults(func=command_split)

	list_parser = subparsers.add_parser("list", help="List benchmark groups from an instances.json file.")
	list_parser.add_argument("--index", type=Path, default=DEFAULT_SPLITS / "instances.json")
	list_parser.set_defaults(func=command_list)

	run_parser = subparsers.add_parser("run", help="Run benchmark groups selected from an instances.json file.")
	run_parser.add_argument("--index", type=Path, default=DEFAULT_SPLITS / "instances.json")
	run_parser.add_argument("--group", action="append", help="Group to run. May be passed more than once.")
	run_parser.add_argument("--max-time", help="Run all groups with upper_seconds <= this value, e.g. 1s or 100ms.")
	run_parser.add_argument("--include-overflow", action="store_true", help="Include the unbounded/capped group with --max-time.")
	run_parser.add_argument("--output", type=Path, default=DEFAULT_RUNS)
	run_parser.add_argument("--name", help="Output basename for combined runs.")
	run_parser.add_argument("--separate-groups", action="store_true", help="Run each selected group separately instead of concatenating them.")
	run_parser.add_argument("--no-build", action="store_true", help="Use the existing benchmark binary without configuring or building.")
	run_parser.add_argument("--threads", type=int)
	run_parser.add_argument("--max-polygons", default="-1")
	run_parser.add_argument("--max-instances", default="-1")
	run_parser.add_argument("--max-calls", default="1000000")
	run_parser.add_argument("--max-branching", default="-1")
	run_parser.add_argument("--repeat-count", default="1")
	run_parser.set_defaults(func=command_run)

	return parser


def main(argv: list[str] | None = None) -> int:
	parser = make_parser()
	args = parser.parse_args(argv)

	try:
		args.func(args)
	except subprocess.CalledProcessError as error:
		return error.returncode
	except Exception as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
