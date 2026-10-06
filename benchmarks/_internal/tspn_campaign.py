"""Run or resume the full Fekete TSPN comparison campaign (558 cases).

Closed tour, free cyclic order, no fixed point: our solver against the pinned
Fekete SOCP B&B. This module owns the setup that used to live in
``scripts/run_tspn_comparison.sh`` (archive and submodule checks, toolchain,
one shared native build, sharding and merging); the per-case engine remains
``tspn_benchmark.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import native_build
import run_layout
import tspn_benchmark

import workspace

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_SOURCE = ROOT / "third_party/tspn-socg"
SUITE = EXTERNAL_SOURCE / "instances/instances_socg_simplified.zip"
EXPECTED_SUITE_SHA256 = (
	"210841184500cb444f332537c4291aeff502dde4e3adc4a4ee307a80db21711f"
)
EXPECTED_CASES = 558
BUILD_DIR = ROOT / ".build/tspn-comparison"
DEFAULT_CAMPAIGN = "tspn-fekete-comparison-v1"
DEFAULT_OPTIMIZATIONS = "cache,features,root,interval"
DEFAULT_MAX_CALLS = 10**8
CYCLE_OPTIMIZATIONS = (
	"cache",
	"dual",
	"features",
	"lazy",
	"root",
	"branch",
	"one-tree",
	"learn",
	"memo",
	"bound-first",
	"dual-screen",
	"interval",
	"share-bounds",
	"proposal-bound",
	"primal-starts",
)
SOLVERS = {"tpp-ours": ["ours"], "tpp-fekete": ["fekete"], "both": ["ours", "fekete"]}
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
POSITIVE_NUMBER = re.compile(r"^(0\.[0-9]+|[1-9][0-9]*(\.[0-9]+)?)$")


@dataclass(frozen=True)
class Options:
	campaign: str = DEFAULT_CAMPAIGN
	solver: str = "both"
	seconds: str = "60"
	external_timeout: str = "75"
	repetitions: int = 1
	workers: int = 1
	relative_gap: str = "1e-6"
	cycle_optimizations: tuple[str, ...] = tuple(DEFAULT_OPTIMIZATIONS.split(","))
	portfolio: str | None = None  # None, "cooperative" or "independent"
	search_strategy: str | None = None
	capture_oracles: bool = False
	max_calls: int = DEFAULT_MAX_CALLS
	progress_interval: str = "60"
	force: bool = False
	setup_only: bool = False
	dry_run: bool = False

	@property
	def backends(self) -> list[str]:
		return SOLVERS[self.solver]

	@property
	def output(self) -> Path:
		return workspace.campaigns_dir() / self.campaign


def positive_number(value: str) -> str:
	if not POSITIVE_NUMBER.match(value):
		raise argparse.ArgumentTypeError("must be a positive number")
	return value


def relative_gap(value: str) -> str:
	"""A relative gap strictly between 0 and 1, kept as text for the command line."""
	try:
		number = float(value)
	except ValueError:
		number = 0.0
	if not 0 < number < 1:
		raise argparse.ArgumentTypeError("must be greater than 0 and less than 1")
	return value


def non_negative_number(value: str) -> str:
	try:
		number = float(value)
	except ValueError:
		number = -1.0
	if not 0 <= number < float("inf"):
		raise argparse.ArgumentTypeError("must be 0 or a positive number")
	return value


def positive_integer(value: str) -> int:
	if not re.fullmatch(r"[1-9][0-9]*", value):
		raise argparse.ArgumentTypeError("must be a positive integer")
	return int(value)


def calls_limit(value: str) -> int:
	if value != "-1" and not re.fullmatch(r"[1-9][0-9]*", value):
		raise argparse.ArgumentTypeError("must be -1 (no limit) or a positive integer")
	return int(value)


def campaign_name(value: str) -> str:
	if not NAME_PATTERN.match(value) or value in {".", ".."}:
		raise argparse.ArgumentTypeError(
			"must be a simple name without path separators"
		)
	return value


def optimization_list(value: str) -> tuple[str, ...]:
	names = tuple(item.strip() for item in value.split(",") if item.strip())
	unknown = [name for name in names if name not in CYCLE_OPTIMIZATIONS]
	if unknown:
		raise argparse.ArgumentTypeError(
			f"unknown optimization(s): {', '.join(unknown)}"
		)
	return names


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="tpp.py tspn-compare",
		description=(
			"Build the solvers, then run or resume the 558-case Fekete TSPN campaign (closed tour, "
			"free cyclic order, no fixed point). Results go to "
			"benchmarks/workspace/campaigns/NAME/. Rerunning the same command resumes."
		),
	)
	parser.add_argument(
		"--campaign",
		type=campaign_name,
		default=DEFAULT_CAMPAIGN,
		help="campaign directory name",
	)
	parser.add_argument(
		"--solver",
		choices=tuple(SOLVERS),
		default="both",
		help="tpp-ours needs no Gurobi (default: both)",
	)
	parser.add_argument(
		"--seconds",
		type=positive_number,
		default="60",
		help="per-instance native limit",
	)
	parser.add_argument(
		"--external-timeout",
		type=positive_number,
		default="75",
		help="per-instance wall-clock watchdog",
	)
	parser.add_argument("--repetitions", type=positive_integer, default=1)
	parser.add_argument(
		"--relative-gap",
		type=relative_gap,
		default="1e-6",
		help="target relative gap",
	)
	parser.add_argument(
		"--cycle-optimizations",
		type=optimization_list,
		default=tuple(DEFAULT_OPTIMIZATIONS.split(",")),
		help="comma-separated opt-ins",
	)
	portfolio = parser.add_mutually_exclusive_group()
	portfolio.add_argument(
		"--portfolio",
		action="store_true",
		help="cooperative portfolio (implies memo opt-in)",
	)
	portfolio.add_argument(
		"--portfolio-no-sharing", action="store_true", help="independent portfolio race"
	)
	parser.add_argument(
		"--search-strategy",
		choices=("best-bound", "dfs-bfs"),
		help="isolated B&B strategy (default: solver default)",
	)
	parser.add_argument(
		"--capture-oracles",
		action="store_true",
		help="record per-call oracle captures (diagnostic)",
	)
	parser.add_argument(
		"--max-calls",
		type=calls_limit,
		default=DEFAULT_MAX_CALLS,
		help="our solver's oracle-call budget (-1: no limit); a non-default value requires --solver tpp-ours",
	)
	parser.add_argument(
		"--progress-interval",
		type=non_negative_number,
		default="60",
		help="seconds between status lines of each running tpp-ours instance (also in live.json, see tpp.py live); 0 disables",
	)
	parser.add_argument(
		"--workers",
		type=positive_integer,
		default=1,
		help="parallel case shards; each instance's real solve keeps one thread",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="move an existing campaign directory aside first",
	)
	parser.add_argument(
		"--setup-only", action="store_true", help="check dependencies and exit"
	)
	parser.add_argument(
		"--dry-run", action="store_true", help="print the plan and exit"
	)
	return parser


def parse_options(argv: Sequence[str]) -> Options:
	args = build_parser().parse_args(argv)
	if args.search_strategy and (args.portfolio or args.portfolio_no_sharing):
		build_parser().error(
			"--search-strategy cannot be combined with portfolio options"
		)
	if args.max_calls != DEFAULT_MAX_CALLS and args.solver != "tpp-ours":
		build_parser().error(
			"a non-default --max-calls requires --solver tpp-ours; Fekete has no matching call budget"
		)
	return Options(
		campaign=args.campaign,
		solver=args.solver,
		seconds=args.seconds,
		external_timeout=args.external_timeout,
		repetitions=args.repetitions,
		workers=args.workers,
		relative_gap=args.relative_gap,
		cycle_optimizations=args.cycle_optimizations,
		portfolio="independent"
		if args.portfolio_no_sharing
		else ("cooperative" if args.portfolio else None),
		search_strategy=args.search_strategy,
		capture_oracles=args.capture_oracles,
		max_calls=args.max_calls,
		progress_interval=args.progress_interval,
		force=args.force,
		setup_only=args.setup_only,
		dry_run=args.dry_run,
	)


def fail(message: str) -> None:
	raise SystemExit(f"Error: {message}")


# --- preflight --------------------------------------------------------------


def suite_sha256(path: Path = SUITE) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_suite(path: Path = SUITE, expected: str = EXPECTED_SUITE_SHA256) -> None:
	if not path.is_file():
		fail(f"Fekete instance archive is missing: {path}")
	actual = suite_sha256(path)
	if actual != expected:
		fail(
			f"Fekete instance archive SHA-256 mismatch: expected {expected}, got {actual}"
		)


def _git(*arguments: str, cwd: Path = ROOT) -> str:
	result = subprocess.run(
		["git", "-C", str(cwd), *arguments], capture_output=True, text=True, check=False
	)
	return result.stdout.strip() if result.returncode == 0 else ""


def pinned_submodule_revision(root: Path = ROOT) -> str:
	"""The commit recorded for the Fekete submodule in this checkout's HEAD."""
	for line in _git(
		"ls-tree", "HEAD", "--", "third_party/tspn-socg", cwd=root
	).splitlines():
		mode, _, rest = line.partition(" ")
		if mode == "160000":
			return rest.split()[1]
	return ""


def verify_submodule() -> str:
	"""Ensure the Fekete submodule is initialized at its pinned revision; return it."""
	if not shutil.which("git"):
		fail("git is required")
	expected = pinned_submodule_revision()
	if not expected:
		fail("third_party/tspn-socg is not pinned as a Git submodule in HEAD")
	if not (EXTERNAL_SOURCE / ".git").exists():
		subprocess.run(
			[
				"git",
				"-C",
				str(ROOT),
				"submodule",
				"update",
				"--init",
				"--recursive",
				"--",
				"third_party/tspn-socg",
			],
			check=True,
		)
	current = _git("rev-parse", "HEAD", cwd=EXTERNAL_SOURCE)
	if current != expected:
		fail(f"Fekete submodule is at {current}; expected pinned revision {expected}")
	return current


def conan_prefix() -> Path:
	return EXTERNAL_SOURCE / ".conan/release"


def conan_packages(backends: Sequence[str]) -> list[tuple[str, ...]]:
	"""CMake package files (any alternative) the selected solvers need from Conan."""
	packages = [
		("BoostConfig.cmake",),
		("nlohmann_json-config.cmake", "nlohmann_jsonConfig.cmake"),
		("Eigen3Config.cmake",),
	]
	if "fekete" in backends:
		packages = [
			("cgal-config.cmake", "cgalConfig.cmake"),
			("fmt-config.cmake", "fmtConfig.cmake"),
			*packages,
		]
	return packages


def verify_conan(backends: Sequence[str]) -> Path:
	prefix = conan_prefix()
	if not prefix.is_dir():
		fail(
			f"Conan C++ dependencies not found at {prefix}\n"
			"Run tpp.py free-compare --setup-only --solver tpp-fekete first to prepare Fekete."
		)
	for alternatives in conan_packages(backends):
		if not any((prefix / name).is_file() for name in alternatives):
			fail(
				f"Conan setup did not generate a CMake package for: {' / '.join(alternatives)}"
			)
	return prefix


def prepare_environment(
	options: Options, *, probe_toolchain: bool = True
) -> dict[str, str]:
	"""Verify prerequisites and export what the CMake build needs; return the changes."""
	verify_suite()
	verify_submodule()
	prefix = verify_conan(options.backends)
	changes = {
		"CMAKE_PREFIX_PATH": os.pathsep.join(
			filter(None, [str(prefix), os.environ.get("CMAKE_PREFIX_PATH", "")])
		)
	}
	if not shutil.which("cmake"):
		fail("CMake is missing; run python3 benchmarks/tpp.py setup")
	if "fekete" in options.backends and not os.environ.get("GUROBI_HOME"):
		home = native_build.gurobi_home()
		if home is not None and home.exists():
			changes["GUROBI_HOME"] = str(home)
	if probe_toolchain:
		toolchain = native_build.select_toolchain()
		changes.update(toolchain)
		print(
			f"Detected C++ standard: {toolchain['TPP_CXX_STANDARD']} ({Path(toolchain['CXX']).name})"
		)
	os.environ.update(changes)
	print(f"Reusing Conan C++ dependencies: {prefix}")
	return changes


# --- command construction ---------------------------------------------------


def engine_arguments(options: Options) -> list[str]:
	"""Arguments for ``tspn-benchmark`` shared by single and sharded runs."""
	arguments = [
		"--seconds",
		options.seconds,
		"--external-timeout",
		options.external_timeout,
		"--repetitions",
		str(options.repetitions),
		"--relative-gap",
		options.relative_gap,
	]
	if options.solver != "both":
		arguments += ["--solver", options.backends[0]]
	if options.portfolio == "cooperative":
		arguments.append("--portfolio")
	elif options.portfolio == "independent":
		arguments.append("--portfolio-no-sharing")
	if options.search_strategy:
		arguments += ["--search-strategy", options.search_strategy]
	if options.capture_oracles:
		arguments.append("--capture-oracles")
	if options.max_calls != DEFAULT_MAX_CALLS:
		arguments += ["--max-calls", str(options.max_calls)]
	arguments += ["--progress-interval", options.progress_interval]
	for optimization in options.cycle_optimizations:
		arguments += ["--cycle-optimization", optimization]
	return arguments


def single_arguments(options: Options, output: Path) -> list[str]:
	arguments = [
		"--all",
		"--instances-zip",
		str(SUITE),
		"--output",
		str(output),
		*engine_arguments(options),
	]
	if (output / "config.json").exists():
		arguments.append("--resume")
	if options.dry_run:
		arguments.append("--dry-run")
	return arguments


def binaries(build_dir: Path = BUILD_DIR) -> dict[str, Path]:
	return {
		"ours": build_dir / "bin/tpp-unordered",
		"fekete": build_dir / "tpp-fekete-cycle",
	}


def shard_arguments(
	options: Options, shard: Path, build_dir: Path = BUILD_DIR
) -> list[str]:
	paths = binaries(build_dir)
	arguments = [
		"--inputs",
		str(shard / "instances.json"),
		"--output",
		str(shard),
		*engine_arguments(options),
		"--skip-build",
		"--build-dir",
		str(build_dir),
		"--ours-binary",
		str(paths["ours"]),
	]
	if "fekete" in options.backends:
		arguments += ["--fekete-binary", str(paths["fekete"])]
	if (shard / "config.json").exists():
		arguments.append("--resume")
	return arguments


# --- execution --------------------------------------------------------------


def build_native(options: Options, output: Path) -> None:
	output.mkdir(parents=True, exist_ok=True)
	commands = tspn_benchmark.native_build_commands(
		BUILD_DIR, EXTERNAL_SOURCE, options.backends
	)
	print(f"Building shared binaries in {BUILD_DIR}")
	with (output / "build.txt").open("w") as log:
		for command in commands:
			result = subprocess.run(
				command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
			)
			if result.returncode:
				log.flush()
				tail = (output / "build.txt").read_text().splitlines()[-40:]
				print("\n".join(tail), file=sys.stderr)
				fail(f"{'configure' if command[1] == '-S' else 'build'} failed")


def write_shard_inputs(output: Path, workers: int) -> int:
	cases, _ = tspn_benchmark.select_all_socg_inputs(SUITE)
	for index in range(workers):
		shard = output / f"shard-{index}"
		shard.mkdir(parents=True, exist_ok=True)
		payload = {
			"formulation": "TSPN, free cyclic order, no fixed point, closed polygon regions",
			"selection": {
				"archive": str(SUITE),
				"policy": "all archive cases",
				"shard": f"{index + 1}/{workers}",
			},
			"instances": cases[index::workers],
		}
		(shard / "instances.json").write_text(json.dumps(payload))
	return len(cases)


def merge_shards(output: Path, workers: int) -> int:
	"""Combine shard records, instances and config into the campaign directory."""
	from tspn_diagnostics import digest

	shards = [output / f"shard-{index}" for index in range(workers)]
	lines, unique, seen, dropped = [], [], set(), 0
	for shard in shards:
		raw = shard / "raw.jsonl"
		if raw.exists():
			lines.extend(line for line in raw.read_text().splitlines() if line.strip())
	for line in lines:
		try:
			row = json.loads(line)
		except ValueError:
			dropped += 1
			continue
		key = (row.get("name"), row.get("sha256"), row.get("solver"), row.get("repeat"))
		if key in seen:
			dropped += 1
			continue
		seen.add(key)
		unique.append(line)
	if dropped:
		print(
			f"Warning: dropped {dropped} duplicate/invalid shard rows before merging."
		)
	(output / "raw.jsonl").write_text("\n".join(unique) + ("\n" if unique else ""))
	instances, formulation = [], None
	for shard in shards:
		payload = json.loads((shard / "instances.json").read_text())
		formulation = formulation or payload["formulation"]
		instances.extend(payload["instances"])
	config = json.loads((shards[0] / "config.json").read_text())
	full = {
		"formulation": formulation,
		"instances": instances,
		"selection": {"policy": "all archive cases", "shards": workers},
	}
	(output / "instances.json").write_text(json.dumps(full))
	config["inputs_sha256"] = digest(full)
	config["plan"]["cases"] = len(instances)
	config["plan"]["planned_runs"] = (
		len(instances)
		* config["repetitions"]
		* len(config.get("solvers", ["ours", "fekete"]))
	)
	config["plan"]["shards"] = workers
	(output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
	print(f"Merged {len(lines)} rows from {workers} shards into {output}")
	return len(unique)


def run_shards(options: Options, output: Path) -> int:
	build_native(options, output)
	total = write_shard_inputs(output, options.workers)
	print(f"Shard instances written for {options.workers} workers: {total} cases total")
	script = ROOT / "benchmarks/tpp.py"
	processes = []
	for index in range(options.workers):
		shard = output / f"shard-{index}"
		arguments = shard_arguments(options, shard)
		print(f"Shard {index}: {len(arguments)} args; log {shard / 'run.log'}")
		log = (shard / "run.log").open("w")
		processes.append(
			(
				subprocess.Popen(
					[sys.executable, str(script), "tspn-benchmark", *arguments],
					stdout=log,
					stderr=subprocess.STDOUT,
				),
				log,
			)
		)
	status = 0
	for process, log in processes:
		while True:
			try:
				code = process.wait()
				break
			except KeyboardInterrupt:
				# The terminal already signalled the whole process group; the
				# shards save their records and exit, so keep waiting for them.
				print(
					"Interrupted; waiting for the shards to save their records...",
					flush=True,
				)
		log.close()
		status = status or code
	if status:
		print(
			f"Warning: at least one shard exited with status {status}; merging available records.",
			file=sys.stderr,
		)
	try:
		merge_shards(output, options.workers)
	except (OSError, ValueError, KeyError) as error:
		print(f"Warning: merge failed: {error}", file=sys.stderr)
		return status or 1
	tspn_benchmark.main(["--report-only", "--output", str(output)])
	return status


def describe(options: Options) -> str:
	return (
		f"Settings: seconds={options.seconds}, external-timeout={options.external_timeout}, "
		f"repetitions={options.repetitions}, gap={options.relative_gap}, "
		f"portfolio={options.portfolio or 'none'}, search={options.search_strategy or 'default'}, "
		f"opts={','.join(options.cycle_optimizations)}"
	)


def run(options: Options) -> int:
	changes = prepare_environment(options, probe_toolchain=not options.dry_run)
	if options.setup_only:
		print(
			f"Setup complete: {EXPECTED_CASES} cases ({EXPECTED_SUITE_SHA256}), "
			f"Fekete {_git('rev-parse', 'HEAD', cwd=EXTERNAL_SOURCE)}, GUROBI_HOME={os.environ.get('GUROBI_HOME', '<unset>')}"
		)
		print(f"Run with: tpp.py tspn-compare --campaign {options.campaign}")
		return 0
	# A run of its own in results/<run-id>/; --force starts a new one instead of resuming.
	output = run_layout.tspn_run_directory(options.output, new=options.force)
	print(f"Archive: {EXPECTED_CASES} cases; campaign: {output}")
	print(describe(options))
	if options.dry_run:
		return tspn_benchmark.main(single_arguments(options, output)) or 0
	parameters = {**asdict(options), "toolchain": changes.get("TPP_CXX_STANDARD")}
	with workspace.recorded_run(output, kind="tspn-comparison", parameters=parameters):
		if options.workers > 1:
			return run_shards(options, output)
		return tspn_benchmark.main(single_arguments(options, output)) or 0


def main(argv: Sequence[str] | None = None) -> int:
	return run(parse_options(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
	raise SystemExit(main())
