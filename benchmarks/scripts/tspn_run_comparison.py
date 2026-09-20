#!/usr/bin/env python3
"""Run TouringPolygons binary suites with the external TSPN solver."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = PROJECT_ROOT / "benchmarks/suites/algorithm-dev-v1.bin"
DEFAULT_TSPN_REPO = PROJECT_ROOT / "tspn-comparison/solver-oracle"
DEFAULT_OUTPUT = PROJECT_ROOT / "tspn-comparison/results"
RESULT_FIELDS = [
	"case_index", "difficulty", "sha256", "mode", "polygons", "vertices", "status",
	"is_optimal", "is_valid_trajectory", "lower_bound", "upper_bound", "absolute_gap",
	"relative_gap", "solve_seconds", "num_iterations", "num_branches", "num_explored",
	"num_discarded_sequences", "soc_num_calls", "tpp_socp_fallback_calls", "tpp_time_limit_calls",
	"soc_total_seconds", "soc_optimize_seconds",
	"soc_seconds_per_call", "soc_optimize_seconds_per_call", "time_limit_seconds", "threads",
	"eps", "oracle_backend", "tspn_version", "error",
	"feasibility_tolerance", "validation_tolerance", "process_seconds",
	"trajectory_json", "snapped_trajectory_json", "start_distance", "target_distance",
	"max_polygon_distance", "recomputed_length", "raw_polygon_valid", "raw_valid",
	"snapped_max_polygon_distance", "snapped_recomputed_length", "snapped_valid",
]

sys.path.insert(0, str(PROJECT_ROOT / "benchmarks/scripts"))
from unordered_validation import orient_path, validate_path


_ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()
_ACTIVE_PROCESSES_LOCK = threading.Lock()


def _stop_process(process: subprocess.Popen[str]) -> None:
	"""Stop one isolated worker process and reap it."""
	if process.poll() is not None:
		return
	try:
		if os.name == "posix":
			os.killpg(process.pid, signal.SIGTERM)
		else:
			process.terminate()
	except ProcessLookupError:
		return
	try:
		process.wait(timeout=5)
	except subprocess.TimeoutExpired:
		try:
			if os.name == "posix":
				os.killpg(process.pid, signal.SIGKILL)
			else:
				process.kill()
		except ProcessLookupError:
			pass
		process.wait()


def stop_active_processes() -> None:
	"""Stop all active case workers, for example after Ctrl-C."""
	with _ACTIVE_PROCESSES_LOCK:
		processes = list(_ACTIVE_PROCESSES)
	for process in processes:
		_stop_process(process)


@dataclass(frozen=True)
class EncodedCase:
	data: bytes
	digest: str
	start: tuple[float, float]
	target: tuple[float, float]
	polygons: tuple[tuple[tuple[float, float], ...], ...]


def read_u64(data: bytes, offset: int, path: Path) -> tuple[int, int]:
	if offset + 8 > len(data):
		raise ValueError(f"Truncated size at byte {offset} in {path}")
	return struct.unpack_from("<Q", data, offset)[0], offset + 8


def read_cases(path: Path) -> list[EncodedCase]:
	data = path.read_bytes()
	offset = 0
	cases: list[EncodedCase] = []
	while offset < len(data):
		case_start = offset
		start_x, start_y, target_x, target_y = struct.unpack_from("<dddd", data, offset)
		offset += 32
		polygon_count, offset = read_u64(data, offset, path)
		polygons = []
		for _ in range(polygon_count):
			vertex_count, offset = read_u64(data, offset, path)
			end = offset + 16 * vertex_count
			if end > len(data):
				raise ValueError(f"Truncated polygon at byte {offset} in {path}")
			polygons.append(tuple(
				struct.unpack_from("<dd", data, offset + 16 * index)
				for index in range(vertex_count)
			))
			offset = end
		solution_count, offset = read_u64(data, offset, path)
		offset += 16 * solution_count
		if offset > len(data):
			raise ValueError(f"Truncated solution at byte {offset} in {path}")
		encoded = data[case_start:offset]
		cases.append(EncodedCase(
			encoded, hashlib.sha256(encoded).hexdigest(), (start_x, start_y),
			(target_x, target_y), tuple(polygons),
		))
	return cases


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run a TPP binary suite as endpoint path or endpoint-free TSPN instances.",
	)
	parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
	parser.add_argument("--metadata", type=Path, help="Suite CSV; defaults beside --suite.")
	parser.add_argument("--tspn-repo", type=Path, default=DEFAULT_TSPN_REPO)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument(
		"--mode", choices=("path", "cycle"), default="path",
		help="path keeps encoded start/target; cycle discards them. Polygon order is free in both.",
	)
	parser.add_argument("--time-limit", type=int, default=60, help="Seconds per TSPN case.")
	parser.add_argument(
		"--oracle-backend", choices=("socp", "tpp"), default="socp",
		help="Ordered-subproblem backend in the oracle-enabled checkout.",
	)
	parser.add_argument("--oracle-tolerance", type=float, default=1e-7)
	parser.add_argument(
		"--threads", type=int, default=0,
		help="TSPN worker threads per case; 0 uses all available threads.",
	)
	parser.add_argument(
		"--workers", type=int, default=1,
		help="Independent instance processes to run concurrently.",
	)
	parser.add_argument("--eps", type=float, default=0.001)
	parser.add_argument("--feasibility-tolerance", type=float, default=0.001)
	parser.add_argument("--validation-tolerance", type=float, default=1e-7)
	parser.add_argument("--max-instances", type=int, default=-1)
	parser.add_argument(
		"--difficulty", action="append", choices=("easy", "medium", "hard"),
		help="Only run this original TPP difficulty; may be repeated.",
	)
	parser.add_argument("--case-index", type=int, action="append", help="Only run this index.")
	parser.add_argument("--resume", type=Path, help="Append missing cases to an existing CSV.")
	parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
	parser.add_argument("--worker-case", type=int, help=argparse.SUPPRESS)
	parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
	return parser


def read_metadata(path: Path) -> dict[int, dict[str, str]]:
	if not path.exists():
		return {}
	with path.open(newline="") as file:
		sample = file.read(4096)
		file.seek(0)
		dialect = csv.Sniffer().sniff(sample, delimiters=",;")
		return {int(row["suite_index"]): row for row in csv.DictReader(file, dialect=dialect)}


def finite_number(value: Any) -> float | None:
	try:
		result = float(value)
	except (TypeError, ValueError):
		return None
	return result if math.isfinite(result) else None


def convert_stat(value: Any) -> Any:
	try:
		return int(value)
	except (TypeError, ValueError):
		try:
			return float(value)
		except (TypeError, ValueError):
			return value


def point_matches(point: Any, expected: tuple[float, float], tolerance: float = 1e-5) -> bool:
	x = point.x if hasattr(point, "x") else point[0]
	y = point.y if hasattr(point, "y") else point[1]
	return math.hypot(x - expected[0], y - expected[1]) <= tolerance


def worker(args: argparse.Namespace) -> int:
	if args.worker_case is None or args.worker_result is None:
		raise SystemExit("--worker requires --worker-case and --worker-result")

	# Load the same native solver directly; the package initializer also imports
	# optional MIP, plotting, and Pydantic dependencies unrelated to this benchmark.
	import importlib.metadata
	import importlib.util
	bindings = next((args.tspn_repo / 'python/tspn_bnb2/core').glob('_tspn_bindings*.so'), None)
	if bindings is None:
		package = importlib.util.find_spec('tspn_bnb2')
		bindings = next((Path(package.origin).parent / 'core').glob('_tspn_bindings*.so'))
	spec = importlib.util.spec_from_file_location('_tspn_bindings', bindings)
	core = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(core)
	Instance, Point, Polygon = core.Instance, core.Point, core.Polygon
	branch_and_bound, set_float_parameter = core.branch_and_bound, core.set_float_parameter
	__version__ = importlib.metadata.version('tspn_bnb2')

	encoded = read_cases(args.suite)[args.worker_case]
	set_float_parameter("FEASIBILITY_TOLERANCE", args.feasibility_tolerance)
	polygons = [Polygon([[Point(x, y) for x, y in vertices]]) for vertices in encoded.polygons]
	if args.mode == "path":
		sites = [Point(*encoded.start), Point(*encoded.target), *polygons]
		instance = Instance(sites, True)
	else:
		instance = Instance(polygons, False)

	started = time.perf_counter()
	options = dict(
		instance=instance,
		callback=lambda _: None,
		initial_solution=None,
		timelimit=args.time_limit,
		branching="FarthestPoly",
		search="DfsBfs",
		root="LongestEdgePlusFurthestSite",
		node_simplification=False,
		rules=[],
		use_cutoff=True,
		num_threads=args.threads,
		decomposition_branch=True,
		skip_convex_hull=False,
		eps=args.eps,
	)
	# The unmodified Fekete et al. binding only has the SOCP backend.  The
	# oracle-comparison checkout adds these two optional keyword arguments.
	if "oracle_backend" in (branch_and_bound.__doc__ or ""):
		options["oracle_backend"] = args.oracle_backend
		options["oracle_tolerance"] = args.oracle_tolerance
	elif args.oracle_backend != "socp":
		raise ValueError("This TSPN binding only supports its native SOCP backend")
	upper_bound, lower_bound, statistics_map = branch_and_bound(**options)
	solve_seconds = time.perf_counter() - started
	statistics_map = {key: convert_stat(value) for key, value in statistics_map.items()}
	if upper_bound is None:
		length = float("inf")
		absolute_gap = float("inf")
		relative_gap = float("inf")
		is_valid = False
		is_optimal = False
	else:
		trajectory = upper_bound.get_trajectory()
		length = trajectory.length()
		absolute_gap = length - lower_bound
		relative_gap = absolute_gap / length if length > 0 else 0.0
		points = orient_path(encoded.start, encoded.target, [(point.x, point.y) for point in trajectory])
		if args.mode == "path":
			raw_validation = validate_path(
				encoded.start, encoded.target, encoded.polygons, points, args.validation_tolerance,
			)
			snapped_points = [point[:] for point in points]
			if len(snapped_points) >= 2:
				snapped_points[0] = list(encoded.start)
				snapped_points[-1] = list(encoded.target)
			snapped_validation = validate_path(
				encoded.start, encoded.target, encoded.polygons, snapped_points, args.validation_tolerance,
			)
			is_valid = point_matches(points[0], encoded.start) and point_matches(points[-1], encoded.target)
		else:
			is_valid = trajectory.is_tour()
		is_optimal = length / lower_bound <= 1 + args.eps if lower_bound > 0 else length == 0

	payload = {
		"status": "optimal" if is_optimal else "limit",
		"is_optimal": is_optimal,
		"is_valid_trajectory": is_valid,
		"lower_bound": lower_bound,
		"upper_bound": length,
		"absolute_gap": absolute_gap,
		"relative_gap": relative_gap,
		"solve_seconds": solve_seconds,
		"statistics": statistics_map,
		"tspn_version": __version__,
	}
	if upper_bound is not None and args.mode == "path":
		payload.update({
			"trajectory": points,
			"snapped_trajectory": snapped_points,
			"validation": raw_validation,
			"snapped_validation": snapped_validation,
		})
	args.worker_result.write_text(json.dumps(payload, allow_nan=True))
	# The native extension owns process-global solver runtimes whose shutdown can
	# stall on macOS after a mixed-backend run. Workers are intentionally isolated,
	# so exit after the result has been durably handed to the parent.
	sys.stdout.flush()
	sys.stderr.flush()
	os._exit(0)


def choose_cases(args: argparse.Namespace, count: int, metadata: dict[int, dict[str, str]]) -> list[int]:
	indices = list(range(count))
	if args.case_index:
		requested = set(args.case_index)
		missing = sorted(index for index in requested if index < 0 or index >= count)
		if missing:
			raise SystemExit(f"Case indices outside suite range: {missing}")
		indices = [index for index in indices if index in requested]
	if args.difficulty:
		allowed = set(args.difficulty)
		indices = [index for index in indices if metadata.get(index, {}).get("difficulty") in allowed]
	if args.max_instances >= 0:
		indices = indices[:args.max_instances]
	return indices


def load_completed(path: Path) -> set[int]:
	if not path.exists():
		return set()
	with path.open(newline="") as file:
		return {
			int(row["case_index"])
			for row in csv.DictReader(file)
			if row.get("status") in {"optimal", "limit"} and not row.get("error")
		}


def result_row(
	args: argparse.Namespace,
	index: int,
	encoded: EncodedCase,
	meta: dict[str, str],
	payload: dict[str, Any],
) -> dict[str, Any]:
	stats = payload.get("statistics", {})
	return {
		"case_index": index,
		"difficulty": meta.get("difficulty", "unknown"),
		"sha256": encoded.digest,
		"mode": args.mode,
		"polygons": len(encoded.polygons),
		"vertices": sum(len(polygon) for polygon in encoded.polygons),
		"status": payload.get("status", "error"),
		"is_optimal": payload.get("is_optimal", False),
		"is_valid_trajectory": payload.get("is_valid_trajectory", False),
		"lower_bound": payload.get("lower_bound", ""),
		"upper_bound": payload.get("upper_bound", ""),
		"absolute_gap": payload.get("absolute_gap", ""),
		"relative_gap": payload.get("relative_gap", ""),
		"solve_seconds": payload.get("solve_seconds", ""),
		"num_iterations": stats.get("num_iterations", ""),
		"num_branches": stats.get("num_branches", ""),
		"num_explored": stats.get("num_explored", ""),
		"num_discarded_sequences": stats.get("num_discarded_sequences", ""),
		"soc_num_calls": stats.get("soc_num_calls", ""),
		"tpp_socp_fallback_calls": stats.get("tpp_socp_fallback_calls", ""),
		"tpp_time_limit_calls": stats.get("tpp_time_limit_calls", ""),
		"soc_total_seconds": stats.get("soc_total_seconds", ""),
		"soc_optimize_seconds": stats.get("soc_optimize_seconds", ""),
		"soc_seconds_per_call": stats.get("soc_seconds_per_call", ""),
		"soc_optimize_seconds_per_call": stats.get("soc_optimize_seconds_per_call", ""),
		"time_limit_seconds": args.time_limit,
		"threads": args.threads,
		"eps": args.eps,
		"oracle_backend": args.oracle_backend,
		"tspn_version": payload.get("tspn_version", ""),
		"error": payload.get("error", ""),
		"feasibility_tolerance": args.feasibility_tolerance,
		"validation_tolerance": args.validation_tolerance,
		"process_seconds": payload.get("process_seconds", ""),
		"trajectory_json": json.dumps(payload.get("trajectory")) if payload.get("trajectory") is not None else "",
		"snapped_trajectory_json": json.dumps(payload.get("snapped_trajectory")) if payload.get("snapped_trajectory") is not None else "",
		"start_distance": payload.get("validation", {}).get("start_distance", ""),
		"target_distance": payload.get("validation", {}).get("target_distance", ""),
		"max_polygon_distance": payload.get("validation", {}).get("max_polygon_distance", ""),
		"recomputed_length": payload.get("validation", {}).get("recomputed_length", ""),
		"raw_polygon_valid": payload.get("validation", {}).get("polygon_valid", ""),
		"raw_valid": payload.get("validation", {}).get("valid", ""),
		"snapped_max_polygon_distance": payload.get("snapped_validation", {}).get("max_polygon_distance", ""),
		"snapped_recomputed_length": payload.get("snapped_validation", {}).get("recomputed_length", ""),
		"snapped_valid": payload.get("snapped_validation", {}).get("valid", ""),
	}


def run_case(
	args: argparse.Namespace, index: int, log_file: Any,
	log_lock: threading.Lock | None = None,
) -> dict[str, Any]:
	def write_log(text: str) -> None:
		if log_lock is None:
			log_file.write(text)
			log_file.flush()
			return
		with log_lock:
			log_file.write(text)
			log_file.flush()

	with tempfile.TemporaryDirectory(prefix="tspn-comparison-") as temp_dir:
		result_path = Path(temp_dir) / "result.json"
		cache_dir = args.tspn_repo / ".cache"
		(cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
		environment = os.environ.copy()
		environment["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
		environment["XDG_CACHE_HOME"] = str(cache_dir)
		command = [
			sys.executable, str(Path(__file__).resolve()), "--worker", "--suite", str(args.suite),
			"--tspn-repo", str(args.tspn_repo), "--worker-case", str(index),
			"--worker-result", str(result_path), "--mode", args.mode,
			"--time-limit", str(args.time_limit), "--threads", str(args.threads),
			"--eps", str(args.eps), "--feasibility-tolerance", str(args.feasibility_tolerance),
			"--validation-tolerance", str(args.validation_tolerance),
			"--oracle-backend", args.oracle_backend,
			"--oracle-tolerance", str(args.oracle_tolerance),
		]
		started = time.perf_counter()
		process = subprocess.Popen(
			command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			text=True, env=environment, start_new_session=(os.name == "posix"),
		)
		with _ACTIVE_PROCESSES_LOCK:
			_ACTIVE_PROCESSES.add(process)
		try:
			try:
				stdout, stderr = process.communicate(timeout=args.time_limit + 120)
			except subprocess.TimeoutExpired:
				_stop_process(process)
				stdout, stderr = process.communicate()
				write_log(f"\n=== case {index}: process timeout ===\n{stdout}{stderr}")
				return {"status": "process_timeout", "solve_seconds": time.perf_counter() - started}
		finally:
			with _ACTIVE_PROCESSES_LOCK:
				_ACTIVE_PROCESSES.discard(process)

		write_log(f"\n=== case {index}: exit {process.returncode} ===\n{stdout}{stderr}")
		if process.returncode != 0:
			message = stderr.strip().splitlines()
			return {
				"status": "error", "solve_seconds": time.perf_counter() - started,
				"error": message[-1] if message else f"worker exited {process.returncode}",
			}
		if not result_path.exists():
			return {"status": "error", "error": "worker produced no result"}
		payload = json.loads(result_path.read_text())
		payload["process_seconds"] = time.perf_counter() - started
		return payload


def format_stats(values: list[float], suffix: str = "") -> str:
	if not values:
		return "n/a"
	return (
		f"min {min(values):.3f}{suffix}, median {statistics.median(values):.3f}{suffix}, "
		f"mean {statistics.fmean(values):.3f}{suffix}, max {max(values):.3f}{suffix}"
	)


def write_summary(csv_path: Path, summary_path: Path, args: argparse.Namespace) -> None:
	with csv_path.open(newline="") as file:
		rows = list(csv.DictReader(file))
	statuses = Counter(row["status"] for row in rows)
	times = [value for row in rows if (value := finite_number(row["solve_seconds"])) is not None]
	gaps = [100 * value for row in rows if (value := finite_number(row["relative_gap"])) is not None]
	soc_calls = [int(value) for row in rows if (value := row.get("soc_num_calls", "")).isdigit()]
	tpp_socp_fallback_calls = [int(value) for row in rows if (value := row.get("tpp_socp_fallback_calls", "")).isdigit()]
	soc_seconds_per_call = [
		1e6 * value for row in rows
		if (value := finite_number(row.get("soc_seconds_per_call"))) is not None
	]
	soc_optimize_seconds_per_call = [
		1e6 * value for row in rows
		if (value := finite_number(row.get("soc_optimize_seconds_per_call"))) is not None
	]
	mode_note = (
		"Encoded start and target points are retained; polygon order remains free."
		if args.mode == "path"
		else "Start and target are discarded; the result is an unordered closed cycle."
	)
	lines = [
		f"# External TSPN {args.mode} comparison", "", f"> {mode_note}", "",
		"| Configuration | Value |", "|---|---:|", f"| Suite | `{args.suite}` |",
		f"| Mode | {args.mode} |", f"| Cases recorded | {len(rows)} |",
		f"| Time limit per case | {args.time_limit}s |",
		f"| Solver threads | {'all available' if args.threads == 0 else args.threads} |",
		f"| Oracle backend | {args.oracle_backend} |",
		f"| TPP oracle tolerance | {args.oracle_tolerance} |",
		f"| Optimality tolerance | {args.eps} |",
		f"| Solver feasibility tolerance | {args.feasibility_tolerance} |",
		f"| Independent validation tolerance | {args.validation_tolerance} |", "",
		"Raw trajectories are exported and independently checked for fixed endpoints, polygon visits, and recomputed length. Endpoint-snapped trajectories are separate diagnostic candidates.", "",
		"| Result | Count |", "|---|---:|",
	]
	for status, count in sorted(statuses.items()):
		lines.append(f"| {status} | {count} ({100 * count / len(rows) if rows else 0:.1f}%) |")
	if args.mode == "path":
		raw_valid = sum(row.get("raw_valid", "").lower() == "true" for row in rows)
		snapped_valid = sum(row.get("snapped_valid", "").lower() == "true" for row in rows)
		lines.extend([
			f"| Independently feasible raw trajectory | {raw_valid} ({100 * raw_valid / len(rows) if rows else 0:.1f}%) |",
			f"| Independently feasible after endpoint snap | {snapped_valid} ({100 * snapped_valid / len(rows) if rows else 0:.1f}%) |",
		])
	lines.extend([
		"", "| Distribution | Value |", "|---|---:|",
		f"| Solve time | {format_stats(times, 's')} |",
		f"| Relative gap | {format_stats(gaps, '%')} |", "",
	])
	if soc_calls:
		oracle_name = "SOCP" if args.oracle_backend == "socp" else "TPP"
		lines.extend([
			f"| {oracle_name} oracle metric | Value |", "|---|---:|",
			f"| Total oracle calls | {sum(soc_calls)} |",
			f"| Oracle calls per case | {format_stats([float(value) for value in soc_calls])} |",
			f"| Full oracle solve per call | {format_stats(soc_seconds_per_call, 'us')} |",
			f"| Backend compute per call | {format_stats(soc_optimize_seconds_per_call, 'us')} |", "",
		])
	if args.oracle_backend == "tpp" and tpp_socp_fallback_calls:
		lines.extend([
			"| TPP robustness metric | Value |", "|---|---:|",
			f"| Calls falling back to SOCP | {sum(tpp_socp_fallback_calls)} / {sum(soc_calls)} |", "",
		])
	lines.extend([
		"| Original TPP difficulty | Cases | Optimal | Median external time |",
		"|---|---:|---:|---:|",
	])
	for difficulty in ("easy", "medium", "hard", "unknown"):
		group = [row for row in rows if row["difficulty"] == difficulty]
		if not group:
			continue
		optimal = sum(row["is_optimal"].lower() == "true" for row in group)
		group_times = [float(row["solve_seconds"]) for row in group]
		lines.append(
			f"| {difficulty} | {len(group)} | {optimal} ({100 * optimal / len(group):.1f}%) | "
			f"{statistics.median(group_times):.3f}s |"
		)
	lines.extend(["", "| Polygons | Cases | Optimal | Median external time |", "|---:|---:|---:|---:|"])
	for polygon_count in sorted({int(row["polygons"]) for row in rows}):
		group = [row for row in rows if int(row["polygons"]) == polygon_count]
		optimal = sum(row["is_optimal"].lower() == "true" for row in group)
		lines.append(
			f"| {polygon_count} | {len(group)} | {optimal} ({100 * optimal / len(group):.1f}%) | "
			f"{statistics.median(float(row['solve_seconds']) for row in group):.3f}s |"
		)
	summary_path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	args.suite = args.suite.resolve()
	args.tspn_repo = args.tspn_repo.resolve()
	if args.worker:
		return worker(args)
	if args.time_limit < 1 or args.threads < 0 or args.workers < 1 or args.eps <= 0 or args.feasibility_tolerance <= 0 or args.validation_tolerance <= 0:
		raise SystemExit("Time, workers, optimality, feasibility, and validation tolerances must be positive; threads must be nonnegative")
	if not args.suite.exists():
		raise SystemExit(f"Suite does not exist: {args.suite}")
	if not (args.tspn_repo / "python/tspn_bnb2").exists():
		raise SystemExit(f"TSPN repository does not look valid: {args.tspn_repo}")

	metadata = read_metadata((args.metadata or args.suite.with_suffix(".csv")).resolve())
	cases = read_cases(args.suite)
	indices = choose_cases(args, len(cases), metadata)
	if args.resume:
		csv_path = args.resume.resolve()
		run_dir = csv_path.parent
	else:
		run_dir = args.output.resolve() / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
		csv_path = run_dir / f"{args.suite.stem}-tspn-{args.mode}.csv"
	run_dir.mkdir(parents=True, exist_ok=True)
	summary_path = csv_path.with_suffix(".md")
	log_path = csv_path.with_suffix(".log")
	pending = [index for index in indices if index not in load_completed(csv_path)]

	write_header = not csv_path.exists() or csv_path.stat().st_size == 0
	with csv_path.open("a", newline="") as csv_file, log_path.open("a") as log_file:
		writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
		if write_header:
			writer.writeheader()
		log_lock = threading.Lock()
		executor = ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending))))
		futures = {}
		try:
			futures = {executor.submit(run_case, args, index, log_file, log_lock): index for index in pending}
			for position, future in enumerate(as_completed(futures), start=1):
				index = futures[future]
				encoded = cases[index]
				row = result_row(args, index, encoded, metadata.get(index, {}), future.result())
				writer.writerow(row)
				csv_file.flush()
				print(f"[{position}/{len(pending)}] case {index}: {row['status']} in {row['solve_seconds']}s", flush=True)
		except KeyboardInterrupt:
			stop_active_processes()
			for future in futures:
				future.cancel()
			executor.shutdown(wait=True, cancel_futures=True)
			print("Interrupted; completed rows are already saved and --resume will skip them.", file=sys.stderr)
			return 130
		else:
			executor.shutdown(wait=True)

	write_summary(csv_path, summary_path, args)
	print(f"Results: {csv_path}")
	print(f"Summary: {summary_path}")
	print(f"Solver log: {log_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
