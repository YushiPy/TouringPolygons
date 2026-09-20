#!/usr/bin/env python3
"""Run the Fekete et al. free-order TSPN solver for up to six hours per case.

The campaign reuses cases already certified by the matching 10-second run and
stores one atomic checkpoint per newly attempted case.  Re-running the same
command resumes the campaign; Ctrl-C loses only currently running cases.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import tspn_run_comparison as comparison


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = ROOT / "benchmarks/suites/german-instances.bin"
DEFAULT_REPO = ROOT / "tspn-comparison/solver"
DEFAULT_BASELINE = ROOT / "tspn-comparison/results/german-gurobi13-20260918/final.csv"
DEFAULT_OUTPUT = ROOT / "tspn-comparison/results/fekete-free-order-6h/final.csv"
DEFAULT_TIME_LIMIT = 6 * 60 * 60
DEFAULT_WORKERS = 8
EPS = 0.001
FEASIBILITY_TOLERANCE = 0.001
VALIDATION_TOLERANCE = 1e-7
TERMINAL_STATUSES = {"optimal", "limit", "process_timeout"}
VENV_MARKER = "TOURING_POLYGONS_FEKETE_VENV"


def parser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(description=__doc__)
	result.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
	result.add_argument("--tspn-repo", type=Path, default=DEFAULT_REPO)
	result.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
	result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	result.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT)
	result.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
	result.add_argument("--case", type=int, action="append", help="Run only this case; repeatable.")
	result.add_argument("--dry-run", action="store_true", help="Validate inputs and show pending work.")
	return result


def sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as file:
		for block in iter(lambda: file.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def source_tree_sha256(repo: Path) -> str:
	"""Fingerprint solver sources without hashing caches or build artifacts."""
	interesting_names = {"CMakeLists.txt", "pyproject.toml", "setup.py", "conanfile.txt"}
	interesting_suffixes = {".cpp", ".h", ".hpp", ".py"}
	digest = hashlib.sha256()
	for top in (repo / "tspn_core", repo / "python"):
		for path in sorted(item for item in top.rglob("*") if item.is_file()):
			if path.name not in interesting_names and path.suffix not in interesting_suffixes:
				continue
			relative = path.relative_to(repo).as_posix().encode()
			digest.update(len(relative).to_bytes(8, "little"))
			digest.update(relative)
			data = path.read_bytes()
			digest.update(len(data).to_bytes(8, "little"))
			digest.update(data)
	return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(
			"w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
		) as file:
			file.write(text)
			file.flush()
			os.fsync(file.fileno())
			temporary = Path(file.name)
		os.replace(temporary, path)
	finally:
		if temporary is not None and temporary.exists():
			temporary.unlink()


def atomic_write_csv(path: Path, rows: dict[int, dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(
			"w", encoding="utf-8", newline="", dir=path.parent,
			prefix=f".{path.name}.", delete=False,
		) as file:
			writer = csv.DictWriter(file, fieldnames=comparison.RESULT_FIELDS, extrasaction="ignore")
			writer.writeheader()
			for index in sorted(rows):
				writer.writerow(rows[index])
			file.flush()
			os.fsync(file.fileno())
			temporary = Path(file.name)
		os.replace(temporary, path)
	finally:
		if temporary is not None and temporary.exists():
			temporary.unlink()


def parse_bool(value: Any) -> bool:
	return str(value).strip().lower() in {"1", "true", "yes"}


def same_float(value: Any, expected: float) -> bool:
	try:
		return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-15)
	except (TypeError, ValueError):
		return False


def load_baseline(path: Path, cases: list[comparison.EncodedCase]) -> dict[int, dict[str, str]]:
	with path.open(newline="") as file:
		rows = list(csv.DictReader(file))
	if len(rows) != len(cases):
		raise SystemExit(f"Baseline has {len(rows)} rows; expected {len(cases)}: {path}")
	indexed: dict[int, dict[str, str]] = {}
	for row in rows:
		try:
			index = int(row["case_index"])
		except (KeyError, ValueError) as error:
			raise SystemExit(f"Invalid case_index in baseline: {error}") from error
		if index in indexed or not 0 <= index < len(cases):
			raise SystemExit(f"Duplicate or out-of-range baseline case: {index}")
		if row.get("sha256") != cases[index].digest:
			raise SystemExit(f"Baseline hash mismatch for case {index}")
		if not (
			row.get("mode") == "path"
			and row.get("threads") == "1"
			and row.get("oracle_backend") == "socp"
			and same_float(row.get("eps"), EPS)
			and same_float(row.get("feasibility_tolerance"), FEASIBILITY_TOLERANCE)
		):
			raise SystemExit(f"Baseline configuration mismatch for case {index}")
		indexed[index] = row
	if len(indexed) != len(cases):
		raise SystemExit("Baseline does not contain every suite case")
	return indexed


def binding_path(repo: Path) -> Path:
	bindings = sorted((repo / "python/tspn_bnb2/core").glob("_tspn_bindings*.so"))
	if len(bindings) != 1:
		raise SystemExit(f"Expected one compiled TSPN binding in {repo}; found {len(bindings)}")
	return bindings[0]


def ensure_solver_python(args: argparse.Namespace) -> None:
	# Keep the venv symlink itself: resolving it would execute the Homebrew base
	# interpreter and silently drop the environment's installed solver metadata.
	python = (args.tspn_repo / ".venv/bin/python").absolute()
	if not python.exists():
		raise SystemExit(f"Solver virtual-environment Python does not exist: {python}")
	if os.environ.get(VENV_MARKER) == str(python):
		return
	environment = os.environ.copy()
	environment[VENV_MARKER] = str(python)
	os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], environment)


def solver_versions() -> dict[str, Any]:
	code = (
		"import importlib.metadata, json; import gurobipy; "
		"print(json.dumps({'tspn_bnb2': importlib.metadata.version('tspn_bnb2'), "
		"'gurobi': list(gurobipy.gurobi.version())}))"
	)
	completed = subprocess.run(
		[sys.executable, "-c", code], check=True, capture_output=True, text=True,
	)
	return json.loads(completed.stdout)


def campaign_config(args: argparse.Namespace, binding: Path) -> dict[str, Any]:
	return {
		"schema_version": 1,
		"problem": "free-order TPP path with fixed endpoints",
		"suite": str(args.suite),
		"suite_sha256": sha256_file(args.suite),
		"baseline": str(args.baseline),
		"baseline_sha256": sha256_file(args.baseline),
		"solver_repo": str(args.tspn_repo),
		"solver_source_sha256": source_tree_sha256(args.tspn_repo),
		"solver_binding": str(binding),
		"solver_binding_sha256": sha256_file(binding),
		"mode": "path",
		"time_limit_seconds": args.time_limit,
		"threads_per_instance": 1,
		"eps": EPS,
		"feasibility_tolerance": FEASIBILITY_TOLERANCE,
		"validation_tolerance": VALIDATION_TOLERANCE,
		"oracle_backend": "socp",
	}


def initialize_manifest(path: Path, config: dict[str, Any], versions: dict[str, Any]) -> None:
	if path.exists():
		manifest = json.loads(path.read_text())
		if manifest.get("config") != config:
			raise SystemExit(
				f"Campaign configuration differs from the existing manifest: {path}\n"
				"Use another --output path instead of mixing experiments."
			)
		return
	payload = {
		"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
		"config": config,
		"versions": versions,
		"host": {
			"platform": platform.platform(),
			"machine": platform.machine(),
			"processor": platform.processor(),
			"logical_cpus": os.cpu_count(),
			"python": sys.version,
		},
	}
	atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def checkpoint_matches(row: dict[str, Any], index: int, case: comparison.EncodedCase, args: argparse.Namespace) -> bool:
	return (
		int(row.get("case_index", -1)) == index
		and row.get("sha256") == case.digest
		and row.get("mode") == "path"
		and int(row.get("threads", -1)) == 1
		and int(row.get("time_limit_seconds", -1)) == args.time_limit
		and row.get("oracle_backend") == "socp"
		and same_float(row.get("eps"), EPS)
		and same_float(row.get("feasibility_tolerance"), FEASIBILITY_TOLERANCE)
		and same_float(row.get("validation_tolerance"), VALIDATION_TOLERANCE)
	)


def load_checkpoints(
	case_dir: Path, cases: list[comparison.EncodedCase], args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
	rows: dict[int, dict[str, Any]] = {}
	if not case_dir.exists():
		return rows
	for path in sorted(case_dir.glob("*.json")):
		try:
			payload = json.loads(path.read_text())
			row = payload["row"]
			index = int(row["case_index"])
		except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
			raise SystemExit(f"Invalid checkpoint {path}: {error}") from error
		if not 0 <= index < len(cases) or not checkpoint_matches(row, index, cases[index], args):
			raise SystemExit(f"Checkpoint does not match this campaign: {path}")
		rows[index] = row
	return rows


def combined_rows(
	baseline: dict[int, dict[str, str]], checkpoints: dict[int, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], set[int]]:
	rows: dict[int, dict[str, Any]] = {
		index: row for index, row in baseline.items()
		if row.get("status") == "optimal" and parse_bool(row.get("is_optimal"))
	}
	seeded = set(rows)
	rows.update(checkpoints)
	return rows, seeded


def write_status(
	output: Path,
	rows: dict[int, dict[str, Any]],
	seeded: set[int],
	total_cases: int,
	workers: int,
	time_limit: int,
) -> None:
	terminal = {index for index, row in rows.items() if row.get("status") in TERMINAL_STATUSES}
	optimal = sum(
		row.get("status") == "optimal" and parse_bool(row.get("is_optimal")) for row in rows.values()
	)
	limits = sum(row.get("status") in {"limit", "process_timeout"} for row in rows.values())
	errors = sum(row.get("status") == "error" for row in rows.values())
	limit_label = f"{time_limit / 3600:g} h" if time_limit >= 3600 else f"{time_limit} s"
	status = {
		"updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
		"total_cases": total_cases,
		"reused_from_10s_baseline": len(seeded),
		"new_terminal_cases": len(terminal - seeded),
		"terminal_cases": len(terminal),
		"optimal_within_0_1_percent": optimal,
		"time_limit_exhausted": limits,
		"errors_to_retry": errors,
		"pending_cases": total_cases - len(terminal),
		"workers": workers,
		"threads_per_instance": 1,
		"time_limit_seconds": time_limit,
	}
	atomic_write_text(output.with_name("status.json"), json.dumps(status, indent=2, sort_keys=True) + "\n")
	lines = [
		"# Fekete et al. — free-order TPP campaign", "",
		"The solver has one thread per instance. Independent instances may run concurrently.", "",
		"| Measure | Value |", "|---|---:|",
		f"| Total instances | {total_cases} |",
		f"| Reused certificates from the matching 10 s run | {len(seeded)} |",
		f"| New terminal cases | {len(terminal - seeded)} |",
		f"| Certified within 0.1% | {optimal} |",
		f"| Did not certify within {limit_label} | {limits} |",
		f"| Errors to retry | {errors} |",
		f"| Pending | {total_cases - len(terminal)} |", "",
		"`optimal` means that the solver's relative upper/lower-bound gap is at most 0.1%.",
		"The campaign is complete only when Pending and Errors are both zero.", "",
	]
	atomic_write_text(output.with_suffix(".md"), "\n".join(lines))


def save_outputs(
	output: Path,
	baseline: dict[int, dict[str, str]],
	checkpoints: dict[int, dict[str, Any]],
	total_cases: int,
	workers: int,
	time_limit: int,
) -> tuple[dict[int, dict[str, Any]], set[int]]:
	rows, seeded = combined_rows(baseline, checkpoints)
	atomic_write_csv(output, rows)
	write_status(output, rows, seeded, total_cases, workers, time_limit)
	return rows, seeded


def pending_order(
	indices: Sequence[int], baseline: dict[int, dict[str, str]], terminal: set[int],
) -> list[int]:
	def key(index: int) -> tuple[float, int, int]:
		try:
			gap = float(baseline[index].get("relative_gap", "inf"))
		except ValueError:
			gap = math.inf
		return gap, int(baseline[index].get("polygons", 0)), index
	return sorted((index for index in indices if index not in terminal), key=key)


def main(argv: Sequence[str] | None = None) -> int:
	args = parser().parse_args(argv)
	args.suite = args.suite.resolve()
	args.tspn_repo = args.tspn_repo.resolve()
	args.baseline = args.baseline.resolve()
	args.output = args.output.resolve()
	if args.time_limit < 1 or args.workers < 1:
		raise SystemExit("--time-limit and --workers must be positive")
	for required in (args.suite, args.baseline, args.tspn_repo / "python/tspn_bnb2"):
		if not required.exists():
			raise SystemExit(f"Required input does not exist: {required}")

	ensure_solver_python(args)
	binding = binding_path(args.tspn_repo)
	cases = comparison.read_cases(args.suite)
	baseline = load_baseline(args.baseline, cases)
	if args.case:
		indices = sorted(set(args.case))
		invalid = [index for index in indices if not 0 <= index < len(cases)]
		if invalid:
			raise SystemExit(f"Case indices outside suite: {invalid}")
	else:
		indices = list(range(len(cases)))

	config = campaign_config(args, binding)
	versions = solver_versions()
	case_dir = args.output.parent / "cases"
	checkpoints = load_checkpoints(case_dir, cases, args)
	rows, seeded = combined_rows(baseline, checkpoints)
	terminal = {index for index, row in rows.items() if row.get("status") in TERMINAL_STATUSES}
	pending = pending_order(indices, baseline, terminal)

	print(f"Suite: {len(cases)} instances")
	print(f"Reused from matching 10 s run: {len(seeded)}")
	print(f"Already checkpointed at {args.time_limit}s: {len(terminal - seeded)}")
	print(f"Pending in this invocation: {len(pending)}")
	print(f"Execution: {args.workers} independent workers, 1 solver thread each")
	print(f"Worst-case remaining wall time: {math.ceil(len(pending) / args.workers) * args.time_limit / 3600:.1f} h")
	print(f"Output: {args.output}")
	if args.dry_run:
		print("Dry run complete; no campaign files were changed.")
		return 0

	args.output.parent.mkdir(parents=True, exist_ok=True)
	manifest_path = args.output.parent / "manifest.json"
	if not manifest_path.exists() and (args.output.exists() or any(case_dir.glob("*.json"))):
		raise SystemExit(f"Refusing to adopt existing output without a manifest: {args.output.parent}")
	initialize_manifest(manifest_path, config, versions)
	rows, seeded = save_outputs(
		args.output, baseline, checkpoints, len(cases), args.workers, args.time_limit,
	)
	if not pending:
		print("Nothing to run; campaign is complete for the selected cases.")
		return 0

	run_args = argparse.Namespace(
		suite=args.suite,
		tspn_repo=args.tspn_repo,
		mode="path",
		time_limit=args.time_limit,
		threads=1,
		eps=EPS,
		feasibility_tolerance=FEASIBILITY_TOLERANCE,
		validation_tolerance=VALIDATION_TOLERANCE,
		oracle_backend="socp",
		oracle_tolerance=VALIDATION_TOLERANCE,
	)
	log_path = args.output.with_suffix(".log")
	log_lock = threading.Lock()
	interrupted = False
	previous_sigterm = signal.getsignal(signal.SIGTERM)

	def interrupt_on_sigterm(_signum: int, _frame: Any) -> None:
		raise KeyboardInterrupt

	signal.signal(signal.SIGTERM, interrupt_on_sigterm)
	with log_path.open("a") as log_file:
		executor = ThreadPoolExecutor(max_workers=min(args.workers, len(pending)))
		futures: dict[Future[dict[str, Any]], int] = {}
		try:
			futures = {
				executor.submit(comparison.run_case, run_args, index, log_file, log_lock): index
				for index in pending
			}
			for position, future in enumerate(as_completed(futures), start=1):
				index = futures[future]
				try:
					payload = future.result()
				except Exception as error:  # preserve the failure and retry on the next invocation
					payload = {"status": "error", "error": f"{type(error).__name__}: {error}"}
				row = comparison.result_row(run_args, index, cases[index], {}, payload)
				checkpoint = {"schema_version": 1, "row": row}
				checkpoint_path = case_dir / f"{index:04d}.json"
				atomic_write_text(checkpoint_path, json.dumps(checkpoint, allow_nan=True, sort_keys=True) + "\n")
				checkpoints[index] = row
				save_outputs(
					args.output, baseline, checkpoints, len(cases), args.workers, args.time_limit,
				)
				print(
					f"[{position}/{len(pending)}] case {index}: {row['status']} "
					f"in {row.get('solve_seconds', 'n/a')} s",
					flush=True,
				)
		except KeyboardInterrupt:
			interrupted = True
			comparison.stop_active_processes()
			for future in futures:
				future.cancel()
			executor.shutdown(wait=True, cancel_futures=True)
		else:
			executor.shutdown(wait=True)
		finally:
			signal.signal(signal.SIGTERM, previous_sigterm)

	if interrupted:
		print("Interrupted safely. Run the same command to resume.", file=sys.stderr)
		return 130
	final_rows, _ = combined_rows(baseline, checkpoints)
	errors = [row for row in final_rows.values() if row.get("status") == "error"]
	if errors:
		print(f"Campaign stopped with {len(errors)} retryable errors; run the same command again.", file=sys.stderr)
		return 1
	print(f"Campaign pass complete. Results: {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
