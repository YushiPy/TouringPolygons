"""Compare the maintained free-order solver at strict and Fekete-equivalent gaps."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import free_order_campaign
from benchmark_cases import EncodedCase, read_encoded_cases
from unordered_runner import run_unordered_solver
from unordered_validation import validate_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = ROOT / "benchmarks/suites/german-instances.bin"
DEFAULT_OUTPUT = ROOT / "benchmarks/results/free-order-gap-comparison"
MAX_CALLS = 100_000_000
VALIDATION_TOLERANCE = 1e-7
FEKETE_GAP = 0.001
FEKETE_EQUIVALENT_RELATIVE_GAP = FEKETE_GAP / (1.0 + FEKETE_GAP)
VARIANTS: dict[str, dict[str, float]] = {
    "strict": {"absolute_gap": 1e-7, "relative_gap": 1e-9},
    "fekete_gap": {
        "absolute_gap": 0.0,
        "relative_gap": FEKETE_EQUIVALENT_RELATIVE_GAP,
    },
}

RESULT_FIELDS = [
    "case", "sha256", "polygons", "variant", "time_limit_seconds",
    "workers", "attempted_at", "status", "exact", "valid", "seconds", "process_seconds",
    "calls", "lower_bound", "upper_bound", "final_absolute_gap",
    "final_relative_gap", "initial_upper_bound", "termination", "validation_error", "error",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
            temporary = Path(file.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(payload, allow_nan=True, indent=2, sort_keys=True) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    result.add_argument("--solver", type=Path, default=free_order_campaign.BINARY)
    result.add_argument("--time-limit", "--seconds", type=float, required=True,
                        help="Maximum solver time per instance and gap setting.")
    result.add_argument("--workers", type=int, default=8,
                        help="Concurrent instances (each solver process is single-threaded).")
    result.add_argument("--max-calls", type=int, default=MAX_CALLS)
    result.add_argument("--case", type=int, action="append",
                        help="Run only this zero-based case; may be repeated.")
    result.add_argument("--limit", type=int, help="Run at most this many selected cases.")
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Campaign directory. Reuse the same directory to resume or extend the time limit.")
    result.add_argument("--no-build", action="store_true",
                        help="Use the existing solver binary without building it.")
    result.add_argument("--dry-run", action="store_true")
    return result


def campaign_config(args: argparse.Namespace, solver: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "problem": "free-order TPP path with fixed endpoints",
        "suite": str(args.suite),
        "suite_sha256": sha256_file(args.suite),
        "solver": str(solver),
        "solver_sha256": sha256_file(solver),
        "max_calls": args.max_calls,
        "validation_tolerance": VALIDATION_TOLERANCE,
        "variants": VARIANTS,
        "fekete_gap_mapping": {
            "criterion": "UB <= (1 + eps) * LB",
            "eps": FEKETE_GAP,
            "our_absolute_gap": 0.0,
            "our_relative_gap": FEKETE_EQUIVALENT_RELATIVE_GAP,
        },
    }


def initialize_manifest(path: Path, config: dict[str, Any], create: bool = True) -> None:
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("config") != config:
            raise SystemExit(
                f"Campaign configuration differs from the manifest: {path}\n"
                "Use a new --output directory instead of mixing solver or suite versions."
            )
        return
    if not create:
        return
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": sys.version,
        },
        "time_limit_seconds": "recorded per attempt; increase it to retry only unfinished cases",
        "workers": "recorded per invocation; each solver run uses the maintained single-threaded binary",
    }
    atomic_json(path, payload)


def checkpoint_path(output: Path, case_index: int, variant: str) -> Path:
    return output / "cases" / f"{case_index:04d}-{variant}.json"


def load_histories(
    output: Path, cases: list[EncodedCase],
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    histories: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for case in cases:
        for variant in VARIANTS:
            path = checkpoint_path(output, case.case_index, variant)
            if not path.exists():
                histories[(case.case_index, variant)] = []
                continue
            try:
                payload = json.loads(path.read_text())
                rows = payload["runs"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise SystemExit(f"Invalid checkpoint {path}: {error}") from error
            if payload.get("case") != case.case_index or payload.get("sha256") != case.digest:
                raise SystemExit(f"Checkpoint does not match suite case {case.case_index}: {path}")
            if payload.get("variant") != variant or not isinstance(rows, list):
                raise SystemExit(f"Checkpoint has an unexpected solver variant: {path}")
            histories[(case.case_index, variant)] = rows
    return histories


def should_run(history: list[dict[str, Any]], time_limit: float) -> bool:
    if any(row.get("exact") is True for row in history):
        return False
    if any(row.get("status") == "call_limit" for row in history):
        return False
    usable = [row for row in history if row.get("status") != "error"]
    if any(float(row.get("time_limit_seconds", 0)) >= time_limit for row in usable):
        return False
    return True


def checkpoint(
    output: Path,
    case: EncodedCase,
    variant: str,
    history: list[dict[str, Any]],
) -> None:
    atomic_json(checkpoint_path(output, case.case_index, variant), {
        "case": case.case_index,
        "sha256": case.digest,
        "variant": variant,
        "runs": history,
    })


def solve_variant(
    case: EncodedCase,
    variant: str,
    solver: Path,
    time_limit: float,
    max_calls: int,
    workers: int,
) -> dict[str, Any]:
    start_x, start_y, target_x, target_y = struct.unpack_from("<dddd", case.data)
    settings = VARIANTS[variant]
    arguments = [
        "--absolute-gap", str(settings["absolute_gap"]),
        "--relative-gap", str(settings["relative_gap"]),
    ]
    row: dict[str, Any] = {
        "case": case.case_index,
        "sha256": case.digest,
        "polygons": case.polygon_count,
        "variant": variant,
        "absolute_gap": settings["absolute_gap"],
        "relative_gap": settings["relative_gap"],
        "time_limit_seconds": time_limit,
        "workers": workers,
        "attempted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    began = time.perf_counter()
    try:
        result = run_unordered_solver(
            solver,
            (start_x, start_y),
            (target_x, target_y),
            case.polygons,
            max_calls,
            time_limit,
            arguments=arguments,
        )
        row.update(result)
        row["status"] = "optimal" if result.get("exact") else result.get("termination", "incomplete")
        path = result.get("path")
        if path:
            try:
                validation = validate_path(
                    (start_x, start_y), (target_x, target_y), case.polygons,
                    path, VALIDATION_TOLERANCE,
                )
                row["validation"] = validation
                row["valid"] = validation["valid"]
            except ImportError as error:
                row["valid"] = None
                row["validation"] = None
                row["validation_error"] = f"Independent path validation unavailable: {error}"
        else:
            row["valid"] = False
            row["validation"] = None
    except Exception as error:
        row["status"] = "error"
        row["exact"] = False
        row["valid"] = False
        row["error"] = f"{type(error).__name__}: {error}"
    row["process_seconds"] = time.perf_counter() - began
    return row


def run_case_pair(
    case: EncodedCase,
    histories: dict[tuple[int, str], list[dict[str, Any]]],
    output: Path,
    solver: Path,
    time_limit: float,
    max_calls: int,
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    order = ["strict", "fekete_gap"] if case.case_index % 2 == 0 else ["fekete_gap", "strict"]
    updated: dict[str, list[dict[str, Any]]] = {}
    for variant in order:
        history = list(histories[(case.case_index, variant)])
        if should_run(history, time_limit):
            row = solve_variant(case, variant, solver, time_limit, max_calls, workers)
            history.append(row)
            checkpoint(output, case, variant, history)
            elapsed = row.get("seconds")
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
                elapsed = row["process_seconds"]
            print(
                f"case {case.case_index} | {variant}: {row['status']} "
                f"in {elapsed:.3f} s "
                f"(limit {time_limit:g} s)",
                flush=True,
            )
        updated[variant] = history
    return updated


def latest_by_case_variant(
    histories: dict[tuple[int, str], list[dict[str, Any]]],
) -> dict[tuple[int, str], dict[str, Any]]:
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for key, rows in histories.items():
        if not rows:
            continue
        exact_rows = [row for row in rows if row.get("exact") is True]
        if exact_rows:
            latest[key] = min(exact_rows, key=lambda row: float(row.get("seconds", math.inf)))
        else:
            latest[key] = max(
                rows,
                key=lambda row: (float(row.get("time_limit_seconds", 0)), row.get("attempted_at", "")),
            )
    return latest


def write_artifacts(
    output: Path,
    histories: dict[tuple[int, str], list[dict[str, Any]]],
    total_cases: int,
) -> None:
    all_rows = [row for key in sorted(histories) for row in histories[key]]
    for row in all_rows:
        if row.get("valid") is None and not row.get("validation_error"):
            row["validation_error"] = "Independent validation unavailable or not recorded for this attempt."
    csv_path = output / "runs.csv"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output, prefix=".runs.", delete=False,
    ) as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())
        temporary = Path(file.name)
    os.replace(temporary, csv_path)

    latest = latest_by_case_variant(histories)
    comparison_fields = [
        "case", "sha256", "polygons",
        "strict_status", "strict_exact", "strict_valid", "strict_time_limit_seconds",
        "strict_seconds", "strict_relative_gap",
        "fekete_gap_status", "fekete_gap_exact", "fekete_gap_valid",
        "fekete_gap_time_limit_seconds", "fekete_gap_seconds", "fekete_gap_relative_gap",
        "strict_over_fekete_gap_speedup",
    ]
    paired_rows = []
    case_indices = sorted({case for case, _variant in latest})
    for index in case_indices:
        strict = latest.get((index, "strict"), {})
        relaxed = latest.get((index, "fekete_gap"), {})
        row: dict[str, Any] = {
            "case": index,
            "sha256": strict.get("sha256", relaxed.get("sha256", "")),
            "polygons": strict.get("polygons", relaxed.get("polygons", "")),
        }
        for prefix, source in (("strict", strict), ("fekete_gap", relaxed)):
            for field in ("status", "exact", "valid", "time_limit_seconds", "seconds", "final_relative_gap"):
                output_field = "relative_gap" if field == "final_relative_gap" else field
                row[f"{prefix}_{output_field}"] = source.get(field, "")
        if strict.get("exact") and relaxed.get("exact"):
            strict_seconds = float(strict.get("seconds", 0))
            relaxed_seconds = float(relaxed.get("seconds", 0))
            row["strict_over_fekete_gap_speedup"] = (
                strict_seconds / relaxed_seconds if relaxed_seconds > 0 else ""
            )
        else:
            row["strict_over_fekete_gap_speedup"] = ""
        paired_rows.append(row)

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output, prefix=".comparison.", delete=False,
    ) as file:
        writer = csv.DictWriter(file, fieldnames=comparison_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paired_rows)
        file.flush()
        os.fsync(file.fileno())
        temporary = Path(file.name)
    os.replace(temporary, output / "comparison.csv")

    lines = [
        "# Free-order gap comparison", "",
        f"Instances in suite: {total_cases}",
        "The strict and permissive settings use the same C++ solver binary and one thread per run.",
        "The permissive relative gap is `0.001 / 1.001`; with zero absolute gap, this matches "
        "Fekete's stopping condition `UB <= 1.001 * LB`.",
        "Exact runs are reused at any later time limit. Non-exact runs are retried only when "
        "the requested per-instance time limit exceeds every prior non-error attempt.", "",
        "| Setting | Cases with results | Closed gap | Incomplete | Errors | Median seconds when exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, label in (("strict", "Strict (1e-7 abs + 1e-9 rel)"),
                           ("fekete_gap", "Fekete-equivalent gap")):
        rows = [latest[(index, variant)] for index in case_indices if (index, variant) in latest]
        exact_rows = [row for row in rows if row.get("exact") is True]
        seconds = [float(row["seconds"]) for row in exact_rows if row.get("seconds") is not None]
        errors = sum(row.get("status") == "error" for row in rows)
        incomplete = sum(not row.get("exact") and row.get("status") != "error" for row in rows)
        lines.append(
            f"| {label} | {len(rows)} | {len(exact_rows)} | {incomplete} | {errors} | "
            f"{statistics.median(seconds):.6g} |" if seconds else
            f"| {label} | {len(rows)} | {len(exact_rows)} | {incomplete} | {errors} | — |"
        )
    both_exact = [
        row for row in paired_rows if row.get("strict_exact") and row.get("fekete_gap_exact")
    ]
    speedups = [float(row["strict_over_fekete_gap_speedup"]) for row in both_exact
                if row.get("strict_over_fekete_gap_speedup") not in ("", None)]
    lines.extend([
        "", f"Paired cases with both settings closing the gap: {len(both_exact)}.",
        "Median strict/permissive runtime ratio among those cases: "
        + (f"{statistics.median(speedups):.4g}×." if speedups else "not available."),
        "Ratios above 1 mean the permissive run was faster. Timed-out rows are censored at their budget.",
        "Independent validation unavailable for "
        f"{sum(row.get('valid') is None for row in all_rows)} result(s); see `validation_error` in `runs.csv`.",
        "", "Files: `runs.csv` contains all attempts; `comparison.csv` contains the latest result per case.", "",
    ])
    atomic_write(output / "summary.md", "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser().error("--time-limit must be finite and positive")
    if args.workers < 1 or args.max_calls < 0 or (args.limit is not None and args.limit < 1):
        parser().error("--workers and --limit must be positive; --max-calls must be nonnegative")
    args.suite = args.suite.resolve()
    args.output = args.output.resolve()
    if not args.suite.is_file():
        raise SystemExit(f"Suite does not exist: {args.suite}")
    if args.solver.resolve() == free_order_campaign.BINARY.resolve():
        solver = free_order_campaign.ensure_binary(args.no_build)
    else:
        if args.no_build:
            parser().error("--no-build applies only to the default solver path")
        solver = args.solver.resolve()
    if not solver.is_file():
        raise SystemExit(f"Solver executable does not exist: {solver}")

    all_cases = read_encoded_cases(args.suite)
    cases = all_cases
    if args.case:
        selected = set(args.case)
        invalid = sorted(index for index in selected if index < 0 or index >= len(cases))
        if invalid:
            raise SystemExit(f"Case indices outside suite: {invalid}")
        cases = [case for case in cases if case.case_index in selected]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("No suite instances selected.")

    config = campaign_config(args, solver)
    manifest_path = args.output / "manifest.json"
    if not manifest_path.exists() and (args.output / "cases").exists():
        raise SystemExit(f"Refusing to adopt checkpoints without a manifest: {args.output}")
    initialize_manifest(manifest_path, config, create=not args.dry_run)
    histories = load_histories(args.output, all_cases)
    pending = [
        case for case in cases
        if any(should_run(histories[(case.case_index, variant)], args.time_limit) for variant in VARIANTS)
    ]
    print(f"Suite instances: {len(cases)} selected; workers: {args.workers}; each solver process uses 1 thread.")
    print(f"Time limit per instance and setting: {args.time_limit:g} s; pending instance pairs: {len(pending)}.")
    print(f"Output: {args.output}")
    if args.dry_run:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    write_artifacts(args.output, histories, len(all_cases))
    if not pending:
        print("Nothing to run; completed or already-budgeted results were reused.")
        return 0

    errors = 0
    try:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = {
                executor.submit(
                    run_case_pair, case, histories, args.output, solver,
                    args.time_limit, args.max_calls, args.workers,
                ): case
                for case in pending
            }
            for future in as_completed(futures):
                case = futures[future]
                try:
                    updated = future.result()
                except Exception as error:
                    print(f"case {case.case_index}: worker error: {type(error).__name__}: {error}", file=sys.stderr)
                    errors += 1
                    continue
                for variant, rows in updated.items():
                    histories[(case.case_index, variant)] = rows
                errors += sum(
                    row.get("status") == "error" or row.get("valid") is False
                    for rows in updated.values() for row in rows[-1:]
                )
                write_artifacts(args.output, histories, len(all_cases))
    except KeyboardInterrupt:
        write_artifacts(args.output, load_histories(args.output, all_cases), len(all_cases))
        print("Interrupted safely; completed variant checkpoints were saved. Re-run the same command to resume.", file=sys.stderr)
        return 130

    latest = latest_by_case_variant(histories)
    selected_indices = {case.case_index for case in cases}
    unresolved_errors = sum(
        row.get("status") == "error" or row.get("valid") is False
        for (index, _variant), row in latest.items() if index in selected_indices
    )
    if errors or unresolved_errors:
        print(
            f"Campaign finished with {max(errors, unresolved_errors)} unresolved error/validation issue(s). "
            "Re-run with the same output directory to retry errors.",
            file=sys.stderr,
        )
        return 1
    print(f"Campaign pass complete. Summary: {args.output / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
