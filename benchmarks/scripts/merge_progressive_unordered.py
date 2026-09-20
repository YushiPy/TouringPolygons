#!/usr/bin/env python3
"""Merge progressive unordered-solver rounds into a complete result artifact.

Each ``--round`` argument has the form ``SECONDS=PATH``.  A later round
replaces an earlier timeout for the same case.  An earlier certificate is
retained if a later run times out, so a transient timeout cannot erase a
valid result.  The selected rows are also written to a compact CSV for
per-instance reporting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            case = int(row["case"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid result row") from error
        if case in rows:
            raise ValueError(f"{path}: duplicate case {case}")
        rows[case] = row
    return rows


def select_row(previous: dict | None, candidate: dict) -> dict:
    if previous is None:
        return candidate
    if previous.get("exact") and not candidate.get("exact"):
        return previous
    return candidate


def parse_round(value: str) -> tuple[float, Path]:
    try:
        seconds_text, path_text = value.split("=", 1)
        seconds = float(seconds_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "round must use SECONDS=PATH, for example 600=round-600s.jsonl"
        ) from error
    if seconds <= 0 or not path_text:
        raise argparse.ArgumentTypeError("round seconds must be positive")
    return seconds, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="complete JSONL result set to update")
    parser.add_argument("--suite", type=Path, required=True,
                        help="encoded suite used by every round")
    parser.add_argument("--round", dest="rounds", type=parse_round, action="append",
                        required=True, help="SECONDS=JSONL, in ascending time order")
    parser.add_argument("--output", type=Path, required=True,
                        help="merged complete JSONL output")
    parser.add_argument("--csv", type=Path, required=True,
                        help="selected focus-case CSV output")
    parser.add_argument("--summary", type=Path, required=True,
                        help="campaign summary JSON output")
    args = parser.parse_args()

    baseline = read_rows(args.baseline)
    selected: dict[int, dict] = {}
    selected_round: dict[int, float] = {}
    round_stats = []
    for seconds, path in args.rounds:
        rows = read_rows(path)
        unknown = sorted(set(rows) - set(baseline))
        if unknown:
            raise ValueError(f"{path}: cases absent from baseline: {unknown}")
        exact_count = sum(bool(row.get("exact")) for row in rows.values())
        round_stats.append({
            "seconds": seconds,
            "path": str(path),
            "sha256": sha256(path),
            "cases": len(rows),
            "exact": exact_count,
            "time_limit": sum(row.get("termination") == "time_limit" for row in rows.values()),
        })
        for case, row in rows.items():
            expected_hash = baseline[case].get("sha256")
            if expected_hash and row.get("sha256") != expected_hash:
                raise ValueError(f"case {case}: input hash changed between runs")
            previous = selected.get(case)
            chosen = select_row(previous, row)
            selected[case] = chosen
            if chosen is row:
                selected_round[case] = seconds

    if not selected:
        raise ValueError("rounds are empty")
    if set(selected) != set().union(*(set(read_rows(path)) for _, path in args.rounds)):
        raise AssertionError("internal case-selection error")

    merged = dict(baseline)
    for case, row in selected.items():
        merged_row = dict(row)
        merged_row["benchmark_phase"] = f"progressive_{selected_round[case]:g}s"
        merged[case] = merged_row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        for case in sorted(merged):
            file.write(json.dumps(merged[case], separators=(",", ":")) + "\n")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case", "polygons", "round_seconds", "exact", "termination", "seconds",
        "lower_bound", "upper_bound", "calls", "nodes", "valid", "sha256",
    ]
    with args.csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for case in sorted(selected):
            row = selected[case]
            writer.writerow({
                "case": case,
                "polygons": row.get("polygons"),
                "round_seconds": selected_round[case],
                "exact": row.get("exact"),
                "termination": row.get("termination"),
                "seconds": row.get("seconds"),
                "lower_bound": row.get("lower_bound"),
                "upper_bound": row.get("upper_bound"),
                "calls": row.get("calls"),
                "nodes": row.get("nodes"),
                "valid": row.get("valid"),
                "sha256": row.get("sha256"),
            })

    summary = {
        "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline), "cases": len(baseline)},
        "suite": {"path": str(args.suite), "sha256": sha256(args.suite)},
        "rounds": round_stats,
        "focus_cases": len(selected),
        "focus_exact": sum(bool(row.get("exact")) for row in selected.values()),
        "focus_time_limit": sum(row.get("termination") == "time_limit" for row in selected.values()),
        "merged_cases": len(merged),
        "merged_exact": sum(bool(row.get("exact")) for row in merged.values()),
        "merged_output": {"path": str(args.output), "sha256": sha256(args.output)},
        "focus_csv": {"path": str(args.csv), "sha256": sha256(args.csv)},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
