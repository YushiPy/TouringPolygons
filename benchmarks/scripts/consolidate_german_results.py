#!/usr/bin/env python3
"""Build the canonical German benchmark CSV from the progressive runs."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/free-order-vs-german-20260919"


def read_rows(path: Path) -> dict[int, dict[str, str]]:
	with path.open(newline="") as file:
		return {int(row["case_index"]): row for row in csv.DictReader(file, delimiter=";")}


def is_true(row: dict[str, str], field: str) -> bool:
	return row.get(field, "").lower() == "true"


def exact(row: dict[str, str]) -> bool:
	return is_true(row, "exhausted") and not is_true(row, "time_limited") and not is_true(row, "branch_limited")


def termination(row: dict[str, str]) -> str:
	if exact(row):
		return "optimal"
	if is_true(row, "tolerance_reached"):
		return "optimality_tolerance"
	if is_true(row, "time_limited"):
		return "time_limit"
	if is_true(row, "branch_limited"):
		return "branch_limit"
	return "call_limit"


def main() -> None:
	base_path = RESULTS / "german-final-1800s.csv"
	round_path = RESULTS / "german-round-7200s-unlimited.csv"
	branch_path = RESULTS / "german-branch-limited-rerun-7200s.csv"
	output_path = RESULTS / "german-results-final.csv"

	base = read_rows(base_path)
	no_branch_round = read_rows(round_path)
	branch_rerun = read_rows(branch_path)
	if set(no_branch_round) & set(branch_rerun):
		raise ValueError("replacement runs overlap")
	if not set(no_branch_round) <= set(base) or not set(branch_rerun) <= set(base):
		raise ValueError("replacement run contains a case outside the base result")

	merged = dict(base)
	merged.update(no_branch_round)
	merged.update(branch_rerun)
	if set(merged) != set(base):
		raise ValueError("consolidated case set differs from german-final-1800s.csv")

	base_header = list(next(iter(base.values())).keys())
	fieldnames = list(base_header)
	for row in (*base.values(), *no_branch_round.values(), *branch_rerun.values()):
		for field in row:
			if field not in fieldnames:
				fieldnames.append(field)

	status_index = fieldnames.index("branch_limited") + 1
	for field in ("exact", "termination"):
		if field in fieldnames:
			fieldnames.remove(field)
		fieldnames.insert(status_index, field)
		status_index += 1
	temporary = output_path.with_suffix(output_path.suffix + ".tmp")
	with temporary.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
		writer.writeheader()
		for index in sorted(merged):
			row = dict(merged[index])
			row["exact"] = "true" if exact(row) else "false"
			row["termination"] = termination(row)
			writer.writerow(row)
	temporary.replace(output_path)

	print(f"Wrote {len(merged)} rows to {output_path}")


if __name__ == "__main__":
	main()
