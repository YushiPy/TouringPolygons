#!/usr/bin/env python3
"""Summarize indexed solver runs from a generated free-order campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CaseMetadata:
	profile: str
	polygons: int


def percentile(values: Sequence[float], fraction: float) -> float:
	return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def parse_result(text: str) -> tuple[str, Path]:
	label, separator, raw_path = text.partition("=")
	if not separator or not label or not raw_path:
		raise argparse.ArgumentTypeError("expected LABEL=PATH")
	path = Path(raw_path)
	if not path.is_file():
		raise argparse.ArgumentTypeError(f"missing result file: {path}")
	return label, path


def load_metadata(path: Path, split: str) -> dict[int, CaseMetadata]:
	with path.open(newline="") as file:
		rows = list(csv.DictReader(file))
	return {
		int(row["case_index"]): CaseMetadata(row["profile"], int(row["polygons"]))
		for row in rows if row["split"] == split
	}


def load_results(path: Path) -> dict[int, dict[str, object]]:
	rows: dict[int, dict[str, object]] = {}
	for line in path.read_text().splitlines():
		if not line.strip():
			continue
		row = json.loads(line)
		case = int(row["case"])
		if case in rows:
			raise ValueError(f"duplicate case {case} in {path}")
		rows[case] = row
	return rows


def relative_gap(row: dict[str, object]) -> float:
	upper = float(row["upper_bound"])
	return (upper - float(row["lower_bound"])) / upper if upper > 0.0 else math.nan


def overall_table(
	results: Sequence[tuple[str, dict[int, dict[str, object]]]],
) -> list[str]:
	lines = [
		"| Solver | Valid | Proven | Time limits | Sum time | Median time | P95 time | Total calls | Median calls | Fallbacks | Median gap |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for label, indexed in results:
		rows = list(indexed.values())
		times = [float(row["seconds"]) for row in rows]
		calls = [int(row["calls"]) for row in rows]
		lines.append(
			f"| {label} | {sum(row.get('valid') is True for row in rows)}/{len(rows)} "
			f"| {sum(bool(row['exact']) for row in rows)}/{len(rows)} "
			f"| {sum(row['termination'] == 'time_limit' for row in rows)} "
			f"| {sum(times):.3f} s | {statistics.median(times):.6f} s "
			f"| {percentile(times, 0.95):.6f} s | {sum(calls):,} "
			f"| {statistics.median(calls):,.1f} "
			f"| {sum(int(row.get('fallback_calls', 0)) for row in rows):,} "
			f"| {statistics.median(relative_gap(row) for row in rows):.4%} |"
		)
	return lines


def profile_table(
	metadata: dict[int, CaseMetadata],
	results: Sequence[tuple[str, dict[int, dict[str, object]]]],
) -> list[str]:
	profiles = sorted({case.profile for case in metadata.values()})
	header = "| Profile | Cases | " + " | ".join(f"{label} proofs | {label} median calls" for label, _ in results) + " |"
	lines = [header, "|---|---:|" + "---:|---:|" * len(results)]
	for profile in profiles:
		indices = [index for index, case in metadata.items() if case.profile == profile]
		cells = [profile, str(len(indices))]
		for _, indexed in results:
			rows = [indexed[index] for index in indices]
			cells.extend((str(sum(bool(row["exact"]) for row in rows)), f"{statistics.median(int(row['calls']) for row in rows):,.1f}"))
		lines.append("| " + " | ".join(cells) + " |")
	return lines


def paired_table(
	results: Sequence[tuple[str, dict[int, dict[str, object]]]],
) -> list[str]:
	baseline_label, baseline = results[0]
	lines = [
		f"| Candidate vs {baseline_label} | Proofs gained | Proofs lost | Lower calls | Better incumbent | Worse incumbent |",
		"|---|---:|---:|---:|---:|---:|",
	]
	for label, candidate in results[1:]:
		gained = lost = lower_calls = better = worse = 0
		for case, before in baseline.items():
			after = candidate[case]
			gained += not bool(before["exact"]) and bool(after["exact"])
			lost += bool(before["exact"]) and not bool(after["exact"])
			lower_calls += int(after["calls"]) < int(before["calls"])
			difference = float(after["upper_bound"]) - float(before["upper_bound"])
			tolerance = 1e-10 * (1.0 + abs(float(before["upper_bound"])))
			better += difference < -tolerance
			worse += difference > tolerance
		lines.append(f"| {label} | {gained} | {lost} | {lower_calls} | {better} | {worse} |")
	return lines


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--index", type=Path, required=True)
	parser.add_argument("--split", choices=("diagnostic", "heldout"), required=True)
	parser.add_argument("--result", type=parse_result, action="append", required=True)
	parser.add_argument("--output", type=Path)
	args = parser.parse_args(argv)

	metadata = load_metadata(args.index, args.split)
	results = [(label, load_results(path)) for label, path in args.result]
	for label, indexed in results:
		missing = sorted(set(metadata) - set(indexed))
		extra = sorted(set(indexed) - set(metadata))
		if missing or extra:
			raise SystemExit(f"{label}: missing {len(missing)} cases and found {len(extra)} unexpected cases")

	lines = [
		f"# Free-order {args.split} comparison",
		"",
		"Rows are joined by `case_index`; result-file order is ignored.",
		"",
		"## Overall",
		"",
		*overall_table(results),
		"",
		"## By profile",
		"",
		*profile_table(metadata, results),
		"",
		"## Paired against the first solver",
		"",
		*paired_table(results),
	]
	text = "\n".join(lines) + "\n"
	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(text)
	print(text, end="")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
