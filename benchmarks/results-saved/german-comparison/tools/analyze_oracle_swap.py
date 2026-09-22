#!/usr/bin/env python3
"""Build a paired report for standalone, German-SOCP, and German-TPP results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(description=__doc__)
	result.add_argument("--standalone", type=Path, required=True)
	result.add_argument("--socp", type=Path, required=True)
	result.add_argument("--tpp", type=Path, required=True)
	result.add_argument("--output", type=Path, required=True)
	return result


def read_jsonl(path: Path) -> dict[str, dict]:
	return {row["sha256"]: row for row in map(json.loads, path.read_text().splitlines())}


def read_csv(path: Path) -> dict[str, dict]:
	with path.open() as file:
		return {row["sha256"]: row for row in csv.DictReader(file)}


def truth(value) -> bool:
	return value is True or str(value).lower() == "true"


def numbers(rows: list[dict], key: str, conversion=float) -> list:
	return [conversion(row[key]) for row in rows if row.get(key) not in (None, "")]


def summarize(rows: list[dict], *, seconds: str, calls: str, exact: str, valid: str) -> dict:
	times = numbers(rows, seconds)
	counts = numbers(rows, calls, lambda value: int(float(value)))
	return {
		"cases": len(rows),
		"optimal": sum(truth(row[exact]) for row in rows),
		"valid": sum(truth(row[valid]) for row in rows),
		"errors": sum(bool(row.get("error")) or row.get("status") == "error" for row in rows),
		"total_seconds": sum(times),
		"median_seconds": statistics.median(times) if times else None,
		"total_calls": sum(counts),
		"call_cases": len(counts),
		"median_calls": statistics.median(counts) if counts else None,
	}


def main(argv: list[str] | None = None) -> int:
	args = parser().parse_args(argv)
	standalone = read_jsonl(args.standalone)
	socp = read_csv(args.socp)
	tpp = read_csv(args.tpp)
	hashes = sorted(standalone.keys() & socp.keys() & tpp.keys())
	if not hashes:
		raise SystemExit("No matching instance hashes")
	groups = {
		"Standalone TPP": ([standalone[key] for key in hashes], "seconds", "calls", "exact", "valid"),
		"German + SOCP": ([socp[key] for key in hashes], "solve_seconds", "soc_num_calls", "is_optimal", "raw_valid"),
		"German + TPP": ([tpp[key] for key in hashes], "solve_seconds", "soc_num_calls", "is_optimal", "raw_valid"),
	}
	summaries = {name: summarize(rows, seconds=seconds, calls=calls, exact=exact, valid=valid)
		for name, (rows, seconds, calls, exact, valid) in groups.items()}
	paired_call_hashes = [key for key in hashes if socp[key].get("soc_num_calls") and tpp[key].get("soc_num_calls")]
	paired = {
		"standalone_faster_than_german_socp": sum(standalone[key]["seconds"] < float(socp[key]["solve_seconds"]) for key in hashes),
		"standalone_faster_than_german_tpp": sum(standalone[key]["seconds"] < float(tpp[key]["solve_seconds"]) for key in hashes),
		"german_tpp_faster_than_german_socp": sum(float(tpp[key]["solve_seconds"]) < float(socp[key]["solve_seconds"]) for key in hashes),
		"german_tpp_fewer_calls_than_german_socp": sum(int(tpp[key]["soc_num_calls"]) < int(socp[key]["soc_num_calls"]) for key in paired_call_hashes),
		"german_call_pairs": len(paired_call_hashes),
	}
	data = {"cases": len(hashes), "summaries": summaries, "paired": paired}
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
	lines = ["# Free-order oracle-swap comparison", "",
		"All rows are matched by encoded-instance SHA-256.", "",
		"| Solver | Optimal | Valid | Errors | Total time | Median time | Total calls | Median calls |",
		"|---|---:|---:|---:|---:|---:|---:|---:|"]
	for name, summary in summaries.items():
		lines.append(f"| {name} | {summary['optimal']}/{summary['cases']} | {summary['valid']}/{summary['cases']} | {summary['errors']} | "
			f"{summary['total_seconds']:.3f}s | {summary['median_seconds']:.6f}s | "
			f"{summary['total_calls']} ({summary['call_cases']} cases) | {summary['median_calls']:.1f} |")
	lines.extend(["", "## Paired outcomes", "",
		f"- Standalone TPP faster than German + SOCP: {paired['standalone_faster_than_german_socp']}/{len(hashes)}.",
		f"- Standalone TPP faster than German + TPP: {paired['standalone_faster_than_german_tpp']}/{len(hashes)}.",
		f"- Inside German search, TPP faster than SOCP: {paired['german_tpp_faster_than_german_socp']}/{len(hashes)}.",
		f"- Inside German search, TPP used fewer calls: {paired['german_tpp_fewer_calls_than_german_socp']}/{paired['german_call_pairs']} recorded pairs.",
		"", "Calls are comparable between the two German-search rows because only the oracle backend changes. "
		"Standalone calls arise from a different search tree and should not be treated as identical work units.", ""])
	args.output.write_text("\n".join(lines))
	print(args.output)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
