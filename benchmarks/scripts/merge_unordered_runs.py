#!/usr/bin/env python3
"""Merge a complete unordered run with a replacement run for selected cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> dict[int, dict]:
	rows = {}
	for line in path.read_text().splitlines():
		if not line.strip():
			continue
		row = json.loads(line)
		case = int(row["case"])
		if case in rows:
			raise ValueError(f"{path}: duplicate case {case}")
		rows[case] = row
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("baseline", type=Path, help="Complete JSONL run")
	parser.add_argument("replacement", type=Path, help="JSONL run containing cases to replace")
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args()

	baseline = read_rows(args.baseline)
	replacement = read_rows(args.replacement)
	if not replacement:
		raise ValueError("replacement run is empty")
	unknown = sorted(set(replacement) - set(baseline))
	if unknown:
		raise ValueError(f"replacement contains cases absent from baseline: {unknown}")
	for case, row in replacement.items():
		if row.get("sha256") != baseline[case].get("sha256"):
			raise ValueError(f"case {case}: input hash changed between runs")
		row["benchmark_phase"] = "refinement_10s"
		baseline[case] = row
	for row in baseline.values():
		row.setdefault("benchmark_phase", "initial_2s")

	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w") as file:
		for case in sorted(baseline):
			file.write(json.dumps(baseline[case], separators=(",", ":")) + "\n")
	print(json.dumps({
		"cases": len(baseline),
		"replaced": len(replacement),
		"output": str(args.output),
	}, indent=2))


if __name__ == "__main__":
	main()
