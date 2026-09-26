#!/usr/bin/env python3
"""Build convex partitions for the static trace simulations."""

from __future__ import annotations

import json
import re
from pathlib import Path

from build_usp_demo import optimal_decomposition

ROOT = Path(__file__).resolve().parents[1]
EVENT_DATA = ROOT / "data/event-data.js"
TRACE_DATA = ROOT / "data/trace-data.js"
OUTPUT = ROOT / "data/trace-decomposition.js"


def read_global(path: Path) -> dict:
	source = path.read_text(encoding="utf-8")
	return json.loads(source.split("=", 1)[1].rstrip(" ;\n"))


def partition_or_skip_invalid(polygons: list[list[list[float]]]) -> tuple[list[list], list[int]]:
	"""Keep validated C++ partitions and report geometries it cannot certify."""
	remaining = list(enumerate(polygons))
	partitions: list[list] = [[] for _ in polygons]
	skipped: list[int] = []
	while remaining:
		try:
			computed = optimal_decomposition([polygon for _, polygon in remaining])
		except ValueError as error:
			match = re.search(r"in region (\d+)", str(error))
			if not match:
				raise
			invalid_index = int(match.group(1))
			if invalid_index >= len(remaining):
				raise
			original_index, _ = remaining.pop(invalid_index)
			skipped.append(original_index)
			continue
		for (original_index, _), pieces in zip(remaining, computed):
			if len(pieces) > 1:
				partitions[original_index] = pieces
		break
	return partitions, skipped


def build() -> tuple[dict[str, list[list]], dict[str, int]]:
	event_data = read_global(EVENT_DATA)
	trace_data = read_global(TRACE_DATA)
	rows = {row["case"]: row for row in event_data["rows"]}
	trace_cases = sorted(
		int(key)
		for key, trace in trace_data["cases"].items()
		if trace.get("omitted_events", 0) == 0 and trace.get("event_count", 0) < 200
	)
	polygon_refs = [
		(case_index, polygon_index, polygon)
		for case_index in trace_cases
		for polygon_index, polygon in enumerate(rows[case_index]["geometry"]["polygons"])
	]
	partitions, skipped_indices = partition_or_skip_invalid([ref[2] for ref in polygon_refs])
	incomplete_cases = {polygon_refs[index][0] for index in skipped_indices}
	by_case: dict[str, list[list]] = {}
	for (case_index, polygon_index, _), pieces in zip(polygon_refs, partitions):
		if case_index in incomplete_cases or not pieces:
			continue
		case_key = str(case_index)
		by_case.setdefault(case_key, [[] for _ in rows[case_index]["geometry"]["polygons"]])
		by_case[case_key][polygon_index] = pieces
	return by_case, {
		"trace_cases": len(trace_cases),
		"polygons": len(polygon_refs),
		"decomposed_polygons": sum(bool(pieces) for regions in by_case.values() for pieces in regions),
		"skipped_polygons": len(skipped_indices),
		"cases_with_uncertified_polygons": len(incomplete_cases),
		"cases_with_decomposition": len(by_case),
	}


def main() -> None:
	decompositions, summary = build()
	OUTPUT.write_text(
		"window.TPPTraceDecompositions = "
		+ json.dumps(decompositions, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
		+ ";\n",
		encoding="utf-8",
	)
	print(json.dumps({"output": str(OUTPUT), **summary}, indent=2))


if __name__ == "__main__":
	main()
