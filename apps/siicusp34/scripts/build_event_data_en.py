#!/usr/bin/env python3
"""Build the English SIICUSP34 page's corpus data from the armijo reference runs.

The English page (``en/``) compares, as the international abstract does, the
original code of Fekete et al. with one thread (configuration B, CHR root)
against our solver with one thread (configuration C) on the 558 instances of
their paper, as closed tours with free order (TSPN). Times follow the summary
in ``benchmarks/results-saved/README.md#fekete-reference-armijo-2026-10-17``:
the median of the three 300 s runs when all three close, otherwise the single
longer run (900 s; 6 h for our case 189). Tours come from the local run
records (``benchmarks/workspace/experiments/fekete-reference@armijo``);
convex decompositions are reused from the Portuguese snapshot, whose polygons
are checked to be identical.

From the repository root:

	python3 apps/siicusp34/scripts/build_event_data_en.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/_internal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_event_data import path_length, polygon_contact, read_legacy_decompositions  # noqa: E402
from fekete_reference import load_polygons  # noqa: E402

CAMPAIGN = "fekete-reference-armijo-2026-10-17"
SIZE_GROUPS = (5, 60)  # sizes quoted on the page, rounded to a multiple of 5 as in the abstract


def solved_times(rows: list[dict[str, str]], configuration: str, root: str) -> dict[int, float]:
	"""Case index -> seconds, for the cases this configuration closes."""
	runs: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
	for row in rows:
		if row["configuration"] == configuration and row["root"] == root:
			runs[int(row["case"]) - 1][int(row["time_limit"])] = row
	times = {}
	for case, by_limit in runs.items():
		if by_limit[300]["solved"] == "3":
			times[case] = float(by_limit[300]["seconds_median"])
			continue
		for limit in sorted(limit for limit in by_limit if limit > 300):
			if by_limit[limit]["solved"] == "1":
				times[case] = float(by_limit[limit]["seconds_median"])
				break
	return times


def solved_within(rows: list[dict[str, str]], configuration: str, root: str, limit: int) -> int:
	"""Cases closed in every run up to ``limit`` seconds (300 s: all three repetitions)."""
	solved = set()
	for row in rows:
		if row["configuration"] != configuration or row["root"] != root or int(row["time_limit"]) > limit:
			continue
		if row["solved"] == row["repetitions"]:
			solved.add(row["case"])
	return len(solved)


def our_tours(experiments: Path) -> dict[str, dict]:
	"""Instance -> the closing run of our solver (first repetition of 300 s, else the longer run)."""
	tours: dict[str, dict] = {}
	for folder in ("C-ours-2467373-300s", "C-ours-2467373-900s-unsolved", "C-ours-2467373-21600s-unsolved"):
		with (experiments / folder / "runs.jsonl").open(encoding="utf-8") as file:
			for line in file:
				run = json.loads(line)
				if run["solved"] and run["instance"] not in tours and run["result"].get("path"):
					tours[run["instance"]] = run
	return tours


def rounded_size(polygons: int) -> int:
	return 5 * max(1, round(polygons / 5))


def stats(values: list[float]) -> dict[str, float]:
	ordered = sorted(values)
	return {
		"n": len(ordered),
		"median": statistics.median(ordered),
		"p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
		"max": ordered[-1],
	}


def build(args: argparse.Namespace) -> dict[str, object]:
	with args.per_case.open(newline="", encoding="utf-8") as file:
		rows = list(csv.DictReader(file))
	cases: dict[int, dict[str, str]] = {}
	for row in rows:
		cases.setdefault(int(row["case"]) - 1, row)
	polygons = load_polygons(args.archive)
	tours = our_tours(args.experiments)
	legacy = read_legacy_decompositions(args.legacy_event_data)
	ours = solved_times(rows, "C", "ours")
	fekete = solved_times(rows, "B", "CHR")
	if set(fekete) - set(ours):
		raise ValueError("an instance is solved only by Fekete et al.")
	if set(ours) != set(cases):
		raise ValueError("our solver should close every instance")

	distances: dict[str, dict[int, float]] = {"ours": {}, "fekete": {}}
	for row in rows:
		if row["time_limit"] == "300" and (row["configuration"], row["root"]) in {("C", "ours"), ("B", "CHR")}:
			distances["ours" if row["configuration"] == "C" else "fekete"][int(row["case"]) - 1] = float(row["tour_max_distance"])

	result_rows = []
	for case, row in sorted(cases.items()):
		instance = row["instance"]
		shape = tuple(tuple(tuple(point) for point in polygon) for polygon in polygons[instance])
		run = tours[instance]
		path = [list(point) for point in run["result"]["path"]]
		if path[0] != path[-1]:
			raise ValueError(f"case {case + 1}: our tour is not closed")
		legacy_shape, decomposition = legacy.get(case, ((), ()))
		if legacy_shape != shape:
			raise ValueError(f"case {case + 1}: polygons differ from the Portuguese snapshot")
		contacts = []
		for polygon in shape:
			contact = polygon_contact(path, polygon)
			contacts.append({"point": list(contact[0]), "fraction": contact[1]} if contact else None)
		if None in contacts:
			raise ValueError(f"case {case + 1}: the tour misses a polygon")
		result_rows.append({
			"case": case,
			"cycle": True,
			"fekete_seconds": fekete.get(case),
			"geometry": {"polygons": shape, "start": path[0], "target": path[0]},
			"instance": instance,
			"length": path_length([tuple(point) for point in path]),
			"lower_bound": run["result"]["lower_bound"],
			"order": run["result"]["order"],
			"path": path,
			"polygons": int(row["polygons"]),
			"seconds": ours[case],
			"source": row["source"],
			"upper_bound": run["result"]["upper_bound"],
			"visualization": {"contacts": contacts, "decomposition": decomposition},
		})

	common = sorted(fekete)
	speedups = {case: fekete[case] / ours[case] for case in common}
	values = list(speedups.values())
	by_size: dict[int, list[float]] = defaultdict(list)
	for case in common:
		by_size[rounded_size(int(cases[case]["polygons"]))].append(speedups[case])
	common_ours = [ours[case] for case in common]
	common_fekete = [fekete[case] for case in common]
	case_189 = next(case for case, row in cases.items() if row["instance"] == "kuala_lumpur_n060_seed3530")
	comparison = {
		"common_completed": len(common),
		"median_speedup_fekete_over_ours": statistics.median(values),
		"mean_speedup_fekete_over_ours": statistics.fmean(values),
		"geometric_mean_speedup": math.exp(statistics.fmean(map(math.log, values))),
		"min_speedup": min(values),
		"ours_faster_count": sum(value > 1 for value in values),
		"fekete_faster_count": sum(value < 1 for value in values),
		"by_size": {
			str(size): {"n": len(by_size[size]), "median": statistics.median(by_size[size]), "mean": statistics.fmean(by_size[size])}
			for size in SIZE_GROUPS
		},
		"solved": {
			"ours_300": solved_within(rows, "C", "ours", 300),
			"ours_all": len(ours),
			"fekete_single_300": solved_within(rows, "B", "CHR", 300),
			"fekete_single_900": len(fekete),
			"fekete_paper_300": solved_within(rows, "A", "CHR", 300),
			"fekete_paper_900": len(solved_times(rows, "A", "CHR")),
		},
		"hardest": {"case": case_189, "seconds": ours[case_189]},
		"time": {
			"common_ours_median_seconds": statistics.median(common_ours),
			"common_fekete_median_seconds": statistics.median(common_fekete),
			"common_ours_mean_seconds": statistics.fmean(common_ours),
			"common_fekete_mean_seconds": statistics.fmean(common_fekete),
			"common_ours_total_seconds": sum(common_ours),
			"common_fekete_total_seconds": sum(common_fekete),
		},
		"precision": {
			"ours": stats(list(distances["ours"].values())),
			"fekete": stats(list(distances["fekete"].values())),
			"ours_above_1e-7": sum(value > 1e-7 for value in distances["ours"].values()),
			"fekete_above_1e-7": sum(value > 1e-7 for value in distances["fekete"].values()),
		},
	}
	return {
		"campaign": CAMPAIGN,
		"comparison": comparison,
		"config": {"fekete": "B (CHR root, 1 thread)", "ours": "C (1 thread)", "eps": 0.001},
		"corpus": "fekete",
		"provenance": {
			"per_case_csv": str(args.per_case.relative_to(ROOT)) if args.per_case.is_relative_to(ROOT) else str(args.per_case),
			"per_case_sha256": hashlib.sha256(args.per_case.read_bytes()).hexdigest(),
			"archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
		},
		"rows": result_rows,
		"schema_version": 1,
		"summary": {"cases": len(result_rows)},
		"title": "TSPN · instances of Fekete et al. · 558 cases",
	}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--per-case", type=Path, default=ROOT / f"benchmarks/results-saved/{CAMPAIGN}/per-case.csv")
	parser.add_argument("--experiments", type=Path, default=ROOT / "benchmarks/workspace/experiments/fekete-reference@armijo")
	parser.add_argument("--archive", type=Path, default=ROOT / "third_party/tspn-socg/instances/instances_socg_simplified.zip")
	parser.add_argument("--legacy-event-data", type=Path, default=ROOT / "apps/siicusp34/data/event-data.js")
	parser.add_argument("--output", type=Path, default=ROOT / "apps/siicusp34/en/data/event-data.js")
	args = parser.parse_args()
	data = build(args)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text("window.TPPEventData = " + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";\n", encoding="utf-8")
	print(json.dumps({"output": str(args.output), "comparison": data["comparison"]}, indent=2))


if __name__ == "__main__":
	main()
