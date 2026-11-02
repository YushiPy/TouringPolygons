#!/usr/bin/env python3
"""Build the English SIICUSP34 page's corpus data from the armijo reference runs.

The English page (``en/``) compares, as the international abstract does, the
original code of Fekete et al. with one thread (configuration B, CHR root)
against our solver with one thread (configuration F) on the 558 instances of
their paper, as closed tours with free order (TSPN). Times follow the summary
in ``benchmarks/results-saved/fekete-reference-armijo-2026-10-30``:
the median of the three 300 s runs when all three close, otherwise the single
longer completed run. Unclosed cases show a feasible tour and a time limit,
not a completion time. Both solvers use U-L <= eps*U + eps. Tours come from the local run
records (``benchmarks/workspace/experiments/fekete-reference@armijo``), and
so do the distances of the precision panel, unrounded. The visiting order is
the order of first contact along each tour, as the page colours it. Convex
decompositions come from the C++ optimal partition, compiled at build time.

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

from build_event_data import path_length, polygon_contact  # noqa: E402
from build_usp_demo import optimal_decomposition  # noqa: E402
from fekete_reference import load_polygons  # noqa: E402

CAMPAIGN = "fekete-reference-armijo-2026-10-30"
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


def our_folders(experiments: Path) -> list[Path]:
	"""Completed F records in increasing budget order; unfinished runs have no records."""
	folders = [folder for folder in experiments.glob("F-ours-*") if (folder / "runs.jsonl").exists()]
	return sorted(folders, key=lambda folder: json.loads((folder / "plan.json").read_text())["time_limit"])


def our_tours(experiments: Path) -> dict[str, dict]:
	"""First closing run, or the latest-budget feasible tour if still unclosed."""
	tours: dict[str, dict] = {}
	for folder in our_folders(experiments):
		with (folder / "runs.jsonl").open(encoding="utf-8") as file:
			for line in file:
				run = json.loads(line)
				previous = tours.get(run["instance"])
				if run["result"].get("path") and (previous is None or
					(not previous["solved"] and (run["solved"] or run["time_limit"] > previous["time_limit"]))):
					tours[run["instance"]] = run
	return tours


def tour_distances(experiments: Path) -> dict[str, dict[str, float]]:
	"""Solver -> instance -> largest distance from a tour to a polygon over the 300 s runs.

	The CSV rounds this column to three digits, which moves a value of 1.003e-7 to 1e-7;
	the threshold counts need the recorded values."""
	distances: dict[str, dict[str, float]] = {"ours": {}, "fekete": {}}
	short = [folder for folder in our_folders(experiments) if json.loads((folder / "plan.json").read_text())["time_limit"] == 300]
	if len(short) != 1:
		raise ValueError("expected exactly one F campaign with a 300 s budget")
	for solver, folder, root in (("ours", short[0], "nosso"), ("fekete", experiments / "B-single-thread-300s", "CHR")):
		with (folder / "runs.jsonl").open(encoding="utf-8") as file:
			for line in file:
				run = json.loads(line)
				value = run.get("validation", {}).get("max_polygon_distance")
				if run["root_label"] != root or value is None:
					continue
				distances[solver][run["instance"]] = max(value, distances[solver].get(run["instance"], 0.0))
	return distances


def without_repeated_vertices(polygon, tolerance: float = 1e-9) -> list[list[float]]:
	points: list[list[float]] = []
	for point in polygon:
		if not points or math.dist(points[-1], point) >= tolerance:
			points.append(list(point))
	if len(points) > 1 and math.dist(points[0], points[-1]) < tolerance:
		points.pop()
	return points


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
	shapes = {row["instance"]: polygons[row["instance"]] for row in cases.values()}
	# 100 regions repeat a vertex within 1e-9, and the partition then returns the whole reflex
	# region as one piece; the drawing layer partitions them without the repeated vertex.
	flat = [without_repeated_vertices(polygon) for instance in sorted(shapes) for polygon in shapes[instance]]
	pieces = iter(optimal_decomposition(flat))
	decompositions = {instance: [next(pieces) for _ in shapes[instance]] for instance in sorted(shapes)}
	ours = solved_times(rows, "F", "ours")
	fekete = solved_times(rows, "B", "CHR")
	if set(fekete) - set(ours):
		raise ValueError("an instance is solved only by Fekete et al.")
	if set(tours) != {row["instance"] for row in cases.values()}:
		raise ValueError("every instance needs a recorded feasible tour")

	distances = tour_distances(args.experiments)
	if any(len(values) != len(cases) for values in distances.values()):
		raise ValueError("every instance needs a recorded tour distance for both solvers")

	result_rows = []
	for case, row in sorted(cases.items()):
		instance = row["instance"]
		shape = tuple(tuple(tuple(point) for point in polygon) for polygon in polygons[instance])
		run = tours[instance]
		path = [list(point) for point in run["result"]["path"]]
		if path[0] != path[-1]:
			raise ValueError(f"case {case + 1}: our tour is not closed")
		decomposition = decompositions[instance]
		contacts = []
		for polygon in shape:
			contact = polygon_contact(path, polygon)
			contacts.append({"point": list(contact[0]), "fraction": contact[1]} if contact else None)
		if None in contacts:
			raise ValueError(f"case {case + 1}: the tour misses a polygon")
		# First contact along the tour; regions met at the same point keep the solver's order.
		solver_rank = {int(index): rank for rank, index in enumerate(run["result"]["order"])}
		order = sorted(range(len(shape)), key=lambda index: (contacts[index]["fraction"], solver_rank.get(index, len(shape) + index)))
		result_rows.append({
			"case": case,
			"cycle": True,
			"fekete_seconds": fekete.get(case),
			"geometry": {"polygons": shape, "start": path[0], "target": path[0]},
			"instance": instance,
			"length": path_length([tuple(point) for point in path]),
			"lower_bound": run["result"]["lower_bound"],
			"order": order,
			"path": path,
			"polygons": int(row["polygons"]),
			"seconds": ours.get(case),
			"solved": case in ours,
			"time_limit": run["time_limit"],
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
			"ours_300": solved_within(rows, "F", "ours", 300),
			"ours_900": solved_within(rows, "F", "ours", 900),
			"ours_all": len(ours),
			"fekete_single_300": solved_within(rows, "B", "CHR", 300),
			"fekete_single_900": len(fekete),
			"fekete_paper_300": solved_within(rows, "A", "CHR", 300),
			"fekete_paper_900": len(solved_times(rows, "A", "CHR")),
		},
		"hardest": {"case": case_189, "seconds": ours.get(case_189), "solved": case_189 in ours,
			"time_limit": tours[cases[case_189]["instance"]]["time_limit"]},
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
		"config": {"fekete": "B (CHR root, 1 thread)", "ours": "F (1 thread)", "eps": 0.001,
			"stopping_rule": "U-L <= eps*U + eps"},
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
	parser.add_argument("--output", type=Path, default=ROOT / "apps/siicusp34/en/data/event-data.js")
	args = parser.parse_args()
	data = build(args)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text("window.TPPEventData = " + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";\n", encoding="utf-8")
	print(json.dumps({"output": str(args.output), "comparison": data["comparison"]}, indent=2))


if __name__ == "__main__":
	main()
