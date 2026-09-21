#!/usr/bin/env python3
"""Build the static SIICUSP34 event snapshot from the comparison CSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/scripts"))

from analyze_german_comparison import build_analysis, orient_path, route_polygon_distance  # noqa: E402
from benchmark_cases import read_encoded_cases  # noqa: E402


Point = tuple[float, float]


def as_bool(value: str) -> bool:
	return value.strip().lower() in {"1", "true", "yes"}


def path_length(path: list[Point]) -> float:
	return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def cross(a: Point, b: Point, c: Point) -> float:
	return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_on_segment(point: Point, a: Point, b: Point, epsilon: float = 1e-10) -> bool:
	return (
		abs(cross(a, b, point)) <= epsilon
		and min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
		and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
	)


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
	inside = False
	for index, a in enumerate(polygon):
		b = polygon[(index + 1) % len(polygon)]
		if point_on_segment(point, a, b):
			return True
		if (a[1] > point[1]) != (b[1] > point[1]):
			x_crossing = (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
			if point[0] <= x_crossing:
				inside = not inside
	return inside


def segment_intersection_fraction(a: Point, b: Point, c: Point, d: Point) -> float | None:
	r = (b[0] - a[0], b[1] - a[1])
	s = (d[0] - c[0], d[1] - c[1])
	denominator = r[0] * s[1] - r[1] * s[0]
	if abs(denominator) <= 1e-14:
		candidates = [
			point for point in (a, b, c, d)
			if point_on_segment(point, a, b) and point_on_segment(point, c, d)
		]
		if not candidates:
			return None
		length_squared = r[0] * r[0] + r[1] * r[1]
		if length_squared <= 1e-24:
			return 0.0
		return max(0.0, min(1.0, ((candidates[0][0] - a[0]) * r[0] + (candidates[0][1] - a[1]) * r[1]) / length_squared))
	q = (c[0] - a[0], c[1] - a[1])
	t = (q[0] * s[1] - q[1] * s[0]) / denominator
	u = (q[0] * r[1] - q[1] * r[0]) / denominator
	return max(0.0, min(1.0, t)) if -1e-10 <= t <= 1.0 + 1e-10 and -1e-10 <= u <= 1.0 + 1e-10 else None


def polygon_contact(path: list[Point], polygon: tuple[Point, ...]) -> tuple[Point, float] | None:
	if not path or not polygon:
		return None
	total = path_length(path)
	traversed = 0.0
	best: tuple[Point, float] | None = None
	for a, b in zip(path, path[1:]):
		segment_length = math.dist(a, b)
		if point_in_polygon(a, polygon):
			fraction = traversed / total if total else 0.0
			return (a, fraction)
		for c, d in zip(polygon, polygon[1:] + polygon[:1]):
			rate = segment_intersection_fraction(a, b, c, d)
			if rate is None:
				continue
			fraction = (traversed + rate * segment_length) / total if total else 0.0
			candidate = ((a[0] + rate * (b[0] - a[0]), a[1] + rate * (b[1] - a[1])), fraction)
			if best is None or fraction < best[1]:
				best = candidate
		traversed += segment_length
	if point_in_polygon(path[-1], polygon):
		return (path[-1], 1.0)
	return best


def validation(start: Point, target: Point, polygons: tuple[tuple[Point, ...], ...], path: list[Point], expected_length: float) -> dict[str, object]:
	actual_length = path_length(path)
	start_distance = math.dist(path[0], start) if path else math.inf
	target_distance = math.dist(path[-1], target) if path else math.inf
	polygon_distances = [route_polygon_distance(path, polygon) for polygon in polygons]
	max_distance = max(polygon_distances, default=math.inf)
	endpoint_valid = start_distance <= 1e-7 and target_distance <= 1e-7
	polygon_valid = max_distance <= 1e-7
	length_valid = abs(actual_length - expected_length) <= max(1e-7, abs(expected_length) * 1e-9)
	return {
		"endpoint_valid": endpoint_valid,
		"length_valid": length_valid,
		"max_polygon_distance": max_distance,
		"polygon_valid": polygon_valid,
		"recomputed_length": actual_length,
		"start_distance": start_distance,
		"target_distance": target_distance,
		"valid": endpoint_valid and polygon_valid and length_valid,
	}


def read_csv(path: Path, delimiter: str) -> dict[int, dict[str, str]]:
	with path.open(newline="", encoding="utf-8") as file:
		return {int(row["case_index"]): row for row in csv.DictReader(file, delimiter=delimiter)}


def read_legacy_decompositions(path: Path | None) -> dict[int, tuple[tuple[tuple[float, float], ...], tuple[tuple[tuple[float, float], ...], ...]]]:
	if path is None:
		return {}
	text = path.read_text(encoding="utf-8")
	legacy = json.loads(text.split("=", 1)[1].rstrip(" ;\n"))
	return {
		int(row["case"]): (
			tuple(tuple(tuple(point) for point in polygon) for polygon in row.get("geometry", {}).get("polygons", [])),
			tuple(
				tuple(tuple(tuple(point) for point in piece) for piece in polygon)
				for polygon in row.get("visualization", {}).get("decomposition", [])
			),
		)
		for row in legacy.get("rows", [])
	}


def build(args: argparse.Namespace) -> dict[str, object]:
	cases = read_encoded_cases(args.instances)
	ours = read_csv(args.ours, ";")
	fekete = read_csv(args.fekete, ",")
	legacy_decompositions = read_legacy_decompositions(args.legacy_event_data)
	if set(ours) != set(range(len(cases))) or set(fekete) != set(ours):
		raise ValueError("Expected the same 558 case indices in both CSVs and instances.bin")

	rows = []
	for case in cases:
		row = ours[case.case_index]
		if row["sha256"] != case.digest or fekete[case.case_index]["sha256"] != case.digest:
			raise ValueError(f"Case {case.case_index} does not match instances.bin")
		start_x, start_y, target_x, target_y = struct.unpack_from("<dddd", case.data)
		start = (start_x, start_y)
		target = (target_x, target_y)
		path = orient_path(json.loads(row["path"]), start, target)
		order = json.loads(row["order"])
		contacts = []
		for polygon in case.polygons:
			contact = polygon_contact(path, polygon)
			contacts.append({"point": list(contact[0]), "fraction": contact[1]} if contact else None)
		expected_length = float(row["final_length"])
		legacy_geometry, decomposition = legacy_decompositions.get(case.case_index, ((), ()))
		if legacy_geometry != case.polygons:
			decomposition = ()
		rows.append({
			"calls": int(row["calls"]),
			"case": case.case_index,
			"exact": as_bool(row["exact"]),
			"fallback_calls": int(row["fallback_calls"]),
			"geometry": {"polygons": case.polygons, "start": list(start), "target": list(target)},
			"length": expected_length,
			"lower_bound": float(row["lower_bound"]),
			"order": order,
			"path": path,
			"polygons": case.polygon_count,
			"relative_gap": float(row["final_relative_gap"]),
			"seconds": float(row["seconds"]),
			"sha256": row["sha256"],
			"termination": row["termination"],
			"upper_bound": float(row["upper_bound"]),
			"validation": validation(start, target, case.polygons, path, expected_length),
			"visualization": {"contacts": contacts, "decomposition": decomposition},
		})

	ours_times = [row["seconds"] for row in rows]
	common = [
		float(fekete[index]["solve_seconds"]) / row["seconds"]
		for index, row in enumerate(rows)
		if as_bool(fekete[index]["is_optimal"]) and row["seconds"] > 0
	]
	comparison = {
		"fekete_completed": sum(as_bool(row["is_optimal"]) for row in fekete.values()),
		"fekete_time_limit_seconds": 21600,
		"fekete_unresolved": sum(not as_bool(row["is_optimal"]) for row in fekete.values()),
		"median_speedup_fekete_over_ours": statistics.median(common),
		"ours_faster_count": sum(value > 1 for value in common),
		"common_completed": len(common),
	}
	analysis = build_analysis(argparse.Namespace(ours=args.ours, fekete=args.fekete, instances=args.instances))
	time = analysis["time_seconds"]
	speedup = analysis["speedup"]
	length = analysis["length"]
	precision = analysis["precision"]
	comparison.update({
		"time": {
			"ours_mean_seconds": time["ours_all"]["mean"],
			"ours_median_seconds": time["ours_all"]["median"],
			"ours_total_hours": time["total_cpu_hours"]["ours_all"],
			"fekete_mean_seconds": time["fekete_all_recorded"]["mean"],
			"fekete_median_seconds": time["fekete_all_recorded"]["median"],
			"fekete_total_hours": time["total_cpu_hours"]["fekete_all_recorded"],
			"common_ours_median_seconds": time["ours_common_completed"]["median"],
			"common_fekete_median_seconds": time["fekete_completed_only"]["median"],
		},
		"geometric_mean_speedup": speedup["stats"]["geometric_mean"],
		"fekete_faster_count": speedup["fekete_faster_count"],
		"ties_count": speedup["ties_count"],
		"length": {
			"common_completed": length["completed_common_instances"],
			"ratio_median": length["ratio_fekete_over_ours"]["median"],
			"ratio_mean": length["ratio_fekete_over_ours"]["mean"],
			"fekete_longer_count": length["fekete_longer_count"],
			"fekete_shorter_count": length["fekete_shorter_count"],
		},
		"precision": {
			"definition": precision["definition"],
			"ours_max_per_instance": precision["ours_max_per_instance"]["stats"],
			"fekete_raw_max_per_instance": precision["fekete_raw_max_per_instance"]["stats"],
			"ours_instances_at_1e-7": precision["ours_max_per_instance"]["thresholds"][5]["count"],
			"fekete_raw_instances_at_1e-7": precision["fekete_raw_max_per_instance"]["thresholds"][5]["count"],
		},
	})
	data = {
		"config": {"solver_threads": 1, "validation_tolerance": 1e-7},
		"corpus": "german",
		"notes": [
			"Este é o corpus de instâncias usado por Fekete et al. no artigo; a página compara os dois solvers no mesmo problema adaptado, com extremos fixos e ordem livre.",
			"Nosso solver certificou 558/558 instâncias. No conjunto comum concluído, o speedup mediano Fekete/nosso é {:.2f}×; o solver de Fekete et al. não concluiu 8 instâncias no limite de 6 horas.".format(comparison["median_speedup_fekete_over_ours"]),
			"Os caminhos e pontos de contato são os exportados pela rodada final.",
		],
		"provenance": {
			"date": "2026-09-21",
			"run_id": "german-comparison-20260921",
			"ours_csv": str(args.ours),
			"fekete_csv": str(args.fekete),
			"binary_sha256": hashlib.sha256(args.instances.read_bytes()).hexdigest(),
			"ours_csv_sha256": hashlib.sha256(args.ours.read_bytes()).hexdigest(),
			"fekete_csv_sha256": hashlib.sha256(args.fekete.read_bytes()).hexdigest(),
		},
		"rows": rows,
		"schema_version": 1,
		"status": "completed",
		"summary": {
			"cases": len(rows),
			"valid_paths": sum(row["validation"]["valid"] for row in rows),
			"exact_certified": sum(row["exact"] for row in rows),
			"resolved_under_10_seconds": sum(row["seconds"] < 10 for row in rows),
			"termination_counts": dict(Counter(row["termination"] for row in rows)),
			"solver_seconds": sum(ours_times),
			"calls": sum(row["calls"] for row in rows),
			"fallback_calls": sum(row["fallback_calls"] for row in rows),
		},
		"title": "TPP · corpus de Fekete et al. · 558 instâncias",
		"visit_order": "free",
		"comparison": comparison,
	}
	return data


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--ours", type=Path, default=Path("benchmarks/results-saved/german-comparison/ours.csv"))
	parser.add_argument("--fekete", type=Path, default=Path("benchmarks/results-saved/german-comparison/fekete.csv"))
	parser.add_argument("--instances", type=Path, default=Path("benchmarks/results-saved/german-comparison/instances.bin"))
	parser.add_argument("--legacy-event-data", type=Path)
	parser.add_argument("--output", type=Path, default=Path("apps/siicusp34/data/event-data.js"))
	args = parser.parse_args()
	data = build(args)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text("window.TPPEventData = " + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";\n", encoding="utf-8")
	print(json.dumps({"output": str(args.output), "summary": data["summary"], "comparison": data["comparison"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
