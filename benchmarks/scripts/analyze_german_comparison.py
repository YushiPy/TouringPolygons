#!/usr/bin/env python3
"""Analyze the saved 558-instance German solver comparison.

The script intentionally treats the two CSVs as different artifacts: the local
solver export contains aggregate metrics and lengths, while the Fekete export
also contains trajectories.  It therefore reports the missing local trajectory
precision as unavailable instead of silently mixing in an older app export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence


Point = tuple[float, float]
Polygon = tuple[Point, ...]

TIME_EDGES = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
TIME_LABELS = ("< 1 ms", "1–10 ms", "10–100 ms", "0.1–1 s", "1–10 s", "10–100 s", "100 s–1 ks", "1–10 ks", "> 10 ks")
SPEEDUP_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1_024.0, math.inf)
SPEEDUP_LABELS = ("< 0.25×", "0.25–0.5×", "0.5–1×", "1–2×", "2–4×", "4–8×", "8–16×", "16–32×", "32–64×", "64–128×", "128–256×", "256–512×", "512–1024×", "> 1024×")
RATIO_EDGES = (0.0, 0.999, 1.0, 1.000001, 1.00001, 1.0001, 1.001, 1.01, 1.1, math.inf)
RATIO_LABELS = ("< 0.999", "0.999–1", "1–1.000001", "1.000001–1.00001", "1.00001–1.0001", "1.0001–1.001", "1.001–1.01", "1.01–1.1", "> 1.1")
PRECISION_EDGES = (0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, math.inf)
PRECISION_LABELS = ("0", "≤ 1e−12", "≤ 1e−10", "≤ 1e−9", "≤ 1e−8", "≤ 1e−7", "≤ 1e−6", "≤ 1e−5", "≤ 1e−4", "> 1e−3")


def finite(value: Any) -> float | None:
	try:
		result = float(value)
	except (TypeError, ValueError):
		return None
	return result if math.isfinite(result) else None


def read_cases(path: Path) -> list[dict[str, Any]]:
	data = path.read_bytes()
	offset = 0
	cases = []
	while offset < len(data):
		case_start = offset
		start = struct.unpack_from("<dd", data, offset)
		target = struct.unpack_from("<dd", data, offset + 16)
		offset += 32
		polygon_count = struct.unpack_from("<Q", data, offset)[0]
		offset += 8
		polygons: list[Polygon] = []
		for _ in range(polygon_count):
			vertex_count = struct.unpack_from("<Q", data, offset)[0]
			offset += 8
			polygon = tuple(
				struct.unpack_from("<dd", data, offset + index * 16)
				for index in range(vertex_count)
			)
			polygons.append(polygon)
			offset += vertex_count * 16
		solution_count = struct.unpack_from("<Q", data, offset)[0]
		offset += 8 + solution_count * 16
		encoded = data[case_start:offset]
		cases.append({
			"start": start,
			"target": target,
			"polygons": tuple(polygons),
			"sha256": hashlib.sha256(encoded).hexdigest(),
		})
	return cases


def orient_path(path: Sequence[Sequence[float]], start: Point, target: Point) -> list[Point]:
	points = [(float(point[0]), float(point[1])) for point in path]
	if len(points) < 2:
		return points
	forward = math.dist(points[0], start) + math.dist(points[-1], target)
	reverse = math.dist(points[0], target) + math.dist(points[-1], start)
	return list(reversed(points)) if reverse < forward else points


def cross(a: Point, b: Point, c: Point) -> float:
	return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_on_segment(point: Point, a: Point, b: Point, epsilon: float = 1e-12) -> bool:
	return (
		abs(cross(a, b, point)) <= epsilon
		and min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
		and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
	)


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
	o1, o2, o3, o4 = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
	if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)):
		return True
	return (
		point_on_segment(c, a, b)
		or point_on_segment(d, a, b)
		or point_on_segment(a, c, d)
		or point_on_segment(b, c, d)
	)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
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


def point_segment_distance(point: Point, a: Point, b: Point) -> float:
	dx, dy = b[0] - a[0], b[1] - a[1]
	length_squared = dx * dx + dy * dy
	if length_squared == 0:
		return math.dist(point, a)
	rate = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_squared))
	return math.hypot(point[0] - (a[0] + rate * dx), point[1] - (a[1] + rate * dy))


def segment_segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
	if segments_intersect(a, b, c, d):
		return 0.0
	return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d), point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def route_polygon_distance(path: Sequence[Point], polygon: Polygon) -> float:
	if not path or not polygon:
		return math.inf
	if any(point_in_polygon(point, polygon) for point in path):
		return 0.0
	route_segments = list(zip(path, path[1:])) or [(path[0], path[0])]
	polygon_segments = list(zip(polygon, polygon[1:] + polygon[:1]))
	return min(segment_segment_distance(a, b, c, d) for a, b in route_segments for c, d in polygon_segments)


def percentile(values: Sequence[float], fraction: float) -> float | None:
	if not values:
		return None
	ordered = sorted(values)
	position = (len(ordered) - 1) * fraction
	lower = math.floor(position)
	upper = math.ceil(position)
	if lower == upper:
		return ordered[lower]
	return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def numeric_stats(values: Iterable[float]) -> dict[str, float | int | None]:
	values = [value for value in values if math.isfinite(value)]
	if not values:
		return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "min": None, "max": None, "geometric_mean": None}
	return {
		"n": len(values),
		"mean": statistics.fmean(values),
		"median": statistics.median(values),
		"p90": percentile(values, 0.90),
		"p95": percentile(values, 0.95),
		"p99": percentile(values, 0.99),
		"min": min(values),
		"max": max(values),
		"geometric_mean": math.exp(statistics.fmean(math.log(value) for value in values if value > 0)) if all(value > 0 for value in values) else None,
	}


def histogram(values: Iterable[float], edges: Sequence[float], labels: Sequence[str]) -> dict[str, Any]:
	counts = [0] * len(labels)
	for value in values:
		for index in range(len(labels)):
			if edges[index] <= value < edges[index + 1]:
				counts[index] += 1
				break
	return {"labels": list(labels), "counts": counts}


def precision_summary(values: Sequence[float]) -> dict[str, Any]:
	thresholds = (0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
	return {
		"stats": numeric_stats(values),
		"histogram": histogram(values, PRECISION_EDGES, PRECISION_LABELS),
		"thresholds": [
			{"threshold": threshold, "count": sum(value <= threshold for value in values), "fraction": sum(value <= threshold for value in values) / len(values) if values else None}
			for threshold in thresholds
		],
	}


def read_csv(path: Path, delimiter: str) -> dict[int, dict[str, str]]:
	with path.open(newline="") as file:
		return {int(row["case_index"]): row for row in csv.DictReader(file, delimiter=delimiter)}


def compact(value: float | None, digits: int = 6) -> str:
	if value is None:
		return "—"
	if value == 0:
		return "0"
	return f"{value:.{digits}g}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
	lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
	lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
	return "\n".join(lines)


def build_analysis(args: argparse.Namespace) -> dict[str, Any]:
	cases = read_cases(args.instances)
	ours = read_csv(args.ours, ";")
	fekete = read_csv(args.fekete, ",")
	if len(cases) != 558 or set(ours) != set(fekete) or set(ours) != set(range(len(cases))):
		raise ValueError("The comparison must contain the same 558 case indices in both CSVs and instances.bin.")
	for case_index, case in enumerate(cases):
		if case["sha256"] != ours[case_index]["sha256"] or case["sha256"] != fekete[case_index]["sha256"]:
			raise ValueError(f"Case {case_index} does not match instances.bin by SHA-256.")

	per_instance = []
	precision_per_polygon = []
	common_completed = []
	all_ours_seconds = []
	all_fekete_seconds = []
	completed_fekete_seconds = []
	ours_lengths = []
	fekete_lengths = []
	raw_max_distances = []
	snapped_max_distances = []
	per_polygon_raw_distances = []
	per_polygon_snapped_distances = []

	for case_index, case in enumerate(cases):
		a = ours[case_index]
		b = fekete[case_index]
		ours_seconds = finite(a.get("seconds"))
		fekete_seconds = finite(b.get("solve_seconds"))
		if ours_seconds is not None:
			all_ours_seconds.append(ours_seconds)
		if fekete_seconds is not None:
			all_fekete_seconds.append(fekete_seconds)
		if b.get("is_optimal") == "True" and fekete_seconds is not None:
			completed_fekete_seconds.append(fekete_seconds)

		ours_length = finite(a.get("final_length"))
		fekete_length = finite(b.get("recomputed_length")) or finite(b.get("upper_bound"))
		if ours_length is not None:
			ours_lengths.append(ours_length)
		if fekete_length is not None:
			fekete_lengths.append(fekete_length)

		path_distances = {"raw": [], "snapped": []}
		for variant, field in path_distances.items():
			path_text = b.get("trajectory_json" if variant == "raw" else "snapped_trajectory_json")
			if not path_text:
				continue
			try:
				path = orient_path(json.loads(path_text), case["start"], case["target"])
			except (TypeError, ValueError, json.JSONDecodeError):
				continue
			path_distances[variant] = [route_polygon_distance(path, polygon) for polygon in case["polygons"]]
		for polygon_index in range(len(case["polygons"])):
			precision_per_polygon.append({
				"case_index": case_index,
				"polygon_index": polygon_index,
				"raw_distance": path_distances["raw"][polygon_index] if path_distances["raw"] else None,
				"snapped_distance": path_distances["snapped"][polygon_index] if path_distances["snapped"] else None,
			})

		raw_case_max = max(path_distances["raw"], default=None)
		snapped_case_max = max(path_distances["snapped"], default=None)
		if raw_case_max is not None:
			raw_max_distances.append(raw_case_max)
			per_polygon_raw_distances.extend(path_distances["raw"])
		if snapped_case_max is not None:
			snapped_max_distances.append(snapped_case_max)
			per_polygon_snapped_distances.extend(path_distances["snapped"])

		is_completed = b.get("is_optimal") == "True"
		speedup = fekete_seconds / ours_seconds if is_completed and fekete_seconds and ours_seconds and ours_seconds > 0 else None
		length_ratio = fekete_length / ours_length if is_completed and fekete_length is not None and ours_length and ours_length > 0 else None
		if speedup is not None:
			common_completed.append({"case_index": case_index, "speedup": speedup, "ours_seconds": ours_seconds, "fekete_seconds": fekete_seconds})
		per_instance.append({
			"case_index": case_index,
			"polygons": len(case["polygons"]),
			"ours_seconds": ours_seconds,
			"fekete_seconds": fekete_seconds,
			"fekete_status": b.get("status"),
			"fekete_completed": is_completed,
			"speedup_fekete_over_ours": speedup,
			"ours_length": ours_length,
			"fekete_length": fekete_length,
			"length_ratio_fekete_over_ours": length_ratio,
			"fekete_relative_gap": finite(b.get("relative_gap")),
			"fekete_raw_max_polygon_distance": raw_case_max,
			"fekete_snapped_max_polygon_distance": snapped_case_max,
			"fekete_csv_raw_max_polygon_distance": finite(b.get("max_polygon_distance")),
			"fekete_csv_snapped_max_polygon_distance": finite(b.get("snapped_max_polygon_distance")),
		})

	common_speedups = [row["speedup"] for row in common_completed]
	common_ours_seconds = [row["ours_seconds"] for row in common_completed if row["ours_seconds"] is not None]
	completed_length_ratios = [row["length_ratio_fekete_over_ours"] for row in per_instance if row["length_ratio_fekete_over_ours"] is not None]
	finished_count = sum(row["fekete_completed"] for row in per_instance)
	resolved_but_invalid_raw = sum(row["fekete_raw_max_polygon_distance"] is not None and row["fekete_raw_max_polygon_distance"] > 1e-7 for row in per_instance)
	resolved_but_invalid_snapped = sum(row["fekete_snapped_max_polygon_distance"] is not None and row["fekete_snapped_max_polygon_distance"] > 1e-7 for row in per_instance)

	by_polygon_bucket = []
	for lower, upper in ((4, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)):
		selected = [row for row in per_instance if lower <= row["polygons"] <= upper]
		ours_times = [row["ours_seconds"] for row in selected if row["ours_seconds"] is not None]
		fekete_times = [row["fekete_seconds"] for row in selected if row["fekete_completed"] and row["fekete_seconds"] is not None]
		by_polygon_bucket.append({
			"bucket": f"{lower}–{upper}",
			"instances": len(selected),
			"ours_median_seconds": statistics.median(ours_times) if ours_times else None,
			"fekete_completed": sum(row["fekete_completed"] for row in selected),
			"fekete_median_seconds_completed": statistics.median(fekete_times) if fekete_times else None,
			"fekete_timeouts": sum(not row["fekete_completed"] for row in selected),
		})

	precision = {
		"definition": "minimum Euclidean distance from the complete trajectory polyline to each polygon; zero means the route touches or crosses the polygon",
		"fekete_raw_per_polygon": precision_summary(per_polygon_raw_distances),
		"fekete_snapped_per_polygon": precision_summary(per_polygon_snapped_distances),
		"fekete_raw_max_per_instance": precision_summary(raw_max_distances),
		"fekete_snapped_max_per_instance": precision_summary(snapped_max_distances),
		"ours": None,
		"ours_unavailable_reason": "ours.csv has no final trajectory or per-polygon contact coordinates; the old app paths are from a different run and were not reused",
	}

	result = {
		"provenance": {
			"ours_csv": str(args.ours),
			"fekete_csv": str(args.fekete),
			"instances_bin": str(args.instances),
			"instances_sha256": hashlib.sha256(args.instances.read_bytes()).hexdigest(),
			"instance_count": len(cases),
			"join_key": "case_index + sha256",
		},
		"resolution": {
			"ours": {"instances": len(ours), "exact_certified": sum(row.get("exact") == "True" for row in ours.values()), "valid_export_rows": len(ours)},
			"fekete": {"instances": len(fekete), "completed_with_is_optimal": finished_count, "unresolved": len(fekete) - finished_count, "status_counts": {status: sum(row.get("status") == status for row in fekete.values()) for status in sorted({row.get("status") for row in fekete.values()})}},
		},
		"time_seconds": {
			"definition": "solver time in seconds; all recorded rows are included in the all-run histograms; direct speedup uses the 550 Fekete-completed instances",
			"ours_all": numeric_stats(all_ours_seconds),
			"ours_common_completed": numeric_stats(common_ours_seconds),
			"fekete_all_recorded": numeric_stats(all_fekete_seconds),
			"fekete_completed_only": numeric_stats(completed_fekete_seconds),
			"histogram": {
				"ours": histogram(all_ours_seconds, TIME_EDGES, TIME_LABELS),
				"fekete": histogram(all_fekete_seconds, TIME_EDGES, TIME_LABELS),
			},
			"total_cpu_hours": {"ours_all": sum(all_ours_seconds) / 3600, "fekete_all_recorded": sum(all_fekete_seconds) / 3600, "fekete_completed_only": sum(completed_fekete_seconds) / 3600},
		},
		"speedup": {
			"definition": "Fekete solve_seconds / ours seconds; greater than 1 means ours is faster",
			"common_completed_instances": len(common_speedups),
			"stats": numeric_stats(common_speedups),
			"ours_faster_count": sum(value > 1 for value in common_speedups),
			"fekete_faster_count": sum(value < 1 for value in common_speedups),
			"ties_count": sum(value == 1 for value in common_speedups),
			"histogram": histogram(common_speedups, SPEEDUP_EDGES, SPEEDUP_LABELS),
			"unresolved_lower_bound_speedups": [
				{"case_index": row["case_index"], "lower_bound_speedup_if_21600s": 21_600 / row["ours_seconds"]}
				for row in per_instance if not row["fekete_completed"] and row["ours_seconds"] and row["ours_seconds"] > 0
			],
		},
		"length": {
			"ours_final_all": numeric_stats(ours_lengths),
			"fekete_recomputed_available": numeric_stats(fekete_lengths),
			"completed_common_instances": len(completed_length_ratios),
			"ratio_fekete_over_ours": numeric_stats(completed_length_ratios),
			"fekete_longer_count": sum(value > 1 for value in completed_length_ratios),
			"fekete_shorter_count": sum(value < 1 for value in completed_length_ratios),
			"histogram": histogram(completed_length_ratios, RATIO_EDGES, RATIO_LABELS),
		},
		"precision": precision,
		"derived": {
			"fekete_raw_max_distance_instances": len(raw_max_distances),
			"fekete_snapped_max_distance_instances": len(snapped_max_distances),
			"fekete_raw_max_over_1e-7": resolved_but_invalid_raw,
			"fekete_snapped_max_over_1e-7": resolved_but_invalid_snapped,
			"by_polygon_bucket": by_polygon_bucket,
		},
	}
	result["_per_instance"] = per_instance
	result["_precision_per_polygon"] = precision_per_polygon
	return result


def build_markdown(analysis: dict[str, Any]) -> str:
	resolution = analysis["resolution"]
	time = analysis["time_seconds"]
	speedup = analysis["speedup"]
	length = analysis["length"]
	precision = analysis["precision"]
	rows = analysis["_per_instance"]
	lines = [
		"# Análise comparativa — German comparison",
		"",
		"Análise das 558 instâncias, unidas por `case_index` e `sha256`. Os histogramas de tempo incluem todos os tempos registrados; speedup e comprimento usam apenas as 550 instâncias concluídas pelo Fekete.",
		"",
		"## Resultado principal",
		"",
		markdown_table(
			["Métrica", "Nosso solver", "Fekete et al."],
			[
				["Instâncias certificadas/concluídas", f"{resolution['ours']['exact_certified']}/558", f"{resolution['fekete']['completed_with_is_optimal']}/558"],
				["Tempo médio registrado", f"{compact(time['ours_all']['mean'])} s", f"{compact(time['fekete_all_recorded']['mean'])} s"],
				["Tempo mediano registrado", f"{compact(time['ours_all']['median'])} s", f"{compact(time['fekete_all_recorded']['median'])} s"],
				["Tempo total registrado", f"{compact(time['total_cpu_hours']['ours_all'], 4)} h", f"{compact(time['total_cpu_hours']['fekete_all_recorded'], 4)} h"],
			],
		),
		"",
		"## Tempos e speedup",
		"",
		f"No conjunto comum concluído, a mediana do speedup Fekete/nosso é **{compact(speedup['stats']['median'])}×** e a média geométrica é **{compact(speedup['stats']['geometric_mean'])}×**. Nosso solver foi mais rápido em {speedup['ours_faster_count']} de {speedup['common_completed_instances']} instâncias; o Fekete foi mais rápido em {speedup['fekete_faster_count']}. Nesse mesmo conjunto, as medianas de tempo são {compact(time['ours_common_completed']['median'])} s contra {compact(time['fekete_completed_only']['median'])} s.",
		"",
		markdown_table(
			["Faixa de regiões", "Casos", "Mediana nosso (s)", "Fekete concluídos", "Mediana Fekete (s)", "Timeouts Fekete"],
			[
				[row["bucket"], row["instances"], compact(row["ours_median_seconds"]), row["fekete_completed"], compact(row["fekete_median_seconds_completed"]), row["fekete_timeouts"]]
				for row in analysis["derived"]["by_polygon_bucket"]
			],
		),
		"",
		"## Comprimento",
		"",
		f"Nas {length['completed_common_instances']} instâncias concluídas por ambos, a razão comprimento(Fekete)/comprimento(nosso) tem mediana **{compact(length['ratio_fekete_over_ours']['median'])}** e média **{compact(length['ratio_fekete_over_ours']['mean'])}**. O Fekete retornou comprimento maior em {length['fekete_longer_count']} casos e menor em {length['fekete_shorter_count']}; diferenças abaixo de 1 podem ser efeito numérico/feasibility tolerance, não evidência de um ótimo melhor que o certificado.",
		"",
		"## Precisão geométrica",
		"",
		"A métrica usada é a distância Euclidiana mínima entre a polilinha completa da trajetória e cada polígono; zero significa que a trajetória toca ou cruza o polígono. A `fekete.csv` permite essa auditoria, mas `ours.csv` não traz a trajetória final nem os pontos de contato.",
		"",
		markdown_table(
			["Métrica Fekete", "Trajetória bruta", "Trajetória snapped"],
			[
				["Instâncias com trajetória", precision["fekete_raw_max_per_instance"]["stats"]["n"], precision["fekete_snapped_max_per_instance"]["stats"]["n"]],
				["Mediana do maior erro por instância", compact(precision["fekete_raw_max_per_instance"]["stats"]["median"]), compact(precision["fekete_snapped_max_per_instance"]["stats"]["median"])],
				["P95 do maior erro por instância", compact(precision["fekete_raw_max_per_instance"]["stats"]["p95"]), compact(precision["fekete_snapped_max_per_instance"]["stats"]["p95"])],
				["Casos ≤ 1e−7", sum(value <= 1e-7 for value in [row["fekete_raw_max_polygon_distance"] for row in rows if row["fekete_raw_max_polygon_distance"] is not None]), sum(value <= 1e-7 for value in [row["fekete_snapped_max_polygon_distance"] for row in rows if row["fekete_snapped_max_polygon_distance"] is not None])],
				["Casos > 1e−7", analysis["derived"]["fekete_raw_max_over_1e-7"], analysis["derived"]["fekete_snapped_max_over_1e-7"]],
			],
		),
		"",
		"A comparação equivalente do nosso solver ficará disponível assim que a trajetória da nova rodada for exportada junto da CSV. Os caminhos antigos em `apps/siicusp34/data/event-data.js` não foram usados porque a própria nova rodada corrige o caso 001 e os tempos/soluções são de outra seleção de runs.",
	]
	return "\n".join(lines) + "\n"


def build_html(analysis: dict[str, Any]) -> str:
	data = {
		"cards": {
			"oursSolved": f"{analysis['resolution']['ours']['exact_certified']}/558",
			"feketeSolved": f"{analysis['resolution']['fekete']['completed_with_is_optimal']}/558",
			"medianSpeedup": analysis["speedup"]["stats"]["median"],
			"medianRawPrecision": analysis["precision"]["fekete_raw_max_per_instance"]["stats"]["median"],
		},
		"charts": {
			"times": [
				{"label": "Nosso solver", "color": "var(--viz-series-1)", **analysis["time_seconds"]["histogram"]["ours"]},
				{"label": "Fekete et al.", "color": "var(--viz-series-2)", **analysis["time_seconds"]["histogram"]["fekete"]},
			],
			"speedup": [analysis["speedup"]["histogram"]],
			"length": [analysis["length"]["histogram"]],
			"precision": [
				{"label": "Bruta", "color": "var(--viz-series-2)", **analysis["precision"]["fekete_raw_max_per_instance"]["histogram"]},
				{"label": "Snapped", "color": "var(--viz-series-1)", **analysis["precision"]["fekete_snapped_max_per_instance"]["histogram"]},
			],
		},
		"precisionThresholds": {
			"labels": ["0", "≤ 1e−7", "≤ 1e−6", "≤ 1e−5", "≤ 1e−4", "≤ 1e−3"],
			"raw": [sum(value <= threshold for value in [row["fekete_raw_max_polygon_distance"] for row in analysis["_per_instance"] if row["fekete_raw_max_polygon_distance"] is not None]) for threshold in (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)],
			"snapped": [sum(value <= threshold for value in [row["fekete_snapped_max_polygon_distance"] for row in analysis["_per_instance"] if row["fekete_snapped_max_polygon_distance"] is not None]) for threshold in (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)],
		},
	}
	payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
	return f'''<section id="german-comparison-analysis" aria-labelledby="german-comparison-title">
<style>
#german-comparison-analysis {{ color: var(--foreground); font-size: var(--font-size-base); }}
#german-comparison-analysis .analysis-intro {{ margin-bottom: 1rem; }}
#german-comparison-analysis .viz-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.75rem; margin: 1rem 0 1.5rem; }}
#german-comparison-analysis .viz-stat {{ min-width: 0; }}
#german-comparison-analysis .viz-stat-value {{ display: block; font-variant-numeric: tabular-nums; }}
#german-comparison-analysis .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5rem 1rem; }}
#german-comparison-analysis figure {{ margin: 0; min-width: 0; }}
#german-comparison-analysis figcaption {{ margin-bottom: 0.35rem; font-weight: 500; }}
#german-comparison-analysis svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
#german-comparison-analysis .chart-frame {{ fill: none; stroke: var(--border); stroke-width: 1; }}
#german-comparison-analysis .grid-line {{ stroke: var(--border); stroke-opacity: 0.55; stroke-width: 1; }}
#german-comparison-analysis .axis-label, #german-comparison-analysis .axis-title {{ fill: var(--foreground); font-size: 12px; }}
#german-comparison-analysis .muted {{ color: var(--muted-foreground); }}
#german-comparison-analysis .legend {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.35rem 0 0; font-size: 12px; }}
#german-comparison-analysis .legend i {{ display: inline-block; width: 0.65rem; height: 0.65rem; margin-right: 0.25rem; background: var(--swatch); }}
@media (max-width: 620px) {{
  #german-comparison-analysis .viz-grid, #german-comparison-analysis .chart-grid {{ grid-template-columns: 1fr 1fr; }}
  #german-comparison-analysis .chart-grid {{ gap: 1rem 0.5rem; }}
}}
@media (max-width: 420px) {{
  #german-comparison-analysis .viz-grid, #german-comparison-analysis .chart-grid {{ grid-template-columns: 1fr; }}
}}
</style>
<h2 id="german-comparison-title">Comparação nas 558 instâncias</h2>
<p class="analysis-intro muted">Tempos: todos os registros. Speedup e comprimento: 550 instâncias concluídas por ambos. Precisão: Fekete, porque a CSV do nosso solver não inclui a trajetória final.</p>
<div class="viz-grid">
  <div class="card viz-stat"><span>Certificadas</span><strong class="viz-stat-value">{data['cards']['oursSolved']} <span class="muted">nosso</span></strong><small class="muted">{data['cards']['feketeSolved']} Fekete concluídas</small></div>
  <div class="card viz-stat"><span>Mediana do tempo</span><strong class="viz-stat-value">{analysis['time_seconds']['ours_all']['median']:.4g} s</strong><small class="muted">nosso solver</small></div>
  <div class="card viz-stat"><span>Mediana do speedup</span><strong class="viz-stat-value">{data['cards']['medianSpeedup']:.4g}×</strong><small class="muted">Fekete / nosso</small></div>
  <div class="card viz-stat"><span>Maior erro geométrico</span><strong class="viz-stat-value">{data['cards']['medianRawPrecision']:.4g}</strong><small class="muted">mediana Fekete por instância</small></div>
</div>
<div class="chart-grid">
  <figure><figcaption>Histograma dos tempos de execução</figcaption><div id="german-times-legend" class="legend" aria-label="Legenda dos tempos"></div><svg id="german-times-chart" role="img" aria-label="Histograma comparativo dos tempos de execução"><title>Histograma dos tempos de execução</title><desc>Distribuição dos tempos registrados dos dois solvers em nove faixas de tempo.</desc></svg></figure>
  <figure><figcaption>Speedup nas 550 instâncias concluídas</figcaption><svg id="german-speedup-chart" role="img" aria-label="Histograma do speedup Fekete sobre nosso solver"><title>Histograma do speedup</title><desc>Distribuição de Fekete dividido pelo tempo do nosso solver; valores acima de um favorecem nosso solver.</desc></svg></figure>
  <figure><figcaption>Razão de comprimento: Fekete / nosso</figcaption><svg id="german-length-chart" role="img" aria-label="Histograma da razão entre comprimentos"><title>Histograma da razão de comprimento</title><desc>Distribuição da razão entre o comprimento recalculado do Fekete e o comprimento final do nosso solver.</desc></svg></figure>
  <figure><figcaption>Maior distância trajetória–polígono por instância</figcaption><div id="german-precision-legend" class="legend" aria-label="Legenda da precisão"></div><svg id="german-precision-chart" role="img" aria-label="Histograma da maior distância da trajetória do Fekete aos polígonos"><title>Histograma de precisão geométrica</title><desc>Distribuição da maior distância mínima da trajetória do Fekete a qualquer polígono.</desc></svg></figure>
</div>
<figure style="margin-top:1.5rem"><figcaption>Casos dentro das tolerâncias geométricas</figcaption><div id="german-threshold-legend" class="legend" aria-label="Legenda das tolerâncias"></div><svg id="german-threshold-chart" role="img" aria-label="Casos do Fekete dentro de tolerâncias de distância"><title>Casos dentro das tolerâncias</title><desc>Contagem cumulativa de instâncias do Fekete cuja maior distância trajetória–polígono está abaixo de cada tolerância.</desc></svg></figure>
<p class="muted">Precisão = distância Euclidiana mínima entre a polilinha completa da trajetória e cada polígono. O arquivo `ours.csv` não contém os pontos necessários para repetir esse cálculo no nosso solver.</p>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const DATA = {payload};
  const draw = (selector, series, options = {{}}) => {{
    const host = document.querySelector(selector);
    if (!host) return;
    const width = Math.max(320, host.parentElement?.clientWidth || 640);
    const height = options.height || 250;
    const margin = {{ top: 12, right: 12, bottom: 58, left: 58 }};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const labels = series[0].labels;
    const svg = d3.select(host).attr('viewBox', `0 0 ${{width}} ${{height}}`);
    svg.selectAll('*').remove();
    const root = svg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);
    const x = d3.scaleBand().domain(labels).range([0, innerWidth]).padding(0.18);
    const maxCount = d3.max(series, s => d3.max(s.counts)) || 1;
    const y = d3.scaleLinear().domain([0, maxCount]).nice().range([innerHeight, 0]);
    root.append('rect').attr('class', 'chart-frame').attr('data-chart-frame', true).attr('width', innerWidth).attr('height', innerHeight);
    root.append('g').selectAll('line').data(y.ticks(4)).join('line').attr('class', 'grid-line').attr('x1', 0).attr('x2', innerWidth).attr('y1', d => y(d)).attr('y2', d => y(d));
    root.append('g').attr('transform', `translate(0,${{innerHeight}})`).call(d3.axisBottom(x).tickValues(labels.filter((d, i) => i % Math.max(1, Math.ceil(labels.length / 4)) === 0)).tickSizeOuter(0)).call(g => g.selectAll('text').attr('class', 'axis-label').attr('transform', 'rotate(-24)').style('text-anchor', 'end'));
    root.append('g').call(d3.axisLeft(y).ticks(4).tickSizeOuter(0)).call(g => g.selectAll('text').attr('class', 'axis-label'));
    const barWidth = x.bandwidth() / series.length;
    series.forEach((s, seriesIndex) => root.append('g').selectAll('rect').data(s.counts).join('rect').attr('x', (d, i) => x(labels[i]) + seriesIndex * barWidth).attr('y', d => y(d)).attr('width', Math.max(1, barWidth - 1)).attr('height', d => innerHeight - y(d)).attr('fill', s.color).attr('opacity', 0.82));
    root.append('text').attr('class', 'axis-title').attr('data-axis', 'x').attr('x', innerWidth / 2).attr('y', innerHeight + 54).attr('text-anchor', 'middle').text(options.xTitle || 'Faixa');
    root.append('text').attr('class', 'axis-title').attr('data-axis', 'y').attr('transform', 'rotate(-90)').attr('x', -innerHeight / 2).attr('y', -42).attr('text-anchor', 'middle').text('Instâncias');
  }};
  const timeLegend = document.querySelector('#german-times-legend');
  const addLegend = (selector, series) => {{ const legend = document.querySelector(selector); series.forEach(s => {{ const item = document.createElement('span'); item.innerHTML = `<i style="--swatch:${{s.color}}"></i>${{s.label}}`; legend.appendChild(item); }}); }};
  addLegend('#german-times-legend', DATA.charts.times);
  addLegend('#german-precision-legend', DATA.charts.precision);
  addLegend('#german-threshold-legend', [{{ label: 'Bruta', color: 'var(--viz-series-2)' }}, {{ label: 'Snapped', color: 'var(--viz-series-1)' }}]);
  draw('#german-times-chart', DATA.charts.times, {{ xTitle: 'Tempo de solver' }});
  draw('#german-speedup-chart', [{{ labels: DATA.charts.speedup[0].labels, counts: DATA.charts.speedup[0].counts, color: 'var(--viz-series-1)' }}], {{ xTitle: 'Fekete / nosso' }});
  draw('#german-length-chart', [{{ labels: DATA.charts.length[0].labels, counts: DATA.charts.length[0].counts, color: 'var(--viz-series-1)' }}], {{ xTitle: 'Razão de comprimento' }});
	  draw('#german-precision-chart', [{{ labels: DATA.charts.precision[0].labels, counts: DATA.charts.precision[0].counts, color: 'var(--viz-series-2)' }}, {{ labels: DATA.charts.precision[1].labels, counts: DATA.charts.precision[1].counts, color: 'var(--viz-series-1)' }}], {{ xTitle: 'Maior distância (faixa)' }});
  draw('#german-threshold-chart', [{{ labels: DATA.precisionThresholds.labels, counts: DATA.precisionThresholds.raw, color: 'var(--viz-series-2)' }}, {{ labels: DATA.precisionThresholds.labels, counts: DATA.precisionThresholds.snapped, color: 'var(--viz-series-1)' }}], {{ xTitle: 'Tolerância' }});
  const redraw = () => {{ draw('#german-times-chart', DATA.charts.times, {{ xTitle: 'Tempo de solver' }}); draw('#german-speedup-chart', [{{ labels: DATA.charts.speedup[0].labels, counts: DATA.charts.speedup[0].counts, color: 'var(--viz-series-1)' }}], {{ xTitle: 'Fekete / nosso' }}); draw('#german-length-chart', [{{ labels: DATA.charts.length[0].labels, counts: DATA.charts.length[0].counts, color: 'var(--viz-series-1)' }}], {{ xTitle: 'Razão de comprimento' }}); draw('#german-precision-chart', [{{ labels: DATA.charts.precision[0].labels, counts: DATA.charts.precision[0].counts, color: 'var(--viz-series-2)' }}, {{ labels: DATA.charts.precision[1].labels, counts: DATA.charts.precision[1].counts, color: 'var(--viz-series-1)' }}], {{ xTitle: 'Maior distância (faixa)' }}); draw('#german-threshold-chart', [{{ labels: DATA.precisionThresholds.labels, counts: DATA.precisionThresholds.raw, color: 'var(--viz-series-2)' }}, {{ labels: DATA.precisionThresholds.labels, counts: DATA.precisionThresholds.snapped, color: 'var(--viz-series-1)' }}], {{ xTitle: 'Tolerância' }}); }};
	  new ResizeObserver(redraw).observe(document.querySelector('#german-comparison-analysis'));
}})();
</script>
</section>\n'''


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--ours", type=Path, default=Path("benchmarks/results-saved/german-comparison/ours.csv"))
	parser.add_argument("--fekete", type=Path, default=Path("benchmarks/results-saved/german-comparison/fekete.csv"))
	parser.add_argument("--instances", type=Path, default=Path("benchmarks/results-saved/german-comparison/instances.bin"))
	parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results-saved/german-comparison/analysis"))
	parser.add_argument("--html-output", type=Path)
	args = parser.parse_args()
	analysis = build_analysis(args)
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "summary.json").write_text(json.dumps({key: value for key, value in analysis.items() if not key.startswith("_")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	with (output_dir / "per-instance.csv").open("w", newline="", encoding="utf-8") as file:
		fields = list(analysis["_per_instance"][0])
		writer = csv.DictWriter(file, fieldnames=fields)
		writer.writeheader()
		writer.writerows(analysis["_per_instance"])
	with (output_dir / "precision-per-polygon.csv").open("w", newline="", encoding="utf-8") as file:
		fields = list(analysis["_precision_per_polygon"][0])
		writer = csv.DictWriter(file, fieldnames=fields)
		writer.writeheader()
		writer.writerows(analysis["_precision_per_polygon"])
	(output_dir / "report.md").write_text(build_markdown(analysis), encoding="utf-8")
	if args.html_output:
		args.html_output.parent.mkdir(parents=True, exist_ok=True)
		args.html_output.write_text(build_html(analysis), encoding="utf-8")
	print(json.dumps({"output_dir": str(output_dir), "summary": analysis["resolution"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
