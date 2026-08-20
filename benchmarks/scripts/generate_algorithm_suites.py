#!/usr/bin/env python3
"""Generate deterministic benchmark suites from the tracked non-convex corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "benchmarks/suites/nonconvex/test_cases.bin"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/suites"


@dataclass(frozen=True)
class EncodedCase:
	data: bytes
	digest: str
	case_index: int
	polygon_vertices: tuple[tuple[tuple[float, float], ...], ...]
	polygons: int
	vertices: int


def read_u64(data: bytes, offset: int, path: Path) -> tuple[int, int]:
	if offset + 8 > len(data):
		raise ValueError(f"Truncated size value at byte {offset} in {path}")
	return struct.unpack_from("<Q", data, offset)[0], offset + 8


def skip_bytes(data: bytes, offset: int, count: int, path: Path) -> int:
	end = offset + count
	if end > len(data):
		raise ValueError(f"Truncated case payload at byte {offset} in {path}")
	return end


def read_encoded_cases(path: Path) -> list[EncodedCase]:
	data = path.read_bytes()
	offset = 0
	cases: list[EncodedCase] = []

	while offset < len(data):
		start = offset
		offset = skip_bytes(data, offset, 32, path)
		polygon_count, offset = read_u64(data, offset, path)
		polygon_vertices: list[tuple[tuple[float, float], ...]] = []
		vertex_total = 0

		for _ in range(polygon_count):
			vertex_count, offset = read_u64(data, offset, path)
			vertex_total += vertex_count
			vertices = tuple(
				struct.unpack_from("<dd", data, offset + 16 * vertex_index)
				for vertex_index in range(vertex_count)
			)
			polygon_vertices.append(vertices)
			offset = skip_bytes(data, offset, 16 * vertex_count, path)

		solution_count, offset = read_u64(data, offset, path)
		offset = skip_bytes(data, offset, 16 * solution_count, path)
		encoded = data[start:offset]
		cases.append(EncodedCase(
			data=encoded,
			digest=hashlib.sha256(encoded).hexdigest(),
			case_index=len(cases),
			polygon_vertices=tuple(polygon_vertices),
			polygons=polygon_count,
			vertices=vertex_total,
		))

	return cases


def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
	return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex_hull(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
	unique = sorted(set(points))
	if len(unique) <= 1:
		return tuple(unique)

	def half_hull(sorted_points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
		hull: list[tuple[float, float]] = []
		for point in sorted_points:
			while len(hull) > 1 and orientation(hull[-2], hull[-1], point) <= 0.0:
				hull.pop()
			hull.append(point)
		return hull[:-1]

	return tuple(half_hull(unique) + half_hull(list(reversed(unique))))


def point_on_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> bool:
	epsilon = 1e-12
	return (
		abs(orientation(a, b, point)) <= epsilon
		and min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
		and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
	)


def segments_intersect_or_touch(
	a: tuple[float, float],
	b: tuple[float, float],
	c: tuple[float, float],
	d: tuple[float, float],
) -> bool:
	o1 = orientation(a, b, c)
	o2 = orientation(a, b, d)
	o3 = orientation(c, d, a)
	o4 = orientation(c, d, b)
	if o1 * o2 < 0.0 and o3 * o4 < 0.0:
		return True
	return (
		point_on_segment(c, a, b)
		or point_on_segment(d, a, b)
		or point_on_segment(a, c, d)
		or point_on_segment(b, c, d)
	)


def point_in_polygon_or_on_boundary(
	point: tuple[float, float],
	polygon: Sequence[tuple[float, float]],
) -> bool:
	inside = False
	px, py = point
	for index, a in enumerate(polygon):
		b = polygon[(index + 1) % len(polygon)]
		if point_on_segment(point, a, b):
			return True
		if (a[1] > py) != (b[1] > py):
			x_crossing = (b[0] - a[0]) * (py - a[1]) / (b[1] - a[1]) + a[0]
			if px <= x_crossing:
				inside = not inside
	return inside


def polygons_intersect_or_touch(
	first: Sequence[tuple[float, float]],
	second: Sequence[tuple[float, float]],
) -> bool:
	first_bounds = (
		min(point[0] for point in first), min(point[1] for point in first),
		max(point[0] for point in first), max(point[1] for point in first),
	)
	second_bounds = (
		min(point[0] for point in second), min(point[1] for point in second),
		max(point[0] for point in second), max(point[1] for point in second),
	)
	if (
		first_bounds[2] < second_bounds[0]
		or second_bounds[2] < first_bounds[0]
		or first_bounds[3] < second_bounds[1]
		or second_bounds[3] < first_bounds[1]
	):
		return False

	for first_index, a in enumerate(first):
		b = first[(first_index + 1) % len(first)]
		for second_index, c in enumerate(second):
			d = second[(second_index + 1) % len(second)]
			if segments_intersect_or_touch(a, b, c, d):
				return True
	return point_in_polygon_or_on_boundary(first[0], second) or point_in_polygon_or_on_boundary(second[0], first)


def case_has_intersecting_hulls(case: EncodedCase) -> bool:
	hulls = [convex_hull(polygon) for polygon in case.polygon_vertices]
	return any(
		polygons_intersect_or_touch(hulls[first], hulls[second])
		for first in range(len(hulls))
		for second in range(first + 1, len(hulls))
	)


def spread_select(cases: Sequence[EncodedCase], count: int) -> list[EncodedCase]:
	if count > len(cases):
		raise SystemExit(f"Requested {count} cases, but only {len(cases)} are available.")

	ordered = sorted(cases, key=lambda case: (case.polygons, case.vertices, case.digest))
	if count == len(ordered):
		return ordered

	last = len(ordered) - 1
	indices = sorted({round(index * last / (count - 1)) for index in range(count)})
	selected = [ordered[index] for index in indices]

	cursor = 0
	while len(selected) < count:
		candidate = ordered[cursor]
		if candidate not in selected:
			selected.append(candidate)
		cursor += 1

	return selected


def prefix_representative_order(cases: Sequence[EncodedCase]) -> list[EncodedCase]:
	ordered = sorted(cases, key=lambda case: (case.polygons, case.vertices, case.digest))
	bucket_count = 3
	bucket_size = (len(ordered) + bucket_count - 1) // bucket_count
	buckets = [
		ordered[index * bucket_size:(index + 1) * bucket_size]
		for index in range(bucket_count)
	]
	result: list[EncodedCase] = []

	while any(buckets):
		for bucket in reversed(buckets):
			if bucket:
				result.append(bucket.pop(0))

	return result


def write_suite(name: str, cases: Sequence[EncodedCase], output_dir: Path, source: Path) -> None:
	bin_path = output_dir / f"{name}.bin"
	csv_path = output_dir / f"{name}.csv"
	md_path = output_dir / f"{name}.md"

	with bin_path.open("wb") as file:
		for case in cases:
			file.write(case.data)

	with csv_path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=[
			"suite_index", "source_case_index", "sha256", "polygons", "vertices",
		])
		writer.writeheader()
		for suite_index, case in enumerate(cases):
			writer.writerow({
				"suite_index": suite_index,
				"source_case_index": case.case_index,
				"sha256": case.digest,
				"polygons": case.polygons,
				"vertices": case.vertices,
			})

	polygon_counts = [case.polygons for case in cases]
	vertex_counts = [case.vertices for case in cases]
	lines = [
		f"# {name}",
		"",
		f"Generated from `{source.relative_to(REPO_ROOT)}`.",
		"",
		f"Cases: {len(cases)}",
		f"Polygons: min {min(polygon_counts)}, median {statistics.median(polygon_counts):.0f}, max {max(polygon_counts)}",
		f"Vertices: min {min(vertex_counts)}, median {statistics.median(vertex_counts):.0f}, max {max(vertex_counts)}",
		"",
	]
	md_path.write_text("\n".join(lines))
	print(f"Wrote {len(cases)} cases to {bin_path}", flush=True)


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate algorithm benchmark suites from the tracked non-convex corpus.")
	parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--canonical-size", type=int, default=300)
	parser.add_argument("--dev-size", type=int, default=60)
	parser.add_argument("--require-disjoint-hulls", action="store_true")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	source = args.source.resolve()
	output = args.output.resolve()
	output.mkdir(parents=True, exist_ok=True)

	if args.dev_size > args.canonical_size:
		raise SystemExit("--dev-size must be less than or equal to --canonical-size")

	source_cases = read_encoded_cases(source)
	if args.require_disjoint_hulls:
		filtered_cases = [case for case in source_cases if not case_has_intersecting_hulls(case)]
		rejected_hull_intersections = len(source_cases) - len(filtered_cases)
		print(
			f"Available hull-disjoint cases: {len(filtered_cases)} of {len(source_cases)} "
			f"({rejected_hull_intersections} rejected).",
			flush=True,
		)
	else:
		filtered_cases = source_cases
		rejected_hull_intersections = 0

	canonical = prefix_representative_order(spread_select(filtered_cases, args.canonical_size))
	dev = prefix_representative_order(spread_select(canonical, args.dev_size))

	write_suite("canonical-v1", canonical, output, source)
	write_suite("algorithm-dev-v1", dev, output, source)

	metadata = {
		"schema_version": 2,
		"generator": "benchmarks/scripts/generate_algorithm_suites.py",
		"source": str(source.relative_to(REPO_ROOT)),
		"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
		"source_cases": len(source_cases),
		"rejected_intersecting_or_touching_hulls": rejected_hull_intersections,
		"require_disjoint_hulls": args.require_disjoint_hulls,
		"canonical_size": len(canonical),
		"development_size": len(dev),
		"selection": (
			"reject cases with intersecting or touching convex hulls; even spread by polygon count, vertex count, and case digest; ordered so prefixes remain representative"
			if args.require_disjoint_hulls
			else "even spread by polygon count, vertex count, and case digest; ordered so prefixes remain representative"
		),
		"canonical_sha256": hashlib.sha256((output / "canonical-v1.bin").read_bytes()).hexdigest(),
		"development_sha256": hashlib.sha256((output / "algorithm-dev-v1.bin").read_bytes()).hexdigest(),
	}
	(output / "suite-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
