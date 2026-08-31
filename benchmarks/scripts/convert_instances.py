#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "tspn-comparison/solver/instances/instances_socg_simplified.zip"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks/suites/nonconvex/test_cases.bin"
POLYGON_RE = re.compile(r"^\s*POLYGON\s*\(\((.*)\)\)\s*$")


Point = tuple[float, float]


@dataclass(frozen=True)
class ConvertedCase:
	name: str
	start: Point
	target: Point
	polygons: list[list[Point]]
	meta: dict


def parse_polygon_wkt(wkt: str) -> list[Point]:
	match = POLYGON_RE.match(wkt)
	if match is None:
		raise ValueError(f"Unsupported polygon WKT: {wkt[:80]}")

	points: list[Point] = []
	for pair in match.group(1).split(","):
		parts = pair.strip().split()
		if len(parts) < 2:
			raise ValueError(f"Malformed WKT coordinate pair: {pair!r}")
		points.append((float(parts[0]), float(parts[1])))

	if len(points) > 1 and points[0] == points[-1]:
		points.pop()

	if len(points) < 3:
		raise ValueError(f"Polygon has fewer than 3 distinct vertices: {wkt[:80]}")

	if signed_area2(points) < 0.0:
		points.reverse()

	return points


def signed_area2(polygon: Sequence[Point]) -> float:
	area = 0.0
	for index, point in enumerate(polygon):
		next_point = polygon[(index + 1) % len(polygon)]
		area += point[0] * next_point[1] - next_point[0] * point[1]
	return area


def bounding_box(polygons: Sequence[Sequence[Point]]) -> tuple[float, float, float, float]:
	xs = [point[0] for polygon in polygons for point in polygon]
	ys = [point[1] for polygon in polygons for point in polygon]
	return min(xs), min(ys), max(xs), max(ys)


def convert_json_case(name: str, payload: bytes) -> ConvertedCase:
	data = json.loads(payload)
	polygons = [parse_polygon_wkt(wkt) for wkt in data["polygons"]]
	min_x, min_y, max_x, max_y = bounding_box(polygons)
	return ConvertedCase(
		name=name,
		start=(min_x, min_y),
		target=(max_x, max_y),
		polygons=polygons,
		meta=data.get("meta", {}),
	)


def write_vector(file, point: Point) -> None:
	file.write(struct.pack("<dd", point[0], point[1]))


def write_size(file, value: int) -> None:
	file.write(struct.pack("<Q", value))


def write_binary_cases(cases: Sequence[ConvertedCase], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("wb") as file:
		for case in cases:
			write_vector(file, case.start)
			write_vector(file, case.target)
			write_size(file, len(case.polygons))
			for polygon in case.polygons:
				write_size(file, len(polygon))
				for point in polygon:
					write_vector(file, point)
			write_size(file, 0)


def load_cases(input_path: Path, *, sort_by_name: bool) -> list[ConvertedCase]:
	with zipfile.ZipFile(input_path) as archive:
		names = [name for name in archive.namelist() if name.endswith(".json")]
		if sort_by_name:
			names.sort()
		return [convert_json_case(name, archive.read(name)) for name in names]


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Convert TSPN JSON archive instances to TouringPolygons binary format.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--zip-order", action="store_true", help="Preserve the zip entry order instead of sorting by filename.")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	input_path = args.input.resolve()
	output_path = args.output.resolve()

	if not input_path.exists():
		raise SystemExit(f"Input zip does not exist: {input_path}")

	cases = load_cases(input_path, sort_by_name=not args.zip_order)
	write_binary_cases(cases, output_path)
	print(f"Wrote {len(cases)} cases to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
