#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


Point = tuple[float, float]
SIZE_STRUCT = struct.Struct("<Q")
POINT_STRUCT = struct.Struct("<dd")


@dataclass(frozen=True)
class BinaryCase:
	start: Point
	target: Point
	polygons: list[list[Point]]
	solution: list[Point]


@dataclass(frozen=True)
class FileReport:
	path: Path
	cases: int
	polygons: int
	reversed_polygons: int
	changed: bool


def signed_area2(polygon: Sequence[Point]) -> float:
	area = 0.0
	for index, point in enumerate(polygon):
		next_point = polygon[(index + 1) % len(polygon)]
		area += point[0] * next_point[1] - next_point[0] * point[1]
	return area


def normalize_polygon(polygon: list[Point]) -> tuple[list[Point], bool]:
	if len(polygon) < 3:
		return polygon, False
	if signed_area2(polygon) < 0.0:
		return list(reversed(polygon)), True
	return polygon, False


def read_size(data: bytes, offset: int, path: Path) -> tuple[int, int]:
	if offset + SIZE_STRUCT.size > len(data):
		raise ValueError(f"Truncated size value at byte {offset} in {path}")
	return SIZE_STRUCT.unpack_from(data, offset)[0], offset + SIZE_STRUCT.size


def read_point(data: bytes, offset: int, path: Path) -> tuple[Point, int]:
	if offset + POINT_STRUCT.size > len(data):
		raise ValueError(f"Truncated point value at byte {offset} in {path}")
	return POINT_STRUCT.unpack_from(data, offset), offset + POINT_STRUCT.size


def read_binary_cases(path: Path) -> list[BinaryCase]:
	data = path.read_bytes()
	offset = 0
	cases: list[BinaryCase] = []
	while offset < len(data):
		start, offset = read_point(data, offset, path)
		target, offset = read_point(data, offset, path)
		polygon_count, offset = read_size(data, offset, path)
		polygons: list[list[Point]] = []
		for _ in range(polygon_count):
			vertex_count, offset = read_size(data, offset, path)
			polygon: list[Point] = []
			for _ in range(vertex_count):
				point, offset = read_point(data, offset, path)
				polygon.append(point)
			polygons.append(polygon)
		solution_count, offset = read_size(data, offset, path)
		solution: list[Point] = []
		for _ in range(solution_count):
			point, offset = read_point(data, offset, path)
			solution.append(point)
		cases.append(BinaryCase(start, target, polygons, solution))
	return cases


def write_point(file: Any, point: Point) -> None:
	file.write(POINT_STRUCT.pack(point[0], point[1]))


def write_size(file: Any, value: int) -> None:
	file.write(SIZE_STRUCT.pack(value))


def write_binary_cases(path: Path, cases: Sequence[BinaryCase]) -> None:
	with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
		temporary_path = Path(file.name)
		for case in cases:
			write_point(file, case.start)
			write_point(file, case.target)
			write_size(file, len(case.polygons))
			for polygon in case.polygons:
				write_size(file, len(polygon))
				for point in polygon:
					write_point(file, point)
			write_size(file, len(case.solution))
			for point in case.solution:
				write_point(file, point)
	try:
		temporary_path.replace(path)
	except Exception:
		temporary_path.unlink(missing_ok=True)
		raise


def normalize_binary(path: Path, *, write: bool) -> FileReport:
	cases = read_binary_cases(path)
	reversed_polygons = 0
	polygon_count = 0
	normalized_cases: list[BinaryCase] = []
	for case in cases:
		normalized_polygons: list[list[Point]] = []
		for polygon in case.polygons:
			polygon_count += 1
			normalized, changed = normalize_polygon(polygon)
			reversed_polygons += int(changed)
			normalized_polygons.append(normalized)
		normalized_cases.append(BinaryCase(case.start, case.target, normalized_polygons, case.solution))
	changed = reversed_polygons > 0
	if write and changed:
		write_binary_cases(path, normalized_cases)
	return FileReport(path, len(cases), polygon_count, reversed_polygons, changed)


def normalize_manual_cases(path: Path, *, write: bool) -> FileReport:
	data = json.loads(path.read_text())
	raw_cases = data.get("cases")
	if not isinstance(raw_cases, list):
		raise ValueError(f"Expected a top-level cases list in {path}")
	reversed_polygons = 0
	polygon_count = 0
	for case in raw_cases:
		if not isinstance(case, dict):
			continue
		raw_polygons = case.get("polygons")
		if not isinstance(raw_polygons, list):
			continue
		for index, raw_polygon in enumerate(raw_polygons):
			if not isinstance(raw_polygon, list):
				continue
			polygon = [(float(point[0]), float(point[1])) for point in raw_polygon]
			polygon_count += 1
			normalized, changed = normalize_polygon(polygon)
			if changed:
				raw_polygons[index] = [[x, y] for x, y in normalized]
				reversed_polygons += 1
	changed = reversed_polygons > 0
	if write and changed:
		with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
			temporary_path = Path(file.name)
			json.dump(data, file, indent=2)
			file.write("\n")
		try:
			temporary_path.replace(path)
		except Exception:
			temporary_path.unlink(missing_ok=True)
			raise
	return FileReport(path, len(raw_cases), polygon_count, reversed_polygons, changed)


def candidate_files(paths: Sequence[Path]) -> list[Path]:
	files: list[Path] = []
	for path in paths:
		if path.is_dir():
			files.extend(sorted(candidate for candidate in path.rglob("*") if candidate.name == "manual-cases.json" or candidate.suffix == ".bin"))
		else:
			files.append(path)
	return files


def normalize_file(path: Path, *, write: bool) -> FileReport:
	if path.name == "manual-cases.json":
		return normalize_manual_cases(path, write=write)
	if path.suffix == ".bin":
		return normalize_binary(path, write=write)
	raise ValueError(f"Unsupported file type: {path}")


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Normalize TouringPolygons instance polygons to counter-clockwise vertex order.")
	parser.add_argument("paths", nargs="+", type=Path, help="Instance .bin files, manual-cases.json files, or directories to scan.")
	parser.add_argument("--check", action="store_true", help="Report clockwise polygons and exit with status 1 if any are found.")
	parser.add_argument("--in-place", action="store_true", help="Rewrite files in place, reversing clockwise polygons.")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	if args.check and args.in_place:
		raise SystemExit("--check and --in-place are mutually exclusive")
	if not args.check and not args.in_place:
		raise SystemExit("Pass --check to inspect files or --in-place to rewrite them.")

	reports: list[FileReport] = []
	for path in candidate_files([path.resolve() for path in args.paths]):
		if not path.exists():
			raise SystemExit(f"Path does not exist: {path}")
		reports.append(normalize_file(path, write=args.in_place))

	for report in reports:
		action = "fixed" if args.in_place and report.changed else "ok"
		if args.check and report.changed:
			action = "needs-fix"
		print(
			f"{action}: {report.path} "
			f"cases={report.cases} polygons={report.polygons} reversed={report.reversed_polygons}"
		)

	total_reversed = sum(report.reversed_polygons for report in reports)
	print(f"summary: files={len(reports)} reversed={total_reversed}")
	return 1 if args.check and total_reversed else 0


if __name__ == "__main__":
	raise SystemExit(main())
