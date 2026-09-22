#!/usr/bin/env python3
"""Convert tspn-bnb2's native tour corpus into anchored endpoint TPP paths."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import struct
import zipfile
from pathlib import Path


Point = tuple[float, float]


def parser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(description=__doc__)
	result.add_argument("--input", type=Path, required=True)
	result.add_argument("--output", type=Path, required=True)
	result.add_argument(
		"--depot-strategy", choices=("auto", "reference-cycle", "bbox-corners"),
		default="auto",
	)
	return result


def parse_points(text: str, geometry: str) -> list[Point]:
	prefix = f"{geometry} ("
	if not text.startswith(prefix) or not text.endswith(")"):
		raise ValueError(f"Unsupported {geometry} WKT: {text[:80]}")
	body = text[len(prefix):-1]
	if geometry == "POLYGON":
		if not body.startswith("(") or not body.endswith(")") or "), (" in body:
			raise ValueError("Only simple polygon exteriors are supported")
		body = body[1:-1]
	points = []
	for coordinate in body.split(","):
		values = coordinate.strip().split()
		if len(values) != 2:
			raise ValueError(f"Expected a 2D coordinate: {coordinate}")
		points.append((float(values[0]), float(values[1])))
	if geometry == "POLYGON" and len(points) > 1 and points[0] == points[-1]:
		points.pop()
	return points


def write_u64(file, value: int) -> None:
	file.write(struct.pack("<Q", value))


def load_items(path: Path) -> list[dict]:
	if path.suffix == ".zip":
		with zipfile.ZipFile(path) as archive:
			return [
				{"name": Path(name).stem, "instance": archive.read(name).decode()}
				for name in sorted(archive.namelist()) if name.endswith(".json")
			]
	return json.load(lzma.open(path, "rt", encoding="utf-8"))


def exterior_depots(polygons: list[list[Point]]) -> tuple[Point, Point]:
	points = [point for polygon in polygons for point in polygon]
	min_x, max_x = min(point[0] for point in points), max(point[0] for point in points)
	min_y, max_y = min(point[1] for point in points), max(point[1] for point in points)
	margin = 0.1 * max(max_x - min_x, max_y - min_y, 1.0)
	return (min_x - margin, min_y - margin), (max_x + margin, max_y + margin)


def main(argv: list[str] | None = None) -> int:
	args = parser().parse_args(argv)
	items = load_items(args.input)
	strategy = args.depot_strategy
	if strategy == "auto":
		strategy = "bbox-corners" if args.input.suffix == ".zip" else "reference-cycle"
	args.output.parent.mkdir(parents=True, exist_ok=True)
	metadata = []
	with args.output.open("wb") as output:
		for index, item in enumerate(items):
			instance = json.loads(item["instance"])
			polygons = [parse_points(wkt, "POLYGON") for wkt in instance["polygons"]]
			if strategy == "reference-cycle":
				reference = json.loads(item["optimal_solution"])
				trajectory = parse_points(reference["trajectory"], "LINESTRING")
				if not trajectory:
					raise ValueError(f"Reference trajectory is empty for {item['name']}")
				start = target = trajectory[0]
			else:
				reference = {}
				trajectory = []
				start, target = exterior_depots(polygons)
			output.write(struct.pack("<dddd", *start, *target))
			write_u64(output, len(polygons))
			for polygon in polygons:
				write_u64(output, len(polygon))
				for point in polygon:
					output.write(struct.pack("<dd", *point))
			write_u64(output, len(trajectory))
			for point in trajectory:
				output.write(struct.pack("<dd", *point))
			meta = instance.get("meta", {})
			metadata.append({
				"suite_index": index,
				"name": item["name"],
				"source": meta.get("source", "geographic" if "geo_information" in meta else "unknown"),
				"polygons": len(polygons),
				"vertices": sum(len(polygon) for polygon in polygons),
				"reference_lower_bound": reference.get("lower_bound"),
				"reference_upper_bound": reference.get("upper_bound"),
				"adaptation": strategy,
			})
	with args.output.with_suffix(".csv").open("w", newline="") as output:
		writer = csv.DictWriter(output, fieldnames=metadata[0].keys())
		writer.writeheader()
		writer.writerows(metadata)
	print(f"Converted {len(items)} instances to {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
