#!/usr/bin/env python3
"""
Generate non-convex TPP benchmark instances from OSM building footprints.

Example:
    python3 packages/instance-generation/source/gen_instances.py sp-city.osm.pbf \
        --output-bin packages/nonconvex-tpp/cpp/tests/osm_buildings.bin \
        --preview packages/instance-generation/source/osm_buildings.png \
        --instances 100 \
        --polygons-per-instance 8 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


EARTH_METERS_PER_DEGREE_LAT = 111_320.0
TESTCASE_ENDIAN = "<"
plt = None
osmium = None
Polygon = None
MultiPolygon = None


@dataclass(frozen=True)
class Candidate:
	polygon: Polygon
	centroid: tuple[float, float]
	area: float
	convexity: float


@dataclass(frozen=True)
class TestCase:
	start: tuple[float, float]
	target: tuple[float, float]
	polygons: list[list[tuple[float, float]]]
	source_candidate_indices: list[int]
	scale: float
	center: tuple[float, float]
	span: tuple[float, float]


def ensure_geometry_dependencies() -> None:
	global MultiPolygon, Polygon, osmium

	if osmium is not None:
		return

	try:
		import osmium as osmium_module
		from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
		from shapely.geometry import Polygon as ShapelyPolygon
	except ImportError as error:
		raise SystemExit(
			"Missing Python dependency. Install osmium and shapely before running this generator."
		) from error

	osmium = osmium_module
	Polygon = ShapelyPolygon
	MultiPolygon = ShapelyMultiPolygon


def ensure_plot_dependency() -> None:
	global plt

	if plt is not None:
		return

	try:
		import matplotlib.pyplot as matplotlib_pyplot
	except ImportError as error:
		raise SystemExit(
			"Missing Python dependency. Install matplotlib or pass --no-preview."
		) from error

	plt = matplotlib_pyplot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate binary TPP test cases from OSM building footprints.")
	parser.add_argument("input_pbf", type=Path, help="Input .osm.pbf file.")
	parser.add_argument("--output-bin", type=Path, default=Path("instances.bin"), help="Binary output compatible with load_test_cases().")
	parser.add_argument("--preview", type=Path, default=Path("instances.png"), help="Preview image output.")
	parser.add_argument("--manifest", type=Path, help="Optional JSON metadata output. Defaults to <output-bin>.manifest.json.")
	parser.add_argument("--no-preview", action="store_true", help="Do not write preview images.")
	parser.add_argument("--no-manifest", action="store_true", help="Do not write the metadata manifest.")
	parser.add_argument("--instances", type=int, default=20, help="Number of TPP instances to generate.")
	parser.add_argument("--polygons-per-instance", type=int, default=8, help="Number of polygons in each instance.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument("--simplify-tolerance", type=float, default=1.0, help="Shapely simplification tolerance in projected meters.")
	parser.add_argument("--scale", type=float, default=1.0, help="Coordinate multiplier after projection and normalization.")
	parser.add_argument("--normalization", choices=("instance", "dataset", "none"), default="instance", help="Coordinate centering mode.")
	parser.add_argument("--order", choices=("spatial", "left-to-right", "random", "angle"), default="spatial", help="Polygon order inside each TPP instance.")
	parser.add_argument("--sampling", choices=("local", "uniform"), default="local", help="Sample each instance from a local neighborhood or from the whole region.")
	parser.add_argument("--local-pool-size", type=int, default=80, help="For local sampling, choose polygons from this many nearest candidates around a random anchor.")
	parser.add_argument("--layout", choices=("geographic", "grid"), default="geographic", help="Keep sampled polygons in map positions or rescale and place them on a non-overlapping grid.")
	parser.add_argument("--grid-polygon-size", type=float, default=1.0, help="For grid layout, scale each polygon so its largest bbox dimension equals this size before --scale.")
	parser.add_argument("--grid-cell-size", type=float, default=3.0, help="For grid layout, distance between grid cell centers before --scale.")
	parser.add_argument("--grid-columns", type=int, default=0, help="For grid layout, number of columns. Defaults to ceil(sqrt(polygons_per_instance)).")
	parser.add_argument("--grid-placement", choices=("row-major", "random"), default="row-major", help="For grid layout, assign visit order to row-major cells or random cells.")
	parser.add_argument("--convex-replacement-fraction", type=float, default=0.0, help="Replace this fraction of sampled polygons with synthetic convex many-vertex polygons.")
	parser.add_argument("--convex-replacement-vertices", type=int, default=64, help="Number of vertices in each synthetic convex replacement polygon.")
	parser.add_argument("--convex-replacement-scale", type=float, default=1.0, help="Size multiplier for synthetic convex replacement polygons.")
	parser.add_argument("--convex-replacement-position", choices=("middle", "random", "alternating"), default="middle", help="Where to place convex replacements in the TPP visit order.")
	parser.add_argument("--endpoint-mode", choices=("ordered", "bbox"), default="ordered", help="How start and target points are placed.")
	parser.add_argument("--candidate-pool", choices=("nonconvex", "all"), default="nonconvex", help="Use only non-convex buildings or all valid buildings.")
	parser.add_argument("--nonconvex-threshold", type=float, default=0.98, help="Area / convex-hull-area below this value is considered non-convex.")
	parser.add_argument("--min-area", type=float, default=10.0, help="Minimum building area in projected square meters.")
	parser.add_argument("--min-vertices", type=int, default=4, help="Minimum vertices after simplification.")
	parser.add_argument("--max-vertices", type=int, default=80, help="Maximum vertices after simplification.")
	parser.add_argument("--single-preview-count", type=int, default=3, help="Number of individual instance preview images to write.")
	parser.add_argument("--single-preview-dir", type=Path, help="Directory for individual instance previews. Defaults to <preview-stem>-instances.")
	parser.add_argument("--cache", type=Path, help="Raw building-ring cache. Defaults to <input>.buildings.pkl.")
	parser.add_argument("--no-cache", action="store_true", help="Do not read or write the building-ring cache.")
	return parser.parse_args(argv)


def load_building_rings(pbf_path: Path, cache_path: Path, use_cache: bool) -> list[list[tuple[float, float]]]:
	if use_cache and cache_path.exists():
		print(f"Loading building rings from cache: {cache_path}", flush=True)
		with cache_path.open("rb") as file:
			rings = pickle.load(file)

		if not isinstance(rings, list):
			raise ValueError(f"Unexpected cache contents: {cache_path}")

		print(f"Loaded {len(rings)} raw building rings.", flush=True)
		return rings

	ensure_geometry_dependencies()

	class BuildingHandler(osmium.SimpleHandler):
		def __init__(self) -> None:
			super().__init__()
			self.rings: list[list[tuple[float, float]]] = []

		def area(self, area: osmium.osm.Area) -> None:
			if "building" not in area.tags:
				return

			try:
				for outer in area.outer_rings():
					coords = [(node.lon, node.lat) for node in outer]

					if len(coords) >= 4:
						self.rings.append(coords)
			except Exception:
				# Some OSM areas have missing locations or malformed rings. They are not useful as
				# polygon test cases, so skip them and keep parsing.
				return

	print(f"Parsing OSM buildings from {pbf_path}...", flush=True)
	handler = BuildingHandler()
	handler.apply_file(str(pbf_path), locations=True)
	print(f"Found {len(handler.rings)} raw building rings.", flush=True)

	if use_cache:
		cache_path.parent.mkdir(parents=True, exist_ok=True)
		with cache_path.open("wb") as file:
			pickle.dump(handler.rings, file)
		print(f"Saved cache: {cache_path}", flush=True)

	return handler.rings


def projection_origin(rings: Sequence[Sequence[tuple[float, float]]]) -> tuple[float, float]:
	if not rings:
		raise ValueError("No building rings found.")

	min_lon = min(lon for ring in rings for lon, _ in ring)
	max_lon = max(lon for ring in rings for lon, _ in ring)
	min_lat = min(lat for ring in rings for _, lat in ring)
	max_lat = max(lat for ring in rings for _, lat in ring)
	return ((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0)


def project_ring(ring: Sequence[tuple[float, float]], origin: tuple[float, float]) -> list[tuple[float, float]]:
	origin_lon, origin_lat = origin
	x_scale = EARTH_METERS_PER_DEGREE_LAT * math.cos(math.radians(origin_lat))
	return [
		((lon - origin_lon) * x_scale, (lat - origin_lat) * EARTH_METERS_PER_DEGREE_LAT)
		for lon, lat in ring
	]


def iter_polygon_parts(geometry: Polygon | MultiPolygon) -> Iterable[Polygon]:
	if isinstance(geometry, Polygon):
		yield geometry
	elif isinstance(geometry, MultiPolygon):
		yield from geometry.geoms


def polygon_without_duplicate_close(poly: Polygon) -> list[tuple[float, float]]:
	coords = list(poly.exterior.coords)

	if len(coords) > 1 and coords[0] == coords[-1]:
		coords.pop()

	return [(float(x), float(y)) for x, y in coords]


def clean_polygon(poly: Polygon, simplify_tolerance: float) -> Polygon | None:
	if simplify_tolerance > 0:
		poly = poly.simplify(simplify_tolerance, preserve_topology=True)

	if poly.is_empty:
		return None

	if not poly.is_valid:
		poly = poly.buffer(0)

	if poly.is_empty:
		return None

	parts = [part for part in iter_polygon_parts(poly) if not part.is_empty]

	if not parts:
		return None

	cleaned = max(parts, key=lambda part: part.area)

	if not cleaned.exterior.is_ccw:
		cleaned = Polygon(list(cleaned.exterior.coords)[::-1])

	return cleaned


def build_candidates(args: argparse.Namespace, rings: Sequence[Sequence[tuple[float, float]]], origin: tuple[float, float]) -> list[Candidate]:
	candidates: list[Candidate] = []

	for ring in rings:
		projected = project_ring(ring, origin)
		poly = clean_polygon(Polygon(projected), args.simplify_tolerance)

		if poly is None:
			continue

		vertices = polygon_without_duplicate_close(poly)
		hull_area = poly.convex_hull.area
		convexity = poly.area / hull_area if hull_area > 0 else 0.0

		if poly.area < args.min_area:
			continue

		if len(vertices) < args.min_vertices or len(vertices) > args.max_vertices:
			continue

		if args.candidate_pool == "nonconvex" and convexity >= args.nonconvex_threshold:
			continue

		centroid = poly.centroid
		candidates.append(Candidate(poly, (float(centroid.x), float(centroid.y)), float(poly.area), float(convexity)))

	return candidates


def order_candidates(indices: list[int], candidates: Sequence[Candidate], rng: random.Random, mode: str) -> list[int]:
	if mode == "random":
		ordered = list(indices)
		rng.shuffle(ordered)
		return ordered

	if mode == "left-to-right":
		return sorted(indices, key=lambda index: (candidates[index].centroid[0], candidates[index].centroid[1]))

	if mode == "angle":
		cx = sum(candidates[index].centroid[0] for index in indices) / len(indices)
		cy = sum(candidates[index].centroid[1] for index in indices) / len(indices)
		return sorted(indices, key=lambda index: math.atan2(candidates[index].centroid[1] - cy, candidates[index].centroid[0] - cx))

	remaining = set(indices)
	current = min(remaining, key=lambda index: (candidates[index].centroid[0], candidates[index].centroid[1]))
	ordered = [current]
	remaining.remove(current)

	while remaining:
		cx, cy = candidates[current].centroid
		current = min(
			remaining,
			key=lambda index: (candidates[index].centroid[0] - cx) ** 2 + (candidates[index].centroid[1] - cy) ** 2,
		)
		ordered.append(current)
		remaining.remove(current)

	return ordered


def sample_candidate_indices(args: argparse.Namespace, candidates: Sequence[Candidate], rng: random.Random) -> list[int]:
	if args.sampling == "uniform":
		return rng.sample(range(len(candidates)), args.polygons_per_instance)

	pool_size = max(args.polygons_per_instance, args.local_pool_size)
	anchor = rng.randrange(len(candidates))
	anchor_x, anchor_y = candidates[anchor].centroid
	nearest = sorted(
		range(len(candidates)),
		key=lambda index: (candidates[index].centroid[0] - anchor_x) ** 2 + (candidates[index].centroid[1] - anchor_y) ** 2,
	)
	local_pool = nearest[:pool_size]
	return rng.sample(local_pool, args.polygons_per_instance)


def bounds_of_polygons(polygons: Sequence[Polygon]) -> tuple[float, float, float, float]:
	minx = min(poly.bounds[0] for poly in polygons)
	miny = min(poly.bounds[1] for poly in polygons)
	maxx = max(poly.bounds[2] for poly in polygons)
	maxy = max(poly.bounds[3] for poly in polygons)
	return minx, miny, maxx, maxy


def bounds_of_point_polygons(polygons: Sequence[Sequence[tuple[float, float]]]) -> tuple[float, float, float, float]:
	minx = min(point[0] for polygon in polygons for point in polygon)
	miny = min(point[1] for polygon in polygons for point in polygon)
	maxx = max(point[0] for polygon in polygons for point in polygon)
	maxy = max(point[1] for polygon in polygons for point in polygon)
	return minx, miny, maxx, maxy


def make_endpoints(ordered: Sequence[Candidate], mode: str) -> tuple[tuple[float, float], tuple[float, float]]:
	polygons = [candidate.polygon for candidate in ordered]
	minx, miny, maxx, maxy = bounds_of_polygons(polygons)
	width = maxx - minx
	height = maxy - miny
	margin = max(20.0, 0.15 * max(width, height))

	if mode == "bbox":
		y = (miny + maxy) / 2.0
		return (minx - margin, y), (maxx + margin, y)

	first = ordered[0].centroid
	last = ordered[-1].centroid
	dx = last[0] - first[0]
	dy = last[1] - first[1]
	length = math.hypot(dx, dy)

	if length < 1e-9:
		dx, dy, length = 1.0, 0.0, 1.0

	ux = dx / length
	uy = dy / length
	return (first[0] - ux * margin, first[1] - uy * margin), (last[0] + ux * margin, last[1] + uy * margin)


def make_point_endpoints(
	polygons: Sequence[Sequence[tuple[float, float]]],
	mode: str,
	min_margin: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
	minx, miny, maxx, maxy = bounds_of_point_polygons(polygons)
	width = maxx - minx
	height = maxy - miny
	margin = max(min_margin, 0.15 * max(width, height))

	if mode == "bbox":
		y = (miny + maxy) / 2.0
		return (minx - margin, y), (maxx + margin, y)

	first = polygon_centroid(polygons[0])
	last = polygon_centroid(polygons[-1])
	dx = last[0] - first[0]
	dy = last[1] - first[1]
	length = math.hypot(dx, dy)

	if length < 1e-9:
		dx, dy, length = 1.0, 0.0, 1.0

	ux = dx / length
	uy = dy / length
	return (first[0] - ux * margin, first[1] - uy * margin), (last[0] + ux * margin, last[1] + uy * margin)


def grid_layout_polygons(args: argparse.Namespace, ordered: Sequence[Candidate], rng: random.Random) -> list[list[tuple[float, float]]]:
	if args.grid_polygon_size <= 0:
		raise ValueError("--grid-polygon-size must be positive.")

	if args.grid_cell_size <= args.grid_polygon_size:
		raise ValueError("--grid-cell-size must be greater than --grid-polygon-size to guarantee separated cells.")

	columns = args.grid_columns if args.grid_columns > 0 else math.ceil(math.sqrt(len(ordered)))
	if columns <= 0:
		raise ValueError("--grid-columns must be positive when provided.")

	cell_centers = [
		((index % columns) * args.grid_cell_size, (index // columns) * args.grid_cell_size)
		for index in range(len(ordered))
	]

	if args.grid_placement == "random":
		rng.shuffle(cell_centers)

	placed: list[list[tuple[float, float]]] = []

	for index, candidate in enumerate(ordered):
		source = polygon_without_duplicate_close(candidate.polygon)
		minx = min(point[0] for point in source)
		miny = min(point[1] for point in source)
		maxx = max(point[0] for point in source)
		maxy = max(point[1] for point in source)
		source_size = max(maxx - minx, maxy - miny)

		if source_size <= 0:
			continue

		# Center the bounding box, not the polygon centroid. With bbox centering,
		# cell_size > polygon_size guarantees that neighboring bboxes are disjoint.
		centroid = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
		target_center = cell_centers[index]
		scale = args.grid_polygon_size / source_size
		placed.append([
			(
				(point[0] - centroid[0]) * scale + target_center[0],
				(point[1] - centroid[1]) * scale + target_center[1],
			)
			for point in source
		])

	return placed


def regular_polygon(center: tuple[float, float], radius: float, vertices: int, rotation: float) -> list[tuple[float, float]]:
	return [
		(
			center[0] + math.cos(rotation + 2.0 * math.pi * index / vertices) * radius,
			center[1] + math.sin(rotation + 2.0 * math.pi * index / vertices) * radius,
		)
		for index in range(vertices)
	]


def convex_replacement_indices(args: argparse.Namespace, count: int, rng: random.Random) -> set[int]:
	if args.convex_replacement_fraction <= 0.0:
		return set()

	if args.convex_replacement_fraction > 1.0:
		raise ValueError("--convex-replacement-fraction must be between 0 and 1.")

	replacement_count = min(count, int(round(count * args.convex_replacement_fraction)))

	if replacement_count == 0:
		return set()

	if args.convex_replacement_position == "random":
		return set(rng.sample(range(count), replacement_count))

	if args.convex_replacement_position == "alternating":
		return set(range(1, count, 2)) if replacement_count >= count // 2 else set(range(1, replacement_count * 2, 2))

	start = max(0, (count - replacement_count) // 2)
	return set(range(start, start + replacement_count))


def apply_convex_replacements(
	args: argparse.Namespace,
	polygons: list[list[tuple[float, float]]],
	rng: random.Random,
) -> list[list[tuple[float, float]]]:
	indices = convex_replacement_indices(args, len(polygons), rng)

	if not indices:
		return polygons

	if args.convex_replacement_vertices < 3:
		raise ValueError("--convex-replacement-vertices must be at least 3.")

	if args.convex_replacement_scale <= 0.0:
		raise ValueError("--convex-replacement-scale must be positive.")

	replaced: list[list[tuple[float, float]]] = []

	for index, polygon in enumerate(polygons):
		if index not in indices:
			replaced.append(polygon)
			continue

		minx, miny, maxx, maxy = bounds_of_point_polygons([polygon])
		size = max(maxx - minx, maxy - miny)
		center = polygon_centroid(polygon)
		radius = size * args.convex_replacement_scale / 2.0
		replaced.append(regular_polygon(center, radius, args.convex_replacement_vertices, rng.random() * 2.0 * math.pi))

	return replaced


def normalize_case(
	start: tuple[float, float],
	target: tuple[float, float],
	polygons: Sequence[Polygon],
	mode: str,
	dataset_origin: tuple[float, float],
	scale: float,
) -> tuple[tuple[float, float], tuple[float, float], list[list[tuple[float, float]]], tuple[float, float]]:
	if mode == "none":
		center = (0.0, 0.0)
	elif mode == "dataset":
		center = dataset_origin
	else:
		minx, miny, maxx, maxy = bounds_of_polygons(polygons)
		minx = min(minx, start[0], target[0])
		miny = min(miny, start[1], target[1])
		maxx = max(maxx, start[0], target[0])
		maxy = max(maxy, start[1], target[1])
		center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

	def transform(point: tuple[float, float]) -> tuple[float, float]:
		return ((point[0] - center[0]) * scale, (point[1] - center[1]) * scale)

	return (
		transform(start),
		transform(target),
		[[transform(point) for point in polygon_without_duplicate_close(poly)] for poly in polygons],
		center,
	)


def normalize_point_case(
	start: tuple[float, float],
	target: tuple[float, float],
	polygons: Sequence[Sequence[tuple[float, float]]],
	mode: str,
	dataset_origin: tuple[float, float],
	scale: float,
) -> tuple[tuple[float, float], tuple[float, float], list[list[tuple[float, float]]], tuple[float, float]]:
	if mode == "none":
		center = (0.0, 0.0)
	elif mode == "dataset":
		center = dataset_origin
	else:
		minx, miny, maxx, maxy = bounds_of_point_polygons(polygons)
		minx = min(minx, start[0], target[0])
		miny = min(miny, start[1], target[1])
		maxx = max(maxx, start[0], target[0])
		maxy = max(maxy, start[1], target[1])
		center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

	def transform(point: tuple[float, float]) -> tuple[float, float]:
		return ((point[0] - center[0]) * scale, (point[1] - center[1]) * scale)

	return (
		transform(start),
		transform(target),
		[[transform(point) for point in polygon] for polygon in polygons],
		center,
	)


def generate_cases(args: argparse.Namespace, candidates: Sequence[Candidate]) -> list[TestCase]:
	if args.instances <= 0:
		raise ValueError("--instances must be positive.")

	if args.polygons_per_instance <= 0:
		raise ValueError("--polygons-per-instance must be positive.")

	if len(candidates) < args.polygons_per_instance:
		raise ValueError(
			f"Only {len(candidates)} candidate polygons available, but {args.polygons_per_instance} are required per instance."
		)

	rng = random.Random(args.seed)
	dataset_origin = (
		sum(candidate.centroid[0] for candidate in candidates) / len(candidates),
		sum(candidate.centroid[1] for candidate in candidates) / len(candidates),
	)
	cases: list[TestCase] = []

	for _ in range(args.instances):
		sampled = sample_candidate_indices(args, candidates, rng)
		ordered_indices = order_candidates(sampled, candidates, rng, args.order)
		ordered_candidates = [candidates[index] for index in ordered_indices]

		if args.layout == "grid":
			point_polygons = grid_layout_polygons(args, ordered_candidates, rng)
			endpoint_margin = args.grid_cell_size * 0.5
			normalization_origin = (0.0, 0.0)
		else:
			point_polygons = [polygon_without_duplicate_close(candidate.polygon) for candidate in ordered_candidates]
			endpoint_margin = 20.0
			normalization_origin = dataset_origin

		point_polygons = apply_convex_replacements(args, point_polygons, rng)
		start, target = make_point_endpoints(point_polygons, args.endpoint_mode, endpoint_margin)
		minx, miny, maxx, maxy = bounds_of_point_polygons(point_polygons)
		norm_start, norm_target, norm_polygons, center = normalize_point_case(
			start,
			target,
			point_polygons,
			args.normalization,
			normalization_origin,
			args.scale,
		)

		cases.append(
			TestCase(
				start=norm_start,
				target=norm_target,
				polygons=norm_polygons,
				source_candidate_indices=ordered_indices,
				scale=args.scale,
				center=center,
				span=((maxx - minx) * args.scale, (maxy - miny) * args.scale),
			)
		)

	return cases


def write_vector(file, point: tuple[float, float]) -> None:
	file.write(struct.pack(f"{TESTCASE_ENDIAN}dd", point[0], point[1]))


def write_size(file, value: int) -> None:
	file.write(struct.pack(f"{TESTCASE_ENDIAN}Q", value))


def write_binary_cases(cases: Sequence[TestCase], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)

	with path.open("wb") as file:
		for case in cases:
			write_vector(file, case.start)
			write_vector(file, case.target)
			write_size(file, len(case.polygons))

			for polygon in case.polygons:
				write_size(file, len(polygon))

				for vertex in polygon:
					write_vector(file, vertex)

			write_size(file, 0)

	print(f"Wrote {len(cases)} binary TPP test cases to {path}", flush=True)


def write_manifest(args: argparse.Namespace, cases: Sequence[TestCase], candidates: Sequence[Candidate], path: Path, projection: tuple[float, float]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	data = {
		"format": "tpp.encode_test.raw-stream",
		"binary": str(args.output_bin),
		"preview": str(args.preview),
		"input_pbf": str(args.input_pbf),
		"projection": {
			"type": "equirectangular",
			"origin_lon_lat": list(projection),
			"projected_units": "meters",
		},
		"parameters": {
			"instances": args.instances,
			"polygons_per_instance": args.polygons_per_instance,
			"seed": args.seed,
			"simplify_tolerance": args.simplify_tolerance,
			"scale": args.scale,
			"normalization": args.normalization,
			"order": args.order,
			"sampling": args.sampling,
			"local_pool_size": args.local_pool_size,
			"layout": args.layout,
			"grid_polygon_size": args.grid_polygon_size,
			"grid_cell_size": args.grid_cell_size,
			"grid_columns": args.grid_columns,
			"grid_placement": args.grid_placement,
			"convex_replacement_fraction": args.convex_replacement_fraction,
			"convex_replacement_vertices": args.convex_replacement_vertices,
			"convex_replacement_scale": args.convex_replacement_scale,
			"convex_replacement_position": args.convex_replacement_position,
			"endpoint_mode": args.endpoint_mode,
			"candidate_pool": args.candidate_pool,
			"nonconvex_threshold": args.nonconvex_threshold,
			"min_area": args.min_area,
			"min_vertices": args.min_vertices,
			"max_vertices": args.max_vertices,
			"single_preview_count": args.single_preview_count,
		},
		"candidate_count": len(candidates),
		"instances": [
			{
				"index": index,
				"polygons": len(case.polygons),
				"vertices": sum(len(polygon) for polygon in case.polygons),
				"span": list(case.span),
				"start": list(case.start),
				"target": list(case.target),
				"normalization_center": list(case.center),
				"source_candidate_indices": case.source_candidate_indices,
			}
			for index, case in enumerate(cases)
		],
	}
	path.write_text(json.dumps(data, indent=2) + "\n")
	print(f"Wrote manifest to {path}", flush=True)


def polygon_centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
	area2 = 0.0
	cx = 0.0
	cy = 0.0

	for index, point in enumerate(points):
		next_point = points[(index + 1) % len(points)]
		cross = point[0] * next_point[1] - next_point[0] * point[1]
		area2 += cross
		cx += (point[0] + next_point[0]) * cross
		cy += (point[1] + next_point[1]) * cross

	if abs(area2) < 1e-12:
		return (
			sum(point[0] for point in points) / len(points),
			sum(point[1] for point in points) / len(points),
		)

	return (cx / (3.0 * area2), cy / (3.0 * area2))


def draw_case(ax, case: TestCase, case_index: int, *, show_route: bool, title: bool) -> None:
	color_map = plt.get_cmap("viridis")
	centroids: list[tuple[float, float]] = []

	for polygon_index, polygon in enumerate(case.polygons):
		xs = [point[0] for point in polygon]
		ys = [point[1] for point in polygon]
		color = color_map(polygon_index / max(1, len(case.polygons) - 1))
		ax.fill(xs, ys, alpha=0.45, facecolor=color, edgecolor="black", linewidth=0.7)
		centroid = polygon_centroid(polygon)
		centroids.append(centroid)
		ax.text(centroid[0], centroid[1], str(polygon_index + 1), ha="center", va="center", fontsize=7)

	if show_route:
		route = [case.start, *centroids, case.target]
		ax.plot(
			[point[0] for point in route],
			[point[1] for point in route],
			color="0.25",
			linewidth=0.8,
			alpha=0.45,
			zorder=2,
		)

	ax.scatter([case.start[0]], [case.start[1]], c="limegreen", marker="o", s=24, edgecolors="black", linewidths=0.5, zorder=3)
	ax.scatter([case.target[0]], [case.target[1]], c="crimson", marker="x", s=32, linewidths=1.2, zorder=3)

	if title:
		ax.set_title(f"case {case_index} ({case.span[0]:.2g} x {case.span[1]:.2g})", fontsize=9)

	ax.set_aspect("equal", adjustable="box")
	ax.axis("off")


def plot_cases(cases: Sequence[TestCase], path: Path, max_cases: int = 50) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	preview_cases = list(cases[:max_cases])
	cols = min(5, len(preview_cases))
	rows = math.ceil(len(preview_cases) / cols)
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), squeeze=False)

	for case_index, case in enumerate(preview_cases):
		ax = axes[case_index // cols][case_index % cols]
		draw_case(ax, case, case_index, show_route=False, title=True)

	for empty_index in range(len(preview_cases), rows * cols):
		axes[empty_index // cols][empty_index % cols].axis("off")

	title = f"{len(cases)} TPP instances from OSM building footprints"
	if len(preview_cases) < len(cases):
		title += f" (showing first {len(preview_cases)})"

	fig.suptitle(title, fontsize=12)
	fig.tight_layout()
	fig.savefig(path, dpi=160)
	plt.close(fig)
	print(f"Wrote preview to {path}", flush=True)


def plot_single_case_previews(cases: Sequence[TestCase], output_dir: Path, count: int) -> None:
	if count <= 0:
		return

	output_dir.mkdir(parents=True, exist_ok=True)

	for case_index, case in enumerate(cases[:count]):
		fig, ax = plt.subplots(1, 1, figsize=(7, 7))
		draw_case(ax, case, case_index, show_route=True, title=True)
		fig.tight_layout()
		path = output_dir / f"case-{case_index:03}.png"
		fig.savefig(path, dpi=180)
		plt.close(fig)
		print(f"Wrote single-instance preview to {path}", flush=True)


def run_generation(args: argparse.Namespace) -> None:
	cache_path = args.cache if args.cache else args.input_pbf.with_suffix(args.input_pbf.suffix + ".buildings.pkl")
	manifest_path = args.manifest if args.manifest else args.output_bin.with_suffix(args.output_bin.suffix + ".manifest.json")
	single_preview_dir = args.single_preview_dir if args.single_preview_dir else args.preview.with_name(f"{args.preview.stem}-instances")

	rings = load_building_rings(args.input_pbf, cache_path, not args.no_cache)
	ensure_geometry_dependencies()
	origin = projection_origin(rings)
	candidates = build_candidates(args, rings, origin)
	print(f"Selected {len(candidates)} candidate polygons from {len(rings)} raw building rings.", flush=True)

	cases = generate_cases(args, candidates)
	write_binary_cases(cases, args.output_bin)

	if not args.no_preview:
		ensure_plot_dependency()
		plot_cases(cases, args.preview)
		plot_single_case_previews(cases, single_preview_dir, args.single_preview_count)

	if not args.no_manifest:
		write_manifest(args, cases, candidates, manifest_path, origin)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	run_generation(args)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
