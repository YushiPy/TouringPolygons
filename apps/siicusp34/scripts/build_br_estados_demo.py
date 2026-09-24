#!/usr/bin/env python3
"""Build the IBGE 27-UF suite and its static SIICUSP demonstration.

Run from the repository root with the C++ unordered solver built. The source
archive is parsed directly, so the conversion has no GIS or Python package
dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "apps/siicusp34"
SUITE = ROOT / "benchmarks/suites/br-estados"
SOURCE = SUITE / "ibge/BR_UF_2025.zip"
MAPPING = SUITE / "polygons.csv"
OUTPUT = APP / "data/br-estados-demo.js"
BINARIO = SUITE / "br-estados.bin"
SIMPLIFICATION_METRES = 5000.0
MAX_CALLS = 1_000_000
MAX_SECONDS = 300
ABSOLUTE_GAP = 1e-7
RELATIVE_GAP = 1e-9
PARTITION_SOURCE = ROOT / "packages/optimal-convex-partition/cpp"
EXPECTED_UFS = 27

sys.path.insert(0, str(ROOT / "benchmarks/_internal"))
sys.path.insert(0, str(APP / "scripts"))

from normalize_polygon_orientation import BinaryCase, write_binary_cases  # noqa: E402
from unordered_runner import encode_instance, run_unordered_solver  # noqa: E402
from benchmark_cases import segments_intersect_or_touch  # noqa: E402
from build_usp_demo import first_contact, optimal_decomposition  # noqa: E402

Point = tuple[float, float]
GRS80_A = 6_378_137.0
GRS80_F = 1 / 298.257222101
GRS80_ES = GRS80_F * (2 - GRS80_F)
FALSE_EASTING = 5_000_000.0
FALSE_NORTHING = 10_000_000.0
CENTRAL_MERIDIAN = math.radians(-54.0)
C00, C02, C04, C06, C08 = 1.0, 0.25, 0.046875, 0.01953125, 0.01068115234375
C22, C44, C46, C48 = 0.75, 0.46875, 0.013020833333333333, 0.007120768229166667
C66, C68, C88 = 0.3645833333333333, 0.005696614583333333, 0.3076171875
EN0 = C00 - GRS80_ES * (C02 + GRS80_ES * (C04 + GRS80_ES * (C06 + GRS80_ES * C08)))
EN1 = GRS80_ES * (C22 - GRS80_ES * (C04 + GRS80_ES * (C06 + GRS80_ES * C08)))
_t = GRS80_ES * GRS80_ES
EN2 = _t * (C44 - GRS80_ES * (C46 + GRS80_ES * C48))
_t *= GRS80_ES
EN3 = _t * (C66 - GRS80_ES * C68)
EN4 = _t * GRS80_ES * C88
del _t


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--solver", type=Path, default=ROOT / ".build/unordered/tpp")
	parser.add_argument("--output", type=Path, default=OUTPUT)
	parser.add_argument("--tolerance-metres", type=float, default=SIMPLIFICATION_METRES)
	parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
	parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
	parser.add_argument("--prepare-only", action="store_true", help="Write source mapping and binary, without solving or exporting app JSON")
	return parser.parse_args()


def signed_area2(ring: list[Point]) -> float:
	return sum(
		x * ring[(index + 1) % len(ring)][1] - ring[(index + 1) % len(ring)][0] * y
		for index, (x, y) in enumerate(ring)
	)


def dbf_rows(archive: zipfile.ZipFile) -> list[dict[str, str]]:
	data = archive.read("BR_UF_2025.dbf")
	encoding = archive.read("BR_UF_2025.cpg").decode("ascii").strip()
	record_count = struct.unpack_from("<I", data, 4)[0]
	header_length, record_length = struct.unpack_from("<HH", data, 8)
	fields: list[tuple[str, int]] = []
	offset = 32
	while data[offset] != 0x0D:
		descriptor = data[offset:offset + 32]
		fields.append((descriptor[:11].split(b"\0", 1)[0].decode("ascii"), descriptor[16]))
		offset += 32
	if offset + 1 != header_length:
		raise ValueError("Unexpected DBF field descriptor header")
	rows = []
	for index in range(record_count):
		record = data[header_length + index * record_length:header_length + (index + 1) * record_length]
		if record[:1] == b"*":
			continue
		cursor = 1
		row = {}
		for name, width in fields:
			row[name] = record[cursor:cursor + width].decode(encoding).strip()
			cursor += width
		rows.append(row)
	return rows


def shapefile_records(archive: zipfile.ZipFile) -> list[list[list[Point]]]:
	data = archive.read("BR_UF_2025.shp")
	if len(data) < 100 or struct.unpack_from(">i", data, 0)[0] != 9994:
		raise ValueError("Invalid IBGE shapefile header")
	if struct.unpack_from("<i", data, 32)[0] != 5:
		raise ValueError("Expected Polygon geometry in BR_UF_2025.shp")
	position = 100
	features: list[list[list[Point]]] = []
	while position < len(data):
		if position + 8 > len(data):
			raise ValueError("Truncated shapefile record header")
		_, content_words = struct.unpack_from(">ii", data, position)
		position += 8
		end = position + content_words * 2
		if end > len(data):
			raise ValueError("Truncated shapefile record")
		record = data[position:end]
		shape_type = struct.unpack_from("<i", record, 0)[0]
		if shape_type != 5:
			raise ValueError(f"Unexpected shapefile record type {shape_type}")
		part_count, point_count = struct.unpack_from("<ii", record, 36)
		parts = list(struct.unpack_from("<" + "i" * part_count, record, 44))
		points_offset = 44 + 4 * part_count
		points = [struct.unpack_from("<dd", record, points_offset + index * 16) for index in range(point_count)]
		parts.append(point_count)
		rings = []
		for index in range(part_count):
			ring = list(points[parts[index]:parts[index + 1]])
			if len(ring) >= 2 and ring[0] == ring[-1]:
				ring.pop()
			if len(ring) >= 3:
				rings.append([(float(x), float(y)) for x, y in ring])
		if not rings:
			raise ValueError("UF feature has no nondegenerate rings")
		features.append(rings)
		position = end
	return features


def polyconic_forward(longitude: float, latitude: float) -> Point:
	"""SIRGAS 2000 / Brazil Polyconic (EPSG:5880) forward projection."""
	phi = math.radians(latitude)
	lam = math.radians(longitude) - CENTRAL_MERIDIAN
	es = GRS80_ES

	def meridional_arc(value: float) -> float:
		sine = math.sin(value)
		cosine = math.cos(value)
		cosine *= sine
		sine *= sine
		return GRS80_A * (
			EN0 * value - cosine * (EN1 + sine * (EN2 + sine * (EN3 + sine * EN4)))
		)

	if abs(phi) <= 1e-10:
		x, y = GRS80_A * lam, 0.0
	else:
		sine, cosine = math.sin(phi), math.cos(phi)
		ms = cosine / (math.sqrt(1.0 - es * sine * sine) * sine)
		angle = lam * sine
		x = GRS80_A * ms * math.sin(angle)
		y = meridional_arc(phi) + GRS80_A * ms * (1.0 - math.cos(angle))
	return x + FALSE_EASTING, y + FALSE_NORTHING


def point_segment_distance2(point: Point, first: Point, last: Point) -> float:
	dx, dy = last[0] - first[0], last[1] - first[1]
	length2 = dx * dx + dy * dy
	if length2 == 0:
		return (point[0] - first[0]) ** 2 + (point[1] - first[1]) ** 2
	fraction = max(0.0, min(1.0, ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / length2))
	return (point[0] - (first[0] + fraction * dx)) ** 2 + (point[1] - (first[1] + fraction * dy)) ** 2


def simplify_open(points: list[Point], tolerance2: float) -> list[Point]:
	if len(points) <= 2:
		return points[:]
	keep = {0, len(points) - 1}
	stack = [(0, len(points) - 1)]
	while stack:
		start, end = stack.pop()
		maximum, selected = tolerance2, None
		for index in range(start + 1, end):
			distance2 = point_segment_distance2(points[index], points[start], points[end])
			if distance2 > maximum:
				maximum, selected = distance2, index
		if selected is not None:
			keep.add(selected)
			stack.extend(((start, selected), (selected, end)))
	return [points[index] for index in sorted(keep)]


def simplify_ring(ring: list[Point], tolerance: float) -> list[Point]:
	if len(ring) <= 3:
		return ring[:]
	anchor = ring[0]
	split = max(range(1, len(ring)), key=lambda index: (ring[index][0] - anchor[0]) ** 2 + (ring[index][1] - anchor[1]) ** 2)
	first = simplify_open(ring[:split + 1], tolerance * tolerance)
	second = simplify_open(ring[split:] + [ring[0]], tolerance * tolerance)
	result = first[:-1] + second[:-1]
	if len(result) < 3:
		return ring[:]
	return result


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
	inside = False
	px, py = point
	for index, (ax, ay) in enumerate(polygon):
		bx, by = polygon[(index + 1) % len(polygon)]
		cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
		if abs(cross) < 1e-5 and min(ax, bx) - 1e-5 <= px <= max(ax, bx) + 1e-5 and min(ay, by) - 1e-5 <= py <= max(ay, by) + 1e-5:
			return True
		if (ay > py) != (by > py) and px < ax + (py - ay) * (bx - ax) / (by - ay):
			inside = not inside
	return inside


def assert_simple_polygon(polygon: list[Point], code: str) -> None:
	count = len(polygon)
	for first_index in range(count):
		a, b = polygon[first_index], polygon[(first_index + 1) % count]
		first_bounds = (
			min(a[0], b[0]), max(a[0], b[0]),
			min(a[1], b[1]), max(a[1], b[1]),
		)
		for second_index in range(first_index + 2, count):
			if first_index == 0 and second_index == count - 1:
				continue
			c, d = polygon[second_index], polygon[(second_index + 1) % count]
			second_min_x, second_max_x = min(c[0], d[0]), max(c[0], d[0])
			second_min_y, second_max_y = min(c[1], d[1]), max(c[1], d[1])
			if (
				first_bounds[1] < second_min_x
				or second_max_x < first_bounds[0]
				or first_bounds[3] < second_min_y
				or second_max_y < first_bounds[2]
			):
				continue
			if segments_intersect_or_touch(a, b, c, d):
				raise ValueError(f"Simplified UF contour {code} self-intersects at edges {first_index} and {second_index}")


def load_suite_geometry(tolerance: float) -> tuple[list[dict[str, str]], list[list[Point]], Point]:
	if not SOURCE.is_file():
		raise FileNotFoundError(f"IBGE source archive not found: {SOURCE}")
	with zipfile.ZipFile(SOURCE) as archive:
		rows = dbf_rows(archive)
		features = shapefile_records(archive)
	if len(rows) != EXPECTED_UFS or len(features) != EXPECTED_UFS:
		raise ValueError(f"Expected {EXPECTED_UFS} UF features, found {len(rows)} DBF rows and {len(features)} geometries")
	if len({row["CD_UF"] for row in rows}) != EXPECTED_UFS:
		raise ValueError("IBGE source contains duplicate UF codes")
	feature_rows = sorted(enumerate(zip(rows, features)), key=lambda item: int(item[1][0]["CD_UF"]))
	selected: list[tuple[dict[str, str], list[Point], int, int, int]] = []
	for source_index, (row, rings) in feature_rows:
		main_ring = max(rings, key=lambda ring: abs(signed_area2(ring)))
		projected = [polyconic_forward(lon, lat) for lon, lat in main_ring]
		simplified = simplify_ring(projected, tolerance)
		if signed_area2(simplified) < 0:
			simplified.reverse()
		if len(simplified) < 3 or abs(signed_area2(simplified)) < 1.0:
			raise ValueError(f"Invalid simplified geometry for {row['SIGLA_UF']}")
		assert_simple_polygon(simplified, row["SIGLA_UF"])
		selected.append((row, simplified, source_index, len(main_ring), len(rings)))

	# EPSG:5880 contains national false offsets; store a compact local copy in metres.
	min_x = min(point[0] for _, polygon, _, _, _ in selected for point in polygon)
	min_y = min(point[1] for _, polygon, _, _, _ in selected for point in polygon)
	polygons = [[(x - min_x, y - min_y) for x, y in polygon] for _, polygon, _, _, _ in selected]
	start_projected = polyconic_forward(-47.8828, -15.7939)
	start = (start_projected[0] - min_x, start_projected[1] - min_y)
	df_index = next(index for index, (row, _, _, _, _) in enumerate(selected) if row["SIGLA_UF"] == "DF")
	if not point_in_polygon(start, polygons[df_index]):
		raise ValueError("The Brasília depot falls outside the simplified Distrito Federal polygon")

	MAPPING.parent.mkdir(parents=True, exist_ok=True)
	with MAPPING.open("w", encoding="utf-8", newline="") as file:
		writer = csv.writer(file, lineterminator="\n")
		writer.writerow([
			"polygon_index", "cd_uf", "sigla_uf", "nome_uf", "source_feature_index",
			"source_parts_total", "source_vertices_selected_ring", "vertices_binario", "area_km2_ibge",
		])
		for index, ((row, polygon, source_index, source_vertices, part_count), binary_polygon) in enumerate(zip(selected, polygons)):
			writer.writerow([
				index, row["CD_UF"], row["SIGLA_UF"], row["NM_UF"], source_index,
				part_count, source_vertices, len(binary_polygon), row["AREA_KM2"],
			])
	write_binary_cases(BINARIO, [BinaryCase(start, start, polygons, [])])
	BINARIO.chmod(0o644)
	return [row for row, _, _, _, _ in selected], polygons, start


def sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, object]:
	if args.tolerance_metres <= 0:
		raise ValueError("Simplification tolerance must be positive")
	rows, polygons, start = load_suite_geometry(args.tolerance_metres)
	if args.prepare_only:
		print(f"prepared {BINARIO}: polygons={len(polygons)} vertices={sum(map(len, polygons))} tolerance_m={args.tolerance_metres:g}")
		return {}
	if not args.solver.is_file():
		raise FileNotFoundError(f"C++ solver not found: {args.solver}")
	result = run_unordered_solver(args.solver, start, start, polygons, args.max_calls, args.max_seconds)
	if result.get("termination") != "optimal" or not result.get("exact"):
		raise ValueError(f"Solver did not certify an optimum: {result.get('termination')}")
	lower, upper = float(result["lower_bound"]), float(result["upper_bound"])
	if upper - lower > ABSOLUTE_GAP + RELATIVE_GAP * abs(upper):
		raise ValueError("The numerical certificate exceeds the configured tolerance")
	path = [[float(x), float(y)] for x, y in result["path"]]
	if len(path) < 2 or math.dist(path[0], start) > 1e-7 or math.dist(path[-1], start) > 1e-7:
		raise ValueError("The solver route endpoints differ from the Brasília depot")
	length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
	if abs(length - upper) > ABSOLUTE_GAP + RELATIVE_GAP * abs(upper):
		raise ValueError("The route length differs from its certified upper bound")
	contacts = [first_contact(path, polygon, 1e-7) for polygon in polygons]
	if any(contact is None for contact in contacts):
		raise ValueError("The route does not intersect every selected UF polygon")
	write_binary_cases(BINARIO, [BinaryCase(start, start, polygons, [tuple(point) for point in path])])
	BINARIO.chmod(0o644)
	solver_rank = {int(index): rank for rank, index in enumerate(result["order"])}
	order = sorted(range(len(polygons)), key=lambda index: (contacts[index]["fraction"], solver_rank.get(index, len(polygons) + index)))
	decomposition = optimal_decomposition(polygons)
	if len(decomposition) != len(polygons) or any(not pieces for pieces in decomposition):
		raise ValueError("The convex decomposition does not cover all UF polygons")
	encoded_input = encode_instance(start, start, polygons, args.max_calls, args.max_seconds)
	return {
		"schema_version": 1,
		"case": "br-estados",
		"title": "Brasil · 27 unidades federativas",
		"corpus": "demonstration",
		"polygons": len(polygons),
		"region_codes": [row["SIGLA_UF"] for row in rows],
		"region_names": [row["NM_UF"] for row in rows],
		"geometry": {"start": start, "target": start, "polygons": polygons},
		"path": path,
		"order": order,
		"length": upper,
		"lower_bound": lower,
		"upper_bound": upper,
		"exact": True,
		"termination": "optimal",
		"seconds": float(result["seconds"]),
		"calls": int(result["calls"]),
		"visualization": {"contacts": contacts, "decomposition": decomposition},
		"depot": {"label": "Brasília"},
		"intro": "Qual caminho um drone deve fazer para sair de Brasília, atravessar os territórios principais das 27 unidades federativas e voltar à capital, minimizando a distância? A seguir mostramos o caminho ótimo para esta versão simplificada do problema:",
		"provenance": {
			"suite_binary_sha256": sha256(BINARIO),
			"mapping_sha256": sha256(MAPPING),
			"source_zip_sha256": sha256(SOURCE),
			"partition_source_sha256": sha256(PARTITION_SOURCE / "src/optimal_convex_partition.cpp"),
			"solver_sha256": sha256(args.solver),
			"input_sha256": hashlib.sha256(encoded_input.encode()).hexdigest(),
			"projection": "SIRGAS 2000 / Brazil Polyconic (EPSG:5880), metres, translated to local origin",
			"simplification_tolerance_metres": args.tolerance_metres,
			"selected_geometry": "largest exterior ring per UF; other disconnected rings and holes omitted",
			"absolute_gap": ABSOLUTE_GAP,
			"relative_gap": RELATIVE_GAP,
		},
	}


def main() -> None:
	args = parse_args()
	data = build(args)
	if not data:
		return
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(
		"window.TPPBrEstadosDemo = "
		+ json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
		+ ";\n",
		encoding="utf-8",
	)
	decomposition = data["visualization"]["decomposition"]
	print(
		f"wrote {args.output}: polygons={data['polygons']} "
		f"vertices={sum(map(len, data['geometry']['polygons']))} "
		f"convex_pieces={sum(map(len, decomposition))} "
		f"seconds={data['seconds']:.3f} exact={data['exact']}"
	)


if __name__ == "__main__":
	main()
