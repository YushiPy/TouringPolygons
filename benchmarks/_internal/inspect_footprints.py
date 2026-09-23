"""Inspect a QGIS GeoPackage layer and compare it with the previous USP export."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "benchmarks/suites/usp-butanta-50/qgis/predios.gpkg"
DEFAULT_BASELINE = REPO_ROOT / "benchmarks/campaigns/usp-campus-85"
LATITUDE_ORIGIN = -23.557
LONGITUDE_ORIGIN = -46.732
ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def quote_identifier(value: str) -> str:
	return '"' + value.replace('"', '""') + '"'


def project_wgs84(point: tuple[float, float]) -> tuple[float, float]:
	lon, lat = point
	phi = math.radians(LATITUDE_ORIGIN)
	eccentricity_squared = 0.0066943799901413165
	radius = 6_378_137.0
	denominator = 1 - eccentricity_squared * math.sin(phi) ** 2
	east_radius = radius / math.sqrt(denominator)
	north_radius = radius * (1 - eccentricity_squared) / denominator**1.5
	return (
		math.radians(lon - LONGITUDE_ORIGIN) * east_radius * math.cos(phi),
		math.radians(lat - LATITUDE_ORIGIN) * north_radius,
	)


def parse_gpkg_polygon(blob: bytes) -> tuple[list[list[tuple[float, float]]], bool]:
	"""Read a 2D GeoPackage POLYGON blob; return rings and its empty flag."""
	if len(blob) < 13 or blob[:2] != b"GP" or blob[2] != 0:
		raise ValueError("cabeçalho GeoPackage inválido")
	flags = blob[3]
	if flags & 0x10:
		return [], True
	envelope_indicator = (flags >> 1) & 0x07
	if envelope_indicator not in ENVELOPE_BYTES:
		raise ValueError(f"envelope GeoPackage não reconhecido ({envelope_indicator})")
	offset = 8 + ENVELOPE_BYTES[envelope_indicator]
	if len(blob) < offset + 9:
		raise ValueError("geometria WKB incompleta")
	byte_order = blob[offset]
	if byte_order not in (0, 1):
		raise ValueError("ordem de bytes WKB inválida")
	order = "<" if byte_order == 1 else ">"
	wkb_type = struct.unpack_from(order + "I", blob, offset + 1)[0]
	if wkb_type != 3:
		raise ValueError(f"esperava POLYGON WKB (tipo 3), encontrei tipo {wkb_type}")
	ring_count = struct.unpack_from(order + "I", blob, offset + 5)[0]
	position = offset + 9
	rings: list[list[tuple[float, float]]] = []
	for ring_index in range(ring_count):
		if position + 4 > len(blob):
			raise ValueError(f"contagem incompleta do anel {ring_index}")
		point_count = struct.unpack_from(order + "I", blob, position)[0]
		position += 4
		byte_count = point_count * 16
		if position + byte_count > len(blob):
			raise ValueError(f"coordenadas incompletas do anel {ring_index}")
		points = [
			struct.unpack_from(order + "dd", blob, position + point_index * 16)
			for point_index in range(point_count)
		]
		if any(not math.isfinite(x) or not math.isfinite(y) for x, y in points):
			raise ValueError("coordenada não finita")
		if points and points[0] == points[-1]:
			points.pop()
		rings.append(points)
		position += byte_count
	if position != len(blob):
		raise ValueError("valores inesperados depois da geometria WKB")
	return rings, False


def read_layer(path: Path, requested_layer: str | None) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
	uri = path.resolve().as_uri() + "?mode=ro"
	connection = sqlite3.connect(uri, uri=True)
	connection.execute("PRAGMA query_only = ON")
	try:
		layers = connection.execute(
			"SELECT c.table_name, g.column_name, g.geometry_type_name, g.srs_id "
			"FROM gpkg_contents AS c JOIN gpkg_geometry_columns AS g "
			"ON c.table_name = g.table_name WHERE c.data_type = 'features' "
			"ORDER BY c.table_name"
		).fetchall()
		if requested_layer:
			matches = [layer for layer in layers if layer[0] == requested_layer]
		else:
			matches = layers
		if len(matches) != 1:
			names = ", ".join(layer[0] for layer in layers) or "nenhuma"
			raise ValueError(f"informe --layer; camadas encontradas: {names}")
		table, geometry_column, declared_type, srs_id = matches[0]
		columns = connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
		column_names = {column[1].lower(): column[1] for column in columns}
		fid_column = next((column[1] for column in columns if column[5]), None)
		if fid_column is None:
			fid_column = column_names.get("fid")
		if fid_column is None:
			raise ValueError("não encontrei a coluna fid da camada")
		id_column = column_names.get("id")
		selected = [quote_identifier(fid_column), quote_identifier(geometry_column)]
		if id_column:
			selected.append(quote_identifier(id_column))
		query = f"SELECT {', '.join(selected)} FROM {quote_identifier(table)} ORDER BY {quote_identifier(fid_column)}"
		rows = connection.execute(query).fetchall()
		features: dict[int, dict[str, Any]] = {}
		for row in rows:
			fid = int(row[0])
			blob = row[1]
			feature_id = row[2] if id_column else None
			record: dict[str, Any] = {"fid": fid, "id": feature_id, "rings": None, "error": None}
			if blob is None:
				record["status"] = "null"
			else:
				try:
					rings, is_empty = parse_gpkg_polygon(blob)
					record["rings"] = rings
					record["status"] = "empty" if is_empty or not rings or not rings[0] else "polygon"
				except ValueError as error:
					record["status"] = "invalid"
					record["error"] = str(error)
			features[fid] = record
		return {
			"table": table,
			"declared_type": declared_type,
			"srs_id": srs_id,
			"fid_column": fid_column,
			"id_column": id_column,
			"geometry_column": geometry_column,
		}, features
	finally:
		connection.close()


def read_baseline(directory: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
	manifest_path = directory / "manifest.json"
	instance_path = directory / "instance.txt"
	if not manifest_path.is_file() or not instance_path.is_file():
		raise ValueError(f"a linha de base precisa conter manifest.json e instance.txt: {directory}")
	manifest = json.loads(manifest_path.read_text())
	polygons = manifest.get("polygons")
	if not isinstance(polygons, list):
		raise ValueError("manifest.json não contém a lista de polígonos esperada")
	tokens = instance_path.read_text().split()
	if len(tokens) < 7:
		raise ValueError("instance.txt incompleto")
	polygon_count = int(tokens[4])
	if polygon_count != len(polygons):
		raise ValueError("a quantidade de polígonos não coincide entre manifesto e instância")
	position = 7
	features: dict[int, dict[str, Any]] = {}
	for index, item in enumerate(polygons):
		if position >= len(tokens):
			raise ValueError(f"faltam os vértices do polígono {index}")
		vertex_count = int(tokens[position])
		position += 1
		needed = 2 * vertex_count
		if position + needed > len(tokens):
			raise ValueError(f"faltam coordenadas do polígono {index}")
		values = [float(value) for value in tokens[position:position + needed]]
		position += needed
		points = [(values[i], values[i + 1]) for i in range(0, needed, 2)]
		fid = int(item["fid"])
		features[fid] = {"fid": fid, "id": item.get("id"), "points": points}
	if position != len(tokens):
		raise ValueError("instance.txt tem valores inesperados depois dos polígonos")
	return manifest, features


def canonical_ring(points: list[tuple[float, float]], digits: int = 4) -> tuple[tuple[float, float], ...]:
	rounded = [(round(x, digits), round(y, digits)) for x, y in points]
	if not rounded:
		return ()
	variants: list[tuple[tuple[float, float], ...]] = []
	for sequence in (rounded, list(reversed(rounded))):
		variants.extend(tuple(sequence[index:] + sequence[:index]) for index in range(len(sequence)))
	return min(variants)


def format_features(records: list[dict[str, Any]]) -> str:
	if not records:
		return "nenhuma"
	return ", ".join(
		f"fid {record['fid']}" + (f" ({record['id']})" if record.get("id") else "")
		for record in records
	)


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Conta feições de uma camada GeoPackage do QGIS e mostra o que mudou desde a instância-base da USP."
	)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"GeoPackage a inspecionar (padrão: {DEFAULT_INPUT})")
	parser.add_argument("--layer", help="nome da camada; obrigatório se o GeoPackage tiver mais de uma")
	parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help=f"pasta com a instância-base (padrão: {DEFAULT_BASELINE})")
	parser.add_argument("--no-baseline", action="store_true", help="só conta feições, sem comparar com a instância-base")
	return parser


def main(argv: list[str] | None = None) -> int:
	parser = make_parser()
	args = parser.parse_args(argv)
	input_path = args.input.expanduser().resolve()
	if not input_path.is_file():
		parser.error(f"GeoPackage não encontrado: {input_path}")
	try:
		layer, current = read_layer(input_path, args.layer)
	except (OSError, sqlite3.Error, ValueError, json.JSONDecodeError) as error:
		parser.error(str(error))

	counts = Counter(record["status"] for record in current.values())
	print(f"Arquivo: {input_path}")
	print(f"Camada: {layer['table']} · {layer['declared_type']} · EPSG:{layer['srs_id']}")
	print(f"Registros na tabela: {len(current)}")
	print(f"Polígonos com geometria: {counts['polygon']}")
	print(f"Sem geometria: {counts['null'] + counts['empty']}")
	if counts["invalid"]:
		invalid_records = [record for record in current.values() if record["status"] == "invalid"]
		print(f"Geometrias que não consegui ler: {counts['invalid']} ({format_features(invalid_records)})")
	if counts["null"] + counts["empty"]:
		empty_records = [record for record in current.values() if record["status"] in {"null", "empty"}]
		print(f"Linhas sem desenho: {format_features(empty_records)}")
	duplicate_ids = {
		feature_id: fids
		for feature_id, fids in _group_ids(current).items()
		if feature_id is not None and len(fids) > 1
	}
	if duplicate_ids:
		print("IDs repetidos: " + "; ".join(f"{feature_id} (fids {', '.join(map(str, fids))})" for feature_id, fids in duplicate_ids.items()))

	if args.no_baseline:
		return 0
	baseline_path = args.baseline.expanduser().resolve()
	try:
		baseline_manifest, baseline = read_baseline(baseline_path)
	except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
		parser.error(str(error))
	if layer["srs_id"] != 4326:
		parser.error(f"comparação geométrica requer EPSG:4326; esta camada usa EPSG:{layer['srs_id']}")
	baseline_ids = set(baseline)
	current_ids = set(current)
	new_rows = sorted(current_ids - baseline_ids)
	deleted_rows = sorted(baseline_ids - current_ids)
	emptied: list[dict[str, Any]] = []
	changed: list[dict[str, Any]] = []
	changed_id: list[dict[str, Any]] = []
	added_shapes: list[dict[str, Any]] = []
	for fid in sorted(current_ids & baseline_ids):
		new_feature = current[fid]
		old_feature = baseline[fid]
		if new_feature["status"] in {"null", "empty"}:
			emptied.append(new_feature)
			continue
		if new_feature["status"] != "polygon":
			continue
		if new_feature.get("id") != old_feature.get("id"):
			changed_id.append({**new_feature, "old_id": old_feature.get("id")})
		new_rings = new_feature["rings"]
		if len(new_rings) != 1:
			changed.append({**new_feature, "reason": f"{len(new_rings)} anéis; só comparo um anel exterior"})
			continue
		new_points = [project_wgs84(point) for point in new_rings[0]]
		if canonical_ring(new_points) != canonical_ring(old_feature["points"]):
			changed.append({**new_feature, "old_vertices": len(old_feature["points"]), "new_vertices": len(new_points)})
	for fid in new_rows:
		feature = current[fid]
		if feature["status"] == "polygon":
			added_shapes.append(feature)
	new_empty_rows = [current[fid] for fid in new_rows if current[fid]["status"] in {"null", "empty"}]
	print(f"\nComparação com a base ({baseline_manifest.get('feature_count', len(baseline))} polígonos, {baseline_path}):")
	print(f"Linhas completamente removidas: {len(deleted_rows)}" + (f" (fids {', '.join(map(str, deleted_rows))})" if deleted_rows else ""))
	print(f"Geometrias de prédios antigos apagadas: {len(emptied)} ({format_features(emptied)})")
	print(f"Polígonos novos: {len(added_shapes)} ({format_features(added_shapes)})")
	if new_empty_rows:
		print(f"Linhas novas sem polígono: {len(new_empty_rows)} ({format_features(new_empty_rows)})")
	print(f"Geometrias alteradas em prédios antigos: {len(changed)}")
	for feature in changed:
		if feature.get("old_vertices") is not None:
			print(f"  fid {feature['fid']} ({feature.get('id')}): {feature['old_vertices']} → {feature['new_vertices']} vértices")
		else:
			print(f"  fid {feature['fid']} ({feature.get('id')}): {feature.get('reason')}")
	if changed_id:
		print("IDs alterados: " + "; ".join(f"fid {f['fid']}: {f['old_id']} → {f['id']}" for f in changed_id))
	print("Nota: a comparação de forma ignora a ordem inicial e o sentido dos vértices; arredonda diferenças menores que 0,1 mm.")
	return 0


def _group_ids(features: dict[int, dict[str, Any]]) -> dict[str | None, list[int]]:
	grouped: dict[str | None, list[int]] = defaultdict(list)
	for fid, feature in features.items():
		if feature["status"] == "polygon":
			grouped[feature.get("id")].append(fid)
	return grouped


if __name__ == "__main__":
	raise SystemExit(main())
