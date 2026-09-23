"""Run the free-order solver directly on the nonempty polygons in a QGIS GeoPackage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_footprints import project_wgs84, read_layer
from unordered_runner import run_unordered_solver


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "benchmarks/suites/usp-butanta-50/qgis/predios.gpkg"
DEFAULT_SOLVER = REPO_ROOT / ".build/unordered/tpp"
SOURCE_DATA = REPO_ROOT / "apps/siicusp34/data/usp-footprints.json"
DEFAULT_MAX_CALLS = 1_000_000


def positive_seconds(value: str) -> float:
	try:
		seconds = float(value)
	except ValueError as error:
		raise argparse.ArgumentTypeError("informe o limite em segundos") from error
	if not math.isfinite(seconds) or seconds <= 0:
		raise argparse.ArgumentTypeError("o limite de segundos deve ser positivo e finito")
	return seconds


def positive_calls(value: str) -> int:
	try:
		calls = int(value)
	except ValueError as error:
		raise argparse.ArgumentTypeError("informe um número inteiro de chamadas") from error
	if calls <= 0:
		raise argparse.ArgumentTypeError("o limite de chamadas deve ser positivo")
	return calls


def centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
	double_area = 0.0
	cx = 0.0
	cy = 0.0
	for index, (x, y) in enumerate(points):
		nx, ny = points[(index + 1) % len(points)]
		cross = x * ny - nx * y
		double_area += cross
		cx += (x + nx) * cross
		cy += (y + ny) * cross
	if abs(double_area) < 1e-10:
		raise ValueError("o polígono do IME Bloco B tem área zero")
	return cx / (3 * double_area), cy / (3 * double_area)


def on_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float], tolerance: float = 1e-6) -> bool:
	cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
	return abs(cross) <= tolerance * max(1.0, math.dist(a, b)) and all(
		min(a[axis], b[axis]) - tolerance <= point[axis] <= max(a[axis], b[axis]) + tolerance
		for axis in (0, 1)
	)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
	inside = False
	for index, a in enumerate(polygon):
		b = polygon[(index + 1) % len(polygon)]
		if on_segment(point, a, b):
			return True
		if (a[1] > point[1]) != (b[1] > point[1]):
			x_at_y = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
			if point[0] < x_at_y:
				inside = not inside
	return inside


def distance_to_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> float:
	distances = []
	for index, a in enumerate(polygon):
		b = polygon[(index + 1) % len(polygon)]
		dx, dy = b[0] - a[0], b[1] - a[1]
		length_squared = dx * dx + dy * dy
		fraction = 0.0 if length_squared == 0 else max(
			0.0,
			min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_squared),
		)
		closest = (a[0] + fraction * dx, a[1] + fraction * dy)
		distances.append(math.dist(point, closest))
	return min(distances)


def endpoint_for(polygons: list[dict[str, Any]]) -> tuple[float, float]:
	try:
		metadata = json.loads(SOURCE_DATA.read_text())
		depot = metadata["depot"]
		entrance_lon_lat = depot["entrance_lon_lat"]
		offset = float(depot["outside_offset_metres"])
	except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
		raise ValueError(f"não consegui carregar o ponto de partida da demonstração USP: {error}") from error
	ime = next((feature for feature in polygons if feature.get("id") == "ime-b"), None)
	if ime is None:
		raise ValueError("a camada precisa manter o polígono `ime-b` para usar a entrada do IME como início e fim")
	points = ime["points"]
	entrance = project_wgs84(tuple(entrance_lon_lat))
	if distance_to_polygon(entrance, points) > 10.0:
		raise ValueError("a entrada padrão do IME está a mais de 10 m do contorno `ime-b`")
	center = centroid(points)
	outward = (entrance[0] - center[0], entrance[1] - center[1])
	length = math.hypot(*outward)
	if length < 1e-9:
		raise ValueError("o ponto de entrada coincide com o centro do IME Bloco B")
	start = (entrance[0] + offset * outward[0] / length, entrance[1] + offset * outward[1] / length)
	if point_in_polygon(start, points):
		raise ValueError("o ponto de início calculado ainda está dentro do IME Bloco B")
	return start


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Resolve todos os polígonos desenhados na camada do GeoPackage, ignorando registros sem geometria."
	)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"GeoPackage do QGIS (padrão: {DEFAULT_INPUT})")
	parser.add_argument("--layer", help="nome da camada se o GeoPackage tiver mais de uma")
	parser.add_argument("--solver", type=Path, default=DEFAULT_SOLVER, help="executável C++ de ordem livre")
	parser.add_argument("--seconds", type=positive_seconds, default=30.0, help="limite de tempo por busca, em segundos (padrão: 30)")
	parser.add_argument("--max-calls", type=positive_calls, default=DEFAULT_MAX_CALLS, help="limite de chamadas ao oráculo (padrão: 1000000)")
	parser.add_argument("--output", type=Path, help="arquivo JSON para salvar o resultado")
	return parser


def main(argv: list[str] | None = None) -> int:
	parser = make_parser()
	args = parser.parse_args(argv)
	input_path = args.input.expanduser().resolve()
	solver_path = args.solver.expanduser().resolve()
	if not input_path.is_file():
		parser.error(f"GeoPackage não encontrado: {input_path}")
	if not solver_path.is_file():
		parser.error(f"solver não encontrado: {solver_path}; compile-o antes de rodar")
	try:
		layer, records = read_layer(input_path, args.layer)
	except (OSError, ValueError) as error:
		parser.error(str(error))
	if layer["srs_id"] != 4326:
		parser.error(f"esta conversão requer EPSG:4326; a camada usa EPSG:{layer['srs_id']}")
	problems = [record for record in records.values() if record["status"] == "invalid"]
	if problems:
		parser.error("geometrias inválidas: " + ", ".join(f"fid {r['fid']} ({r.get('id')}): {r['error']}" for r in problems))
	features = []
	for record in records.values():
		if record["status"] in {"null", "empty"}:
			continue
		rings = record["rings"]
		if len(rings) != 1:
			parser.error(f"fid {record['fid']} ({record.get('id')}) tem buracos; o solver não representa polígonos com buracos")
		points = [project_wgs84(point) for point in rings[0]]
		if len(points) < 3:
			parser.error(f"fid {record['fid']} ({record.get('id')}) tem menos de 3 vértices")
		features.append({"fid": record["fid"], "id": record.get("id"), "points": points})
	if not features:
		parser.error("não há polígonos desenhados nessa camada")
	try:
		start = endpoint_for(features)
	except ValueError as error:
		parser.error(str(error))

	print(f"Polígonos: {len(features)} (linhas vazias ignoradas)", flush=True)
	print(f"Início = fim: {start[0]:.3f}, {start[1]:.3f} m (junto à entrada do IME)", flush=True)
	print(f"Limites: {args.seconds:g} segundos ou {args.max_calls} chamadas", flush=True)
	started = time.perf_counter()
	try:
		result = run_unordered_solver(
			solver_path,
			start,
			start,
			[feature["points"] for feature in features],
			args.max_calls,
			args.seconds,
		)
	except (RuntimeError, json.JSONDecodeError, OSError, subprocess.TimeoutExpired) as error:
		print(f"Falha no solver: {error}")
		return 1
	wall_seconds = time.perf_counter() - started
	print(
		f"Resultado: {result.get('termination')} · certificado={result.get('exact')} · "
		f"{result.get('seconds', wall_seconds):.3f} s · chamadas={result.get('calls')}",
		flush=True,
	)
	if result.get("lower_bound") is not None and result.get("upper_bound") is not None:
		print(f"Limites: LB={result['lower_bound']:.6f} m; UB={result['upper_bound']:.6f} m", flush=True)
	if args.output:
		output_path = args.output.expanduser().resolve()
	else:
		timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
		output_path = REPO_ROOT / "benchmarks/results" / f"usp-footprints-{timestamp}.json"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output = {
		"input": str(input_path),
		"input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
		"layer": layer["table"],
		"crs": f"EPSG:{layer['srs_id']}",
		"solver": str(solver_path),
		"solver_sha256": hashlib.sha256(solver_path.read_bytes()).hexdigest(),
		"seconds_limit": args.seconds,
		"max_calls": args.max_calls,
		"polygon_count": len(features),
		"polygons": [{"index": index, "fid": feature["fid"], "id": feature["id"]} for index, feature in enumerate(features)],
		"endpoint": {"start": start, "target": start, "source": "IME Bloco B entrance; 3 m outward offset"},
		"projection": "WGS84 local tangent plane; origin 23.557 S, 46.732 W; metres",
		"wall_seconds": wall_seconds,
		"result": result,
	}
	output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
	print(f"Resultado salvo: {output_path}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
