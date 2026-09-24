#!/usr/bin/env python3
"""Export the São Paulo subprefecture suite as a static SIICUSP case.

Run from the repository root with the C++ unordered solver built. The app gets
the route, visit contacts, and Greene convex decomposition as static JSON; no
solver or GIS dependency is needed in the browser.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
APP = REPOSITORY / "apps/siicusp34"
SUITE = REPOSITORY / "benchmarks/suites/sp-bairros"
SOURCE = SUITE / "sp-bairros.bin"
MAPPING = SUITE / "polygons.csv"
SOURCE_GEOPACKAGE = SUITE / "geosampa/subprefeituras.gpkg"
OUTPUT = APP / "data/sp-bairros-demo.js"
MAX_CALLS = 1_000_000
MAX_SECONDS = 300
ABSOLUTE_GAP = 1e-7
RELATIVE_GAP = 1e-9

sys.path.insert(0, str(REPOSITORY / "benchmarks/_internal"))
sys.path.insert(0, str(APP / "scripts"))

from normalize_polygon_orientation import read_binary_cases  # noqa: E402
from unordered_runner import encode_instance, run_unordered_solver  # noqa: E402
from build_usp_demo import first_contact, optimal_decomposition  # noqa: E402


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--solver", type=Path, default=REPOSITORY / ".build/unordered/tpp")
	parser.add_argument("--output", type=Path, default=OUTPUT)
	parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
	parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
	return parser.parse_args()


def sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, object]:
	if not args.solver.is_file():
		raise FileNotFoundError(f"C++ solver not found: {args.solver}")
	cases = read_binary_cases(SOURCE)
	if len(cases) != 1 or len(cases[0].polygons) != 32:
		raise ValueError("Expected the 32-subprefecture binary suite case")
	case = cases[0]
	with MAPPING.open(encoding="utf-8", newline="") as file:
		mapping = list(csv.DictReader(file))
	if len(mapping) != len(case.polygons):
		raise ValueError("The polygon mapping does not match the suite binary")
	for index, (row, polygon) in enumerate(zip(mapping, case.polygons)):
		if int(row["polygon_index"]) != index or int(row["vertices_binario"]) != len(polygon):
			raise ValueError(f"The mapping differs from binary polygon {index}")

	polygons = [[[float(x), float(y)] for x, y in polygon] for polygon in case.polygons]
	result = run_unordered_solver(
		args.solver, case.start, case.target, polygons,
		args.max_calls, args.max_seconds,
	)
	if result.get("termination") != "optimal" or not result.get("exact"):
		raise ValueError(f"Solver did not certify an optimum: {result.get('termination')}")
	lower, upper = float(result["lower_bound"]), float(result["upper_bound"])
	if upper - lower > ABSOLUTE_GAP + RELATIVE_GAP * abs(upper):
		raise ValueError("The numerical certificate exceeds the configured tolerance")
	path = [[float(x), float(y)] for x, y in result["path"]]
	if len(path) < 2 or math.dist(path[0], case.start) > 1e-7 or math.dist(path[-1], case.target) > 1e-7:
		raise ValueError("The solver route endpoints differ from the suite instance")
	length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
	if abs(length - upper) > ABSOLUTE_GAP + RELATIVE_GAP * abs(upper):
		raise ValueError("The solver route length differs from its upper bound")
	contacts = [first_contact(path, polygon, 1e-7) for polygon in polygons]
	if any(contact is None for contact in contacts):
		raise ValueError("The route does not intersect every subprefecture")
	# Order the labels by the first point where the route reaches each region.
	# A solver order breaks ties where neighboring closed polygons share a boundary.
	solver_rank = {int(index): rank for rank, index in enumerate(result["order"])}
	order = sorted(
		range(len(polygons)),
		key=lambda index: (contacts[index]["fraction"], solver_rank.get(index, len(polygons) + index)),
	)
	decomposition = optimal_decomposition(polygons)
	if len(decomposition) != len(polygons) or any(not pieces for pieces in decomposition):
		raise ValueError("The convex decomposition does not cover all subprefectures")

	encoded_input = encode_instance(case.start, case.target, polygons, args.max_calls, args.max_seconds)
	return {
		"schema_version": 1,
		"case": "sp-bairros",
		"title": "São Paulo · 32 subprefeituras",
		"corpus": "demonstration",
		"polygons": len(polygons),
		"region_codes": [row["cd_subprefeitura"] for row in mapping],
		"region_names": [row["nome_oficial"] for row in mapping],
		"geometry": {"start": case.start, "target": case.target, "polygons": polygons},
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
		"depot": {"label": "Sé"},
		"provenance": {
			"suite_binary_sha256": sha256(SOURCE),
			"mapping_sha256": sha256(MAPPING),
			"geopackage_sha256": sha256(SOURCE_GEOPACKAGE),
			"partition_source_sha256": sha256(REPOSITORY / "packages/optimal-convex-partition/cpp/src/optimal_convex_partition.cpp"),
			"solver_sha256": sha256(args.solver),
			"input_sha256": hashlib.sha256(encoded_input.encode()).hexdigest(),
			"projection": "SIRGAS 2000 / UTM zone 23S (EPSG:31983), metres, translated to local origin",
			"simplification_tolerance_metres": 20,
			"absolute_gap": ABSOLUTE_GAP,
			"relative_gap": RELATIVE_GAP,
		},
	}


def main() -> None:
	args = parse_args()
	data = build(args)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(
		"window.TPPSpBairrosDemo = "
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
