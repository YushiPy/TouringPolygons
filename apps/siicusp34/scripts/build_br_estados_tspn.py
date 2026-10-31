#!/usr/bin/env python3
"""Solve the Brazil highlight of the English page as a TSPN: a closed tour with no fixed point.

The Portuguese page's ``data/br-estados-demo.js`` fixes Brasília as start and end
(a free-order TPP with s = t). This script takes the same 27 simplified regions,
names and convex decompositions from that file, solves them with ``--cycle`` and
writes ``en/data/br-estados-tspn.js`` in the same format, so the English page and
the international poster show the problem they describe. Time is the median of
three runs. The tour has no start; it is written beginning where it first
touches the Federal District, so the visiting order counts from there. From
the repository root:

	python3 apps/siicusp34/scripts/build_br_estados_tspn.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/_internal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_usp_demo import first_contact  # noqa: E402
from unordered_runner import encode_instance, run_unordered_solver  # noqa: E402

ABSOLUTE_GAP = 1e-7
RELATIVE_GAP = 1e-9
REPETITIONS = 3


def read_script_json(path: Path) -> dict:
	text = path.read_text(encoding="utf-8")
	return json.loads(text.split("=", 1)[1].strip().rstrip(";"))


def rotate_closed_path(path: list[list[float]], point: list[float]) -> list[list[float]]:
	"""The same closed tour, starting and ending at ``point``, which lies on one of its segments."""
	cycle = path[:-1]
	def distance(index: int) -> float:
		a, b = cycle[index], cycle[(index + 1) % len(cycle)]
		ab = (b[0] - a[0], b[1] - a[1])
		length2 = ab[0] ** 2 + ab[1] ** 2
		t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((point[0] - a[0]) * ab[0] + (point[1] - a[1]) * ab[1]) / length2))
		return math.dist(point, (a[0] + t * ab[0], a[1] + t * ab[1]))
	segment = min(range(len(cycle)), key=distance)
	if distance(segment) > 1e-6:
		raise ValueError("The rotation point is not on the tour")
	start = [float(point[0]), float(point[1])]
	rotated = [start, *cycle[segment + 1:], *cycle[:segment + 1], start]
	return [point for index, point in enumerate(rotated) if index == 0 or math.dist(point, rotated[index - 1]) > 0]


def build(args: argparse.Namespace) -> dict[str, object]:
	source = read_script_json(args.source)
	polygons = [[tuple(point) for point in polygon] for polygon in source["geometry"]["polygons"]]
	# --cycle ignores the endpoints; any point will do.
	anchor = tuple(source["geometry"]["start"])
	runs = [run_unordered_solver(args.solver, anchor, anchor, polygons, -1, args.max_seconds, arguments=["--cycle"]) for _ in range(REPETITIONS)]
	for result in runs:
		if result.get("termination") != "optimal" or not result.get("exact"):
			raise ValueError(f"Solver did not certify an optimum: {result.get('termination')}")
		lower, upper = float(result["lower_bound"]), float(result["upper_bound"])
		if upper - lower > ABSOLUTE_GAP + RELATIVE_GAP * abs(upper):
			raise ValueError("The numerical certificate exceeds the configured tolerance")
	result = runs[0]
	if any(run["path"] != result["path"] for run in runs):
		raise ValueError("The repetitions returned different tours")
	path = [[float(x), float(y)] for x, y in result["path"]]
	if len(path) < 3 or path[0] != path[-1]:
		raise ValueError("The solver tour is not closed")
	lower, upper = float(result["lower_bound"]), float(result["upper_bound"])
	length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
	if abs(length - upper) > ABSOLUTE_GAP + RELATIVE_GAP * abs(upper):
		raise ValueError("The tour length differs from its certified upper bound")
	contacts = [first_contact(path, polygon, 1e-7) for polygon in polygons]
	if any(contact is None for contact in contacts):
		raise ValueError("The tour does not intersect every federative unit")
	# A TSPN tour has no start. Begin it where it first touches the Federal District, so the
	# visiting order (and the page's colours and animation) counts from Brasília's unit.
	path = rotate_closed_path(path, contacts[source["region_codes"].index("DF")]["point"])
	if abs(sum(math.dist(a, b) for a, b in zip(path, path[1:])) - length) > ABSOLUTE_GAP + RELATIVE_GAP * abs(upper):
		raise ValueError("Rotating the tour changed its length")
	contacts = [first_contact(path, polygon, 1e-7) for polygon in polygons]
	if any(contact is None for contact in contacts):
		raise ValueError("The rotated tour does not intersect every federative unit")
	solver_rank = {int(index): rank for rank, index in enumerate(result["order"])}
	order = sorted(range(len(polygons)), key=lambda index: (contacts[index]["fraction"], solver_rank.get(index, len(polygons) + index)))
	encoded_input = encode_instance(anchor, anchor, polygons, -1, args.max_seconds)
	return {
		"schema_version": 1,
		"case": "br-estados",
		"title": "Brazil · 27 federative units",
		"corpus": "demonstration",
		"cycle": True,
		"polygons": len(polygons),
		"region_codes": source["region_codes"],
		"region_names": source["region_names"],
		"geometry": {"start": path[0], "target": path[0], "polygons": polygons},
		"path": path,
		"order": order,
		"length": upper,
		"lower_bound": lower,
		"upper_bound": upper,
		"exact": True,
		"termination": "optimal",
		"seconds": statistics.median(float(run["seconds"]) for run in runs),
		"calls": int(result["calls"]),
		"visualization": {"contacts": contacts, "decomposition": source["visualization"]["decomposition"]},
		"provenance": {
			"source": str(args.source.relative_to(ROOT)),
			"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
			"solver_sha256": hashlib.sha256(args.solver.read_bytes()).hexdigest(),
			"solver_arguments": ["--cycle"],
			"input_sha256": hashlib.sha256(encoded_input.encode()).hexdigest(),
			"repetitions": REPETITIONS,
			"absolute_gap": ABSOLUTE_GAP,
			"relative_gap": RELATIVE_GAP,
			"depot_tour_length": source["length"],
		},
	}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--source", type=Path, default=ROOT / "apps/siicusp34/data/br-estados-demo.js")
	parser.add_argument("--solver", type=Path, default=ROOT / ".build/tools/bin/tpp-unordered")
	parser.add_argument("--max-seconds", type=float, default=600)
	parser.add_argument("--output", type=Path, default=ROOT / "apps/siicusp34/en/data/br-estados-tspn.js")
	args = parser.parse_args()
	if not args.solver.is_file():
		raise FileNotFoundError(f"C++ solver not found: {args.solver}")
	data = build(args)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text("window.TPPBrEstadosDemo = " + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";\n", encoding="utf-8")
	print(json.dumps({"output": str(args.output), "length": data["length"], "lower_bound": data["lower_bound"], "seconds": data["seconds"], "depot_tour_length": data["provenance"]["depot_tour_length"]}, indent=2))


if __name__ == "__main__":
	main()
