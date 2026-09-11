#!/usr/bin/env python3
"""Compare the SOCP and specialized TPP backends on identical ordered relaxations."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks/scripts"))
from benchmark_cases import read_encoded_cases
from unordered_validation import validate_path


def parser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(description=__doc__)
	result.add_argument("--suite", type=Path, required=True)
	result.add_argument("--tspn-repo", type=Path, required=True)
	result.add_argument("--output", type=Path, required=True)
	result.add_argument("--backend", choices=("socp", "tpp"), action="append")
	result.add_argument("--repeat", type=int, default=3)
	result.add_argument("--limit", type=int)
	result.add_argument("--stride", type=int, default=1)
	result.add_argument("--case-index", type=int, action="append")
	result.add_argument("--validation-tolerance", type=float, default=1e-7)
	result.add_argument("--oracle-tolerance", type=float, default=1e-7)
	result.add_argument("--objective-tolerance", type=float, default=1e-7)
	result.add_argument("--strict", action="store_true",
		help="Fail if either backend emits an invalid path or objectives disagree.")
	return result


def load_core(repository: Path):
	bindings = next((repository / "python/tspn_bnb2/core").glob("_tspn_bindings*.so"), None)
	if bindings is None:
		package = importlib.util.find_spec("tspn_bnb2")
		if package is None:
			raise RuntimeError("tspn_bnb2 is not installed")
		bindings = next((Path(package.origin).parent / "core").glob("_tspn_bindings*.so"))
	spec = importlib.util.spec_from_file_location("_tspn_bindings", bindings)
	core = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(core)
	return core, bindings


def convex_hulls(polygons):
	from shapely.geometry import MultiPoint

	result = []
	for polygon in polygons:
		hull = MultiPoint(polygon).convex_hull
		if hull.geom_type == "Point":
			result.append([tuple(hull.coords[0])])
		elif hull.geom_type == "LineString":
			result.append([tuple(point) for point in hull.coords])
		else:
			result.append([tuple(point) for point in list(hull.exterior.coords)[:-1]])
	return result


def main(argv: list[str] | None = None) -> int:
	args = parser().parse_args(argv)
	if args.repeat < 1 or args.stride < 1:
		raise SystemExit("--repeat and --stride must be positive")
	backends = args.backend or ["socp", "tpp"]
	core, bindings = load_core(args.tspn_repo.resolve())
	cases = read_encoded_cases(args.suite.resolve())
	if args.case_index:
		selected = set(args.case_index)
		cases = [case for index, case in enumerate(cases) if index in selected]
	cases = cases[:: args.stride]
	if args.limit is not None:
		cases = cases[: args.limit]
	args.output.parent.mkdir(parents=True, exist_ok=True)
	rows = []
	with args.output.open("w") as output:
		for position, case in enumerate(cases):
			sx, sy, tx, ty = struct.unpack_from("<dddd", case.data)
			start, target = (sx, sy), (tx, ty)
			sites = [core.Point(*start), core.Point(*target)]
			sites.extend(core.Polygon([[core.Point(x, y) for x, y in polygon]]) for polygon in case.polygons)
			instance = core.Instance(sites, True)
			sequence = [core.TourElement(instance, 0)]
			sequence.extend(core.TourElement(instance, index + 2) for index in range(len(case.polygons)))
			sequence.append(core.TourElement(instance, 1))
			hulls = convex_hulls(case.polygons)
			for repeat in range(args.repeat):
				offset = (position + repeat) % len(backends)
				for backend in backends[offset:] + backends[:offset]:
					began = time.perf_counter()
					row = {"sha256": case.digest, "case": position, "polygons": len(case.polygons),
						"backend": backend, "repeat": repeat}
					try:
						solver = core.SocSolver(False, backend, args.oracle_tolerance)
						lower, upper, trajectory = solver.compute_trajectory_with_information(sequence, True)
						points = [[point.x, point.y] for point in trajectory]
						row.update({"seconds": time.perf_counter() - began, "lower_bound": lower,
							"upper_bound": upper, "path": points,
							"validation": validate_path(start, target, hulls, points, args.validation_tolerance)})
						row["valid"] = row["validation"]["valid"]
					except Exception as error:
						row.update({"seconds": time.perf_counter() - began, "error": repr(error), "valid": False})
					rows.append(row)
					output.write(json.dumps(row) + "\n")
					output.flush()
					print(json.dumps({key: row.get(key) for key in ("case", "backend", "repeat", "seconds", "valid", "error")}), flush=True)
	metadata = {"suite": str(args.suite.resolve()), "bindings": str(bindings),
		"version": importlib.metadata.version("tspn_bnb2"), "repeat": args.repeat,
		"backends": backends, "validation_tolerance": args.validation_tolerance,
		"oracle_tolerance": args.oracle_tolerance,
		"objective_tolerance": args.objective_tolerance}
	args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
	for backend in backends:
		group = [row for row in rows if row["backend"] == backend and "error" not in row]
		print(json.dumps({"backend": backend, "runs": len(group),
			"valid": sum(row["valid"] for row in group),
			"total_seconds": sum(row["seconds"] for row in group),
			"median_seconds": statistics.median(row["seconds"] for row in group) if group else None}))
	paired = {}
	for row in rows:
		paired.setdefault((row["sha256"], row["repeat"]), {})[row["backend"]] = row
	disagreements = []
	for key, pair in paired.items():
		if set(backends) <= pair.keys() and all("upper_bound" in pair[backend] for backend in backends):
			values = [pair[backend]["upper_bound"] for backend in backends]
			if max(values) - min(values) > args.objective_tolerance * (1 + max(map(abs, values))):
				disagreements.append({"pair": key, "objectives": values})
	print(json.dumps({"objective_disagreements": len(disagreements),
		"invalid_runs": sum(not row.get("valid") for row in rows),
		"errors": sum(bool(row.get("error")) for row in rows)}))
	for disagreement in disagreements[:10]:
		print(json.dumps({"objective_disagreement": disagreement}), file=sys.stderr)
	failed = any(row.get("error") for row in rows)
	if args.strict:
		failed |= bool(disagreements) or any(not row.get("valid") for row in rows)
	return int(failed)


if __name__ == "__main__":
	status = main()
	sys.stdout.flush()
	sys.stderr.flush()
	os._exit(status)
