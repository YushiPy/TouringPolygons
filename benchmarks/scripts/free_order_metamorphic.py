"""Metamorphic correctness checks for the free-order endpoint TPP solver."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

from benchmark_cases import read_encoded_cases
from unordered_runner import run_unordered_solver
from unordered_validation import validate_path


Point = tuple[float, float]
Polygon = list[Point]


def transform(point: Point, matrix: tuple[float, float, float, float], offset: Point) -> Point:
	x, y = point
	a, b, c, d = matrix
	return a * x + b * y + offset[0], c * x + d * y + offset[1]


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--suite", type=Path, required=True)
	parser.add_argument("--solver", type=Path, required=True)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--seconds", type=float, default=2)
	parser.add_argument("--max-calls", type=int, default=10_000_000)
	parser.add_argument("--limit", type=int, default=20)
	parser.add_argument("--case-index", type=int, action="append")
	parser.add_argument("--tolerance", type=float, default=1e-7)
	return parser


def main(argv: list[str] | None = None) -> int:
	args = make_parser().parse_args(argv)
	angle = 0.731
	transforms = [
		("identity", (1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0, False),
		("translate", (1.0, 0.0, 0.0, 1.0), (1729.25, -913.5), 1.0, False),
		("rotate", (math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle)), (0.0, 0.0), 1.0, False),
		("reflect", (-1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0, True),
		("scale-small", (1e-3, 0.0, 0.0, 1e-3), (0.0, 0.0), 1e-3, False),
		("scale-large", (1e3, 0.0, 0.0, 1e3), (0.0, 0.0), 1e3, False),
	]
	cases = read_encoded_cases(args.suite)
	selected = set(args.case_index or range(len(cases)))
	cases = [(index, case) for index, case in enumerate(cases) if index in selected][: args.limit]
	args.output.parent.mkdir(parents=True, exist_ok=True)
	failed = False
	with args.output.open("w") as output:
		for index, case in cases:
			sx, sy, tx, ty = struct.unpack_from("<dddd", case.data)
			start = (sx, sy)
			target = (tx, ty)
			baseline = None
			for name, matrix, offset, factor, reverse in transforms:
				s = transform(start, matrix, offset)
				t = transform(target, matrix, offset)
				polygons = [[transform(tuple(point), matrix, offset) for point in polygon] for polygon in case.polygons]
				if reverse:
					polygons = [list(reversed(polygon)) for polygon in reversed(polygons)]
				row = {"case": index, "sha256": case.digest, "transform": name, "factor": factor}
				try:
					result = run_unordered_solver(args.solver, s, t, polygons, args.max_calls, args.seconds)
					validation = validate_path(s, t, polygons, result["path"], args.tolerance * max(1.0, factor))
					row.update(result)
					row["validation"] = validation
					row["valid"] = validation["valid"]
					if name == "identity":
						baseline = result
					elif baseline and baseline["exact"] and result["exact"]:
						expected = factor * baseline["upper_bound"]
						row["objective_error"] = abs(result["upper_bound"] - expected)
						row["objective_consistent"] = row["objective_error"] <= args.tolerance * max(1.0, abs(expected))
					expected_order = list(range(len(polygons)))
					row["order_is_permutation"] = sorted(result["order"]) == expected_order
				except Exception as error:
					row["error"] = repr(error)
				failed |= bool(row.get("error") or not row.get("valid") or
					not row.get("order_is_permutation") or row.get("objective_consistent") is False)
				output.write(json.dumps(row) + "\n")
				output.flush()
				print(json.dumps({key: row.get(key) for key in
					("case", "transform", "exact", "valid", "objective_consistent", "error")}), flush=True)
	return int(failed)


if __name__ == "__main__":
	raise SystemExit(main())
