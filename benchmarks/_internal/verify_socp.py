"""Independent exhaustive-order/piece SOCP verification of small endpoint TPP.

This numerical reference requires gurobipy and Shapely >= 2.1 in its own Python
runtime. It does not use the C++ convex solver or its polygon partition.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import native_build
import workspace

ROOT = Path(__file__).resolve().parents[2]


def independent_pieces(region):
	from shapely import constrained_delaunay_triangles
	from shapely.geometry import Polygon
	from shapely.ops import unary_union

	if len(region) <= 2:
		return [region]
	polygon = Polygon(region)
	if not polygon.is_valid or polygon.area == 0:
		raise ValueError("Reference expects a valid simple polygon")
	if polygon.equals(polygon.convex_hull):
		return [region]
	triangles = list(constrained_delaunay_triangles(polygon).geoms)
	if not unary_union(triangles).equals(polygon):
		raise ValueError("Independent triangulation does not cover the original region")
	return [list(t.exterior.coords)[:-1] for t in triangles]


def baseline(start, target, pieces, env, seconds=30):
	"""Enumerate all orders and alternative pieces; each leaf is a convex SOCP.

	p_i = sum_v lambda_iv v, sum lambda_iv = 1, lambda >= 0;
	||p_(i+1)-p_i||_2 <= d_i; minimize sum d_i. Singletons and
	segments use exactly the same convex-combination constraint as polygons.
	"""
	import gurobipy as gp

	count = math.factorial(len(pieces)) * math.prod(len(p) for p in pieces)
	if count > 20000:
		raise ValueError(
			f"{count} SOCPs exceed the small-instance guard; use smaller subsets"
		)
	lower = upper = math.inf
	# Independent normalization keeps the numerical reference well conditioned.
	vertices = [start, target, *(q for region in pieces for p in region for q in p)]
	xmin, xmax = min(q[0] for q in vertices), max(q[0] for q in vertices)
	ymin, ymax = min(q[1] for q in vertices), max(q[1] for q in vertices)
	center = (xmin / 2 + xmax / 2, ymin / 2 + ymax / 2)
	scale = max(xmax - xmin, ymax - ymin, 1.0)

	def normalized(q):
		return ((q[0] - center[0]) / scale, (q[1] - center[1]) / scale)

	s, t = normalized(start), normalized(target)
	best_path = None
	for order in itertools.permutations(range(len(pieces))):
		for selection in itertools.product(*(pieces[i] for i in order)):
			with gp.Model(env=env) as model:
				model.Params.Threads = 1
				model.Params.NonConvex = 0
				model.Params.TimeLimit = seconds
				model.Params.BarQCPConvTol = 1e-9
				model.Params.FeasibilityTol = 1e-9
				xy = [s]
				for piece in selection:
					weights = model.addVars(len(piece), lb=0)
					model.addConstr(weights.sum() == 1)
					p = [normalized(q) for q in piece]
					x = model.addVar(lb=min(q[0] for q in p), ub=max(q[0] for q in p))
					y = model.addVar(lb=min(q[1] for q in p), ub=max(q[1] for q in p))
					model.addConstr(
						x == gp.quicksum(weights[v] * p[v][0] for v in range(len(p)))
					)
					model.addConstr(
						y == gp.quicksum(weights[v] * p[v][1] for v in range(len(p)))
					)
					xy.append((x, y))
				xy.append(t)
				costs = []
				for a, b in itertools.pairwise(xy):
					dx = model.addVar(lb=-gp.GRB.INFINITY)
					dy = model.addVar(lb=-gp.GRB.INFINITY)
					d = model.addVar(lb=0)
					model.addConstr(dx == b[0] - a[0])
					model.addConstr(dy == b[1] - a[1])
					model.addQConstr(dx * dx + dy * dy <= d * d)
					costs.append(d)
				model.setObjective(gp.quicksum(costs))
				model.optimize()
				if model.Status != gp.GRB.OPTIMAL:
					model.Params.NumericFocus = 3
					model.Params.BarQCPConvTol = 1e-7
					model.reset()
					model.optimize()
				if model.Status != gp.GRB.OPTIMAL:
					raise RuntimeError(
						f"SOCP leaf status {model.Status}; verification incomplete"
					)
				lower = min(lower, model.ObjBound * scale)
				if model.ObjVal * scale < upper:
					upper = model.ObjVal * scale

					def value(v):
						return v.X if isinstance(v, gp.Var) else v

					best_path = [
						[
							value(q[0]) * scale + center[0],
							value(q[1]) * scale + center[1],
						]
						for q in xy
					]
	return {
		"lower": lower,
		"upper": upper,
		"path": best_path,
		"socp_models": count,
		"status": "all_leaves_numerically_optimal",
	}


def synthetic_cases():
	def case(name, regions, start=(0.0, 0.0), target=(0.0, 0.0)):
		return {
			"name": name,
			"start": start,
			"target": target,
			"polygons": regions,
			"origin": "synthetic",
		}

	cases = [
		case("point-return", [[(2.0, 1.0)]]),
		case("segment-interior-reflection", [[(2.0, -1.0), (2.0, 1.0)]]),
		case("segment-endpoint", [[(3.0, 2.0), (4.0, 2.0)]], target=(4.0, 0.0)),
		case(
			"crossing-segments", [[(-2.0, 0.0), (2.0, 0.0)], [(0.0, -2.0), (0.0, 2.0)]]
		),
		case(
			"collinear-segments", [[(1.0, 0.0), (3.0, 0.0)], [(2.0, 0.0), (4.0, 0.0)]]
		),
		case("coincident-points", [[(2.0, 1.0)], [(2.0, 1.0)], [(2.0, -1.0)]]),
		case(
			"point-segment-polygon",
			[
				[(4.0, 1.0)],
				[(2.0, -1.0), (2.0, 2.0)],
				[(1.0, -1.0), (3.0, -1.0), (2.0, 1.0)],
			],
		),
		case(
			"nonconvex-alternatives",
			[
				[
					(0.0, 0.0),
					(2.0, 0.0),
					(2.0, 0.6),
					(0.6, 0.6),
					(0.6, 2.0),
					(0.0, 2.0),
				],
				[(2.0, 2.0)],
				[(1.0, 1.0), (3.0, 1.0)],
			],
			start=(1.0, 1.2),
			target=(1.8, 1.3),
		),
	]
	rng = random.Random(20261003)
	for k in range(24):
		regions = []
		for j in range(1 + k % 3):
			x, y = rng.uniform(-3, 3), rng.uniform(-3, 3)
			kind = (k + j) % 4
			if kind == 0:
				p = [(x, y)]
			elif kind == 1:
				p = [(x, y), (x + rng.uniform(-2, 2), y + rng.uniform(-2, 2))]
			elif kind == 2:
				p = [(x, y), (x + 1, y), (x + 0.4, y + 1)]
			else:
				p = [
					(x, y),
					(x + 1, y),
					(x + 1, y + 0.3),
					(x + 0.3, y + 0.3),
					(x + 0.3, y + 1),
					(x, y + 1),
				]
			if k % 2:
				p.reverse()
			regions.append(p)
		cases.append(
			case(
				f"random-{k:02}",
				regions,
				start=(-1.0, -0.5),
				target=(-1.0, -0.5) if k % 2 else (3.0, 2.0),
			)
		)
	return cases


def worker(request: Path, output: Path) -> int:
	import gurobipy as gp
	import shapely
	from unordered_validation import validate_path

	payload = json.loads(request.read_text())
	rows = []
	with gp.Env(empty=True) as env:
		env.setParam("OutputFlag", 0)
		env.start()
		for case in payload["cases"]:
			began = time.monotonic()
			reference = baseline(
				case["start"],
				case["target"],
				[independent_pieces(p) for p in case["polygons"]],
				env,
				payload["seconds"],
			)
			reference["validation"] = validate_path(
				case["start"],
				case["target"],
				case["polygons"],
				reference["path"],
				payload["comparison_tolerance"] * (1 + reference["upper"]),
			)
			reference["seconds"] = time.monotonic() - began
			reference["runtime"] = {
				"gurobi": ".".join(map(str, gp.gurobi.version())),
				"shapely": shapely.__version__,
				"geos": shapely.geos_version_string,
				"python": sys.version.split()[0],
			}
			rows.append({"name": case["name"], "reference": reference})
			output.write_text(json.dumps(rows, indent=2) + "\n")
			print(
				f"SOCP {case['name']}: {reference['upper']:.10g} ({reference['socp_models']} models)",
				flush=True,
			)
	return 0


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--solver", type=Path, default=native_build.tool_path("tpp-unordered"))
	parser.add_argument(
		"--reference-python",
		type=Path,
		default=ROOT / "third_party/tspn-socg/.venv/bin/python",
	)
	parser.add_argument(
		"--output", type=Path, default=workspace.run_path("socp-verification")
	)
	parser.add_argument("--manifest", type=Path)
	parser.add_argument(
		"--subset-size",
		type=int,
		default=3,
		help="Deterministic subsets retain the original depot; 0 selects full instances",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=12,
		help="Number of Paula cases in addition to 32 synthetic checks",
	)
	parser.add_argument("--seconds", type=float, default=30)
	parser.add_argument("--comparison-tolerance", type=float, default=2e-6)
	args = parser.parse_args(argv)
	if (
		args.subset_size < 0
		or args.limit < 0
		or not math.isfinite(args.seconds)
		or args.seconds <= 0
		or not math.isfinite(args.comparison_tolerance)
		or args.comparison_tolerance <= 0
	):
		parser.error("Invalid verification limits or tolerances")
	cases = synthetic_cases()
	if args.manifest:
		manifest = json.loads(args.manifest.read_text())

		def subset_indices(source):
			indices = list(range(len(source["polygons"])))
			if args.subset_size and len(indices) > args.subset_size:
				indices = sorted(
					indices, key=lambda i: (len(source["polygons"][i]), i)
				)[: args.subset_size]
			return sorted(indices)

		def complexity(source):
			indices = subset_indices(source)
			return math.factorial(len(indices)) * math.prod(
				max(1, len(source["polygons"][i]) - 2) for i in indices
			)

		# Select manageable leaves before launching the exhaustive reference.
		# Short regions exercise the point/segment contract without a large
		# product of triangulation alternatives. No source geometry is changed.
		selected = sorted(
			manifest["cases"],
			key=lambda c: (complexity(c), len(c["polygons"]), c["name"]),
		)[: args.limit]
		for source in selected:
			indices = subset_indices(source)
			cases.append(
				{
					"name": source["name"]
					+ (
						f"/subset-{','.join(map(str, indices))}"
						if len(indices) != len(source["polygons"])
						else ""
					),
					"start": source["start"],
					"target": source["target"],
					"polygons": [source["polygons"][i] for i in indices],
					"origin": "paula",
					"source_indices": indices,
				}
			)
	args.output.mkdir(parents=True, exist_ok=True)
	request = args.output / "socp-inputs.json"
	request.write_text(
		json.dumps(
			{
				"cases": cases,
				"seconds": args.seconds,
				"comparison_tolerance": args.comparison_tolerance,
			},
			indent=2,
		)
		+ "\n"
	)
	results = []
	from unordered_validation import validate_path

	for case in cases:
		text = (
			" ".join(
				map(
					str,
					(
						*case["start"],
						*case["target"],
						len(case["polygons"]),
						1000000,
						args.seconds,
					),
				)
			)
			+ "\n"
		)
		text += "".join(
			str(len(p)) + " " + " ".join(str(v) for q in p for v in q) + "\n"
			for p in case["polygons"]
		)
		process = subprocess.run(
			[str(args.solver.resolve())],
			input=text,
			text=True,
			capture_output=True,
			timeout=args.seconds + 60,
			check=False,
		)
		if process.returncode:
			raise RuntimeError(f"C++ failed for {case['name']}: {process.stderr}")
		result = json.loads(process.stdout)
		result["independent_validation"] = validate_path(
			case["start"],
			case["target"],
			case["polygons"],
			result["path"],
			1e-7 * (1 + result["upper_bound"]),
		)
		results.append({"name": case["name"], "ours": result})
	(args.output / "native-results.json").write_text(
		json.dumps(results, indent=2) + "\n"
	)
	response = args.output / "socp-results.json"
	subprocess.run(
		[
			str(args.reference_python),
			str(Path(__file__).resolve()),
			"--worker",
			str(request),
			str(response),
		],
		check=True,
	)
	references = json.loads(response.read_text())
	for row, reference in zip(results, references, strict=True):
		if row["name"] != reference["name"]:
			raise RuntimeError("Reference result order mismatch")
		ours, ref = row["ours"], reference["reference"]
		tolerance = args.comparison_tolerance * (1 + ref["upper"])
		row["reference"] = ref
		row["passed"] = (
			ours["exact"]
			and ours["independent_validation"]["valid"]
			and ref["validation"]["valid"]
			and abs(ours["upper_bound"] - ref["upper"]) <= tolerance
			and ours["lower_bound"] <= ref["upper"] + tolerance
			and ref["lower"] <= ours["upper_bound"] + tolerance
		)
	summary = {
		"formulation": "free-order endpoint TPP; exhaustive orders and independent convex alternatives; continuous SOCP leaves",
		"reference": "Gurobi numerical SOCP; this comparison is not an exact arithmetic proof",
		"comparison_tolerance": args.comparison_tolerance,
		"socp_feasibility_tolerance": 1e-9,
		"socp_barrier_tolerance": 1e-9,
		"socp_retry_barrier_tolerance": 1e-7,
		"solver": str(args.solver.resolve()),
		"seconds_per_solve": args.seconds,
		"solver_sha256": hashlib.sha256(args.solver.read_bytes()).hexdigest(),
		"inputs_sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
		"native_max_calls": 1000000,
		"native_absolute_gap": 1e-7,
		"native_relative_gap": 1e-9,
		"native_feasibility_tolerance": 1e-8,
		"passed": sum(r["passed"] for r in results),
		"count": len(results),
		"results": results,
	}
	(args.output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
	print(
		f"Passed {summary['passed']}/{len(results)} independent SOCP comparisons. Report: {args.output / 'verification.json'}"
	)
	return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
	if len(sys.argv) == 4 and sys.argv[1] == "--worker":
		raise SystemExit(worker(Path(sys.argv[2]), Path(sys.argv[3])))
	raise SystemExit(main())
