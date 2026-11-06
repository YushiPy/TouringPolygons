"""Paper 1, Section 7.2: cycles through random sequences of polygons, by the
boundary-anchor method (Algorithm 4.1, no sweeps) against Gurobi's SOCP.

    python3 benchmarks/tpp.py paper1-cycles                 # starts a detached job
    python3 benchmarks/tpp.py paper1-cycles --plan          # sizes and output, nothing run
    python3 benchmarks/tpp.py paper1-cycles --smoke         # a few instances, minutes
    python3 benchmarks/tpp.py paper1-cycles --foreground --limit 200

Instances: for each of the 558 instances of Fekete et al. and each k in
{4, 8, 16, 32} not larger than its number of regions, k regions drawn with a
fixed seed (random.Random("paper1-cycles-<case>-<k>")), their exact convex
hulls, in a random cyclic order: 1638 cycles. --limit N keeps the first N
cases in name order (all their k), and states the cut in the summary.

Steps (each writes STEP/done.json; cycles resumes instance by instance):

  build    tpp-convex-cycle-sequences (with Gurobi)
  check    the corpus archive, and a one-triangle Gurobi solve
  cycles   ours (refine_contacts=false, gap zero, no polish; 3 repetitions,
           calls under 0.05 s repeated back to back) and Gurobi (unit
           normals, default parameters, one thread), one process per
           instance, --seconds caps each of our calls
  summary  summary.json and summary.md

Output: benchmarks/workspace/experiments/paper1-cycles/<host>-<commit>[-smoke]/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import native_build  # noqa: E402
import paper1_experiments as paper1  # noqa: E402
import workspace  # noqa: E402

KS = (4, 8, 16, 32)
REPETITIONS = 3
TOOL = "tpp-convex-cycle-sequences"


def hull(points) -> list[list[float]]:
	"""Counterclockwise convex hull of binary64 points, decided exactly
	(Fractions), without collinear vertices; the vertices stay binary64."""
	unique = sorted({(float(x), float(y)) for x, y in points})
	exact = {point: (Fraction(point[0]), Fraction(point[1])) for point in unique}

	def cross(o, a, b):
		(ox, oy), (ax, ay), (bx, by) = exact[o], exact[a], exact[b]
		return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)

	if len(unique) < 3:
		return [list(point) for point in unique]
	lower: list = []
	for point in unique:
		while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
			lower.pop()
		lower.append(point)
	upper: list = []
	for point in reversed(unique):
		while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
			upper.pop()
		upper.append(point)
	return [list(point) for point in lower[:-1] + upper[:-1]]


def cycle_instances(cases: dict, smoke: bool, limit: int | None) -> list[dict]:
	names = sorted(cases)
	if limit:
		names = names[:limit]
	instances = []
	for name in names:
		regions = cases[name].polygons
		for k in KS:
			if k > len(regions):
				continue
			rng = random.Random(f"paper1-cycles-{name}-{k}")
			chosen = rng.sample(range(len(regions)), k)
			rng.shuffle(chosen)  # the cyclic order
			instances.append({"name": f"{Path(name).stem}-k{k}", "case": name, "k": k, "regions": chosen,
				"polygons": [hull(regions[index]) for index in chosen]})
	if smoke:
		instances = [item for item in instances if item["k"] in (4, 8)][:8]
	return instances


class Context(paper1.Context):
	def __init__(self, args: argparse.Namespace):
		if not args.output:
			git = workspace.git_state()
			commit = (git.get("revision") or "unknown")[:7] + ("-dirty" if git.get("dirty") else "")
			label = f"{platform.node().split('.')[0]}-{commit}" + ("-smoke" if args.smoke else "")
			args.output = str(workspace.experiments_dir() / "paper1-cycles" / label)
		super().__init__(args)


def power_state() -> str | None:
	"""macOS: the power source and Low Power Mode, which slows timings."""
	if platform.system() != "Darwin":
		return None
	try:
		battery = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[:1]
		mode = [line.strip() for line in subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout.splitlines()
			if "powermode" in line or "lowpowermode" in line]
		return "; ".join(battery + mode)
	except OSError:
		return None


def step_build(context: Context) -> dict:
	path = native_build.ensure_tools([TOOL], gurobi=True)[0]
	return {"tools": {TOOL: str(path)}}


def load_tools(context: Context) -> None:
	context.tools = {TOOL: native_build.tool_path(TOOL, gurobi=True),
		"tpp-convex-path-benchmark": native_build.tool_path("tpp-convex-path-benchmark", gurobi=True)}


def step_check(context: Context) -> dict:
	native_build.ensure_tools(["tpp-convex-path-benchmark"], gurobi=True)
	return paper1.step_check(context)


def step_cycles(context: Context) -> dict:
	directory = context.step_dir("cycles")
	instances = cycle_instances(context.cases(), context.smoke, context.args.limit)
	(directory / "plan.json").write_text(json.dumps([{key: item[key] for key in ("name", "case", "k", "regions")}
		for item in instances]))
	tasks = []
	for item in sorted(instances, key=lambda item: -sum(len(polygon) for polygon in item["polygons"])):
		path = directory / "inputs" / f"{item['name']}.json"
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps({"instances": [{"name": item["name"], "polygons": item["polygons"]}]}))
		tasks.append({"command": [str(context.tools[TOOL]), str(path), "1" if context.smoke else str(REPETITIONS),
			str(context.args.seconds)], "stdout": directory / "results" / f"{item['name']}.jsonl",
			"timeout": REPETITIONS * (context.args.seconds + 600)})
	failures = paper1.run_parallel(context, tasks, "cycles")
	return {"instances": len(instances), "limit": context.args.limit, "seconds_cap": context.args.seconds,
		"failures": failures, "git": workspace.git_state(), "power": power_state()}


# --------------------------------------------------------------------------- summary

STATUS = {0: "InvalidInput", 1: "UnsupportedIntersection", 2: "Optimal", 3: "FloatingPointLimit", 4: "OracleFailure",
	5: "CertifiedBound", 6: "Interrupted", 7: "ProposalLimit", 8: "GapClosed"}


def summarize_cycles(rows: list[dict]) -> dict:
	"""Per k and overall. Time per instance: the median over repetitions of our
	mean and of Gurobi's total; the speed is their ratio, in geometric mean."""
	by_instance: dict = {}
	for row in rows:
		by_instance.setdefault(row["name"], []).append(row)
	groups: dict = {}
	for name, repetitions in by_instance.items():
		groups.setdefault(repetitions[0]["k"], []).append(repetitions)
	groups["all"] = [repetitions for group in list(groups.values()) for repetitions in group]
	out = {}
	for key, instances in sorted(groups.items(), key=lambda item: (isinstance(item[0], str), item[0])):
		calls = [row for repetitions in instances for row in repetitions]
		first = [repetitions[0] for repetitions in instances]
		ours = [paper1.median(row["ours"]["mean"] for row in repetitions) for repetitions in instances]
		gurobi = [paper1.median(row["gurobi"]["total"] for row in repetitions) for repetitions in instances]
		exact = [row for row in first if row["ours"]["exact"]]
		# Gurobi's objective against our proved lower bound L, per call.
		below = []
		errors = []
		for row in calls:
			lower, objective = row["ours"].get("exact_lower"), row["gurobi"].get("objective")
			if lower is None or objective is None or not row["ours"]["exact"]:
				continue
			errors.append(abs(objective - lower) / max(1e-300, abs(lower)) if lower else abs(objective))
			if objective < lower:
				below.append((lower - objective) / max(1.0, abs(lower)))
		with_exact = sum(1 for row in calls if row["ours"]["exact"] and row["gurobi"].get("objective") is not None)
		entry = {
			"instances": len(instances), "calls": len(calls),
			"vertices_median": paper1.median(row["vertices"] for row in first),
			"status": paper1.count_by(STATUS.get(row["ours"]["status"], str(row["ours"]["status"])) for row in first),
			"exact_optimal": len(exact),
			"step": paper1.count_by(row["ours"]["step"] for row in first),
			"step2_share": (sum(1 for row in exact if row["ours"]["step"] == 2) / len(exact)) if exact else None,
			"step3_share": (sum(1 for row in exact if row["ours"]["step"] == 3) / len(exact)) if exact else None,
			"ours_median_s": paper1.median(ours), "ours_max_s": max(value for value in ours if value is not None) if ours else None,
			"gurobi_median_s": paper1.median(gurobi),
			"gurobi_over_ours_geomean": paper1.geometric_mean(g / o for g, o in zip(gurobi, ours) if g and o),
			"gurobi_faster_instances": sum(1 for g, o in zip(gurobi, ours) if g and o and g < o),
			"gurobi_status": paper1.count_by(row["gurobi"]["status"] for row in first),
			"gurobi_below_L": {"calls": len(below), "of": with_exact, "share": len(below) / with_exact if with_exact else None,
				"max_relative": max(below) if below else 0.0},
			"gurobi_relative_error": {"median": paper1.median(errors), "max": max(errors) if errors else None},
			"anchored_tpps": {"median": paper1.median(row["ours"]["anchored"] for row in first),
				"max": max((row["ours"]["anchored"] for row in first), default=None),
				"total": sum(row["ours"]["anchored"] for row in first)},
			"bisection_probes": {"median": paper1.median(row["ours"]["probes"] for row in first),
				"max": max((row["ours"]["probes"] for row in first), default=None),
				"total": sum(row["ours"]["probes"] for row in first)},
			"rational_anchor_recoveries": {"instances": sum(1 for row in first if row["ours"]["rational_anchor_recoveries"]),
				"total": sum(row["ours"]["rational_anchor_recoveries"] for row in first)},
			"rational_search": {"instances": sum(1 for row in first if row["ours"]["rational_cycle_recoveries"]),
				"anchored_total": sum(row["ours"]["rational_anchored"] for row in first),
				"probes_total": sum(row["ours"]["rational_probes"] for row in first)},
		}
		out[str(key)] = entry
	return out


def write_summary(context: Context) -> dict:
	rows = paper1.jsonl(sorted((context.out / "cycles" / "results").glob("*.jsonl")))
	cycles_done = context.out / "cycles" / "done.json"
	summary = {"output": str(context.out), "machine": workspace.machine_state(), "git": workspace.git_state(),
		"power": power_state(), "smoke": context.smoke,
		"cycles_step": json.loads(cycles_done.read_text()) if cycles_done.exists() else None,
		"cycles": summarize_cycles(rows) if rows else None}
	(context.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
	lines = [f"# Paper 1 cycles: {context.out.name}", ""]
	for key, value in summary.items():
		lines += [f"## {key}", "", "```json", json.dumps(value, indent=1, default=str), "```", ""]
	(context.out / "summary.md").write_text("\n".join(lines))
	return {"rows": len(rows)}


STEPS: list[tuple[str, Callable[[Context], dict]]] = [
	("build", step_build),
	("check", step_check),
	("cycles", step_cycles),
	("summary", write_summary),
]


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(prog="tpp.py paper1-cycles", description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--smoke", action="store_true", help="a few instances (separate output directory)")
	parser.add_argument("--plan", action="store_true", help="print the sizes and output directory, and stop")
	parser.add_argument("--only", action="append", choices=[name for name, _ in STEPS], help="run only this step (repeatable)")
	parser.add_argument("--redo", action="append", choices=[name for name, _ in STEPS], default=[],
		help="run this step again even if it is done (repeatable)")
	parser.add_argument("--workers", type=int, help="parallel processes (default: one per physical core)")
	parser.add_argument("--limit", type=int, help="only the first N corpus cases in name order (all their k)")
	parser.add_argument("--seconds", type=float, default=600.0, help="cap on each of our calls (default 600)")
	parser.add_argument("--output", help="output directory (default: workspace/experiments/paper1-cycles/<host>-<commit>)")
	parser.add_argument("--foreground", action="store_true", help="run here instead of starting a detached job")
	parser.add_argument("--archive", help="the 558-case archive (default: Fekete's upstream checkout, from fekete-setup)")
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse(argv)
	if not args.foreground and not args.plan:
		import jobs
		return jobs.main(["start", "--name", "paper1-cycles", "--", "paper1-cycles", "--foreground",
			*(argv if argv is not None else sys.argv[1:])])
	context = Context(args)
	steps = [(name, function) for name, function in STEPS if not args.only or name in args.only]
	if args.plan:
		instances = cycle_instances(context.cases(), args.smoke, args.limit)
		print(f"output: {context.out}\nworkers: {context.workers}\ncycles: {len(instances)} instances "
			f"({paper1.count_by(item['k'] for item in instances)} by k), {REPETITIONS} repetitions, cap {args.seconds} s")
		for name, _ in steps:
			print(f"  {name}{' (done)' if (context.out / name / 'done.json').exists() else ''}")
		return 0
	context.out.mkdir(parents=True, exist_ok=True)
	(context.out / "plan.json").write_text(json.dumps({"argv": list(argv or sys.argv[1:]), "git": workspace.git_state(),
		"machine": workspace.machine_state(), "power": power_state(), "workers": context.workers, "smoke": args.smoke,
		"limit": args.limit, "seconds": args.seconds, "steps": [name for name, _ in steps]}, indent=2) + "\n")
	context.log(f"paper1-cycles: {context.out} ({context.workers} workers; {power_state()})")
	began = time.monotonic()
	for name, function in steps:
		done = context.out / name / "done.json"
		if done.exists() and name not in args.redo and name != "summary":
			context.log(f"{name}: done before, skipped")
			if name == "check":
				license_file = json.loads(done.read_text()).get("license_file")
				if license_file:
					os.environ["GRB_LICENSE_FILE"] = license_file
			continue
		if name in args.redo and name not in ("build", "summary"):
			import shutil
			shutil.rmtree(context.out / name, ignore_errors=True)
		if name != "build" and not context.tools:
			load_tools(context)
		context.log(f"{name}: started")
		started = time.monotonic()
		result = function(context)
		result["seconds"] = round(time.monotonic() - started, 1)
		context.step_dir(name)
		done.write_text(json.dumps(result, indent=2, default=str) + "\n")
		context.log(f"{name}: done in {result['seconds']:.0f} s")
	context.log(f"all steps done in {(time.monotonic() - began) / 3600:.2f} h; summary: {context.out / 'summary.md'}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
