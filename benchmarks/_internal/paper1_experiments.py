"""Paper 1 (Section 7): every experiment in one resumable command.

    python3 benchmarks/tpp.py paper1-experiments            # starts a detached job
    python3 benchmarks/tpp.py paper1-experiments --plan     # steps, sizes and output, nothing run
    python3 benchmarks/tpp.py paper1-experiments --smoke    # small sizes, a few minutes
    python3 benchmarks/tpp.py paper1-experiments --foreground --only paths-random

Steps, in order (each writes STEP/done.json and is skipped when run again; a
step killed midway resumes case by case where it writes one file per case):

  build            tpp-unordered (as configuration C) and the Gurobi tools
  check            the corpus archive, and a one-instance Gurobi solve
  paths-random     Section 7.1: generated paths (disjoint and intersecting,
                   growing k and vertices per polygon): our certified oracle,
                   the lazy and eager constructions with the cones they build,
                   and Gurobi's SOCP (tpp-convex-path-benchmark)
  bnb-cycle        Section 7.2: our TSPN B&B in configuration C on the 558
                   corpus cases, with how each oracle call closed
  capture-cycle    cycle oracle calls captured from that B&B on 40 cases
  replay-cycle     the captured calls: ours against Gurobi's SOCP
  capture-path     fixed-endpoint calls from the B&B in path mode, same cases
  replay-path      those calls: ours against Gurobi's SOCP
  vertex-scaling   one cycle call as the vertices per region grow, against Gurobi
  anchor-probes    the exact path on captured calls: anchored solves and probes
  no-polish        the B&B with and without the interior-point polish (60 s)
  summary          summary.json, summary.md and a tarball of the outputs

Output: benchmarks/workspace/experiments/paper1/<host>-<commit>[-smoke]/.
Timing steps run one process per physical core, pinned on Linux.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import platform
import queue
import random
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import native_build  # noqa: E402
import workspace  # noqa: E402

CORPUS_SEED = 20261021
CAPTURE_CASES = 40
CAPTURE_MIN_CALLS = 1000
CAPTURE_MAX_CALLS = 200_000
CAPTURE_TARGET = 2000  # captured calls per case before reduction
REPLAY_CALLS = 400
GUROBI_TOOLS = ["tpp-convex-path-benchmark", "tpp-convex-vertex-scaling", "tpp-convex-anchor-probes"]


# --------------------------------------------------------------------------- context

class Context:
	def __init__(self, args: argparse.Namespace):
		self.smoke = args.smoke
		self.args = args
		git = workspace.git_state()
		commit = (git.get("revision") or "unknown")[:7] + ("-dirty" if git.get("dirty") else "")
		label = f"{platform.node().split('.')[0]}-{commit}" + ("-smoke" if self.smoke else "")
		self.out = Path(args.output) if args.output else workspace.experiments_dir() / "paper1" / label
		self.cores = physical_cores()
		self.workers = args.workers or max(1, len(self.cores) if self.cores else (os.cpu_count() or 2) // 2)
		self.tools: dict[str, Path] = {}
		self._cases = None

	def log(self, message: str) -> None:
		line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
		print(line, flush=True)
		self.out.mkdir(parents=True, exist_ok=True)
		with (self.out / "log.txt").open("a") as file:
			file.write(line + "\n")

	def step_dir(self, step: str) -> Path:
		path = self.out / step
		path.mkdir(parents=True, exist_ok=True)
		return path

	def cases(self) -> dict:
		"""Corpus case name -> converted case (polygons; start and target for the path mode)."""
		if self._cases is None:
			import fekete_reference
			from convert_instances import convert_json_case
			if self.args.archive:
				fekete_reference.ARCHIVE_OVERRIDE = Path(self.args.archive)
			archive = fekete_reference.instance_archive()
			if not archive.is_file():
				raise SystemExit(f"The 558-case archive is missing ({archive}). Run "
					"'python3 benchmarks/tpp.py fekete-setup' once, or pass --archive PATH.")
			cases = {}
			with zipfile.ZipFile(archive) as file:
				for member in sorted(file.namelist()):
					if member.endswith(".json"):
						cases[Path(member).stem] = convert_json_case(member, file.read(member))
			if self.smoke:
				# Six quick cases that still need many oracle calls (the 5-polygon ones need one).
				chosen = sorted((name for name in cases if len(cases[name].polygons) >= 15),
					key=lambda name: (len(cases[name].polygons), name))[:6]
				cases = {name: cases[name] for name in chosen}
			self._cases = cases
		return self._cases


def physical_cores() -> list[tuple[int, ...]]:
	"""Logical CPUs of each physical core (Linux), empty where pinning is unavailable."""
	if not hasattr(os, "sched_setaffinity"):
		return []
	try:
		from cpu_affinity import detect_cores
		return [core.cpus for core in detect_cores()]
	except Exception:
		return []


def run_parallel(context: Context, tasks: Sequence[dict], label: str) -> list[dict]:
	"""Run {command, stdout, stdin?, timeout?} tasks, one per core, pinned on Linux.
	A task whose stdout file exists and is complete (done marker) is skipped."""
	pending = [task for task in tasks if not Path(str(task["stdout"]) + ".done").exists()]
	if not pending:
		return []
	cores: queue.Queue = queue.Queue()
	for index in range(context.workers):
		cores.put(context.cores[index % len(context.cores)] if context.cores else None)
	failures: list[dict] = []
	finished = 0
	started = time.monotonic()

	def run(task: dict) -> dict | None:
		cpus = cores.get()
		try:
			out = Path(task["stdout"])
			out.parent.mkdir(parents=True, exist_ok=True)
			stdin = open(task["stdin"], "rb") if task.get("stdin") else subprocess.DEVNULL
			with open(out, "wb") as stdout, open(str(out) + ".log", "wb") as stderr:
				process = subprocess.Popen(task["command"], stdin=stdin, stdout=stdout, stderr=stderr,
					preexec_fn=(lambda: os.sched_setaffinity(0, set(cpus))) if cpus else None)
				try:
					code = process.wait(timeout=task.get("timeout"))
				except subprocess.TimeoutExpired:
					process.kill()
					process.wait()
					code = "killed"
			if stdin is not subprocess.DEVNULL:
				stdin.close()
			if code == 0 or task.get("accept_nonzero"):
				Path(str(out) + ".done").write_text(json.dumps({"exit": code}) + "\n")
				return None
			return {"task": str(out), "exit": code}
		finally:
			cores.put(cpus)

	with concurrent.futures.ThreadPoolExecutor(max_workers=context.workers) as pool:
		for result in pool.map(run, pending):
			finished += 1
			if result:
				failures.append(result)
				context.log(f"  {label}: failed {result['task']} (exit {result['exit']})")
			if finished % max(1, len(pending) // 20) == 0 or finished == len(pending):
				context.log(f"  {label}: {finished}/{len(pending)} in {time.monotonic() - started:.0f} s")
	return failures


# --------------------------------------------------------------------------- generators

def exact_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
	"""Counterclockwise convex hull of integer points (monotone chain, exact)."""
	points = sorted(set(points))
	if len(points) < 3:
		return points

	def cross(o, a, b):
		return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

	lower: list = []
	for p in points:
		while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
			lower.pop()
		lower.append(p)
	upper: list = []
	for p in reversed(points):
		while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
			upper.pop()
		upper.append(p)
	return lower[:-1] + upper[:-1]


SCALE = 2 ** 13


def convex_polygon(rng: random.Random, center: tuple[float, float], radius: float, m: int,
		aspect: float = 1.0) -> list[list[float]]:
	"""The exact hull of m integer points (coordinates times 2^13) on an ellipse:
	convex by construction, with about m vertices; binary64-exact coordinates."""
	rotation = rng.uniform(0, math.pi)
	points = []
	for _ in range(m):
		angle = rng.uniform(0, 2 * math.pi)
		x, y = radius * math.cos(angle), radius * aspect * math.sin(angle)
		x, y = x * math.cos(rotation) - y * math.sin(rotation), x * math.sin(rotation) + y * math.cos(rotation)
		points.append((round((center[0] + x) * SCALE), round((center[1] + y) * SCALE)))
	return [[px / SCALE, py / SCALE] for px, py in exact_hull(points)]


def path_instances(smoke: bool) -> list[dict]:
	"""Section 7.1: k polygons between s and t. Disjoint: centers 2.4 apart in x
	(radii at most 1). Intersecting: 1.1 apart, so consecutive polygons overlap."""
	ks, ms, seeds = ((4, 8), (8, 64), 1) if smoke else ((4, 8, 16, 32, 64), (8, 64, 512, 4096), 10)
	instances = []
	for family, spacing, wobble in (("disjoint", 2.4, 1.5), ("intersecting", 1.1, 0.3)):
		for k in ks:
			for m in ms:
				for seed in range(seeds):
					rng = random.Random(f"{family}-{k}-{m}-{seed}")
					polygons = [convex_polygon(rng, (i * spacing, rng.uniform(-wobble, wobble)), rng.uniform(0.6, 1.0), m,
						rng.uniform(0.5, 1.0)) for i in range(k)]
					instances.append({"name": f"{family}-k{k}-m{m}-s{seed}", "family": family, "k": k, "m": m,
						"start": [-2.0, rng.uniform(-1, 1)], "target": [(k - 1) * spacing + 2.0, rng.uniform(-1, 1)],
						"polygons": polygons})
	return instances


def cycle_instances(smoke: bool) -> list[dict]:
	"""Section 7.2, as oracle-vertex-scaling-2026-10-21: k disjoint regions on a
	circle, each the hull of m points on a disk or an ellipse."""
	ks, ms, samples = ((3,), (8, 128), 1) if smoke else ((3, 5, 10), (8, 32, 128, 512, 2048, 8192, 32768), 2)
	instances = []
	for k in ks:
		for m in ms:
			for shape in ("disk", "ellipse"):
				for sample in range(samples):
					rng = random.Random(f"cycle-{k}-{m}-{shape}-{sample}")
					ring = 3.0 * k / math.pi
					polygons = [convex_polygon(rng, (ring * math.cos(2 * math.pi * i / k), ring * math.sin(2 * math.pi * i / k)),
						1.0, m, 1.0 if shape == "disk" else 0.4) for i in range(k)]
					instances.append({"name": f"cycle-k{k}-m{m}-{shape}-{sample}", "polygons": polygons})
	return instances


# --------------------------------------------------------------------------- steps

def unordered_arguments(cycle: bool) -> list[str]:
	import fekete_reference
	if cycle:
		return list(fekete_reference.OUR_ARGUMENTS)
	eps = fekete_reference.EPS
	return ["--absolute-gap", "0", "--relative-gap", repr(eps / (1 + eps)), "--feasibility-tolerance", "1e-08"]


def write_input(path: Path, start, target, polygons, max_calls: int, seconds: float) -> Path:
	from unordered_runner import encode_instance
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(encode_instance(tuple(start), tuple(target), polygons, max_calls, seconds))
	return path


def step_build(context: Context) -> dict:
	unordered = native_build.ensure_tool("tpp-unordered")
	gurobi = native_build.ensure_tools(GUROBI_TOOLS, gurobi=True)
	return {"tools": {"tpp-unordered": str(unordered), **{tool: str(path) for tool, path in zip(GUROBI_TOOLS, gurobi)}}}


def load_tools(context: Context) -> None:
	context.tools = {"tpp-unordered": native_build.tool_path("tpp-unordered"),
		**{tool: native_build.tool_path(tool, gurobi=True) for tool in GUROBI_TOOLS}}


def step_check(context: Context) -> dict:
	"""The corpus, and Gurobi: a one-triangle path, first with Gurobi's own license
	lookup, then with the license files the Fekete runs try (fekete_reference)."""
	cases = context.cases()
	probe = context.step_dir("check") / "probe.json"
	probe.write_text(json.dumps({"instances": [{"name": "probe", "start": [0, 0], "target": [4, 0],
		"polygons": [[[1, 1], [3, 1], [2, 3]]]}]}))

	def solve(license_file: str | None) -> subprocess.CompletedProcess:
		environment = dict(os.environ)
		if license_file:
			environment["GRB_LICENSE_FILE"] = license_file
		return subprocess.run([str(context.tools["tpp-convex-path-benchmark"]), str(probe), "1"],
			capture_output=True, text=True, env=environment)

	import fekete_reference
	license_file = os.environ.get("GRB_LICENSE_FILE")
	result = solve(license_file)
	for candidate in ([] if result.returncode == 0 else fekete_reference.license_candidates()):
		result = solve(str(candidate))
		if result.returncode == 0:
			license_file = str(candidate)
			break
	if result.returncode != 0:
		raise SystemExit("Gurobi could not solve a one-triangle path with any license file. Put the academic "
			"gurobi.lic in the home directory (or set GRB_LICENSE_FILE) and run again.\n" + result.stderr)
	if license_file:
		os.environ["GRB_LICENSE_FILE"] = license_file  # every later step inherits it
	return {"cases": len(cases), "workers": context.workers, "cores": [list(core) for core in context.cores],
		"license_file": license_file, "gurobi_probe": json.loads(result.stdout)["gurobi"]}


def step_paths_random(context: Context) -> dict:
	directory = context.step_dir("paths-random")
	instances = path_instances(context.smoke)
	(directory / "instances.json").write_text(json.dumps({"instances": instances}))
	# Shards balanced by size: the largest instances first, dealt round robin.
	ordered = sorted(instances, key=lambda item: -item["k"] * item["m"])
	shards = [ordered[index::context.workers] for index in range(context.workers)]
	tasks = []
	for index, shard in enumerate(shards):
		if not shard:
			continue
		path = directory / f"shard-{index:02d}.json"
		path.write_text(json.dumps({"instances": shard}))
		tasks.append({"command": [str(context.tools["tpp-convex-path-benchmark"]), str(path), "1" if context.smoke else "3"],
			"stdout": directory / f"shard-{index:02d}.jsonl"})
	failures = run_parallel(context, tasks, "paths-random")
	return {"instances": len(instances), "failures": failures}


def bnb_tasks(context: Context, directory: Path, names: Sequence[str], *, cycle: bool, seconds: float,
		max_calls: int = -1, extra: Sequence[str] = (), capture_every: int | None = None) -> list[dict]:
	cases = context.cases()
	tasks = []
	for name in names:
		case = cases[name]
		start, target = ((0.0, 0.0), (0.0, 0.0)) if cycle else (case.start, case.target)
		stdin = write_input(directory / "inputs" / f"{name}.txt", start, target, case.polygons, max_calls, seconds)
		command = [str(context.tools["tpp-unordered"]), *unordered_arguments(cycle), *extra]
		if capture_every:
			command += ["--oracle-capture", str(directory / "captures" / f"{name}.jsonl"),
				"--oracle-capture-every", str(capture_every)]
			(directory / "captures").mkdir(parents=True, exist_ok=True)
		tasks.append({"command": command, "stdin": stdin, "stdout": directory / "results" / f"{name}.json",
			"timeout": seconds + max(120.0, 0.5 * seconds)})
	return tasks


def read_result(path: Path) -> dict | None:
	try:
		return json.loads(path.read_text())
	except (OSError, json.JSONDecodeError):
		return None


def step_bnb_cycle(context: Context) -> dict:
	directory = context.step_dir("bnb-cycle")
	seconds = 10.0 if context.smoke else 300.0
	failures = run_parallel(context, bnb_tasks(context, directory, list(context.cases()), cycle=True, seconds=seconds),
		"bnb-cycle")
	return {"seconds": seconds, "failures": failures}


def bnb_calls(context: Context) -> dict[str, int]:
	"""Oracle calls of each configuration-C run (bnb-cycle)."""
	rows = {path.stem: read_result(path) for path in (context.out / "bnb-cycle" / "results").glob("*.json")}
	return {name: int(row.get("calls", 0)) for name, row in rows.items() if row}


def capture_cases(context: Context) -> list[str]:
	"""40 cases, seeded, among those whose configuration-C run made at least 1000 oracle calls."""
	rows = {path.stem: read_result(path) for path in (context.out / "bnb-cycle" / "results").glob("*.json")}
	eligible = sorted(name for name, row in rows.items() if row and row.get("convex_calls", row.get("calls", 0)) >= (
		10 if context.smoke else CAPTURE_MIN_CALLS))
	count = 2 if context.smoke else CAPTURE_CASES
	return sorted(random.Random(CORPUS_SEED).sample(eligible, min(count, len(eligible))))


def reduce_capture(full: Path, reduced: Path, keep: int) -> int:
	"""Keep at most `keep` "begin" records, evenly spaced, in their original order."""
	begins = [line for line in full.read_text().splitlines() if '"event":"begin"' in line]
	if len(begins) > keep:
		step = len(begins) / keep
		begins = [begins[int(index * step)] for index in range(keep)]
	reduced.write_text("\n".join(begins) + ("\n" if begins else ""))
	return len(begins)


def capture_step(context: Context, step: str, *, cycle: bool) -> dict:
	directory = context.step_dir(step)
	names = capture_cases(context)
	seconds = 5.0 if context.smoke else 120.0
	# One call in `every`, so that about CAPTURE_TARGET calls are captured per case
	# (from the call counts of bnb-cycle, also for the path mode).
	calls = bnb_calls(context)
	every = {name: max(1, min(calls.get(name, 0), CAPTURE_MAX_CALLS) // CAPTURE_TARGET) for name in names}
	tasks = []
	for name in names:
		tasks += bnb_tasks(context, directory, [name], cycle=cycle, seconds=seconds, max_calls=CAPTURE_MAX_CALLS,
			capture_every=every[name])
	failures = run_parallel(context, tasks, step)
	kept = {}
	for name in names:
		full, reduced = directory / "captures" / f"{name}.jsonl", directory / f"{name}.jsonl"
		if full.exists():
			kept[name] = reduce_capture(full, reduced, REPLAY_CALLS)
			full.unlink()  # the full capture can reach hundreds of MB; the reduced one is what replays
		elif reduced.exists():
			kept[name] = sum(1 for _ in reduced.open())
	return {"cases": names, "calls_kept": kept, "max_calls": CAPTURE_MAX_CALLS, "every": every,
		"failures": failures}


def replay_step(context: Context, step: str, source: str, tool: str) -> dict:
	directory = context.step_dir(step)
	tasks = [{"command": [str(context.tools[tool]), "--replay", str(capture)], "stdout": directory / f"{capture.stem}.jsonl"}
		for capture in sorted((context.out / source).glob("*.jsonl"))]
	return {"captures": len(tasks), "failures": run_parallel(context, tasks, step)}


def step_vertex_scaling(context: Context) -> dict:
	directory = context.step_dir("vertex-scaling")
	instances = cycle_instances(context.smoke)
	tasks = []
	for item in instances:
		path = directory / "inputs" / f"{item['name']}.json"
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps({"instances": [item]}))
		tasks.append({"command": [str(context.tools["tpp-convex-vertex-scaling"]), str(path), "1" if context.smoke else "3"],
			"stdout": directory / f"{item['name']}.jsonl"})
	tasks.sort(key=lambda task: -sum(len(polygon) for polygon in json.loads(Path(task["command"][1]).read_text())["instances"][0]["polygons"]))
	return {"instances": len(instances), "failures": run_parallel(context, tasks, "vertex-scaling")}


def step_anchor_probes(context: Context) -> dict:
	directory = context.step_dir("anchor-probes")
	samples, seconds = ("2", "5") if context.smoke else ("8", "30")
	tasks = [{"command": [str(context.tools["tpp-convex-anchor-probes"]), samples, seconds, str(capture)],
		"stdout": directory / f"{capture.stem}.jsonl"} for capture in sorted((context.out / "capture-cycle").glob("*.jsonl"))]
	return {"captures": len(tasks), "failures": run_parallel(context, tasks, "anchor-probes")}


def step_no_polish(context: Context) -> dict:
	directory = context.step_dir("no-polish")
	seconds = 5.0 if context.smoke else 60.0
	names = list(context.cases())
	tasks = []
	# Interleaved by case, so both variants see the same machine state.
	for name in names:
		tasks += bnb_tasks(context, directory / "default", [name], cycle=True, seconds=seconds)
		tasks += bnb_tasks(context, directory / "no-polish", [name], cycle=True, seconds=seconds, extra=["--no-cycle-polish"])
	return {"seconds": seconds, "failures": run_parallel(context, tasks, "no-polish")}


# --------------------------------------------------------------------------- summary

def jsonl(paths) -> list[dict]:
	rows = []
	for path in paths:
		for line in Path(path).read_text().splitlines():
			if line.strip().startswith("{"):
				try:
					rows.append(json.loads(line))
				except json.JSONDecodeError:
					pass
	return rows


def median(values) -> float | None:
	values = [value for value in values if value is not None and math.isfinite(value)]
	return statistics.median(values) if values else None


def geometric_mean(values) -> float | None:
	values = [value for value in values if value and value > 0 and math.isfinite(value)]
	return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def below_lower(rows, ours_key: str) -> dict:
	"""How often Gurobi's objective falls below our proved lower bound, and by how much (relative)."""
	gaps = []
	for row in rows:
		lower, objective = row[ours_key].get("lower"), row["gurobi"].get("objective")
		if lower is not None and objective is not None and objective < lower:
			gaps.append((lower - objective) / max(1.0, abs(lower)))
	return {"count": len(gaps), "of": len(rows), "max_relative": max(gaps) if gaps else 0.0}


def count_by(values) -> dict:
	out: dict = {}
	for value in values:
		out[value] = out.get(value, 0) + 1
	return dict(sorted(out.items()))


def summarize(context: Context) -> dict:
	out = context.out
	summary: dict = {"output": str(out), "machine": workspace.machine_state(), "git": workspace.git_state(),
		"smoke": context.smoke}
	paths = jsonl(sorted((out / "paths-random").glob("shard-*.jsonl")))
	if paths:
		groups: dict = {}
		for row in paths:
			family = row["name"].split("-k")[0]
			groups.setdefault((family, row["k"], row["vertices"] // max(1, row["k"])), []).append(row)
		table = []
		for (family, k, per_polygon), rows in sorted(groups.items()):
			table.append({"family": family, "k": k, "vertices_per_polygon": per_polygon, "rows": len(rows),
				"certified_mean_s": median(r["certified"]["mean"] for r in rows),
				"tolerant_mean_s": median(r["tolerant"]["mean"] for r in rows),
				"lazy_mean_s": median(r["lazy"]["mean"] for r in rows),
				"eager_mean_s": median(r["eager"]["mean"] for r in rows),
				"gurobi_total_s": median(r["gurobi"]["total"] for r in rows),
				"gurobi_over_certified": geometric_mean(r["gurobi"]["total"] / r["certified"]["mean"] for r in rows if r["certified"]["mean"]),
				"lazy_cones": median(r["lazy"]["cones"] for r in rows), "eager_cones": median(r["eager"]["cones"] for r in rows),
				"certified_closed_by": count_by(r["certified"]["closed_by"] for r in rows),
				"certified_optimal": sum(1 for r in rows if r["certified"]["lower"] == r["certified"]["upper"]
					or (r["certified"]["upper"] is not None and r["certified"]["upper"] - r["certified"]["lower"] <= 4e-16 * abs(r["certified"]["upper"]))),
				"gurobi_below_lower": below_lower(rows, "certified")})
		summary["paths_random"] = table
	for step, key in (("replay-path", "replay_path"), ("replay-cycle", "replay_cycle")):
		rows = jsonl(sorted((out / step).glob("*.jsonl")))
		if rows:
			summary[key] = {"calls": len(rows), "ours_median_s": median(r["ours"]["seconds"] for r in rows),
				"gurobi_median_s": median(r["gurobi"]["total"] for r in rows),
				"gurobi_over_ours": geometric_mean(r["gurobi"]["total"] / r["ours"]["seconds"] for r in rows if r["ours"]["seconds"]),
				"closed_by": count_by(r["ours"].get("closed_by") or ("interval" if r["ours"].get("interval_closed")
					else "polish" if r["ours"].get("polish_closed") else "exact") for r in rows),
				"gurobi_below_lower": below_lower(rows, "ours")}
	rows = [row for row in (read_result(path) for path in (out / "bnb-cycle" / "results").glob("*.json")) if row]
	if rows:
		total = sum(r.get("convex_calls", r.get("calls", 0)) for r in rows)
		import fekete_reference
		closed = sum(1 for r in rows if r.get("lower_bound") is not None and r.get("upper_bound") is not None
			and r["upper_bound"] <= (1 + fekete_reference.EPS) * r["lower_bound"])
		summary["bnb_cycle"] = {"cases": len(rows), "closed_ub_le_1.001_lb": closed, "oracle_calls": total,
			"closed_by_construction": sum(r.get("oracle_interval_bound_calls", 0) for r in rows),
			"closed_by_polish": sum(r.get("oracle_float_calls", 0) for r in rows),
			"exact_path": sum(r.get("oracle_rational_calls", 0) for r in rows)}
	rows = jsonl(sorted((out / "vertex-scaling").glob("*.jsonl")))
	if rows:
		groups = {}
		for row in rows:
			groups.setdefault((row["k"], row["largest"]), []).append(row)
		summary["vertex_scaling"] = [{"k": k, "largest": largest, "rows": len(group),
			"ours_ids_mean_s": median(r["ids"]["mean"] for r in group), "ours_cold_s": median(r["cold"]["seconds"] for r in group),
			"gurobi_total_s": median(r["gurobi"]["total"] for r in group)} for (k, largest), group in sorted(groups.items())]
	rows = jsonl(sorted((out / "anchor-probes").glob("*.jsonl")))
	if rows:
		summary["anchor_probes"] = {"calls": len(rows), "keys": sorted(rows[0].keys())}
	variants = {}
	for variant in ("default", "no-polish"):
		variants[variant] = {path.stem: read_result(path) for path in (out / "no-polish" / variant / "results").glob("*.json")}
	if variants["default"]:
		both = [name for name in variants["default"] if variants["default"][name] and variants["no-polish"].get(name)]
		summary["no_polish"] = {"cases": len(both), "time_ratio_geometric": geometric_mean(
			variants["no-polish"][name].get("seconds", 0) / variants["default"][name].get("seconds", 1) for name in both
			if variants["default"][name].get("seconds"))}
	return summary


def write_summary(context: Context) -> dict:
	summary = summarize(context)
	(context.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
	lines = [f"# Paper 1 experiments: {context.out.name}", "",
		f"Machine: {summary['machine'].get('cpu_model')} ({summary['machine'].get('host')}); "
		f"commit {(summary['git'].get('revision') or '?')[:10]}{' (dirty)' if summary['git'].get('dirty') else ''}.", ""]
	for key, value in summary.items():
		if key in ("output", "machine", "git", "smoke"):
			continue
		lines += [f"## {key}", "", "```json", json.dumps(value, indent=1, default=str), "```", ""]
	(context.out / "summary.md").write_text("\n".join(lines))
	# Everything but the inputs and logs, to bring back to the Mac.
	archive = context.out.parent / f"{context.out.name}.tar.gz"
	with tarfile.open(archive, "w:gz") as tar:
		for path in sorted(context.out.rglob("*")):
			if path.is_file() and "inputs" not in path.parts and not path.name.endswith((".log", ".done")):
				tar.add(path, arcname=str(path.relative_to(context.out.parent)))
	return {"archive": str(archive)}


# --------------------------------------------------------------------------- driver

STEPS: list[tuple[str, Callable[[Context], dict]]] = [
	("build", step_build),
	("check", step_check),
	("paths-random", step_paths_random),
	("bnb-cycle", step_bnb_cycle),
	("capture-cycle", lambda context: capture_step(context, "capture-cycle", cycle=True)),
	("replay-cycle", lambda context: replay_step(context, "replay-cycle", "capture-cycle", "tpp-convex-vertex-scaling")),
	("capture-path", lambda context: capture_step(context, "capture-path", cycle=False)),
	("replay-path", lambda context: replay_step(context, "replay-path", "capture-path", "tpp-convex-path-benchmark")),
	("vertex-scaling", step_vertex_scaling),
	("anchor-probes", step_anchor_probes),
	("no-polish", step_no_polish),
	("summary", write_summary),
]


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(prog="tpp.py paper1-experiments", description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--smoke", action="store_true", help="small sizes, a few minutes (separate output directory)")
	parser.add_argument("--plan", action="store_true", help="print the steps, sizes and output directory, and stop")
	parser.add_argument("--only", action="append", choices=[name for name, _ in STEPS], help="run only this step (repeatable)")
	parser.add_argument("--redo", action="append", choices=[name for name, _ in STEPS], default=[],
		help="run this step again even if it is done (repeatable)")
	parser.add_argument("--workers", type=int, help="parallel processes (default: one per physical core)")
	parser.add_argument("--output", help="output directory (default: workspace/experiments/paper1/<host>-<commit>)")
	parser.add_argument("--foreground", action="store_true", help="run here instead of starting a detached job")
	parser.add_argument("--archive", help="the 558-case archive (default: Fekete's upstream checkout, from fekete-setup)")
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse(argv)
	if not args.foreground and not args.plan:
		import jobs
		return jobs.main(["start", "--name", "paper1-experiments", "--", "paper1-experiments", "--foreground",
			*(argv if argv is not None else sys.argv[1:])])
	context = Context(args)
	steps = [(name, function) for name, function in STEPS if not args.only or name in args.only]
	if args.plan:
		print(f"output: {context.out}\nworkers: {context.workers} (cores: {len(context.cores) or 'no pinning'})")
		print(f"paths-random: {len(path_instances(args.smoke))} instances; vertex-scaling: {len(cycle_instances(args.smoke))}")
		for name, _ in steps:
			print(f"  {name}{' (done)' if (context.out / name / 'done.json').exists() else ''}")
		return 0
	context.out.mkdir(parents=True, exist_ok=True)
	(context.out / "plan.json").write_text(json.dumps({"argv": list(argv or sys.argv[1:]), "git": workspace.git_state(),
		"machine": workspace.machine_state(), "workers": context.workers, "smoke": args.smoke,
		"steps": [name for name, _ in steps]}, indent=2) + "\n")
	context.log(f"paper1-experiments: {context.out} ({context.workers} workers)")
	began = time.monotonic()
	for name, function in steps:
		done = context.out / name / "done.json"
		if done.exists() and name not in args.redo and name != "summary":
			context.log(f"{name}: done before, skipped")
			if name == "build":
				load_tools(context)
			if name == "check":
				license_file = json.loads(done.read_text()).get("license_file")
				if license_file:
					os.environ["GRB_LICENSE_FILE"] = license_file
			continue
		if name in args.redo and name not in ("build", "summary"):
			import shutil
			shutil.rmtree(context.out / name, ignore_errors=True)  # its per-case results too
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
