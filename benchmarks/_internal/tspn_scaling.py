"""TSPN scaling campaign: random nonconvex regions of growing count, B&B and ILS on one budget.

Formulation: closed tour, free cyclic order, no fixed point (``tpp-unordered --cycle``),
over n simple nonconvex polygons with the same number of vertices each.

Instances (``generate_instance``): each region is a random star-shaped polygon around
its centre, with VERTICES vertices at increasing angles (angular gaps uniform in
[0.5, 1.5] before normalization, so no two vertices are closer than a third of the mean
gap) and radii uniform in [INNER_RADIUS, 1] times the region's radius, itself uniform
in [0.5, 1]. Convex draws are rejected, so every region has a reflex vertex. The plate
is the square [0, L]^2 with L chosen so the regions' total area is COVERAGE times the
plate's area, and the centres are uniform on the plate. Density is therefore the same at
every n; regions may overlap or contain each other. Coordinates lie on the grid 2^-24
(exact in binary64 and in the text the solver reads), and simplicity and nonconvexity
are checked exactly on that grid. The polygon of (n, seed) depends only on those two
numbers and the generator's parameters.

Phases, one single-threaded process per run and the same wall limit SECONDS for both:
- ``bnb``: the B&B from scratch with the maintained defaults (its own initial heuristic,
  eager evaluation with dives) and the flags of the Fekete comparison;
- ``ils``: the same binary with ``--primal-ils 0.99``: the iterated local search gets 99%
  of the limit, and the B&B the rest, which also gives the run a (weak) lower bound.
Every tour is checked independently against the original regions, every run records its
incumbent history and its bound trajectory (one point per progress report), and rerunning
the same command resumes. ``--report`` rewrites report.md/report.json and the figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import cpu_affinity
import workspace
from live_progress import LiveStatus
from paula_campaign import KEEP, default_frontier_mib, default_workers, fail, finite, prepare_native, read_rows
from unordered_runner import interrupt_running_solvers, run_unordered_solver, terminate_running_solvers
from unordered_validation import validate_cycle

Point = tuple[float, float]

VERTICES = 8
COVERAGE = 0.2
INNER_RADIUS = 0.35
GRID = 2 ** 24
GENERATOR = "star-uniform-v1"

DEFAULT_CAMPAIGN = "tspn-scaling-v1"
DEFAULT_SIZES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150)
DEFAULT_SEEDS = 10
DEFAULT_SECONDS = 600.0
RELATIVE_GAP = 1e-6
# The corpus comparison's stopping rule (Fekete's eps = 0.001), read from the bound trajectory.
LOOSE_GAP = 1e-3
VALIDATION_TOLERANCE = 1e-7
# Flags of the Fekete comparison (fekete_reference.OUR_ARGUMENTS) with a 1e-6 relative gap.
BASE_ARGUMENTS = (
	"--cycle", "--absolute-gap", "0", "--relative-gap", repr(RELATIVE_GAP), "--feasibility-tolerance", "1e-08",
	"--cycle-optimization", "cache", "--cycle-optimization", "features",
	"--cycle-optimization", "root", "--cycle-optimization", "interval",
)
ILS_ARGUMENTS = ("--primal-ils", "0.99", "--primal-ils-seed", "0")
PHASES = ("bnb", "ils")
PROGRESS_INTERVAL = 2.0
RUNS_FILE = "runs.jsonl"
PROTOCOL_FILE = "protocol.json"
SHUTDOWN_BEFORE_START = "shutdown requested before solver start"


def _cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
	return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(a, b, c, d) -> bool:
	"""Closed segments ab and cd share a point (exact on integers)."""
	d1, d2, d3, d4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
	if ((d1 > 0) != (d2 > 0) and d1 and d2) and ((d3 > 0) != (d4 > 0) and d3 and d4):
		return True
	def on(p, q, r):  # r on segment pq, given collinear
		return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
	return (not d1 and on(a, b, c)) or (not d2 and on(a, b, d)) or (not d3 and on(c, d, a)) or (not d4 and on(c, d, b))


def simple_ccw_nonconvex(ring: Sequence[tuple[int, int]]) -> bool:
	"""A simple counterclockwise polygon with at least one reflex vertex and no collinear one."""
	k = len(ring)
	twice_area = sum(ring[i - 1][0] * ring[i][1] - ring[i][0] * ring[i - 1][1] for i in range(k))
	turns = [_cross(ring[i - 1], ring[i], ring[(i + 1) % k]) for i in range(k)]
	if twice_area <= 0 or 0 in turns or all(turn > 0 for turn in turns):
		return False
	edges = [(ring[i], ring[(i + 1) % k]) for i in range(k)]
	for i in range(k):
		for j in range(i + 2, k):
			if i == 0 and j == k - 1:
				continue  # adjacent through the closing edge
			if _segments_cross(*edges[i], *edges[j]):
				return False
	return True


def _star(rng: random.Random, vertices: int, radius: float) -> list[Point]:
	gaps = [rng.uniform(0.5, 1.5) for _ in range(vertices)]
	scale = 2 * math.pi / sum(gaps)
	angle = rng.uniform(0, 2 * math.pi)
	points = []
	for gap in gaps:
		r = radius * rng.uniform(INNER_RADIUS, 1.0)
		points.append((r * math.cos(angle), r * math.sin(angle)))
		angle += gap * scale
	return points


def _area(points: Sequence[Point]) -> float:
	return abs(sum(points[i - 1][0] * points[i][1] - points[i][0] * points[i - 1][1] for i in range(len(points)))) / 2


def generate_instance(n: int, seed: int, *, vertices: int = VERTICES, coverage: float = COVERAGE) -> dict:
	"""The n regions of instance (n, seed) and the plate side, deterministic in its arguments."""
	rng = random.Random(f"{GENERATOR}:{vertices}:{coverage!r}:{n}:{seed}")
	shapes = []
	while len(shapes) < n:
		shape = _star(rng, vertices, rng.uniform(0.5, 1.0))
		# Rejection on the shape alone (at the origin, on the grid): convex draws are redrawn.
		if simple_ccw_nonconvex([(round(x * GRID), round(y * GRID)) for x, y in shape]):
			shapes.append(shape)
	side = math.sqrt(sum(_area(shape) for shape in shapes) / coverage)
	polygons = []
	for shape in shapes:
		while True:
			cx, cy = rng.uniform(0, side), rng.uniform(0, side)
			ring = [(round((cx + x) * GRID), round((cy + y) * GRID)) for x, y in shape]
			if simple_ccw_nonconvex(ring):  # translation by a rounded centre keeps it, barring rounding ties
				break
		polygons.append([(x / GRID, y / GRID) for x, y in ring])
	return {"n": n, "seed": seed, "vertices": vertices, "coverage": coverage, "side": side,
		"generator": GENERATOR, "polygons": polygons}


def instance_name(n: int, seed: int) -> str:
	return f"n{n:03d}-s{seed:02d}"


def instance_digest(instance: dict) -> str:
	return hashlib.sha256(json.dumps(instance["polygons"]).encode()).hexdigest()


def load_instances(output: Path, sizes: Sequence[int], seeds: int, vertices: int, coverage: float) -> dict[str, dict]:
	"""Generate the instances into ``instances/``; a stored file must match its regeneration."""
	directory = output / "instances"
	directory.mkdir(parents=True, exist_ok=True)
	instances = {}
	for n in sizes:
		for seed in range(seeds):
			name = instance_name(n, seed)
			instance = generate_instance(n, seed, vertices=vertices, coverage=coverage)
			instance["sha256"] = instance_digest(instance)
			path = directory / f"{name}.json"
			try:
				stored = json.loads(path.read_text())
			except (OSError, ValueError):
				path.write_text(json.dumps(instance) + "\n")
			else:
				if stored.get("sha256") != instance["sha256"]:
					fail(f"{path} differs from what the generator gives now; use a new --campaign.")
			instances[name] = instance
	return instances


# --- protocol ---------------------------------------------------------------


def protocol_of(args: argparse.Namespace) -> dict:
	return {
		"formulation": "TSPN, closed tour, free cyclic order, no fixed point",
		"generator": GENERATOR,
		"vertices": args.vertices,
		"coverage": args.coverage,
		"inner_radius": INNER_RADIUS,
		"grid": GRID,
		"seconds": args.seconds,
		"base_arguments": list(BASE_ARGUMENTS),
		"ils_arguments": list(ILS_ARGUMENTS),
		"relative_gap": RELATIVE_GAP,
		"validation_tolerance": VALIDATION_TOLERANCE,
	}


def check_protocol(output: Path, protocol: dict) -> None:
	"""A campaign keeps one protocol; more sizes or seeds may be added to it."""
	try:
		previous = json.loads((output / PROTOCOL_FILE).read_text())
	except (OSError, ValueError):
		return
	if previous != protocol:
		fail(f"{output} was started with another protocol ({output / PROTOCOL_FILE}); use a new --campaign.")


def solver_arguments(phase: str, frontier: int) -> list[str]:
	return [*BASE_ARGUMENTS, "--max-frontier-mib", str(frontier), *(ILS_ARGUMENTS if phase == "ils" else ())]


# --- per-instance quantities -------------------------------------------------


def closed(row: dict | None) -> bool:
	"""The run closed the requested 1e-6 gap with an independently valid tour."""
	return bool(row and row.get("exact") and row.get("valid"))


def time_to_length(row: dict, target: float) -> float | None:
	"""Seconds until the run's incumbent first came within the 1e-6 gap of ``target``."""
	limit = target * (1 + RELATIVE_GAP)
	return next((t for t, length in row.get("history", []) if length <= limit), None)


def time_to_gap(row: dict, gap: float) -> float | None:
	"""Seconds until the run's proved gap first reached ``gap`` (resolution: the progress interval)."""
	hit = next((t for t, value, _ in row.get("trajectory", []) if value is not None and value <= gap), None)
	if hit is not None:
		return hit
	final = row.get("final_gap")
	return row.get("seconds") if final is not None and final <= gap else None


def instance_summary(rows: Sequence[dict]) -> dict:
	"""Both runs of one instance: the best bound of either, the optimum if either closed."""
	by_phase = {row["phase"]: row for row in rows}
	bnb, ils = by_phase.get("bnb"), by_phase.get("ils")
	lowers = [row["lower_bound"] for row in rows if finite(row.get("lower_bound"))]
	lengths = [row["length"] for row in rows if row.get("valid") and finite(row.get("length"))]
	optimum = min((row["length"] for row in rows if closed(row)), default=None)
	item = {"bnb": bnb, "ils": ils, "lower": max(lowers, default=None), "best": min(lengths, default=None),
		"optimum": optimum}
	if ils and ils.get("valid") and finite(ils.get("length")) and finite(item["lower"]) and item["lower"] > 0:
		item["ils_excess_lower"] = ils["length"] / item["lower"] - 1
	if bnb and bnb.get("valid") and finite(bnb.get("length")) and finite(item["lower"]) and item["lower"] > 0:
		item["bnb_excess_lower"] = bnb["length"] / item["lower"] - 1
	if optimum is not None and ils and ils.get("valid"):
		item["ils_excess_optimum"] = ils["length"] / optimum - 1
		item["ils_found_optimum"] = ils["length"] <= optimum * (1 + RELATIVE_GAP)
		item["ils_time_to_optimum"] = time_to_length(ils, optimum)
	if bnb and ils and bnb.get("valid") and ils.get("valid"):
		band = RELATIVE_GAP * min(bnb["length"], ils["length"])
		item["ils_vs_bnb"] = ("better" if ils["length"] < bnb["length"] - band
			else "worse" if ils["length"] > bnb["length"] + band else "equal")
	return item


# --- report -----------------------------------------------------------------


def quartiles(values: Sequence[float]) -> tuple[float, float, float] | None:
	if not values:
		return None
	if len(values) == 1:
		return (values[0],) * 3
	q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
	return q1, median, q3


def summarize(rows: Sequence[dict], protocol: dict) -> dict:
	limit = protocol["seconds"]
	groups: dict[str, list[dict]] = {}
	for row in rows:
		if row.get("complete"):
			groups.setdefault(row["name"], []).append(row)
	instances = {name: instance_summary(group) for name, group in groups.items()}
	sizes = sorted({row["n"] for row in rows if row.get("complete")})
	table = []
	for n in sizes:
		items = [item for item in instances.values() if (item["bnb"] or item["ils"])["n"] == n]
		bnb = [item["bnb"] for item in items if item["bnb"]]
		ils = [item["ils"] for item in items if item["ils"]]
		solved = [row for row in bnb if closed(row)]
		open_runs = [row for row in bnb if not closed(row)]
		# A run that did not close counts at the limit: quartiles there are lower bounds.
		times = sorted(row["seconds"] if closed(row) else limit for row in bnb)
		loose = [time_to_gap(row, LOOSE_GAP) for row in bnb]
		entry = {
			"n": n,
			"bnb_runs": len(bnb),
			"bnb_closed": len(solved),
			"bnb_closed_loose": sum(value is not None for value in loose),
			"bnb_seconds": quartiles(times),
			"bnb_seconds_closed": quartiles(sorted(row["seconds"] for row in solved)),
			"bnb_loose_seconds": quartiles(sorted(limit if value is None else value for value in loose)),
			"bnb_nodes": quartiles(sorted(row["nodes"] for row in bnb if finite(row.get("nodes")))),
			"bnb_nodes_closed": quartiles(sorted(row["nodes"] for row in solved)),
			"bnb_open_gap": quartiles(sorted(row["final_gap"] for row in open_runs if finite(row.get("final_gap")))),
			"bnb_excess_lower": quartiles(sorted(item["bnb_excess_lower"] for item in items if "bnb_excess_lower" in item)),
			"ils_runs": len(ils),
			"ils_excess_lower": quartiles(sorted(item["ils_excess_lower"] for item in items if "ils_excess_lower" in item)),
			"with_optimum": sum(item["optimum"] is not None for item in items),
			"ils_found_optimum": sum(bool(item.get("ils_found_optimum")) for item in items),
			"ils_excess_optimum": quartiles(sorted(item["ils_excess_optimum"] for item in items if "ils_excess_optimum" in item)),
			"ils_time_to_optimum": quartiles(sorted(item["ils_time_to_optimum"] for item in items
				if item.get("ils_time_to_optimum") is not None)),
			"ils_vs_bnb": {key: sum(item.get("ils_vs_bnb") == key for item in items) for key in ("better", "equal", "worse")},
			"invalid_runs": sum(row.get("valid") is False for row in [*bnb, *ils]),
			"memory_stops": sum("memory_guard_bytes" in row for row in [*bnb, *ils]),
		}
		table.append(entry)
	return {"protocol": protocol, "sizes": table, "instances": {
		name: {key: value for key, value in item.items() if key not in ("bnb", "ils")} for name, item in sorted(instances.items())}}


def markdown(report: dict, machines: Sequence[str]) -> str:
	protocol = report["protocol"]

	def triple(values, fmt) -> str:
		return "–" if values is None else " / ".join(fmt(value) for value in values)

	def seconds(value: float) -> str:
		return f"≥{protocol['seconds']:g}" if value >= protocol["seconds"] else f"{value:.3g}"

	def percent(value: float) -> str:
		return f"{100 * value:.3g}%"

	lines = [
		"# TSPN scaling: B&B and ILS on random nonconvex regions",
		"",
		f"**Formulation.** {protocol['formulation']}. Instances `{protocol['generator']}`: n random star-shaped polygons "
		f"with {protocol['vertices']} vertices and a reflex vertex each, centres uniform on a square whose area is "
		f"1/{protocol['coverage']:g} times the regions' total area (same density at every n), coordinates on the grid "
		f"2^-{int(math.log2(protocol['grid']))}; regions may overlap.",
		"",
		f"**Protocol.** Each instance runs twice, one single-threaded process each, at most {protocol['seconds']:g} s: "
		f"the B&B from scratch (`{' '.join(protocol['base_arguments'])}`) and the same binary with "
		f"`{' '.join(protocol['ils_arguments'])}` (ILS for 99% of the limit, then the B&B).",
		"",
		f"**Machines.** {'; '.join(machines) or 'not recorded'}.",
		"",
		f"**Tolerances and exactness.** \"Closed\" means the B&B proved a relative gap of at most "
		f"{protocol['relative_gap']:g} with a tour that passes the independent check (binary-rational intersection and "
		f"containment tests, distance diagnostic {protocol['validation_tolerance']:g}); that tour is the optimum within "
		f"that gap. The {LOOSE_GAP:g} column is the time the same run's proved gap first reached {LOOSE_GAP:g} (the "
		f"corpus comparison's eps), read from progress reports every {PROGRESS_INTERVAL:g} s. A run that did not close "
		f"counts at the limit, so quartiles marked ≥ are lower bounds. ILS quality is measured against the best "
		f"certified lower bound of either run, and against the optimum where a run closed.",
		"",
		"## Per size (quartiles q1 / median / q3)",
		"",
		"| n | B&B closed | B&B s | nodes | gap ≤ 0.1% | open gap | ILS = opt | ILS over opt | ILS s to opt | ILS over LB | B&B UB over LB | ILS vs B&B UB (b/e/w) |",
		"|---|---|---|---|---|---|---|---|---|---|---|---|",
	]
	for entry in report["sizes"]:
		lines.append("| " + " | ".join([
			str(entry["n"]),
			f"{entry['bnb_closed']}/{entry['bnb_runs']}",
			triple(entry["bnb_seconds"], seconds),
			triple(entry["bnb_nodes"], lambda value: f"{value:.3g}"),
			f"{entry['bnb_closed_loose']}/{entry['bnb_runs']}",
			triple(entry["bnb_open_gap"], percent),
			f"{entry['ils_found_optimum']}/{entry['with_optimum']}",
			triple(entry["ils_excess_optimum"], percent),
			triple(entry["ils_time_to_optimum"], lambda value: f"{value:.3g}"),
			triple(entry["ils_excess_lower"], percent),
			triple(entry["bnb_excess_lower"], percent),
			"/".join(str(entry["ils_vs_bnb"][key]) for key in ("better", "equal", "worse")),
		]) + " |")
	invalid = sum(entry["invalid_runs"] for entry in report["sizes"])
	memory = sum(entry["memory_stops"] for entry in report["sizes"])
	lines += ["", f"Invalid tours: {invalid}. Runs stopped by the memory guard: {memory}.", ""]
	return "\n".join(lines)


def write_figure(report: dict, rows: Sequence[dict], path: Path) -> None:
	"""Four panels against n: B&B time, nodes, fraction closed, tour quality."""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	limit = report["protocol"]["seconds"]
	sizes = report["sizes"]
	if not sizes:
		return
	blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
	plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
		"ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
	figure, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
	ns = [entry["n"] for entry in sizes]

	def band(axis, key, color, label, scale=1.0):
		points = [(entry["n"], entry[key]) for entry in sizes if entry[key] is not None]
		if not points:
			return
		x = [n for n, _ in points]
		axis.fill_between(x, [q[0] * scale for _, q in points], [q[2] * scale for _, q in points], color=color,
			alpha=0.18, linewidth=0)
		axis.plot(x, [q[1] * scale for _, q in points], color=color, linewidth=2, marker="o", markersize=4, label=label)

	def runs(axis, field, color):
		done = [(row["n"], row[field]) for row in rows if row.get("phase") == "bnb" and row.get("complete") and closed(row)]
		left = [(row["n"], row[field] if field == "nodes" else limit) for row in rows
			if row.get("phase") == "bnb" and row.get("complete") and not closed(row) and finite(row.get(field))]
		if done:
			axis.scatter(*zip(*done), s=10, color=color, alpha=0.35, linewidths=0)
		if left:
			axis.scatter(*zip(*left), s=22, facecolors="none", edgecolors=color, linewidths=1, label="not closed")

	axis = axes[0][0]
	runs(axis, "seconds", blue)
	band(axis, "bnb_seconds", blue, "B&B: time to close (gap 1e-6)")
	band(axis, "ils_time_to_optimum", orange, "ILS: time to reach the optimum")
	axis.axhline(limit, color=muted, linewidth=1, linestyle="--")
	axis.annotate(f"limit {limit:g} s", (ns[0], limit), xytext=(0, 3), textcoords="offset points", color=muted, fontsize=8)
	axis.set_yscale("log")
	axis.set_ylabel("seconds (log)")
	axis.set_title("Time", loc="left", fontweight="bold")
	axis.legend(frameon=False, fontsize=8, loc="lower right")

	axis = axes[0][1]
	runs(axis, "nodes", blue)
	band(axis, "bnb_nodes", blue, "B&B nodes (at the limit if not closed)")
	axis.set_yscale("log")
	axis.set_ylabel("nodes (log)")
	axis.set_title("Search effort", loc="left", fontweight="bold")
	axis.legend(frameon=False, fontsize=8, loc="lower right")

	axis = axes[1][0]
	axis.plot(ns, [entry["bnb_closed"] / entry["bnb_runs"] for entry in sizes], color=blue, linewidth=2, marker="o",
		markersize=4, label="gap ≤ 1e-6 (closed)")
	axis.plot(ns, [entry["bnb_closed_loose"] / entry["bnb_runs"] for entry in sizes], color=blue, linewidth=1.5,
		linestyle="--", marker="o", markersize=4, markerfacecolor="white", label="gap ≤ 0.1%")
	axis.set_ylim(-0.03, 1.03)
	axis.set_ylabel(f"fraction of instances within {limit:g} s")
	axis.set_title("B&B: instances closed within the limit", loc="left", fontweight="bold")
	axis.legend(frameon=False, fontsize=8, loc="lower left")

	axis = axes[1][1]
	band(axis, "bnb_excess_lower", blue, "B&B tour", scale=100)
	band(axis, "ils_excess_lower", orange, "ILS tour", scale=100)
	axis.set_ylabel("% above the best lower bound")
	axis.set_title(f"Tour at {limit:g} s against the best lower bound", loc="left", fontweight="bold")
	axis.legend(frameon=False, fontsize=8, loc="upper left")

	for axis in axes.flat:
		axis.set_xlabel("regions n")
		axis.grid(True, color=grid, linewidth=0.6)
		axis.set_axisbelow(True)
	figure.suptitle("Median and quartiles per size; dots: individual B&B runs", fontsize=9, color=muted)
	for suffix in (".svg", ".png"):
		figure.savefig(path.with_suffix(suffix), dpi=150)
	plt.close(figure)


def write_report(output: Path, rows: Sequence[dict], protocol: dict) -> dict:
	report = summarize(rows, protocol)
	machines = []
	try:
		for attempt in json.loads((output / workspace.MANIFEST).read_text()).get("attempts", []):
			machine = attempt.get("machine") or {}
			text = f"{machine.get('host')} ({machine.get('cpu_model')}, {machine.get('system')})"
			if text not in machines:
				machines.append(text)
	except (OSError, ValueError):
		pass
	(output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
	(output / "report.md").write_text(markdown(report, machines))
	return report


def per_run_csv(rows: Sequence[dict], path: Path) -> None:
	"""One compact line per run: what results-saved keeps and what the paper's figure needs."""
	import csv

	instances = {}
	for row in rows:
		if row.get("complete"):
			instances.setdefault(row["name"], []).append(row)
	summaries = {name: instance_summary(group) for name, group in instances.items()}
	fields = ("n", "seed", "phase", "termination", "closed", "seconds", "nodes", "calls", "lower_bound", "length",
		"final_gap", "seconds_to_gap_1e-3", "instance_lower_bound", "instance_optimum", "seconds_to_optimum", "valid")
	with path.open("w", newline="") as file:
		writer = csv.writer(file, lineterminator="\n")
		writer.writerow(fields)
		for row in sorted((row for row in rows if row.get("complete")), key=lambda row: (row["n"], row["seed"], row["phase"])):
			item = summaries[row["name"]]
			optimum = item["optimum"]
			writer.writerow([row["n"], row["seed"], row["phase"], row.get("termination"), int(closed(row)),
				f"{row['seconds']:.3f}" if finite(row.get("seconds")) else "",
				row.get("nodes", ""), row.get("calls", ""),
				repr(row["lower_bound"]) if finite(row.get("lower_bound")) else "",
				repr(row["length"]) if finite(row.get("length")) else "",
				f"{row['final_gap']:.3e}" if finite(row.get("final_gap")) else "",
				"" if (value := time_to_gap(row, LOOSE_GAP)) is None else f"{value:.2f}",
				repr(item["lower"]) if finite(item["lower"]) else "", "" if optimum is None else repr(optimum),
				"" if optimum is None or (value := time_to_length(row, optimum)) is None else f"{value:.3f}",
				int(bool(row.get("valid")))])


# --- execution --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="tpp.py tspn-scaling", formatter_class=argparse.RawDescriptionHelpFormatter,
		description=__doc__, epilog=(
			"On the Mac a fresh build needs TPP_CXX_STANDARD=23.\n"
			"Follow: python3 benchmarks/tpp.py live CAMPAIGN; report: tpp.py tspn-scaling --report [--campaign NAME]"))
	parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN, help=f"campaign name or directory (default: {DEFAULT_CAMPAIGN})")
	parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)), help="region counts, comma-separated")
	parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help=f"instances per size (default: {DEFAULT_SEEDS})")
	parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
		help=f"wall limit of every run, B&B and ILS alike (default: {DEFAULT_SECONDS:g})")
	parser.add_argument("--vertices", type=int, default=VERTICES, help=f"vertices per region (default: {VERTICES})")
	parser.add_argument("--coverage", type=float, default=COVERAGE,
		help=f"regions' total area over the plate's (default: {COVERAGE:g})")
	parser.add_argument("--phases", default=",".join(PHASES), help="bnb, ils or bnb,ils (default)")
	parser.add_argument("--workers", type=int, help="concurrent runs (default: half the logical CPUs)")
	parser.add_argument("--frontier-mib", type=int, help="B&B frontier budget per run (default: 60%% of memory / workers)")
	parser.add_argument("--cpu-affinity", choices=cpu_affinity.MODES, default="auto")
	parser.add_argument("--allow-new-binary", action="store_true",
		help="resume with a solver binary different from the one of earlier runs (each row records its binary)")
	parser.add_argument("--verbose", action="store_true", help="print every progress report of the running solvers")
	actions = parser.add_mutually_exclusive_group()
	actions.add_argument("--detach", action="store_true", help="prepare and build here, then run as a background job")
	actions.add_argument("--report", action="store_true", help="only rewrite the report, the figure and per-run.csv")
	actions.add_argument("--dry-run", action="store_true", help="generate the instances, print the plan and exit")
	return parser


def parse_sizes(text: str) -> list[int]:
	try:
		sizes = sorted({int(item) for item in text.split(",") if item.strip()})
	except ValueError:
		fail("--sizes takes comma-separated integers")
	if not sizes or sizes[0] < 3:
		fail("--sizes needs region counts of at least 3")
	return sizes


def run(args: argparse.Namespace, argv: Sequence[str]) -> int:
	output = workspace.campaign_path(args.campaign)
	rows = read_rows(output / RUNS_FILE)
	if args.report:
		try:
			protocol = json.loads((output / PROTOCOL_FILE).read_text())
		except (OSError, ValueError):
			fail(f"{output} has no {PROTOCOL_FILE}; nothing to report")
		write_report(output, rows, protocol)
		write_figure(json.loads((output / "report.json").read_text()), rows, output / "scaling")
		per_run_csv(rows, output / "per-run.csv")
		print((output / "report.md").read_text())
		return 0

	sizes = parse_sizes(args.sizes)
	phases = [phase for phase in PHASES if phase in {item.strip() for item in args.phases.split(",")}]
	if not phases or set(item.strip() for item in args.phases.split(",") if item.strip()) - set(PHASES):
		fail("--phases takes bnb, ils or bnb,ils")
	if args.seeds < 1 or not args.seconds > 0 or args.vertices < 4 or not 0 < args.coverage < 1:
		fail("--seeds and --seconds must be positive, --vertices at least 4 and --coverage in (0, 1)")
	protocol = protocol_of(args)
	check_protocol(output, protocol)
	instances = load_instances(output, sizes, args.seeds, args.vertices, args.coverage)
	workers = args.workers or default_workers()
	frontier = args.frontier_mib or default_frontier_mib(workers)
	done = {(row["phase"], row["name"]) for row in rows if row.get("complete")}
	# Seed by seed, so an interrupted campaign has every size; larger instances first within a seed.
	tasks = [(phase, instance_name(n, seed)) for seed in range(args.seeds) for n in reversed(sizes) for phase in phases
		if (phase, instance_name(n, seed)) not in done]
	worst = len(tasks) * args.seconds / workers / 3600
	print(f"Campaign: {output}")
	print(f"Sizes: {','.join(map(str, sizes))}; seeds: {args.seeds}; phases: {','.join(phases)}; limit {args.seconds:g} s")
	print(f"Pending: {len(tasks)} runs, {workers} at a time: at most {worst:.1f} h; frontier budget {frontier} MiB per run")
	if args.dry_run:
		return 0

	binary, arithmetic = prepare_native()
	if args.detach:
		import jobs

		command = ["tspn-scaling", *[item for item in argv if item != "--detach"]]
		status = jobs.command_start(argparse.Namespace(name="tspn-scaling", command=command))
		print(f"  live: python3 benchmarks/tpp.py live {output}", file=sys.stderr)
		return status

	binary_sha256 = workspace.file_sha256(binary)
	previous = {row.get("binary_sha256") for row in rows} - {None}
	if previous and previous != {binary_sha256} and not args.allow_new_binary:
		fail("the solver binary changed since the earlier runs of this campaign; "
			"use a new --campaign, or --allow-new-binary to mix them (each row records its binary).")
	(output / PROTOCOL_FILE).write_text(json.dumps(protocol, indent=2) + "\n")
	queue = list(tasks)
	lock = threading.Lock()
	live = LiveStatus(output / "live.json", echo=None if args.verbose else (lambda line: None))
	pool = cpu_affinity.CorePool(workers, 1, args.cpu_affinity)
	guard = int(frontier * 2**20 * 1.25) + 2**30
	stopped = threading.Event()
	finished = [0]
	failures = [0]

	def record(row: dict) -> None:
		with lock:
			rows.append(row)
			with (output / RUNS_FILE).open("a") as file:
				file.write(json.dumps(row, allow_nan=False) + "\n")
			finished[0] += 1
			parts = [f"[{finished[0]}/{len(tasks)}] {row['phase']} {row['name']}"]
			if "error" in row:
				parts.append(f"ERROR {row['error']}")
			if finite(row.get("length")):
				parts.append(f"tour {row['length']:.6f}")
			if finite(row.get("final_gap")):
				parts.append(f"gap {100 * row['final_gap']:.4f}%")
			if row.get("valid") is False:
				parts.append("INVALID TOUR")
			parts.append(f"{row.get('termination', 'no result')}, {row.get('seconds', 0) or 0:.1f} s, "
				f"{row.get('nodes', 0) or 0} nodes")
			print(", ".join(parts), flush=True)

	def execute(phase: str, name: str) -> dict | None:
		instance = instances[name]
		polygons = instance["polygons"]
		row = {"phase": phase, "name": name, "n": instance["n"], "seed": instance["seed"], "sha256": instance["sha256"],
			"seconds_limit": args.seconds, "binary_sha256": binary_sha256, "arithmetic": arithmetic,
			"started_at": workspace.now()}
		key = f"{phase}-{name}"
		report = live.reporter(key, f"{phase} {name}", max_seconds=args.seconds, target_gap=RELATIVE_GAP)
		trajectory: list[list] = []

		def progress(raw: dict) -> None:
			report(raw)
			lower, upper = raw.get("lower_bound"), raw.get("upper_bound")
			gap = 1 - lower / upper if finite(lower) and finite(upper) and upper > 0 else None
			trajectory.append([round(raw.get("elapsed_seconds") or 0.0, 2), gap, raw.get("nodes")])

		lease = pool.acquire()
		memory_hits: list[int] = []

		def started(pid: int) -> None:
			live.set_pid(key, pid)
			pool.pin(pid, lease)

		began = time.monotonic()
		try:
			result = run_unordered_solver(binary, (0, 0), (0, 0), polygons, -1, args.seconds,
				solver_arguments(phase, frontier), process_timeout=args.seconds + max(120.0, 0.05 * args.seconds),
				progress=progress, progress_interval=PROGRESS_INTERVAL, on_start=started, max_memory_bytes=guard,
				on_memory_limit=memory_hits.append)
			if result.get("error") == SHUTDOWN_BEFORE_START:
				return None
			row.update({field: result[field] for field in KEEP if field in result})
			history = result.get("incumbent_history") or []
			values = [length for _, length in history if finite(length) and length > 0]
			if values and finite(result.get("upper_bound")):
				# The solver records normalized lengths; the scale maps them back to the input's units.
				scale = result["upper_bound"] / min(values)
				row["history"] = [[round(t, 4), length * scale] for t, length in history]
		except Exception as error:
			partial = live.last(key)
			if partial:
				row.update({field: partial.get(field) for field in ("lower_bound", "upper_bound")})
			row["error"] = f"{type(error).__name__}: {error}"[:500]
		finally:
			live.finish(key)
			if (placement := pool.release(lease)) is not None:
				row["cpu_affinity"] = placement
		row["process_seconds"] = round(time.monotonic() - began, 3)
		row["trajectory"] = trajectory
		if memory_hits:
			row["memory_guard_bytes"] = memory_hits[0]
		if row.get("path"):
			validation = validate_cycle(polygons, row["path"], VALIDATION_TOLERANCE)
			row["valid"] = bool(validation.get("valid"))
			row["validation"] = {key: validation.get(key) for key in ("valid", "recomputed_length", "max_polygon_distance")}
			row["length"] = validation.get("recomputed_length")
			if finite(row.get("lower_bound")) and finite(row["length"]) and row["length"] > 0:
				row["final_gap"] = max(0.0, 1 - row["lower_bound"] / row["length"])
		# A run stopped by Ctrl+C or `jobs stop` is repeated on resume; one stopped by the memory guard is not.
		row["complete"] = "error" not in row and (row.get("termination") != "interrupted" or bool(memory_hits))
		return row

	def worker() -> None:
		while not stopped.is_set():
			with lock:
				if not queue:
					return
				task = queue.pop(0)
			try:
				row = execute(*task)
			except Exception as error:
				row = {"phase": task[0], "name": task[1], "complete": False,
					"error": f"runner: {type(error).__name__}: {error}"[:500]}
			if row is not None:
				if "error" in row or row.get("valid") is False:
					failures[0] += 1
				record(row)

	status = 0
	parameters = {**protocol, "sizes": sizes, "seeds": args.seeds, "phases": phases, "workers": workers,
		"frontier_mib": frontier, "arithmetic": arithmetic, "cpu_affinity": pool.describe()}
	with workspace.recorded_run(output, kind="tspn-scaling", argv=["tspn-scaling", *argv], tools=[binary],
			parameters=parameters) as attempt:
		threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(workers, max(1, len(tasks))))]
		for thread in threads:
			thread.start()
		try:
			while any(thread.is_alive() for thread in threads):
				for thread in threads:
					thread.join(timeout=1.0)
		except KeyboardInterrupt:
			status = 130
			attempt["status"] = "interrupted"
			stopped.set()
			print("Stopping: running solvers save their tours and bounds (Ctrl+C again forces them)...", flush=True)
			interrupt_running_solvers()
			try:
				while any(thread.is_alive() for thread in threads):
					for thread in threads:
						thread.join(timeout=1.0)
			except KeyboardInterrupt:
				terminate_running_solvers()
				for thread in threads:
					thread.join(timeout=30.0)
		finally:
			live.close()
		report = write_report(output, rows, protocol)
	write_figure(report, rows, output / "scaling")
	per_run_csv(rows, output / "per-run.csv")
	print()
	print((output / "report.md").read_text())
	if status:
		print("Interrupted; repeat the same command to resume.")
		return status
	return int(bool(failures[0]))


def main(argv: Sequence[str] | None = None) -> int:
	arguments = list(sys.argv[1:] if argv is None else argv)
	return run(build_parser().parse_args(arguments), arguments)


if __name__ == "__main__":
	raise SystemExit(main())
