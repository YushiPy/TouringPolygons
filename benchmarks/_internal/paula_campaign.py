"""Long TSPN campaign on Paula's instances: seeded ILS runs, then a certifying B&B.

Formulation: closed tour, free cyclic order, no fixed point, over the regions of
the 235 ``npol-le-100`` files of Paula's draft (points, segments and simple
polygons, exactly as in the files). Those files are local and may not be
redistributed: the campaign reads them from ``--source`` and keeps everything
it derives in the ignored workspace.

Phase ``ils`` repeats the protocol of her appendix table with our solver:
SEEDS independent runs per instance (``--primal-ils-seed``), each of at most
ILS_SECONDS, almost all of it in the iterated local search (``--primal-ils
0.99``). Phase ``bnb`` restarts the branch and bound from the shortest valid
tour of any run of the instance and spends BNB_SECONDS raising the certified
lower bound (lazy evaluation without dives, the recommended pipeline for these
instances). Every run is one single-threaded process, pinned to a whole core on
Linux, and every tour is checked independently against the original regions.
Rerunning the same command resumes; ``--detach`` continues it as a background
job that survives the end of the SSH session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import cpu_affinity
import native_build
from convert_paula import parse_dat
from live_progress import LiveStatus
from unordered_runner import (
	interrupt_running_solvers,
	run_unordered_solver,
	terminate_running_solvers,
)
from unordered_validation import validate_cycle

import workspace

ROOT = Path(__file__).resolve().parents[2]
PAULA = ROOT / "third_party/paula-tspn"
DEFAULT_SOURCE = PAULA / "tspn-instances/tspn-instances/npol-le-100"
DEFAULT_REFERENCE = PAULA / "ber2025_tspn/ber2025.tex"
DEFAULT_CAMPAIGN = "paula-long-v1"
EXPECTED_INSTANCES = 235
# Not certified by any run up to 2026-10-09 (60 s with the defaults, then 600 s of
# --primal-ils 0.5 --cycle-optimization lazy --lazy-oracles --dive-interval 0); see
# results-saved/README.md#tspn-search-retune-2026-10-09.
OPEN_INSTANCES = (
	"50eil101", "64lin318", "72vm1084-8x9", "75lin105", "80rd400", "81vm1084-9x9", "88pr439",
	"89pcb442", "99d493", "100C10k.0", "100i1000-410", "100i2000-604", "100i2500-708",
	"100i3000-803", "100nrw1379", "100pcb1173", "100pr1002", "100prb1173-10x10", "100rat783",
	"100rat783-10x10", "100vm1084",
)
BASE_ARGUMENTS = (
	"--cycle", "--absolute-gap", "0", "--relative-gap", "1e-6",
	"--cycle-optimization", "cache", "--cycle-optimization", "features",
	"--cycle-optimization", "root", "--cycle-optimization", "interval",
)
ILS_ARGUMENTS = ("--primal-ils", "0.99")
BNB_ARGUMENTS = ("--cycle-optimization", "lazy", "--lazy-oracles", "--dive-interval", "0")
RELATIVE_GAP = 1e-6
VALIDATION_TOLERANCE = 1e-7
PROGRESS_INTERVAL = 60
RUNS_FILE = "runs.jsonl"
PROTOCOL_FILE = "protocol.json"
SHUTDOWN_BEFORE_START = "shutdown requested before solver start"
# Run once per build before a campaign: the proofs in binary64 must hold with this compiler and CPU.
SOLVER_TESTS = ("tpp-convex-cycle-tests", "tpp-convex-cycle-certificate-tests", "tpp-unordered-tests", "tpp-tspn-tests")
TESTS_MARKER = ".paula-campaign-tests-passed"
# Her table prints 100pcb1173-10x10 for the instance whose file is 100prb1173-10x10 (same GTSP length).
REFERENCE_ALIASES = {"100pcb1173-10x10": "100prb1173-10x10"}
REFERENCE_ROW = re.compile(
	r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*&\s*(\d+)\s*&\s*([\d,]+\.\d+)\s*&\s*([\d,]+\.\d+)\s*&\s*(\d+)\s*&"
	r"\s*([\d,]+\.\d+)\s*&\s*([-+]?[\d.]+)\s*&\s*([-+]?[\d.]+)\s*&\s*([-+]?[\d.]+)\s*\\\\"
)
KEEP = (
	"lower_bound", "upper_bound", "termination", "exact", "seconds", "nodes", "calls", "peak_queue",
	"peak_frontier_bytes", "frontier_strips", "frontier_trims", "frontier_discarded_nodes",
	"frontier_discarded_bound", "frontier_completions", "frontier_completion_nodes", "peak_completion_stack",
	"incumbent_updates", "first_incumbent_seconds", "primal_ils_iterations",
	"primal_ils_improvements", "primal_ils_seconds", "primal_ils_reorder_calls",
	"primal_ils_reorder_improvements", "subtree_threads", "subtree_seeds", "subtree_donations", "subtree_seconds",
	"checkpoint_writes", "checkpoint_resumed", "checkpoint_prior_nodes", "checkpoint_prior_seconds", "path",
)


def fail(message: str) -> None:
	raise SystemExit(f"Error: {message}")


def same_length_band(length: float) -> float:
	"""Her lengths have two decimals; ours also carry the solver's relative gap."""
	return 0.005 + RELATIVE_GAP * abs(length)


def compare(ours: float, reference: float) -> str:
	band = same_length_band(reference)
	return "better" if ours < reference - band else "worse" if ours > reference + band else "equal"


# --- inputs -----------------------------------------------------------------


def load_instances(source: Path) -> dict[str, dict]:
	paths = sorted(source.rglob("*.dat")) if source.is_dir() else []
	if len(paths) != EXPECTED_INSTANCES:
		fail(f"expected the {EXPECTED_INSTANCES} npol-le-100 instance files in {source}; found {len(paths)}.\n"
			"They are not in Git: copy third_party/paula-tspn/ from a machine that has it (see --help).")
	instances = {}
	for path in paths:
		raw = path.read_bytes()
		instances[path.stem] = {"polygons": parse_dat(raw.decode("utf-8")), "sha256": hashlib.sha256(raw).hexdigest()}
	if len(instances) != EXPECTED_INSTANCES:
		fail(f"duplicate instance names in {source}")
	return instances


def parse_reference(text: str) -> dict[str, dict]:
	"""Per instance, the ILS-BCD row of her appendix table: best of ten runs and the gap spread."""
	def number(value: str) -> float:
		return float(value.replace(",", ""))

	rows = {}
	for line in text.splitlines():
		match = REFERENCE_ROW.match(line)
		if not match:
			continue
		name, m, gtsp, best, runs, cpu, low, mean, high = match.groups()
		gtsp_length, best_length = number(gtsp), number(best)
		gaps = [float(low), float(mean), float(high)]
		# Mean and worst come from gaps with three decimals (about 1e-5 relative), so they
		# are clamped to the printed best.
		mean_length = max(best_length, gtsp_length * (1 + gaps[1] / 100))
		rows[REFERENCE_ALIASES.get(name, name)] = {
			"m": int(m), "gtsp": gtsp_length, "best": best_length, "runs_le_gtsp": int(runs),
			"cpu_seconds": number(cpu), "gap_percent": gaps, "mean": mean_length,
			"worst": max(mean_length, gtsp_length * (1 + gaps[2] / 100)),
		}
	return rows


def load_reference(path: Path | None) -> dict[str, dict]:
	if path is None or not path.is_file():
		return {}
	rows = parse_reference(path.read_text(errors="replace"))
	if len(rows) != EXPECTED_INSTANCES:
		print(f"WARNING: {len(rows)} rows read from Paula's table in {path} (expected {EXPECTED_INSTANCES}).", flush=True)
	return rows


def select(value: str, names: Sequence[str]) -> list[str]:
	if value == "open":
		chosen = list(OPEN_INSTANCES)
	elif value == "all":
		chosen = sorted(names)
	else:
		chosen = [item.strip() for item in value.split(",") if item.strip()]
	unknown = sorted(set(chosen) - set(names))
	if unknown:
		fail(f"unknown instance(s): {', '.join(unknown)}")
	if len(set(chosen)) != len(chosen):
		fail("an instance is listed twice")
	return chosen


# --- rows -------------------------------------------------------------------


def read_rows(path: Path) -> list[dict]:
	rows = []
	try:
		lines = path.read_text().splitlines()
	except OSError:
		return rows
	for line in lines:
		try:
			rows.append(json.loads(line))
		except ValueError:
			pass  # a write cut short by a crash; that run is repeated
	return rows


def finite(value) -> bool:
	return isinstance(value, (int, float)) and math.isfinite(value)


def best_tour(rows: Sequence[dict], name: str) -> dict | None:
	"""The shortest independently valid tour of any run of ``name``."""
	tours = [row for row in rows if row.get("name") == name and row.get("valid") and finite(row.get("length"))]
	return min(tours, key=lambda row: row["length"], default=None)


def lower_bound(rows: Sequence[dict], name: str) -> float | None:
	"""Every run searches the whole feasible set, so the largest bound of any run is valid."""
	values = [row["lower_bound"] for row in rows if row.get("name") == name and finite(row.get("lower_bound"))]
	return max(values, default=None)


def completed(rows: Sequence[dict]) -> set[tuple[str, str, int]]:
	return {(row["phase"], row["name"], row["seed"]) for row in rows if row.get("complete")}


def time_to(row: dict, target: float) -> float | None:
	"""Seconds until the run's incumbent first reached ``target`` (within the comparison band)."""
	limit = target + same_length_band(target)
	return next((t for t, length in row.get("history", []) if length <= limit), None)


# --- report -----------------------------------------------------------------


def summarize(rows: Sequence[dict], selection: Sequence[str], reference: dict, protocol: dict) -> dict:
	instances = []
	for name in selection:
		ils = [row for row in rows if row.get("name") == name and row.get("phase") == "ils" and row.get("complete")]
		lengths = [row["length"] for row in ils if row.get("valid") and finite(row.get("length"))]
		best = best_tour(rows, name)
		lower = lower_bound(rows, name)
		item = {"name": name, "m": next((row.get("m") for row in rows if row.get("name") == name), None),
			"ils_runs": len(ils), "invalid_runs": sum(1 for row in rows if row.get("name") == name and row.get("valid") is False),
			"best": best["length"] if best else None, "best_from": f"{best['phase']} seed {best['seed']}" if best else None,
			"lower_bound": lower, "bnb_done": any(row.get("name") == name and row.get("phase") == "bnb" and row.get("complete") for row in rows)}
		item["ils_valid_runs"] = len(lengths)
		if lengths:
			item.update(ils_best=min(lengths), ils_mean=statistics.fmean(lengths), ils_worst=max(lengths))
		if best and finite(lower) and best["length"] > 0:
			item["gap"] = max(0.0, 1 - lower / best["length"])
		elif best and best["length"] == 0:
			item["gap"] = 0.0
		item["certified"] = item.get("gap") is not None and item["gap"] <= RELATIVE_GAP
		paula = reference.get(name)
		if paula:
			item["paula"] = {key: paula[key] for key in ("best", "mean", "worst", "gap_percent", "cpu_seconds")}
			if best:
				item["vs_paula_best"] = compare(best["length"], paula["best"])
				item["delta_percent"] = (best["length"] / paula["best"] - 1) * 100 if paula["best"] > 0 else 0.0
			if lengths:
				item["seeds_reaching_paula_best"] = sum(compare(length, paula["best"]) != "worse" for length in lengths)
				times = [t for t in (time_to(row, paula["best"]) for row in ils if row.get("valid")) if t is not None]
				item["median_seconds_to_paula_best"] = statistics.median(times) if times else None
				item["mean_vs_paula_mean"] = ("better" if item["ils_mean"] < paula["mean"] - 1e-5 * paula["gtsp"]
					else "worse" if item["ils_mean"] > paula["mean"] + 1e-5 * paula["gtsp"] else "equal")
		instances.append(item)
	instances.sort(key=lambda item: (item["m"] or 0, item["name"]))
	compared = [item for item in instances if "vs_paula_best" in item]
	gaps = [item["gap"] for item in instances if item.get("gap") is not None]
	return {
		"protocol": protocol,
		"instances": instances,
		"summary": {
			"instances": len(instances),
			"with_tour": sum(item["best"] is not None for item in instances),
			"ils_runs": sum(item["ils_runs"] for item in instances),
			"bnb_done": sum(item["bnb_done"] for item in instances),
			"bnb_limits": sorted({row["seconds_limit"] for row in rows if row.get("phase") == "bnb" and row.get("seconds_limit")}),
			"certified": sum(item["certified"] for item in instances),
			"median_gap": statistics.median(gaps) if gaps else None,
			"vs_paula_best": {key: sum(item["vs_paula_best"] == key for item in compared) for key in ("better", "equal", "worse")},
			"worse_than_paula": [item["name"] for item in compared if item["vs_paula_best"] == "worse"],
			"better_than_paula": [item["name"] for item in compared if item["vs_paula_best"] == "better"],
			"mean_vs_paula_mean": {key: sum(item.get("mean_vs_paula_mean") == key for item in compared)
				for key in ("better", "equal", "worse")},
			"invalid_runs": sum(item["invalid_runs"] for item in instances),
		},
	}


def markdown(report: dict, machines: Sequence[str], has_reference: bool) -> str:
	protocol, summary = report["protocol"], report["summary"]

	def length(value) -> str:
		return "–" if value is None else f"{value:,.2f}"

	lines = [
		"# Long TSPN campaign on Paula's instances",
		"",
		"**Formulation.** Closed tour, free cyclic order, no fixed point; regions exactly as in the "
		"`npol-le-100` files (points, segments, simple polygons, possibly nonconvex and overlapping).",
		"",
		f"**Protocol.** ILS: {protocol['seeds']} seeds per instance, each run at most {protocol['ils_seconds']:g} s "
		f"(`{' '.join(protocol['ils_arguments'])}`). B&B: "
		f"{' or '.join(f'{limit:g}' for limit in summary.get('bnb_limits') or [protocol['bnb_seconds']])} s per attempt, from the shortest valid tour "
		f"(`{' '.join(protocol['bnb_arguments'])}`). Common flags: `{' '.join(protocol['base_arguments'])}`. "
		"One single-threaded process per run, pinned to a whole core on Linux; several runs at a time.",
		"",
		f"**Machines.** {'; '.join(machines) or 'not recorded'}.",
		"",
		f"**Tolerances and exactness.** Every tour is checked against the original regions with binary-rational "
		f"intersection and containment tests (distance diagnostic tolerance {VALIDATION_TOLERANCE:g}); its length is "
		f"recomputed independently. Lower bounds are the B&B's certified bounds; a relative gap of at most "
		f"{RELATIVE_GAP:g} counts as a certified optimum, anything else is a feasible tour with a proved gap.",
		"",
	]
	if has_reference:
		lines += [
			"**Paula.** Best, mean and worst of ten ILS-BCD runs from her draft's appendix table (lengths with two "
			"decimals, gaps with three); i9-13900, at most 1,200 s of CPU per run. Two lengths count as equal within "
			"0.005 + 1e-6 × length.",
			"",
			"**Limitations.** Different machines and compilers; her times are CPU seconds per run, ours wall seconds "
			"with concurrent runs, so our times are conservative. Her mean and worst are reconstructed from rounded gaps.",
			"",
		]
	lines += [
		"## Summary",
		"",
		f"- Instances: {summary['instances']}; with a valid tour: {summary['with_tour']}; ILS runs recorded: "
		f"{summary['ils_runs']}; B&B runs finished: {summary['bnb_done']}; invalid tours: {summary['invalid_runs']}.",
		f"- Certified optimal: {summary['certified']}; median certified gap: "
		+ ("–" if summary["median_gap"] is None else f"{100 * summary['median_gap']:.2f}%") + ".",
	]
	if has_reference:
		versus = summary["vs_paula_best"]
		lines += [
			f"- Best tour against her best of ten: {versus['better']} better, {versus['equal']} equal, {versus['worse']} worse"
			+ (f" ({', '.join(summary['worse_than_paula'])})" if summary["worse_than_paula"] else "") + ".",
			f"- Mean run against her mean run: {summary['mean_vs_paula_mean']['better']} better, "
			f"{summary['mean_vs_paula_mean']['equal']} equal, {summary['mean_vs_paula_mean']['worse']} worse.",
		]
	lines += ["", "## Per instance", ""]
	header = ["instance", "m", "runs", "best tour", "our mean", "LB", "gap", "status"]
	if has_reference:
		header[4:4] = ["Paula best", "Δ best", "seeds ≤ Paula", "time to Paula (median s)", "Paula mean"]
	lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
	for item in report["instances"]:
		paula = item.get("paula", {})
		cells = [item["name"], str(item["m"] or "–"), str(item["ils_runs"]), length(item["best"]), length(item.get("ils_mean")),
			length(item["lower_bound"]), "–" if item.get("gap") is None else f"{100 * item['gap']:.3f}%",
			"optimal" if item["certified"] else ("open" if item["best"] is not None else "no tour")]
		if has_reference:
			seconds = item.get("median_seconds_to_paula_best")
			cells[4:4] = [length(paula.get("best")),
				"–" if item.get("delta_percent") is None else f"{item['delta_percent']:+.4f}%",
				"–" if item.get("seeds_reaching_paula_best") is None else f"{item['seeds_reaching_paula_best']}/{item['ils_valid_runs']}",
				"–" if seconds is None else f"{seconds:.1f}", length(paula.get("mean"))]
		lines.append("| " + " | ".join(cells) + " |")
	return "\n".join(lines) + "\n"


def write_report(output: Path, rows: Sequence[dict], selection: Sequence[str], reference: dict, protocol: dict) -> dict:
	report = summarize(rows, selection, reference, protocol)
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
	(output / "report.md").write_text(markdown(report, machines, bool(reference)))
	return report


# --- execution --------------------------------------------------------------


def default_workers() -> int:
	if platform.system() == "Linux":
		cores = cpu_affinity.detect_cores()
		return max(1, sum(core.performance for core in cores) or len(cores))
	return max(1, (os.cpu_count() or 2) // 2)


def default_frontier_mib(workers: int) -> int:
	"""About 60% of the machine's memory, shared by the concurrent B&B frontiers."""
	memory = workspace.machine_state().get("memory_bytes") or 16 * 2**30
	return max(512, min(65536, int(0.6 * memory / workers / 2**20)))


def prepare_native() -> tuple[Path, str]:
	"""Fetch what a machine without root lacks, then build the solver once; return it and its arithmetic."""
	if platform.system() != "Darwin":
		system = [Path("/usr/include"), Path("/usr/local/include")]
		markers = {"eigen": "eigen3/Eigen/Core", "boost": "boost/multiprecision/cpp_bin_float.hpp"}
		fetched = native_build.header_dependency_dirs()
		missing = [name for name, spec in native_build.HEADER_DEPENDENCIES.items()
			if spec["cmake"] not in fetched and not any((root / markers[name]).exists() for root in system)]
		if missing:
			print(f"==> Headers missing on this machine ({', '.join(missing)}): fetching pinned releases (no root)", flush=True)
			native_build.fetch_header_dependencies(missing)
		if native_build.source_dependency_prefix("gmp") is None:
			cc = native_build.select_toolchain()["CC"]
			if not native_build.system_gmp_available(cc):
				print("==> GMP headers missing: building the pinned GMP into .cache/deps (no root, a few minutes)", flush=True)
				try:
					native_build.build_gmp()
				except SystemExit as error:
					print(f"WARNING: {error}; exact arithmetic falls back to Boost cpp_rational (slower).", flush=True)
	print("==> Building tpp-unordered and its tests (the first build takes a few minutes)", flush=True)
	binary, *tests = native_build.ensure_tools(["tpp-unordered", *SOLVER_TESTS], quiet=True)
	run_solver_tests([binary, *tests])
	arithmetic = {True: "gmp", False: "boost-cpp_rational"}.get(native_build.build_uses_gmp(), "unknown")
	if arithmetic != "gmp" and platform.system() != "Darwin":
		print("WARNING: the solver was built without GMP; rational fallbacks will be slower.", flush=True)
	return binary, arithmetic


def run_solver_tests(tools: Sequence[Path]) -> None:
	"""Run the solver's own tests once per build; a failure stops the campaign before any run."""
	marker = tools[0].parent / TESTS_MARKER
	fingerprint = json.dumps({str(tool): workspace.file_sha256(tool) for tool in tools}, sort_keys=True)
	try:
		if marker.read_text() == fingerprint:
			return
	except OSError:
		pass
	for tool in tools[1:]:
		print(f"==> {tool.name}", flush=True)
		result = subprocess.run([str(tool)], cwd=ROOT, capture_output=True, text=True, check=False)
		if result.returncode:
			print("\n".join((result.stdout + result.stderr).splitlines()[-40:]), file=sys.stderr)
			fail(f"{tool.name} failed on this machine (exit {result.returncode}); do not run the campaign with this build.")
	marker.write_text(fingerprint)


class Scheduler:
	"""ILS runs first, seed by seed so every instance gets runs early; an instance's B&B
	attempt is released once none of its ILS runs is queued or running."""

	def __init__(self, ils: Sequence[tuple[str, int]], bnb: Sequence[tuple[str, int]]) -> None:
		self.ils = deque(ils)
		self.bnb = list(bnb)
		self.outstanding: dict[str, int] = {}
		for name, _ in ils:
			self.outstanding[name] = self.outstanding.get(name, 0) + 1
		self.condition = threading.Condition()
		self.stopped = False

	def next(self) -> tuple[str, str, int] | None:
		with self.condition:
			while not self.stopped:
				if self.ils:
					name, seed = self.ils.popleft()
					return ("ils", name, seed)
				ready = next((item for item in self.bnb if not self.outstanding.get(item[0])), None)
				if ready is not None:
					self.bnb.remove(ready)
					return ("bnb", *ready)
				if not self.bnb:
					return None
				self.condition.wait()
			return None

	def done(self, task: tuple[str, str, int]) -> None:
		with self.condition:
			if task[0] == "ils":
				self.outstanding[task[1]] -= 1
			self.condition.notify_all()

	def stop(self) -> None:
		with self.condition:
			self.stopped = True
			self.condition.notify_all()


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="tpp.py paula-campaign",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		description=__doc__,
		epilog=(
			"Paula's files are not in Git. On a new machine copy them first, for example from the Mac:\n"
			"  rsync -a third_party/paula-tspn/ HOST:TouringPolygons/third_party/paula-tspn/\n"
			"Then: scripts/run_paula_campaign.sh --detach   (prepares everything without sudo)\n"
			"Follow: python3 benchmarks/tpp.py jobs log JOB --follow; python3 benchmarks/tpp.py live CAMPAIGN_DIR\n"
			"Report: scripts/run_paula_campaign.sh --report"
		),
	)
	parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN, help=f"campaign name or directory (default: {DEFAULT_CAMPAIGN})")
	parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="directory with the 235 npol-le-100 .dat files")
	parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
		help="Paula's draft (.tex) with the ILS-BCD appendix table; only used by the report")
	parser.add_argument("--instances", default="open",
		help="open (the 21 instances not certified by 2026-10-09), all, or a comma-separated list")
	parser.add_argument("--seeds", type=int, default=10, help="ILS runs per instance (default: 10, as in her table)")
	parser.add_argument("--ils-seconds", type=float, default=1200, help="limit of each ILS run (default: 1200, her CPU limit)")
	parser.add_argument("--bnb-seconds", type=float, default=14400, help="limit of each B&B run (default: 14400 = 4 h)")
	parser.add_argument("--phases", default="ils,bnb", help="ils, bnb or ils,bnb (default)")
	parser.add_argument("--workers", type=int, help="concurrent runs (default: physical cores / --bnb-threads)")
	parser.add_argument("--bnb-threads", type=int, default=1,
		help="threads per B&B run that finish subtrees in parallel past the frontier budget (needs --phases bnb)")
	parser.add_argument("--frontier-mib", type=int, help="B&B frontier budget per run (default: 60%% of memory / workers)")
	parser.add_argument("--cpu-affinity", choices=cpu_affinity.MODES, default="auto")
	parser.add_argument("--ils-args", default="", help="extra solver flags for ILS runs (part of the protocol)")
	parser.add_argument("--bnb-args", default="", help="extra solver flags for B&B runs (part of the protocol)")
	parser.add_argument("--repeat-bnb", action="store_true",
		help="run one more B&B attempt per selected instance, from its best tour (resumes an interrupted one)")
	parser.add_argument("--allow-new-binary", action="store_true",
		help="resume with a solver binary different from the one of earlier runs (each row records its binary)")
	parser.add_argument("--verbose", action="store_true", help="print every progress report of the running solvers")
	actions = parser.add_mutually_exclusive_group()
	actions.add_argument("--detach", action="store_true", help="prepare and build here, then run as a background job")
	actions.add_argument("--report", action="store_true", help="only rewrite and print the report")
	actions.add_argument("--setup-only", action="store_true", help="prepare and build, then exit")
	actions.add_argument("--dry-run", action="store_true", help="print the plan and exit")
	return parser


def solver_arguments(phase: str, protocol: dict, frontier: int, seed: int, bnb_threads: int, checkpoint: Path) -> list[str]:
	"""The solver flags of one run. A B&B run checkpoints its open subtrees, so a later
	attempt resumes them (the solver removes the file once the gap closes)."""
	arguments = [*BASE_ARGUMENTS, "--max-frontier-mib", str(frontier)]
	if phase == "ils":
		return [*arguments, *protocol["ils_arguments"], "--primal-ils-seed", str(seed)]
	return [*arguments, *protocol["bnb_arguments"], "--subtree-threads", str(bnb_threads), "--checkpoint", str(checkpoint)]


def protocol_of(args: argparse.Namespace) -> dict:
	return {
		"formulation": "TSPN, closed tour, free cyclic order, no fixed point; Paula npol-le-100 regions as in the files",
		"seeds": args.seeds,
		"ils_seconds": args.ils_seconds,
		"bnb_seconds": args.bnb_seconds,
		"base_arguments": list(BASE_ARGUMENTS),
		"ils_arguments": [*ILS_ARGUMENTS, *args.ils_args.split()],
		"bnb_arguments": [*BNB_ARGUMENTS, *args.bnb_args.split()],
		"relative_gap": RELATIVE_GAP,
		"validation_tolerance": VALIDATION_TOLERANCE,
	}


# May change between invocations: more seeds, and longer (or shorter) B&B attempts,
# whose limit each row records as seconds_limit. Every B&B bound is valid alone.
FLEXIBLE_PROTOCOL = ("seeds", "bnb_seconds")


def check_protocol(output: Path, protocol: dict) -> None:
	"""A campaign keeps one protocol; only the number of seeds and the B&B limit may change."""
	path = output / PROTOCOL_FILE
	try:
		previous = json.loads(path.read_text())
	except (OSError, ValueError):
		return
	fixed = {key: value for key, value in protocol.items() if key not in FLEXIBLE_PROTOCOL}
	if {key: previous.get(key) for key in fixed} != fixed:
		fail(f"{output} was started with another protocol ({path}); use a new --campaign.")


def run(args: argparse.Namespace, argv: Sequence[str]) -> int:
	output = workspace.campaign_path(args.campaign)
	phases = {phase.strip() for phase in args.phases.split(",") if phase.strip()}
	if not phases or phases - {"ils", "bnb"}:
		fail("--phases takes ils, bnb or ils,bnb")
	if args.seeds < 1 or not args.ils_seconds > 0 or not args.bnb_seconds > 0:
		fail("--seeds, --ils-seconds and --bnb-seconds must be positive")
	protocol = protocol_of(args)
	reference = load_reference(args.reference)
	rows = read_rows(output / RUNS_FILE)
	if not args.report:
		check_protocol(output, protocol)

	if args.report:
		# Every instance with a recorded run; the protocol is the campaign's own.
		try:
			protocol = json.loads((output / PROTOCOL_FILE).read_text())
		except (OSError, ValueError):
			pass
		report = write_report(output, rows, sorted({row["name"] for row in rows}), reference, protocol)
		print((output / "report.md").read_text())
		return int(bool(report["summary"]["invalid_runs"]))

	instances = load_instances(args.source)
	selection = select(args.instances, list(instances))
	if args.bnb_threads < 1 or (args.bnb_threads > 1 and "ils" in phases):
		fail("--bnb-threads takes a positive count, and more than 1 needs --phases bnb (ILS runs use one thread)")
	bnb_threads = args.bnb_threads
	workers = args.workers or max(1, default_workers() // bnb_threads)
	frontier = args.frontier_mib or default_frontier_mib(workers)
	done = completed(rows)
	ils = [(name, seed) for seed in range(args.seeds) for name in selection
		if "ils" in phases and ("ils", name, seed) not in done]
	def attempt(name: str) -> int:
		"""B&B rows use the seed field as an attempt number."""
		if not args.repeat_bnb:
			return 0
		finished = [row["seed"] for row in rows if row.get("phase") == "bnb" and row.get("name") == name and row.get("complete")]
		return max(finished) + 1 if finished else 0

	bnb = [(name, attempt(name)) for name in selection
		if "bnb" in phases and ("bnb", name, attempt(name)) not in done]
	hours = (len(ils) * args.ils_seconds + len(bnb) * args.bnb_seconds) / workers / 3600
	print(f"Campaign: {output}")
	print(f"Instances: {len(selection)} ({args.instances}); seeds: {args.seeds}; phases: {','.join(sorted(phases))}")
	print(f"Pending: {len(ils)} ILS runs of {args.ils_seconds:g} s, {len(bnb)} B&B runs of {args.bnb_seconds:g} s; "
		f"{workers} at a time, about {hours:.1f} h; frontier budget {frontier} MiB per run"
		+ (f"; {bnb_threads} subtree threads per B&B run" if bnb_threads > 1 else ""))
	if not reference:
		print(f"Note: Paula's table not found at {args.reference}; the report will not compare with her.")
	if args.dry_run:
		return 0

	binary, arithmetic = prepare_native()
	if args.setup_only:
		print(f"Ready: {binary} (exact arithmetic: {arithmetic}). Start with: scripts/run_paula_campaign.sh --detach")
		return 0
	if args.detach:
		import jobs

		command = ["paula-campaign", *[item for item in argv if item != "--detach"]]
		status = jobs.command_start(argparse.Namespace(name="paula-campaign", command=command))
		print(f"  live: python3 benchmarks/tpp.py live {output}", file=sys.stderr)
		print(f"  report: scripts/run_paula_campaign.sh --report --campaign {args.campaign}", file=sys.stderr)
		return status

	binary_sha256 = workspace.file_sha256(binary)
	previous = {row.get("binary_sha256") for row in rows} - {None}
	if previous and previous != {binary_sha256} and not args.allow_new_binary:
		fail("the solver binary changed since the earlier runs of this campaign; "
			"use a new --campaign, or --allow-new-binary to mix them (each row records its binary).")
	output.mkdir(parents=True, exist_ok=True)
	(output / PROTOCOL_FILE).write_text(json.dumps(protocol, indent=2) + "\n")
	scheduler = Scheduler(ils, bnb)
	total = len(ils) + len(bnb)
	lock = threading.Lock()
	live = LiveStatus(output / "live.json", echo=None if args.verbose else (lambda line: None))
	pool = cpu_affinity.CorePool(workers, bnb_threads, args.cpu_affinity)
	# The frontier budget, allocator slack, and each subtree thread's own workspaces.
	guard = int(frontier * 2**20 * 1.25) + 2**30 + bnb_threads * 2**28
	checkpoints = output / "checkpoints"
	finished = [0]
	failures = [0]

	def record(row: dict) -> None:
		with lock:
			rows.append(row)
			with (output / RUNS_FILE).open("a") as file:
				file.write(json.dumps(row, allow_nan=False) + "\n")
			finished[0] += 1
			write_report(output, rows, selection, reference, protocol)
			paula = reference.get(row["name"], {}).get("best")
			label = f" seed {row['seed']}" if row["phase"] == "ils" else f" attempt {row['seed']}" if row["seed"] else ""
			parts = [f"[{finished[0]}/{total}] {row['phase']} {row['name']}{label}"]
			if "error" in row:
				parts.append(f"ERROR {row['error']}")
			if finite(row.get("length")):
				parts.append(f"tour {row['length']:,.4f}" + (f" ({(row['length'] / paula - 1) * 100:+.4f}% vs Paula's best)" if paula else ""))
			if finite(row.get("lower_bound")) and finite(row.get("length")) and row["length"] > 0:
				parts.append(f"gap {100 * max(0.0, 1 - row['lower_bound'] / row['length']):.3f}%")
			if row.get("valid") is False:
				parts.append("INVALID TOUR")
			parts.append(f"{row.get('termination', 'no result')}, {row.get('process_seconds', 0):.0f} s")
			print(", ".join(parts), flush=True)

	def execute(task: tuple[str, str, int]) -> dict | None:
		phase, name, seed = task
		polygons = instances[name]["polygons"]
		initial = None
		if phase == "bnb":
			checkpoints.mkdir(parents=True, exist_ok=True)
		arguments = solver_arguments(phase, protocol, frontier, seed, bnb_threads, checkpoints / f"{name}.ckpt")
		if phase == "ils":
			seconds = args.ils_seconds
		else:
			seconds = args.bnb_seconds
			with lock:
				start = best_tour(rows, name)
			initial = start["path"] if start else None
		row = {"phase": phase, "name": name, "seed": seed, "m": len(polygons), "sha256": instances[name]["sha256"],
			"seconds_limit": seconds, "binary_sha256": binary_sha256, "arithmetic": arithmetic,
			"started_at": workspace.now()}
		if phase == "bnb":
			row["initial_length"] = start["length"] if start else None
			row["initial_from"] = f"{start['phase']} seed {start['seed']}" if start else None
		key = f"{phase}-{name}-{seed}"
		report = live.reporter(key, f"{phase} {name}" + (f" seed {seed}" if phase == "ils" else f" attempt {seed}" if seed else ""),
			max_seconds=seconds, target_gap=RELATIVE_GAP)
		lease = pool.acquire()
		memory_hits: list[int] = []

		def started(pid: int) -> None:
			live.set_pid(key, pid)
			pool.pin(pid, lease)

		began = time.monotonic()
		try:
			result = run_unordered_solver(binary, (0, 0), (0, 0), polygons, -1, seconds, arguments,
				initial_path=initial, process_timeout=seconds + max(120.0, 0.05 * seconds), progress=report,
				progress_interval=PROGRESS_INTERVAL, on_start=started, max_memory_bytes=guard,
				on_memory_limit=memory_hits.append)
			if result.get("error") == SHUTDOWN_BEFORE_START:
				return None
			row.update({field: result[field] for field in KEEP if field in result})
			history = result.get("incumbent_history") or []
			values = [length for _, length in history if finite(length) and length > 0]
			if values and finite(result.get("upper_bound")):
				# The solver records normalized lengths; the scale maps them back to the input's units.
				scale = result["upper_bound"] / min(values)
				row["history"] = [[round(t, 3), length * scale] for t, length in history]
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
		if memory_hits:
			row["memory_guard_bytes"] = memory_hits[0]
		if row.get("path"):
			validation = validate_cycle(polygons, row["path"], VALIDATION_TOLERANCE)
			row["valid"] = bool(validation.get("valid"))
			row["validation"] = {key: validation.get(key) for key in ("valid", "recomputed_length", "max_polygon_distance")}
			row["length"] = validation.get("recomputed_length")
		# A run stopped by Ctrl+C or `jobs stop` is repeated on resume; one stopped by the memory guard is not.
		row["complete"] = "error" not in row and (row.get("termination") != "interrupted" or bool(memory_hits))
		return row

	def worker() -> None:
		while (task := scheduler.next()) is not None:
			try:
				row = execute(task)
			except Exception as error:
				row = {"phase": task[0], "name": task[1], "seed": task[2], "complete": False,
					"error": f"runner: {type(error).__name__}: {error}"[:500]}
			if row is not None:
				if "error" in row or row.get("valid") is False:
					failures[0] += 1
				record(row)
			scheduler.done(task)

	status = 0
	parameters = {**protocol, "workers": workers, "frontier_mib": frontier, "arithmetic": arithmetic,
		"instances": selection, "cpu_affinity": pool.describe()}
	with workspace.recorded_run(output, kind="paula-campaign", argv=["paula-campaign", *argv], tools=[binary],
			parameters=parameters) as attempt:
		threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(workers, max(1, total)))]
		for thread in threads:
			thread.start()
		try:
			while any(thread.is_alive() for thread in threads):
				for thread in threads:
					thread.join(timeout=1.0)
		except KeyboardInterrupt:
			status = 130
			attempt["status"] = "interrupted"
			scheduler.stop()
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
		with lock:
			report = write_report(output, rows, selection, reference, protocol)
	print()
	print((output / "report.md").read_text())
	if status:
		print("Interrupted; repeat the same command to resume.")
		return status
	return int(bool(failures[0] or report["summary"]["invalid_runs"]))


def main(argv: Sequence[str] | None = None) -> int:
	arguments = list(sys.argv[1:] if argv is None else argv)
	return run(build_parser().parse_args(arguments), arguments)


if __name__ == "__main__":
	raise SystemExit(main())
