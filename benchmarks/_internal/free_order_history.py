"""Attribute the free-order speedup of 2026-09-21..10-06 to individual revisions.

One command builds the free-order solver at every milestone in ``STEPS`` from
``git archive`` (no checkout is touched), then runs all of them on the same
stratified sample of the Fekete corpus through ``free-order-ablation``: same
machine, one solver thread, variants interleaved per case, each solver process
pinned to its own idle performance core (Linux). It finishes by printing and
writing ``summary.md``/``summary.json`` with, per step, the geometric-mean
speedup over the 2026-09-21 baseline and over the previous step, split into
oracle calls and time per call.

    python3 benchmarks/tpp.py free-order-history            # foreground
    python3 benchmarks/tpp.py free-order-history --detach   # background job
    python3 benchmarks/tpp.py free-order-history --plan     # show, run nothing

Rerunning the same command resumes: built binaries and finished runs are kept
in ``workspace/experiments/free-order-history`` (local data, never versioned).

Old revisions predate the Linux/GCC fixes of 2026-10-01; two build-only
compatibility edits are applied to the exported sources and recorded in
``plan.json``: the C++ standard is lowered to the probed one (C++23 where the
compiler lacks C++26), and the designated initializers of trace events are put
in declaration order (what 7c6e29c did by hand). Neither changes the solver.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import native_build
import workspace

ROOT = workspace.ROOT
SAVED = ROOT / "benchmarks/results-saved"
SUITE = SAVED / "fekete-comparison/instances.bin"
OLD_TIMES = SAVED / "fekete-comparison/ours.csv"  # 2026-09-21, macOS, 2 workers
NEW_TIMES = SAVED / "free-order-dantzig-2026-10-06/per-case.csv"  # 470a0c9, dantzig, 1 process
# Standard protocol of docs/algorithms/unordered-tpp-experiments.md: UB <= 1.001 LB.
SOLVER_ARGUMENTS = ["--absolute-gap", "0", "--relative-gap", "0.0009990009990009992"]

# (label, commit, what the step adds). Each commit descends from the previous one.
STEPS = [
	("0921-base", "bb1c44a", "Baseline of the saved Fekete comparison (fekete-comparison/ours.csv)"),
	("0923-search", "e3ac002", "Dive at every expansion instead of every 128; sibling pruning"),
	("0924-dual-cutoff", "3e1b4eb", "Certified dual cutoff before the rational fallback"),
	("0925-polygon-cache", "0447f77", "Exact convex polygons cached across oracle calls (+ fc295af)"),
	("0928-cycles", "924e8b6", "Certified convex cycles merged into the shared branch and bound"),
	("0930-gmp", "63a5124", "GMP exact arithmetic (was Boost cpp_rational); cycle optimizations"),
	("1002-intervals", "e759fca", "Rigorous interval certificates; revision of the 2026-10-02 run"),
	("1003-filtered", "d004d64", "Interval primal-dual bounds, filtered rational recovery, pair cache"),
	("1003-geometry", "31a208b", "Borrowed geometry, segment cache, IEEE rounding, compact pair proofs"),
	("1005-visit-bounds", "f4c99cb", "Anchored visit upper bounds"),
	("1005-exact-arith", "82cb31e", "Arena and homogeneous-integer KKT; solver of the 2026-10-06 comparison"),
	("1006-allocation", "8f8241a", "Allocation and conversion savings in the oracle (current main)"),
]

# Stratified sample: per polygon-count band, k cases per source at evenly spaced
# quantiles of the 2026-09-21 time, among cases that the baseline closed in at
# most OLD_SECONDS_CAP and that still take NEW_SECONDS_FLOOR on 2026-10-06 (so
# the timing is not dominated by noise). Bands 4-10 run in under a millisecond.
BANDS = [(11, 20, 3), (21, 40, 4), (41, 60, 5)]
SOURCES = ("OSM", "random", "tessellation")
OLD_SECONDS_CAP = 60.0
NEW_SECONDS_FLOOR = 0.002

TRACE_FILE = "packages/nonconvex-tpp/cpp/src/solvers/unordered.cpp"
TRACE_HEADER = "packages/nonconvex-tpp/cpp/include/tpp/nonconvex/unordered.h"


def experiment_dir() -> Path:
	return workspace.experiments_dir() / "free-order-history"


def band_of(polygons: int) -> str:
	for low, high, _ in BANDS:
		if low <= polygons <= high:
			return f"{low}-{high}"
	return "other"


def saved_times() -> dict[int, dict]:
	"""Per 0-based case: polygons, source and the saved 09-21 and 10-06 times."""
	cases: dict[int, dict] = {}
	with NEW_TIMES.open(newline="") as file:
		for row in csv.DictReader(file):
			cases[int(row["case"]) - 1] = {  # per-case.csv numbers cases from 1
				"polygons": int(row["polygons"]), "source": row["source"],
				"new_seconds": float(row["ours_seconds"]), "new_calls": int(row["ours_calls"])}
	with OLD_TIMES.open(newline="") as file:
		for row in csv.DictReader(file, delimiter=";"):
			case = cases[int(row["case_index"])]
			if int(row["polygons"]) != case["polygons"]:
				raise SystemExit(f"Saved comparisons disagree on case {row['case_index']}.")
			case.update(old_seconds=float(row["seconds"]), old_calls=int(row["calls"]),
				old_exact=row["exact"] == "True")
	return cases


def select_cases(times: dict[int, dict]) -> list[int]:
	selected = []
	for low, high, per_source in BANDS:
		for source in SOURCES:
			eligible = sorted((case["old_seconds"], index) for index, case in times.items()
				if case["source"] == source and low <= case["polygons"] <= high and case.get("old_exact")
				and case["old_seconds"] <= OLD_SECONDS_CAP and case["new_seconds"] >= NEW_SECONDS_FLOOR)
			if len(eligible) < per_source:
				raise SystemExit(f"Only {len(eligible)} eligible cases for {source} {low}-{high}.")
			selected += [eligible[round((i + 0.5) * len(eligible) / per_source - 0.5)][1] for i in range(per_source)]
	return sorted(selected)


# --- exporting and building old revisions -------------------------------------------------------

def export_sources(commit: str, destination: Path) -> None:
	stamp = destination / ".export.json"
	if stamp.exists() and json.loads(stamp.read_text()).get("commit") == commit:
		return
	shutil.rmtree(destination, ignore_errors=True)
	destination.mkdir(parents=True)
	archive = subprocess.Popen(["git", "-C", str(ROOT), "archive", "--format=tar", commit, "packages"],
		stdout=subprocess.PIPE)
	with tarfile.open(fileobj=archive.stdout, mode="r|") as bundle:
		bundle.extractall(destination, filter="data")
	if archive.wait():
		raise SystemExit(f"git archive {commit} failed; is this a full clone with that commit?")
	stamp.write_text(json.dumps({"commit": commit}) + "\n")


def split_top_level(text: str) -> list[str]:
	"""Split at commas outside brackets, strings and character literals."""
	parts, depth, start, quote, index = [], 0, 0, "", 0
	while index < len(text):
		character = text[index]
		if quote:
			if character == "\\":
				index += 1
			elif character == quote:
				quote = ""
		elif character in "\"'":
			quote = character
		elif character in "([{":
			depth += 1
		elif character in ")]}":
			depth -= 1
		elif character == "," and depth == 0:
			parts.append(text[start:index])
			start = index + 1
		index += 1
	parts.append(text[start:])
	return parts


def matching_brace(text: str, opening: int) -> int:
	depth, quote, index = 0, "", opening
	while index < len(text):
		character = text[index]
		if quote:
			if character == "\\":
				index += 1
			elif character == quote:
				quote = ""
		elif character in "\"'":
			quote = character
		elif character in "([{":
			depth += 1
		elif character in ")]}":
			depth -= 1
			if depth == 0:
				return index
		index += 1
	raise ValueError("unbalanced braces")


def order_trace_initializers(source: str, fields: list[str]) -> tuple[str, int]:
	"""Put the designators of every trace_event({...}) in declaration order (GCC requires it)."""
	rank = {name: position for position, name in enumerate(fields)}
	output, cursor, changed = [], 0, 0
	for match in re.finditer(r"trace_event\(\{", source):
		opening = match.end() - 1
		if opening < cursor:
			continue
		closing = matching_brace(source, opening)
		body = source[opening + 1:closing]
		items = split_top_level(body)
		tail = ""
		if not items[-1].strip():  # trailing comma
			tail = "," + items.pop()
		stripped = items[-1].rstrip()
		tail = items[-1][len(stripped):] + tail
		items[-1] = stripped
		names = [re.match(r"\s*\.(\w+)\s*=", item) for item in items]
		if not all(names) or any(name.group(1) not in rank for name in names):
			continue
		ordered = sorted(items, key=lambda item: rank[re.match(r"\s*\.(\w+)", item).group(1)])
		if ordered != items:
			changed += 1
			output += [source[cursor:opening + 1], ",".join(ordered), tail]
			cursor = closing
	output.append(source[cursor:])
	return "".join(output), changed


def apply_compatibility(tree: Path, standard: str) -> list[str]:
	"""Build-only edits so old sources compile with this toolchain; returns what was changed."""
	applied = []
	if standard != "26":
		for cmake in sorted(tree.glob("packages/*/cpp/CMakeLists.txt")):
			text = cmake.read_text()
			updated = text.replace("set(CMAKE_CXX_STANDARD 26)", f"set(CMAKE_CXX_STANDARD {standard})")
			updated = updated.replace("cxx_std_26", f"cxx_std_{standard}")
			if updated != text:
				cmake.write_text(updated)
				applied.append(f"{cmake.relative_to(tree)}: C++26 -> C++{standard}")
	header, solver = tree / TRACE_HEADER, tree / TRACE_FILE
	if header.exists() and solver.exists():
		block = re.search(r"struct UnorderedTppTraceEvent\s*\{(.*?)\};", header.read_text(), re.S)
		if block:
			fields = re.findall(r"^\s*[\w:<>, ]+?\s+(\w+)\s*(?:=[^;]*)?;", block.group(1), re.M)
			text, changed = order_trace_initializers(solver.read_text(), fields)
			if changed:
				solver.write_text(text)
				applied.append(f"{TRACE_FILE}: {changed} trace-event initializer(s) put in declaration order")
	return applied


def dependency_arguments(directory: Path) -> list[str]:
	"""Eigen/Boost for every revision: old CMake files only use find_package(... CONFIG)."""
	arguments = []
	shims = {
		"Eigen3": ("TPP_EIGEN_INCLUDE_DIR", "Eigen3::Eigen"),
		"Boost": ("TPP_BOOST_INCLUDE_DIR", "Boost::headers"),
	}
	found = native_build.header_dependency_dirs()
	for package, (variable, target) in shims.items():
		include = found.get(variable)
		if include is None:
			continue  # installed system package: every revision finds it the same way
		shim = directory / "cmake-shims" / package
		shim.mkdir(parents=True, exist_ok=True)
		(shim / f"{package}Config.cmake").write_text(
			f"if(NOT TARGET {target})\n"
			f"  add_library({target} INTERFACE IMPORTED GLOBAL)\n"
			f"  set_target_properties({target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"{include}\")\n"
			f"endif()\n"
			f"set({package}_FOUND TRUE)\n")
		arguments += [f"-D{package}_DIR={shim}", f"-D{variable}={include}"]
	return arguments


def build_step(label: str, commit: str, directory: Path, toolchain: dict, jobs: int) -> dict:
	tree = directory / "src" / label
	build = directory / "build" / label
	binary = directory / "bin" / f"tpp-unordered-{label}"
	record_path = directory / "bin" / f"tpp-unordered-{label}.json"
	if binary.exists() and record_path.exists():
		record = json.loads(record_path.read_text())
		if record.get("commit") == commit and record.get("sha256") == workspace.file_sha256(binary):
			return record
	export_sources(commit, tree)
	compatibility = apply_compatibility(tree, toolchain["TPP_CXX_STANDARD"])
	shutil.rmtree(build, ignore_errors=True)
	build.mkdir(parents=True)
	log = directory / "build" / f"{label}.log"
	environment = os.environ | {"CC": toolchain["CC"], "CXX": toolchain["CXX"]}
	configure = ["cmake", "-S", str(tree / "packages/nonconvex-tpp/cpp"), "-B", str(build),
		"-DCMAKE_BUILD_TYPE=Release", f"-DTPP_CXX_STANDARD={toolchain['TPP_CXX_STANDARD']}",
		"-DTPP_ENABLE_GUROBI=OFF", *dependency_arguments(directory),
		*shlex.split(os.environ.get("TPP_CMAKE_ARGS", ""))]
	compile_ = ["cmake", "--build", str(build), "--target", "tpp-unordered", "--parallel", str(jobs)]
	began = time.monotonic()
	with log.open("w") as output:
		for command in (configure, compile_):
			output.write("+ " + " ".join(command) + "\n")
			output.flush()
			if subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT).returncode:
				tail = "\n".join(log.read_text().splitlines()[-15:])
				return {"label": label, "commit": commit, "error": f"build failed; see {log}\n{tail}"}
	built = next((path for path in build.rglob("tpp-unordered") if path.is_file() and os.access(path, os.X_OK)), None)
	if built is None:
		return {"label": label, "commit": commit, "error": f"no tpp-unordered produced; see {log}"}
	binary.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(built, binary)
	text = log.read_text()
	arithmetic = ("GMP" if "TPP exact arithmetic: GMP" in text
		else "Boost cpp_rational")  # the only option before 63a5124, and its fallback afterwards
	record = {"label": label, "commit": commit, "binary": str(binary), "sha256": workspace.file_sha256(binary),
		"exact_arithmetic": arithmetic, "compatibility_edits": compatibility, "toolchain": toolchain,
		"build_seconds": round(time.monotonic() - began, 1)}
	record_path.write_text(json.dumps(record, indent=2) + "\n")
	shutil.rmtree(build, ignore_errors=True)  # keep only the binary
	return record


def build_all(steps: list[tuple[str, str, str]], directory: Path) -> list[dict]:
	missing = [commit for _, commit, _ in steps
		if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
			capture_output=True).returncode]
	if missing:
		raise SystemExit(f"Commits not in this clone: {', '.join(missing)}. Run 'git fetch' (a full, not shallow, clone).")
	toolchain = native_build.select_toolchain()
	print(f"Toolchain: {toolchain['CXX']} with C++{toolchain['TPP_CXX_STANDARD']}", flush=True)
	cpus = os.cpu_count() or 4
	concurrent = min(4, len(steps))
	jobs = max(2, cpus // concurrent)
	print(f"Building {len(steps)} revisions ({concurrent} at a time, {jobs} jobs each)...", flush=True)

	def build(step):
		label, commit, _ = step
		record = build_step(label, commit, directory, toolchain, jobs)
		state = "FAILED" if "error" in record else f"ok ({record['exact_arithmetic']})"
		print(f"  {label:<20} {commit}  {state}", flush=True)
		return record

	with ThreadPoolExecutor(max_workers=concurrent) as executor:
		return list(executor.map(build, steps))


# --- summary ------------------------------------------------------------------------------------

def geometric_mean(values: list[float]) -> float | None:
	values = [value for value in values if value > 0 and math.isfinite(value)]
	return math.exp(statistics.fmean(map(math.log, values))) if values else None


def summarize(directory: Path, labels: list[str], times: dict[int, dict]) -> dict:
	runs: dict[tuple[int, str], list[dict]] = {}
	for line in (directory / "runs.jsonl").read_text().splitlines():
		row = json.loads(line)
		runs.setdefault((int(row["case"]), row["solver"]), []).append(row)
	cases = sorted({case for case, _ in runs})

	def cell(case: int, label: str) -> dict | None:
		rows = runs.get((case, label), [])
		if not rows or any(row.get("error") or not row.get("exact") or row.get("valid") is False for row in rows):
			return None
		return {"seconds": statistics.median(row["seconds"] for row in rows),
			"calls": statistics.median(row["calls"] for row in rows),
			"spread": max(row["seconds"] for row in rows) / max(1e-9, min(row["seconds"] for row in rows))}

	table = {(case, label): cell(case, label) for case in cases for label in labels}

	def ratios(first: str, second: str, subset: list[int]) -> dict:
		pairs = [(table[case, first], table[case, second]) for case in subset
			if table[case, first] and table[case, second]]
		speedup = geometric_mean([a["seconds"] / b["seconds"] for a, b in pairs])
		calls = geometric_mean([a["calls"] / b["calls"] for a, b in pairs])
		return {"cases": len(pairs), "speedup": speedup, "fewer_calls": calls,
			"cheaper_calls": speedup / calls if speedup and calls else None}

	base, final = labels[0], labels[-1]
	steps = []
	for position, label in enumerate(labels):
		steps.append({"label": label,
			"closed": sum(table[case, label] is not None for case in cases),
			"vs_previous": ratios(labels[position - 1], label, cases) if position else None,
			"vs_base": ratios(base, label, cases) if position else None})
	total = ratios(base, final, cases)
	for step in steps[1:]:
		speedup = step["vs_previous"]["speedup"]
		step["share_of_log_gain"] = (math.log(speedup) / math.log(total["speedup"])
			if speedup and total["speedup"] and total["speedup"] > 1 else None)
	bands = {name: ratios(base, final, [case for case in cases if band_of(times[case]["polygons"]) == name])
		for name in sorted({band_of(times[case]["polygons"]) for case in cases})}
	# Machine and protocol effect on the two saved comparisons, read on the same binaries.
	machine = {
		"saved_0921_mac_over_measured_base": geometric_mean([times[case]["old_seconds"] / table[case, base]["seconds"]
			for case in cases if base == "0921-base" and table[case, base]]),
		"saved_1006_dantzig_over_measured_1005": geometric_mean([times[case]["new_seconds"] / table[case, "1005-exact-arith"]["seconds"]
			for case in cases if ("1005-exact-arith" in labels) and table[case, "1005-exact-arith"]]),
	}
	spreads = [value["spread"] for value in table.values() if value]
	other_load = [row["cpu"]["other_load"] for rows in runs.values() for row in rows if row.get("cpu")]
	return {"cases": cases, "steps": steps, "total": total, "bands": bands, "machine": machine,
		"repeat_spread_median": statistics.median(spreads) if spreads else None,
		"repeat_spread_p90": sorted(spreads)[int(0.9 * (len(spreads) - 1))] if spreads else None,
		"pinned_runs": len(other_load),
		"runs_with_other_load": sum(load > 0.10 for load in other_load)}


def render(summary: dict, plan: dict) -> str:
	def number(value, digits=2):
		return "—" if value is None else f"{value:.{digits}f}×"
	steps = {step[0]: step for step in plan["steps"]}
	builds = {build["label"]: build for build in plan["builds"]}
	lines = [
		"# Free-order speedup by revision",
		"",
		f"{len(summary['cases'])} cases of `fekete-comparison/instances.bin` (stratified; see plan.json), "
		f"{plan['repeats']} repeat(s), one solver thread, gap UB ≤ 1.001·LB (absolute 0), "
		f"independent validation 1e-7. Speedups are geometric means of per-case median times over cases that "
		f"both binaries closed; *fewer calls* × *cheaper calls* = speedup.",
		"",
		"| Step | Commit | Arithmetic | Closed | vs previous | fewer calls | cheaper calls | share of log gain | vs 09-21 |",
		"|---|---|---|---:|---:|---:|---:|---:|---:|",
	]
	for step in summary["steps"]:
		label = step["label"]
		previous, base = step["vs_previous"] or {}, step["vs_base"] or {}
		share = step.get("share_of_log_gain")
		lines.append(f"| {label} | `{steps[label][1]}` | {builds[label].get('exact_arithmetic', '—')} | "
			f"{step['closed']}/{len(summary['cases'])} | {number(previous.get('speedup'))} | "
			f"{number(previous.get('fewer_calls'))} | {number(previous.get('cheaper_calls'))} | "
			f"{'—' if share is None else f'{100 * share:.0f}%'} | {number(base.get('speedup'))} |")
	total = summary["total"]
	lines += ["", f"**Total** 09-21 → final: {number(total['speedup'])} "
		f"({number(total['fewer_calls'])} fewer calls × {number(total['cheaper_calls'])} cheaper calls, "
		f"{total['cases']} cases).", "", "| Polygons | Cases | Speedup | fewer calls | cheaper calls |", "|---|---:|---:|---:|---:|"]
	for band, values in summary["bands"].items():
		lines.append(f"| {band} | {values['cases']} | {number(values['speedup'])} | "
			f"{number(values['fewer_calls'])} | {number(values['cheaper_calls'])} |")
	machine = summary["machine"]
	lines += ["", "Machine/protocol check (saved times over the same revision measured here): "
		f"09-21 Mac run {number(machine['saved_0921_mac_over_measured_base'])}, "
		f"10-06 dantzig run {number(machine['saved_1006_dantzig_over_measured_1005'])}.",
		f"Repeat spread (max/min time per case and binary): median {number(summary['repeat_spread_median'], 3)}, "
		f"p90 {number(summary['repeat_spread_p90'], 3)}. Pinned runs: {summary['pinned_runs']}, "
		f"{summary['runs_with_other_load']} saw other load above 10% of a CPU on their core.",
		"", "Steps:", ""]
	for label, commit, description in plan["steps"]:
		edits = builds[label].get("compatibility_edits") or []
		lines.append(f"- `{label}` (`{commit}`): {description}." + (f" Build edits: {'; '.join(edits)}." if edits else ""))
	lines += ["", "Limitations: cases the 09-21 baseline needed more than "
		f"{OLD_SECONDS_CAP:.0f} s for are excluded (the largest speedups are there); per-step gains "
		"are measured in sequence and depend on the steps before them; one machine."]
	return "\n".join(lines) + "\n"


# --- command ------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="tpp.py free-order-history", description=__doc__.split("\n\n")[0])
	parser.add_argument("--workers", type=int, default=4,
		help="cases solved at once, each solver pinned to its own performance core (default 4; 1 = strictly one process)")
	parser.add_argument("--repeats", type=int, default=3)
	parser.add_argument("--seconds", type=float, default=600, help="time limit per solver run (default 600)")
	parser.add_argument("--case", type=int, action="append", help="0-based case index; replaces the default sample")
	parser.add_argument("--step", action="append", help="only these step labels (the first one listed is the baseline)")
	parser.add_argument("--output", type=Path, help=f"experiment directory (default {experiment_dir()})")
	parser.add_argument("--plan", action="store_true", help="print revisions, cases and estimated time, then exit")
	parser.add_argument("--detach", action="store_true", help="run as a background job (tpp.py jobs)")
	args = parser.parse_args(argv)
	argv = list(sys.argv[2:] if argv is None else argv)
	if args.workers < 1 or args.repeats < 1 or args.seconds <= 0:
		parser.error("--workers, --repeats and --seconds must be positive.")
	if args.detach and not args.plan:
		import jobs
		command = ["free-order-history", *[argument for argument in argv if argument != "--detach"]]
		return jobs.command_start(argparse.Namespace(name="free-order-history", command=command))

	steps = STEPS
	if args.step:
		known = {step[0]: step for step in STEPS}
		unknown = [label for label in args.step if label not in known]
		if unknown:
			parser.error(f"Unknown step(s): {', '.join(unknown)}. Known: {', '.join(known)}")
		steps = [known[label] for label in args.step]
	directory = args.output or experiment_dir()
	times = saved_times()
	cases = sorted(set(args.case)) if args.case else select_cases(times)
	old = sum(times[case]["old_seconds"] for case in cases)
	new = sum(times[case]["new_seconds"] for case in cases)
	# Log-linear interpolation from the 09-21 to the 10-06 time across the steps.
	per_repeat = sum(times[case]["old_seconds"] * (times[case]["new_seconds"] / times[case]["old_seconds"])
		** (position / max(1, len(steps) - 1)) for case in cases for position in range(len(steps)))
	estimate = per_repeat * args.repeats / min(args.workers, len(cases))
	print(f"{len(steps)} revisions x {len(cases)} cases x {args.repeats} repeat(s), {args.workers} worker(s).")
	print(f"Saved times of these cases: {old:.0f} s on 2026-09-21, {new:.1f} s on 2026-10-06; "
		f"estimated solving time ~{estimate / 60:.0f} min (+ building).")
	print(f"Cases: {' '.join(map(str, cases))}")
	print(f"Output: {directory}")
	if args.plan:
		for label, commit, description in steps:
			print(f"  {label:<20} {commit}  {description}")
		return 0

	directory.mkdir(parents=True, exist_ok=True)
	with workspace.recorded_run(directory, kind="free-order-history", argv=["free-order-history", *argv],
			inputs=[SUITE], parameters=workspace.jsonable(vars(args))):
		builds = build_all(steps, directory)
		failed = [build for build in builds if "error" in build]
		for build in failed:
			print(f"\n{build['label']}: {build['error']}", file=sys.stderr)
		ready = [build for build in builds if "error" not in build]
		if len(ready) < 2:
			raise SystemExit("Fewer than two revisions built; nothing to compare.")
		plan = {"steps": steps, "cases": cases, "repeats": args.repeats, "workers": args.workers,
			"solver_arguments": SOLVER_ARGUMENTS, "selection": {"bands": BANDS, "sources": SOURCES,
			"old_seconds_cap": OLD_SECONDS_CAP, "new_seconds_floor": NEW_SECONDS_FLOOR}, "builds": builds,
			"machine": workspace.machine_state()}
		(directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

		import free_order_ablation
		ablation = ["--suite", str(SUITE), "--output", str(directory / "runs.jsonl"), "--resume",
			"--repeats", str(args.repeats), "--workers", str(args.workers), "--cpu-affinity", "p-cores",
			"--seconds", str(args.seconds), "--max-calls", str(10**15),
			"--absolute-gap", SOLVER_ARGUMENTS[1], "--relative-gap", SOLVER_ARGUMENTS[3]]
		for build in ready:
			ablation += ["--solver", f"{build['label']}={build['binary']}"]
		for case in cases:
			ablation += ["--case", str(case)]
		# Same options resume; different ones start over, keeping the earlier rows aside.
		key, runs = directory / "runs.key.json", directory / "runs.jsonl"
		if runs.exists() and (not key.exists() or json.loads(key.read_text()) != ablation):
			stamp = time.strftime("%Y%m%d-%H%M%S")
			for path in (runs, runs.with_suffix(".meta.json"), key):
				if path.exists():
					path.rename(path.with_name(f"{path.stem}-{stamp}{path.suffix}"))
			print(f"Options changed: earlier rows moved to runs-{stamp}.jsonl.", flush=True)
		key.write_text(json.dumps(ablation) + "\n")
		print(f"Solving ({len(ready)} binaries; per-solver totals follow)...", flush=True)
		status = free_order_ablation.main(ablation)

		labels = [build["label"] for build in ready]
		summary = summarize(directory, labels, times)
		(directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
		text = render(summary, plan)
		(directory / "summary.md").write_text(text)
		print("\n" + text)
		print(f"Written to {directory / 'summary.md'} (raw rows: runs.jsonl).")
		if failed:
			print(f"WARNING: {len(failed)} revision(s) did not build: {', '.join(b['label'] for b in failed)}.")
		return status


if __name__ == "__main__":
	raise SystemExit(main())
