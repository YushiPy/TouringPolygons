"""Replay captured native convex-cycle oracle calls through the public TPP CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

from tspn_diagnostics import atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks/_internal/cycle_replay"
INTEGER_FIELDS = {"call_id", "repeat"}
NUMBER_FIELDS = {"capture_tolerance", "deadline_seconds", "wall_seconds",
	"construction_seconds", "certification_seconds", "rational_recovery_seconds", "lower_bound",
	"upper_bound", "lower_bound_cutoff"}
BOOLEAN_FIELDS = {"precise", "cache", "features", "interval", "bound_first", "used_fallback",
	"dual_cutoff_pruned", "gap_satisfied", "upper_bound_infinite", "independent_certificate_valid",
	"intervals_overlap", "interval_bounds_used"}


def _number(value):
	try:
		result = float(value)
	except (TypeError, ValueError):
		return None
	return result if math.isfinite(result) else None


def _boolean(value):
	return value if isinstance(value, bool) else str(value).lower() in {"true", "1", "yes"}


def normalize_row(row: dict) -> dict:
	"""Property-tree emits scalars as strings; restore JSON numeric/boolean types."""
	for key in INTEGER_FIELDS:
		if key in row:
			try:
				row[key] = int(row[key])
			except (TypeError, ValueError):
				row[key] = None
	for key in NUMBER_FIELDS:
		if key in row:
			row[key] = _number(row[key])
	for key in BOOLEAN_FIELDS:
		if key in row:
			row[key] = _boolean(row[key])
	for key in ("independent_certificate", "filtered_certificate"):
		certificate = row.get(key)
		if isinstance(certificate, dict):
			for name in ("status", "lower_bound", "upper_bound", "exact_predicate_evaluations"):
				if name in certificate:
					certificate[name] = _number(certificate[name])
			for name in ("interval_bounds_used",):
				if name in certificate:
					certificate[name] = _boolean(certificate[name])
	if "contacts" in row:
		if row["contacts"] in ("", None):
			row["contacts"] = []
		else:
			row["contacts"] = [[_number(axis) for axis in point] for point in row["contacts"]]
	return row


def sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def read_capture(path: Path) -> tuple[list[dict], list[dict]]:
	begins: dict[int, dict] = {}
	ends: dict[int, dict] = {}
	for line_number, line in enumerate(path.read_text().splitlines(), 1):
		if not line.strip():
			continue
		try:
			event = json.loads(line)
			call_id = int(event["id"])
			kind = event["event"]
		except (ValueError, KeyError, TypeError) as error:
			raise ValueError(f"invalid capture event at line {line_number}") from error
		if kind == "begin":
			if call_id in begins:
				raise ValueError(f"duplicate begin event for call {call_id}")
			begins[call_id] = event
		elif kind == "end":
			if call_id in ends:
				raise ValueError(f"duplicate end event for call {call_id}")
			ends[call_id] = event
		else:
			raise ValueError(f"unknown capture event {kind!r} at line {line_number}")
	if not begins or any(call_id not in begins for call_id in ends):
		raise ValueError("capture has no begin events or contains an end without a begin")
	return list(begins.values()), list(ends.values())


def select_calls(begins: list[dict], ends: list[dict], call_ids: list[int] | None,
		minimum_seconds: float | None) -> list[dict]:
	ended = {int(event["id"]): event for event in ends}
	by_id = {int(event["id"]): event for event in begins}
	if call_ids:
		missing = [call_id for call_id in call_ids if call_id not in by_id]
		if missing:
			raise ValueError(f"requested capture call id(s) not found: {missing}")
		return [by_id[call_id] for call_id in dict.fromkeys(call_ids)]
	if minimum_seconds is None:
		raise ValueError("select calls with --call-id or --min-seconds")
	return [event for event in begins
		if int(event["id"]) in ended and float(ended[int(event["id"])].get("seconds", 0)) >= minimum_seconds]


def source_hashes() -> dict[str, str]:
	paths = sorted([*SOURCE.glob("*.cpp"), *SOURCE.glob("*.txt"), ROOT / "benchmarks/_internal/cycle_replay.py",
		ROOT / "benchmarks/_internal/tpp.py", ROOT / "benchmarks/tpp.py"])
	for package in ("convex-tpp", "nonconvex-tpp"):
		base = ROOT / "packages" / package / "cpp"
		paths += sorted(base.rglob("*.cpp"))
		paths += sorted(base.rglob("*.h"))
		paths += [base / "CMakeLists.txt"]
	return {str(path.relative_to(ROOT)): sha256(path) for path in paths if path.is_file()}


def persist_complete_rows(stdout: str, path: Path) -> tuple[int, int]:
	"""Normalize all complete JSON rows, retaining malformed output in stdout.log."""
	rows = []
	malformed = 0
	for line in stdout.splitlines():
		if not line.strip():
			continue
		try:
			rows.append(normalize_row(json.loads(line)))
		except (json.JSONDecodeError, TypeError):
			malformed += 1
	path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
	return len(rows), malformed


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--capture", type=Path, required=True)
	selection = parser.add_mutually_exclusive_group(required=True)
	selection.add_argument("--min-seconds", type=float,
		help="Select completed captured calls whose recorded duration meets this threshold.")
	selection.add_argument("--call-id", type=int, action="append",
		help="Replay a specific captured call; also accepts begin-only records from hard-killed processes.")
	parser.add_argument("--seconds", type=float, required=True,
		help="Finite per-replay solver deadline; Interrupted results are retained as incomplete.")
	parser.add_argument("--repetitions", type=int, default=1)
	parser.add_argument("--binary", type=Path, help="Use a previously built replay runner.")
	parser.add_argument("--skip-build", action="store_true")
	parser.add_argument("--build-dir", type=Path, default=ROOT / ".build/cycle-replay")
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--cache", action="store_true")
	parser.add_argument("--features", action="store_true")
	parser.add_argument("--interval", action="store_true")
	parser.add_argument("--bound-first", action="store_true")
	args = parser.parse_args(argv)
	if (not math.isfinite(args.seconds) or args.seconds <= 0 or args.repetitions < 1 or
		(args.min_seconds is not None and (not math.isfinite(args.min_seconds) or args.min_seconds < 0))):
		parser.error("seconds and repetitions must be positive; min-seconds cannot be negative")
	capture = args.capture.resolve()
	output = args.output.resolve()
	if not capture.is_file():
		parser.error(f"capture file does not exist: {capture}")
	if output.exists() and any(output.iterdir()):
		parser.error("output directory must be new or empty")
	try:
		begins, ends = read_capture(capture)
		selected = select_calls(begins, ends, args.call_id, args.min_seconds)
	except ValueError as error:
		parser.error(str(error))
	if not selected:
		parser.error("selection matched no completed calls")
	if args.skip_build and not args.binary:
		parser.error("--skip-build requires --binary")
	output.mkdir(parents=True, exist_ok=True)
	selected_ids = [int(record["id"]) for record in selected]
	inputs = {"schema": "tspn-cycle-replay-inputs-v1", "capture": str(capture),
		"capture_sha256": sha256(capture), "selection": {"call_ids": selected_ids,
		"min_seconds": args.min_seconds, "call_id": args.call_id},
		"calls": [{"begin": record,
			"end": next((end for end in ends if int(end["id"]) == int(record["id"])), None)}
			for record in selected]}
	atomic_json(output / "inputs.json", inputs)
	input_path = output / "calls.jsonl"
	input_path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in selected))
	defaults = {"cache": args.cache, "features": args.features, "interval": args.interval,
		"bound_first": args.bound_first}
	options_by_call = {str(record["id"]): {key: bool(record.get(key, value)) for key, value in defaults.items()}
		for record in selected}
	binary = args.binary.resolve() if args.binary else (args.build_dir.resolve() / "tpp-cycle-replay")
	commands = []
	if not args.skip_build and not args.binary:
		configure = ["cmake", "-S", str(SOURCE), "-B", str(args.build_dir.resolve()),
			"-DCMAKE_BUILD_TYPE=Release", "-DTPP_ENABLE_GUROBI=OFF"]
		commands = [configure, ["cmake", "--build", str(args.build_dir.resolve()), "--target", "tpp-cycle-replay", "-j", "4"]]
		with (output / "build.log").open("w") as log:
			for command in commands:
				subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
	if not binary.is_file():
		parser.error(f"replay binary not found: {binary}")
	config = {"schema": "tspn-cycle-replay-v1", "formulation": "fixed-order convex cycle oracle calls captured from free-order TSPN B&B",
		"capture": str(capture), "capture_sha256": inputs["capture_sha256"], "selected_call_ids": selected_ids,
		"seconds": args.seconds, "repetitions": args.repetitions, "options_by_call": options_by_call,
		"binary": str(binary), "binary_sha256": sha256(binary), "build_commands": commands,
		"platform": platform.platform(), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
		"inputs_sha256": digest(inputs), "source_sha256": source_hashes(),
		"timing": "whole shared convex-cycle adapter call; each replay starts with a fresh workspace, so prepared-geometry cache is cold unlike repeated calls in a B&B worker; independent feasible-contact and interval certificates run outside timing; a solver deadline interruption is incomplete, not a process failure"}
	atomic_json(output / "config.json", config)
	command = [str(binary), str(input_path), str(args.seconds), str(args.repetitions),
		str(int(args.cache)), str(int(args.features)), str(int(args.interval)), str(int(args.bound_first))]
	try:
		process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
			timeout=args.seconds * args.repetitions * len(selected) + 30)
	except subprocess.TimeoutExpired as error:
		stdout = error.stdout or ""
		if isinstance(stdout, bytes):
			stdout = stdout.decode(errors="replace")
		(output / "stdout.log").write_text(stdout)
		stderr = error.stderr or ""
		if isinstance(stderr, bytes):
			stderr = stderr.decode(errors="replace")
		(output / "stderr.log").write_text(stderr)
		complete_rows, malformed = persist_complete_rows(stdout, output / "raw.jsonl")
		atomic_json(output / "progress.json", {"status": "process_timeout", "censored": True,
			"complete_records": complete_rows, "malformed_complete_lines": malformed,
			"planned_records": len(selected) * args.repetitions,
			"timeout_seconds": args.seconds * args.repetitions * len(selected) + 30})
		(output / "analysis.md").write_text("# Captured convex-cycle oracle replay\n\n"
			f"Status: process timeout; {complete_rows} complete rows were retained, while the unfinished process tail is censored. "
			"No solver reason is inferred from a hard process timeout.\n")
		return 1
	(output / "stdout.log").write_text(process.stdout)
	(output / "stderr.log").write_text(process.stderr)
	complete_rows, malformed = persist_complete_rows(process.stdout, output / "raw.jsonl")
	if process.returncode != 0:
		atomic_json(output / "progress.json", {"status": "failed", "censored": True,
			"return_code": process.returncode, "complete_records": complete_rows,
			"malformed_complete_lines": malformed, "planned_records": len(selected) * args.repetitions})
		return process.returncode
	if malformed:
		atomic_json(output / "progress.json", {"status": "failed", "malformed_complete_lines": malformed,
			"complete_records": complete_rows, "planned_records": len(selected) * args.repetitions})
		return 1
	rows = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines() if line.strip()]
	if len(rows) != len(selected) * args.repetitions:
		atomic_json(output / "progress.json", {"status": "failed", "censored": True,
			"complete_records": len(rows), "planned_records": len(selected) * args.repetitions})
		return 1
	failed_rows = [row for row in rows if row.get("status") in {"oracle_failure", "invalid_input", "unsupported_intersection"}]
	incomplete = any(row.get("status") == "interrupted" for row in rows)
	status = "failed" if failed_rows else "incomplete" if incomplete else "complete"
	atomic_json(output / "summary.json", {"status": status, "records": len(rows),
		"planned_records": len(selected) * args.repetitions,
		"interrupted_records": sum(row.get("status") == "interrupted" for row in rows),
		"invalid_or_missing_certificates": sum(not row.get("independent_certificate_valid") and
			not (row.get("status") == "interrupted" and not row.get("contacts")) for row in rows)})
	lines = ["# Captured convex-cycle oracle replay", "", f"Status: {status}.", "",
		f"Calls: {len(selected)}; repetitions: {args.repetitions}; solver deadline: {args.seconds:g} s.",
		"Options by call: " + json.dumps(options_by_call, sort_keys=True), "",
		"| Call id | Repeat | Status | Gap met | UB infinite | Seconds | Construction ms | Certification ms | Rational recovery ms | Bound interval | Independent certificate | Interval overlap |",
		"|---:|---:|---|---|---|---:|---:|---:|---:|---|---|---|"]
	for row in rows:
		lines.append(f"| {row['call_id']} | {row['repeat']} | {row['status']} | {row['gap_satisfied']} | {row['upper_bound_infinite']} | {row['wall_seconds']:.6f} | "
			f"{row['construction_seconds']*1000:.3f} | {row['certification_seconds']*1000:.3f} | "
			f"{row['rational_recovery_seconds']*1000:.3f} | [{row['lower_bound']}, {row['upper_bound']}] | "
			f"{row['independent_certificate_valid']} | {row['intervals_overlap']} |")
	lines += ["", "A row with status `interrupted` retains only completed certificates and makes the report incomplete; it is not treated as an executable failure.",
		"Independent contact feasibility and interval checks are outside the measured solver interval.", ""]
	(output / "analysis.md").write_text("\n".join(lines))
	atomic_json(output / "progress.json", {"status": status, "records": len(rows), "planned_records": len(rows)})
	print("\n".join(lines))
	return 1 if failed_rows else 0


if __name__ == "__main__":
	raise SystemExit(main())
