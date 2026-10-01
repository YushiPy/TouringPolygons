#!/usr/bin/env python3
"""Audit cross-run native objectives and matched Fekete intervals for this campaign."""
import glob
import json
import math
import os
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "runs")
rows_by_input = {}
run_rows = {}
configs = {}
for config_path in sorted(glob.glob(os.path.join(RUNS, "*", "config.json"))):
    run = os.path.basename(os.path.dirname(config_path))
    with open(config_path) as f:
        configs[run] = json.load(f)
    raw_path = os.path.join(os.path.dirname(config_path), "raw.jsonl")
    rows = [json.loads(line) for line in open(raw_path) if line.strip()]
    run_rows[run] = rows
    for row in rows:
        if row.get("solver") not in ("ours", "fekete"):
            continue
        key = row.get("sha256") or row.get("name")
        rows_by_input.setdefault(key, {"name": row.get("name"), "k": row.get("k"), "rows": []})["rows"].append((run, row))

native_rows = [(run, r) for run, rows in run_rows.items() for r in rows if r.get("solver") == "ours"]
fekete_rows = [(run, r) for run, rows in run_rows.items() for r in rows if r.get("solver") == "fekete"]
valid = sum(bool(r.get("validation", {}).get("valid")) for _, r in native_rows)
closed = [(run, r) for run, r in native_rows if r.get("gap_closed")]
interval_violations = []
objective_spreads = []
external_interval_checks = 0
external_interval_violations = []
max_native_width_rel = 0.0
max_native_closed_spread_rel = 0.0
max_external_bracket_rel = 0.0
for key, group in rows_by_input.items():
    rs = group["rows"]
    nrows = [(run, r) for run, r in rs if r.get("solver") == "ours"]
    frows = [(run, r) for run, r in rs if r.get("solver") == "fekete"]
    objectives = [r.get("validation", {}).get("recomputed_length", r.get("upper_bound")) for _, r in nrows if r.get("gap_closed")]
    objectives = [float(x) for x in objectives if x is not None and math.isfinite(float(x))]
    if objectives:
        spread = max(objectives) - min(objectives)
        rel = spread / max(1.0, abs(statistics.mean(objectives)))
        max_native_closed_spread_rel = max(max_native_closed_spread_rel, rel)
        objective_spreads.append({"name": group["name"], "k": group["k"], "closed_objective_count": len(objectives), "min": min(objectives), "max": max(objectives), "relative_spread": rel})
        known = statistics.median(objectives)
        for run, r in nrows:
            tol = float(configs[run].get("validation_tolerance", 1e-7)) * max(1.0, abs(known))
            lb, ub = float(r["lower_bound"]), float(r["upper_bound"])
            width_rel = (ub - lb) / max(1.0, abs(ub))
            max_native_width_rel = max(max_native_width_rel, width_rel)
            if lb > known + tol or ub < known - tol:
                interval_violations.append({"name": group["name"], "run": run, "known_closed_objective": known, "lower_bound": lb, "upper_bound": ub, "tolerance": tol})
        for run, r in frows:
            external_interval_checks += 1
            tol = float(configs[run].get("validation_tolerance", 1e-7)) * max(1.0, abs(known))
            lb, ub = float(r["lower_bound"]), float(r["upper_bound"])
            if lb > known + tol or ub < known - tol:
                external_interval_violations.append({"name": group["name"], "run": run, "known_native_closed_objective": known, "fekete_lower_bound": lb, "fekete_upper_bound": ub, "tolerance": tol})
            max_external_bracket_rel = max(max_external_bracket_rel, (ub - lb) / max(1.0, abs(ub)))

by_name = {}
for key, group in rows_by_input.items():
    by_name.setdefault(group["name"], []).append(group)

summary = {
    "audit": "cross-run native objective consistency and matched external interval containment",
    "campaign": os.path.basename(ROOT),
    "native_rows": len(native_rows),
    "native_valid_rows": valid,
    "native_closed_rows": len(closed),
    "native_time_limit_rows": sum(r.get("termination") == "time_limit" for _, r in native_rows),
    "native_process_timeout_rows": sum(r.get("termination") == "process_timeout" for _, r in native_rows),
    "fekete_reference_rows_reused": len(fekete_rows),
    "distinct_input_keys": len(rows_by_input),
    "distinct_case_names": len(by_name),
    "native_interval_contains_known_closed_objective_checks": sum(len([1 for _, r in rows_by_input[k]["rows"] if r.get("solver") == "ours"]) for k in rows_by_input if any(r.get("solver") == "ours" and r.get("gap_closed") for _, r in rows_by_input[k]["rows"])),
    "native_interval_violations": interval_violations,
    "matched_fekete_interval_contains_native_closed_objective_checks": external_interval_checks,
    "matched_fekete_interval_violations": external_interval_violations,
    "max_native_relative_interval_width": max_native_width_rel,
    "max_native_closed_objective_relative_spread": max_native_closed_spread_rel,
    "max_matched_fekete_relative_interval_width": max_external_bracket_rel,
    "objective_spreads_by_case": objective_spreads,
    "tolerance_contract": "Use each run config validation_tolerance (1e-7) times max(1, |known closed native objective|). Native exact-cycle/B&B intervals are checked against cross-run independently validated tour objectives; Fekete rows remain external numerical bounds checked under the same declared native validation tolerance for cross-solver consistency only, not treated as exact rational certificates.",
}
out = os.path.join(os.path.dirname(__file__), "objective-interval-audit.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps({k: v for k, v in summary.items() if k not in ("objective_spreads_by_case", "native_interval_violations", "matched_fekete_interval_violations")}, indent=2))
print("wrote", out)
