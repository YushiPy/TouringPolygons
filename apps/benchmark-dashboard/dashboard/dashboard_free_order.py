from __future__ import annotations

import asyncio
import csv
import json
import math
import struct
import sys
import threading
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "benchmarks/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_cases import read_encoded_cases  # noqa: E402
from free_order_campaign import ensure_binary  # noqa: E402
from unordered_runner import run_unordered_solver  # noqa: E402

_build_lock = threading.Lock()


def free_command(request, campaign: Path, cli: Path, *, comparison: bool = False) -> list[str]:
    solvers = request.solvers if comparison else [request.solver or "unordered"]
    if not solvers or any(s not in {"unordered", "tspn"} for s in solvers):
        raise HTTPException(400, "Free order requires Unordered TPP B&B or External TSPN.")
    try:
        seconds = float(request.max_seconds or 30)
        calls = int(request.max_calls)
    except ValueError as error:
        raise HTTPException(400, "Free order requires one numeric time/call limit.") from error
    if not math.isfinite(seconds) or seconds <= 0 or calls < 0:
        raise HTTPException(400, "Seconds must be positive and calls nonnegative.")
    if request.threads not in (None, 1):
        raise HTTPException(400, "Free-order runs currently use one worker.")
    if "tspn" in solvers and seconds != int(seconds):
        raise HTTPException(400, "External TSPN requires whole seconds.")
    command = [sys.executable, str(cli), "free-order", str(campaign), "--threads", "1",
               "--max-calls", str(calls), "--max-seconds", str(seconds)]
    for solver in solvers:
        command += ["--solver", solver]
    if request.max_instances:
        command += ["--max-instances", str(request.max_instances)]
    for option in ("no_build", "force", "dry_run"):
        if getattr(request, option, False):
            command.append("--" + option.replace("_", "-"))
    return command


def free_results(campaign: Path) -> dict:
    files = sorted((campaign / "results/free-order").glob("*/report.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    if not files:
        return {"visit_order": "free", "title": campaign.name, "rows": [], "notes": ["No free-order run for this campaign yet."]}
    report = json.loads(files[0].read_text())
    report["path"] = str(files[0].relative_to(campaign))
    return report


def recorded_results() -> dict:
    path = ROOT / "benchmarks/results/unordered/final-dev.jsonl"
    if not path.exists():
        return {"visit_order": "free", "rows": [], "title": "Recorded comparison", "notes": ["Recorded results are not installed in this checkout."]}
    rows = [dict(json.loads(line), solver="unordered") for line in path.read_text().splitlines()]
    suite = ROOT / "benchmarks/suites/algorithm-dev-v1.bin"
    if suite.exists():
        cases = read_encoded_cases(suite)
        for row in rows:
            index = row["case"]
            if index < len(cases) and cases[index].digest == row["sha256"]:
                coords = struct.unpack_from("<dddd", cases[index].data)
                row["geometry"] = {"start": coords[:2], "target": coords[2:], "polygons": cases[index].polygons}
    external_files = sorted((ROOT / "tspn-comparison/results/unordered-final").glob("*/*-tspn-path.csv"))
    own_hashes = {r["case"]: r["sha256"] for r in rows}
    if external_files:
        with external_files[-1].open() as file:
            for external in csv.DictReader(file):
                index = int(external["case_index"])
                if external["mode"] != "path" or own_hashes.get(index) != external["sha256"]:
                    raise HTTPException(409, "Recorded comparison contains mismatched instances.")
                rows.append({"case": index, "sha256": external["sha256"], "solver": "tspn",
                             "polygons": int(external["polygons"]), "seconds": float(external["solve_seconds"]),
                             "lower_bound": float(external["lower_bound"]), "upper_bound": float(external["upper_bound"]),
                             "exact": external["is_optimal"] == "True", "endpoint_valid": external["is_valid_trajectory"] == "True",
                             "valid": None, "calls": int(external["soc_num_calls"]), "termination": external["status"]})
    return {"visit_order": "free", "title": "Recorded comparison · algorithm-dev-v1 · 2026-09-05",
            "status": "completed", "config": {"threads": 1, "max_seconds": 2}, "rows": rows,
            "notes": ["60 matched instances; fixed endpoints; one worker; 2 seconds per instance.",
                      "Our optimality tolerance: 1e-7 + 1e-9 × UB; external: 1e-6 relative, geometry tolerance 0.001.",
                      "External optimal flags are solver-reported. Its endpoint check at 1e-5 failed in 27 cases. Polygon coverage was independently checked only for our paths.",
                      "Compare both solved counts and gaps. Speedup on jointly solved cases does not describe total suite time."]}


async def solve_free_editor(case) -> dict:
    def run():
        with _build_lock:
            binary = ensure_binary()
        return run_unordered_solver(binary, case[0], case[1], case[2], 200000, 3)
    try:
        return await asyncio.to_thread(run)
    except Exception as error:
        raise HTTPException(500, f"Free-order solve failed: {error}") from error
