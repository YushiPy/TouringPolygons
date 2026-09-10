from __future__ import annotations

import asyncio
import csv
import json
import math
import struct
import sys
import threading
from copy import deepcopy
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
_live_binary: Path | None = None
_VISUALIZATION_FIELDS = ("geometry", "path")


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
    threads = request.threads or 1
    if "tspn" in solvers and seconds != int(seconds):
        raise HTTPException(400, "External TSPN requires whole seconds.")
    command = [
        sys.executable,
        str(cli),
        "free-order",
        str(campaign),
        "--threads",
        str(threads),
        "--max-calls",
        str(calls),
        "--max-seconds",
        str(seconds),
    ]
    for solver in solvers:
        command += ["--solver", solver]
    if request.max_instances:
        command += ["--max-instances", str(request.max_instances)]
    for option in ("no_build", "force", "dry_run"):
        if getattr(request, option, False):
            command.append("--" + option.replace("_", "-"))
    return command


def _latest_free_report(campaign: Path) -> Path | None:
    files = sorted(
        (campaign / "results/free-order").glob("*/report.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True
    )
    return files[0] if files else None


def _geometry_index(report_path: Path) -> tuple[set[str], dict[str, dict]]:
    path = report_path.with_name("geometry.json")
    hashes = {item.stem for item in report_path.with_name("geometry").glob("*.json")}
    if not path.exists():
        return hashes, {}
    data = json.loads(path.read_text())
    legacy = data.get("cases", data if "hashes" not in data else {})
    return hashes | set(data.get("hashes", [])) | set(legacy), legacy


def _geometry_for_hash(report_path: Path, digest: str | None) -> dict | None:
    if not digest:
        return None
    path = report_path.with_name("geometry") / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text())
    return _geometry_index(report_path)[1].get(digest)


def _compact_report(report: dict, endpoint: str) -> dict:
    catalog_hashes = set(report.get("geometry_catalog", []))
    compact = {key: deepcopy(value) for key, value in report.items() if key not in {"rows", "geometry_catalog"}}
    compact["rows"] = []
    for source in report.get("rows", []):
        has_geometry = (
            bool(source.get("geometry")) or (source.get("geometry_sha256") or source.get("sha256")) in catalog_hashes
        )
        row = {key: deepcopy(value) for key, value in source.items() if key not in _VISUALIZATION_FIELDS}
        row["visualization_available"] = bool(source.get("path")) and has_geometry
        compact["rows"].append(row)
    compact["visualizations_deferred"] = True
    compact["visualization_endpoint"] = endpoint
    return compact


def free_results(campaign: Path, *, endpoint: str = "") -> dict:
    report_path = _latest_free_report(campaign)
    if report_path is None:
        return {
            "visit_order": "free",
            "title": campaign.name,
            "rows": [],
            "notes": ["No free-order run for this campaign yet."],
        }
    report = json.loads(report_path.read_text())
    report["path"] = str(report_path.relative_to(campaign))
    report["geometry_catalog"] = list(_geometry_index(report_path)[0])
    return _compact_report(report, endpoint)


def free_result_case(campaign: Path, case_index: int) -> dict:
    report_path = _latest_free_report(campaign)
    if report_path is None:
        raise HTTPException(404, "No free-order report exists for this campaign.")
    report = json.loads(report_path.read_text())
    rows = [deepcopy(row) for row in report.get("rows", []) if int(row.get("case", -1)) == case_index]
    if not rows:
        raise HTTPException(404, "The requested case is absent from the latest report.")
    geometry = next((row.get("geometry") for row in rows if row.get("geometry")), None)
    geometry_hash = next((row.get("geometry_sha256") or row.get("sha256") for row in rows), None)
    geometry = geometry or _geometry_for_hash(report_path, geometry_hash)
    if geometry:
        for row in rows:
            row["geometry"] = geometry
    return {"case": case_index, "rows": rows}


def _recorded_results_full(*, include_geometry: bool = True) -> dict:
    path = ROOT / "benchmarks/results/unordered/final-dev.jsonl"
    if not path.exists():
        return {
            "visit_order": "free",
            "rows": [],
            "title": "Recorded comparison",
            "notes": ["Recorded results are not installed in this checkout."],
        }
    rows = [dict(json.loads(line), solver="unordered") for line in path.read_text().splitlines()]
    suite = ROOT / "benchmarks/suites/algorithm-dev-v1.bin"
    if suite.exists() and include_geometry:
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
                rows.append(
                    {
                        "case": index,
                        "sha256": external["sha256"],
                        "solver": "tspn",
                        "polygons": int(external["polygons"]),
                        "seconds": float(external["solve_seconds"]),
                        "lower_bound": float(external["lower_bound"]),
                        "upper_bound": float(external["upper_bound"]),
                        "exact": external["is_optimal"] == "True",
                        "endpoint_valid": external["is_valid_trajectory"] == "True",
                        "valid": None,
                        "calls": int(external["soc_num_calls"]),
                        "termination": external["status"],
                    }
                )
    return {
        "visit_order": "free",
        "title": "Recorded comparison · algorithm-dev-v1 · 2026-09-05",
        "status": "completed",
        "config": {"threads": 1, "max_seconds": 2},
        "rows": rows,
        "geometry_catalog": [row["sha256"] for row in rows if row["solver"] == "unordered"] if suite.exists() else [],
        "notes": [
            "60 matched instances; fixed endpoints; one worker; 2 seconds per instance.",
            "Our optimality tolerance: 1e-7 + 1e-9 × UB; external: 1e-6 relative, geometry tolerance 0.001.",
            "External optimal flags are solver-reported. Its endpoint check at 1e-5 failed in 27 cases. Polygon coverage was independently checked only for our paths.",
            "Compare both solved counts and gaps. Speedup on jointly solved cases does not describe total suite time.",
        ],
    }


def recorded_results(*, endpoint: str = "/api/free-order/reference/cases") -> dict:
    return _compact_report(_recorded_results_full(include_geometry=False), endpoint)


def recorded_result_case(case_index: int) -> dict:
    report = _recorded_results_full()
    rows = [row for row in report.get("rows", []) if int(row.get("case", -1)) == case_index]
    if not rows:
        raise HTTPException(404, "The requested recorded case does not exist.")
    geometry = next((row.get("geometry") for row in rows if row.get("geometry")), None)
    if geometry:
        for row in rows:
            row["geometry"] = geometry
    return {"case": case_index, "rows": rows}


async def solve_free_editor(case) -> dict:
    def run():
        global _live_binary
        with _build_lock:
            if _live_binary is None:
                _live_binary = ensure_binary()
            binary = _live_binary
        return run_unordered_solver(binary, case[0], case[1], case[2], 200000, 3)

    try:
        return await asyncio.to_thread(run)
    except Exception as error:
        raise HTTPException(500, f"Free-order solve failed: {error}") from error
