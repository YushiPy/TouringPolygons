from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
VISUALIZER_STATIC_ROOT = REPO_ROOT / "apps/visualizer-server/static"
BENCHMARK_CLI = REPO_ROOT / "benchmarks/tpp.py"
CONVERT_INSTANCES_SCRIPT = REPO_ROOT / "benchmarks/scripts/convert_instances.py"
CAMPAIGNS_ROOT = REPO_ROOT / "benchmarks/campaigns"
RESULTS_ROOT = REPO_ROOT / "benchmarks/results"
JOBS_PATH = APP_ROOT / ".jobs.json"
CANONICAL_SUITE = REPO_ROOT / "benchmarks/suites/canonical-v1.bin"
TRACKED_NONCONVEX_SUITE = REPO_ROOT / "benchmarks/suites/nonconvex/test_cases.bin"
GERMAN_INSTANCES_ZIP = REPO_ROOT / "tspn-comparison/solver/instances/instances_socg_simplified.zip"
SOLVERS = {
    "linear": "linear_search_lazy",
    "linear_disjoint": "linear_search_disjoint",
    "binary": "binary_search_lazy",
    "binary_disjoint": "binary_search_disjoint",
    "tan": "tan_jiang",
    "gurobi": "gurobi",
    "linear_search_lazy": "linear_search_lazy",
    "linear_search_disjoint": "linear_search_disjoint",
    "binary_search_lazy": "binary_search_lazy",
    "binary_search_disjoint": "binary_search_disjoint",
    "binary_search_eager": "binary_search_eager",
    "tan_jiang": "tan_jiang",
}
SOLVER_BINARY = REPO_ROOT / "build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp"
SOLVER_BUILD_CACHE = REPO_ROOT / "build/nonconvex-release/CMakeCache.txt"
OSM_SEARCH_ROOTS = [
    REPO_ROOT,
    Path.home() / "Downloads",
    Path.home() / "Documents",
    Path.home() / "Desktop",
]
OSM_SEARCH_EXCLUDES = {
    ".cache",
    ".git",
    ".venv",
    "Library",
    "node_modules",
    "__pycache__",
}


if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
from dashboard.dashboard_binary import (  # noqa: E402
    _binary_offset_cache,
    binary_case_count,
    counter_clockwise_case,
    read_binary_case,
    read_binary_cases,
    write_binary_cases,
)
from dashboard.dashboard_campaign_routes import register_campaign_routes  # noqa: E402
from dashboard.dashboard_campaigns import (  # noqa: E402
    campaign_input_label,
    first_input_file,
    instance_preview_list,
    preview_map,
    read_campaign_case,
    read_campaign_cases,
    result_preview_list,
    total_instance_count,
)
from dashboard.dashboard_files import (  # noqa: E402
    FILE_CACHE_LIMIT,
    _csv_cache,
    _json_cache,
    file_signature,
    read_csv_rows,
    read_json,
    read_result_rows,
    read_run_index,
    trim_file_caches,
)
from dashboard.dashboard_jobs import JobController  # noqa: E402
from dashboard.dashboard_models import (  # noqa: E402
    MAX_MANUAL_CASES,
    MAX_POLYGONS_PER_CASE,
    MAX_VERTICES_PER_CASE,
    CaseData,
    CompareSolversRequest,
    CreateOsmRequest,
    Job,
    ManualCaseRequest,
    RunCampaignRequest,
)
from dashboard.dashboard_previews import (  # noqa: E402
    PREVIEW_VERSION,
    campaign_previews_are_stale,
    case_bounds,
    ensure_instance_preview,
    preview_grid_metrics,
    preview_svg_is_stale,
    refresh_stale_previews,
    rewrite_dashboard_previews,
    sample_cases,
    svg_line,
    svg_points,
    write_case_preview,
    write_imported_previews,
)
from dashboard.dashboard_reports import (  # noqa: E402
    comparison_rows,
    completed_instance_count,
    latest_comparison_path,
    parse_float,
    parse_markdown_tables,
    summary_files,
    summary_result_rows,
)
from dashboard.dashboard_routes import register_support_routes  # noqa: E402

_COMPAT_EXPORTS = (
    FILE_CACHE_LIMIT,
    _binary_offset_cache,
    _csv_cache,
    file_signature,
    trim_file_caches,
    completed_instance_count,
    comparison_rows,
    PREVIEW_VERSION,
    campaign_previews_are_stale,
    case_bounds,
    instance_preview_list,
    preview_grid_metrics,
    preview_svg_is_stale,
    read_binary_case,
    read_campaign_case,
    read_campaign_cases,
    sample_cases,
    svg_line,
    svg_points,
    write_case_preview,
    CreateOsmRequest,
)

app = FastAPI(title="TPP Benchmark Dashboard")
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")
if VISUALIZER_STATIC_ROOT.exists():
    app.mount(
        "/visualizer-static",
        StaticFiles(directory=VISUALIZER_STATIC_ROOT),
        name="visualizer-static",
    )
templates = Jinja2Templates(directory=str(APP_ROOT / "templates"))


jobs: dict[str, Job] = {}


def campaign_path(name: str) -> Path:
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid campaign name.")
    return CAMPAIGNS_ROOT / name


def validate_manual_cases(cases: list[ManualCaseRequest]) -> None:
    if len(cases) > MAX_MANUAL_CASES:
        raise HTTPException(
            status_code=413,
            detail=f"A campaign may contain at most {MAX_MANUAL_CASES} cases.",
        )
    for case_index, case in enumerate(cases):
        if len(case.polygons) > MAX_POLYGONS_PER_CASE:
            raise HTTPException(status_code=413, detail=f"Case {case_index} has too many polygons.")
        vertex_count = sum(len(polygon) for polygon in case.polygons)
        if vertex_count > MAX_VERTICES_PER_CASE:
            raise HTTPException(status_code=413, detail=f"Case {case_index} has too many vertices.")


def manual_cases_path(path: Path) -> Path:
    return path / "manual-cases.json"


def manual_input_path(path: Path) -> Path:
    campaign_file = path / "campaign.json"
    if campaign_file.exists():
        data = read_json(campaign_file)
        for input_record in data.get("inputs", []):
            file_value = input_record.get("file")
            if isinstance(file_value, str) and file_value:
                input_path = path / file_value
                if input_path.exists() or file_value != "inputs/manual.bin":
                    return input_path
        campaign_named_input = path / "inputs" / f"{path.name}.bin"
        if campaign_named_input.exists():
            return campaign_named_input
    return path / "inputs" / "manual.bin"


def ensure_manual_binary_cache(path: Path) -> Path:
    """Rebuild the solver compatibility file when the editable store is present."""
    input_path = manual_input_path(path)
    cases_path = manual_cases_path(path)
    if not cases_path.exists():
        return input_path
    input_path.parent.mkdir(parents=True, exist_ok=True)
    write_binary_cases(input_path, read_manual_cases(path))
    campaign_file = path / "campaign.json"
    if campaign_file.exists():
        data = read_json(campaign_file)
        relative_input = str(input_path.relative_to(path))
        if data.get("inputs", [{}])[0].get("file") != relative_input:
            inputs = data.get("inputs")
            if not isinstance(inputs, list) or not inputs:
                inputs = [{}]
            inputs[0] = {**inputs[0], "file": relative_input}
            data["inputs"] = inputs
            campaign_file.write_text(json.dumps(data, indent=2) + "\n")
            _json_cache.pop(campaign_file, None)
    return input_path


def manual_case_to_data(case: CaseData, name: str | None = None, generated: bool = False) -> dict[str, Any]:
    start, target, polygons = case
    data: dict[str, Any] = {
        "start": [start[0], start[1]],
        "target": [target[0], target[1]],
        "polygons": [[[x, y] for x, y in polygon] for polygon in polygons],
    }
    if name:
        data["name"] = name
    if generated:
        data["generated"] = True
    return data


def manual_case_from_request(request: ManualCaseRequest) -> CaseData:
    polygons = [[(float(x), float(y)) for x, y in polygon] for polygon in request.polygons if len(polygon) >= 3]
    return (
        (float(request.start[0]), float(request.start[1])),
        (float(request.target[0]), float(request.target[1])),
        polygons,
    )


def manual_case_request_to_json(request: ManualCaseRequest) -> dict[str, Any]:
    return request.model_dump() if hasattr(request, "model_dump") else request.dict()


def read_manual_cases(path: Path) -> list[CaseData]:
    return [manual_case_from_request(case) for case in read_editable_case_requests(path)]


def read_editable_case_requests(path: Path) -> list[ManualCaseRequest]:
    if not manual_cases_path(path).exists():
        data = read_json(path / "campaign.json")
        cases: list[ManualCaseRequest] = []
        for input_record in data.get("inputs", []):
            file = input_record.get("file")
            instances = input_record.get("instances")
            if not isinstance(file, str):
                continue
            limit = instances if isinstance(instances, int) and instances >= 0 else 1000000
            for case in read_binary_cases(path / file, limit=limit):
                cases.append(ManualCaseRequest(**manual_case_to_data(case, generated=data.get("type") != "manual")))
        return cases
    raw_cases = read_json(manual_cases_path(path)).get("cases", [])
    if not isinstance(raw_cases, list):
        raise HTTPException(status_code=500, detail="Invalid manual case store.")
    cases: list[ManualCaseRequest] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise HTTPException(status_code=500, detail=f"Invalid manual case {index}.")
        try:
            request = ManualCaseRequest(**raw_case)
        except ValueError as error:
            raise HTTPException(status_code=500, detail=f"Invalid manual case {index}.") from error
        cases.append(request)
    return cases


def write_manual_case_requests(path: Path, cases: list[ManualCaseRequest]) -> None:
    manual_cases_path(path).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        **manual_case_to_data(manual_case_from_request(case), name=case.name),
                        **({"generated": True} if getattr(case, "generated", False) else {}),
                    }
                    for case in cases
                ],
            },
            indent=2,
        )
        + "\n"
    )


def write_manual_cases(path: Path, cases: list[CaseData]) -> None:
    write_manual_case_requests(path, [ManualCaseRequest(**manual_case_to_data(case)) for case in cases])


def rebuild_manual_campaign(
    path: Path,
    cases: list[ManualCaseRequest] | list[CaseData],
    *,
    rebuild_previews: bool = True,
) -> dict[str, Any]:
    case_requests = [
        (case if isinstance(case, ManualCaseRequest) else ManualCaseRequest(**manual_case_to_data(case)))
        for case in cases
    ]
    case_data = [manual_case_from_request(case) for case in case_requests]
    write_manual_case_requests(path, case_requests)
    results_dir = path / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    preview_dir = path / "previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    if case_data and rebuild_previews:
        previews, instance_previews = write_imported_previews(path, case_data)
    else:
        previews, instance_previews = {}, []

    campaign_file = path / "campaign.json"
    data = read_json(campaign_file)
    if data.get("type") != "manual":
        data["edited_from_type"] = data.get("edited_from_type") or data.get("type")
    data["generation"] = {
        "instances": len(case_data),
        "polygons": None,
        "format": "manual-json-v1",
        "edited": True,
    }
    data["inputs"] = [
        {
            "file": str(manual_input_path(path).relative_to(path)),
            "instances": len(case_data),
            "polygons_per_instance": None,
            "source": "manual-editor",
        }
    ]
    data["preview"] = previews.get("all")
    data["previews"] = previews
    data["instance_previews"] = instance_previews
    campaign_file.write_text(json.dumps(data, indent=2) + "\n")
    _json_cache.pop(campaign_file, None)
    _json_cache.pop(manual_cases_path(path), None)
    return campaign_summary(path)


def create_manual_campaign_data(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "inputs").mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "name": path.name,
        "type": "manual",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation": {
            "instances": 0,
            "polygons": None,
            "format": "manual-json-v1",
        },
        "inputs": [
            {
                "file": "inputs/manual.bin",
                "instances": 0,
                "polygons_per_instance": None,
                "source": "manual-editor",
            }
        ],
        "preview": None,
        "previews": {},
        "instance_previews": [],
        "benchmark_runs": [],
    }
    (path / "campaign.json").write_text(json.dumps(data, indent=2) + "\n")
    write_manual_cases(path, [])


def find_osm_files() -> list[dict[str, Any]]:
    paths: set[Path] = set()
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["mdfind", "kMDItemFSName == '*.osm.pbf'"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        for line in completed.stdout.splitlines():
            path = Path(line).expanduser()
            if path.is_file():
                paths.add(path.resolve())

    for root in OSM_SEARCH_ROOTS:
        if not root.exists():
            continue
        for directory, names, files in os.walk(root):
            names[:] = [name for name in names if name not in OSM_SEARCH_EXCLUDES and not name.startswith(".")]
            for file in files:
                if file.endswith(".osm.pbf"):
                    paths.add((Path(directory) / file).resolve())
            if len(paths) >= 200:
                break

    return [
        {
            "name": path.name,
            "path": str(path),
            "size": path.stat().st_size,
            "mtime": path.stat().st_mtime,
        }
        for path in sorted(paths, key=lambda item: (-item.stat().st_size, item.name.lower()))
    ]


def benchmarked_instances(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    data = read_json(path / "campaign.json")
    previews = result_preview_list(path, data)
    index = read_run_index(path / "results/run-index.csv")
    instances: list[dict[str, Any]] = []
    for run_row in index["rows"]:
        if run_row.get("status") != "completed":
            continue
        csv_value = run_row.get("csv_output", "")
        if not csv_value:
            continue
        csv_path = Path(csv_value)
        if not csv_path.is_absolute():
            csv_path = path / csv_path
        for result_row in read_result_rows(csv_path):
            try:
                case_index = int(result_row["case_index"])
            except (KeyError, ValueError):
                continue
            preview = previews[case_index] if 0 <= case_index < len(previews) else None
            solution_preview = solution_preview_path(path, csv_path, case_index, result_row.get("repeat_index", "0"))
            exhausted = result_row.get("exhausted") == "true"
            branch_limited = result_row.get("branch_limited") == "true"
            time_limited = result_row.get("time_limited") == "true"
            decomposition_seconds = parse_float(result_row.get("decomposition_seconds"))
            approximation_seconds = parse_float(result_row.get("approximation_seconds"))
            bnb_seconds = parse_float(result_row.get("bnb_seconds"))
            instances.append(
                {
                    "case_index": case_index,
                    "repeat_index": result_row.get("repeat_index", "0"),
                    "status": ("solved" if exhausted and not branch_limited and not time_limited else "capped"),
                    "preview": preview,
                    "final_length": result_row.get("final_length"),
                    "initial_length": result_row.get("initial_length"),
                    "decomposed_pieces": result_row.get("decomposed_pieces"),
                    "grouped_pieces": result_row.get("grouped_pieces"),
                    "calls": result_row.get("calls"),
                    "pruned_nodes": result_row.get("pruned_nodes"),
                    "visited_nodes": result_row.get("visited_nodes"),
                    "decomposition_seconds": result_row.get("decomposition_seconds"),
                    "approximation_seconds": result_row.get("approximation_seconds"),
                    "bnb_seconds": result_row.get("bnb_seconds"),
                    "solver_seconds": result_row.get("solver_seconds"),
                    "seconds_per_call": result_row.get("seconds_per_call"),
                    "total_seconds": f"{decomposition_seconds + approximation_seconds + bnb_seconds:.6f}",
                    "solution_preview": (
                        str(solution_preview.relative_to(path))
                        if solution_preview and solution_preview.exists()
                        else None
                    ),
                    "solution_available": bool(solution_preview and solution_preview.exists()),
                }
            )
            if len(instances) >= limit:
                return instances
    return instances


def solution_preview_path(path: Path, csv_path: Path, case_index: int, repeat_index: str) -> Path | None:
    try:
        repeat = int(repeat_index)
    except ValueError:
        repeat = 0
    solution_path = csv_path.parent / f"{csv_path.stem}-solutions" / f"case-{case_index:04}-repeat-{repeat:03}.svg"
    if solution_path.exists():
        return solution_path

    results_dir = path / "results"
    matches = (
        sorted(results_dir.glob(f"*-solutions/case-{case_index:04}-repeat-{repeat:03}.svg"))
        if results_dir.exists()
        else []
    )
    return matches[-1] if matches else solution_path


def comparison_data(path: Path) -> dict[str, Any]:
    comparison_path = latest_comparison_path(path)
    if not comparison_path:
        return {"rows": [], "input_file": campaign_input_label(path), "path": None}
    return {
        "rows": read_csv_rows(comparison_path),
        "input_file": campaign_input_label(path),
        "path": str(comparison_path.relative_to(path)),
    }


def campaign_summary(path: Path) -> dict[str, Any]:
    data = read_json(path / "campaign.json")
    campaign_file = path / "campaign.json"
    inputs = data.get("inputs", [])
    if data.get("type") == "manual" and manual_cases_path(path).exists():
        existing_inputs = len(inputs)
    else:
        existing_inputs = sum((path / record["file"]).exists() for record in inputs)
    previews = preview_map(data)
    inputs_available = (
        manual_cases_path(path).exists()
        if data.get("type") == "manual"
        else any((path / record["file"]).exists() for record in inputs if isinstance(record.get("file"), str))
    )
    run_index = read_run_index(path / "results/run-index.csv")
    total_instances = total_instance_count(data)
    completed_instances = completed_instance_count(path, run_index)
    return {
        "name": data.get("name", path.name),
        "order": data.get("display_order"),
        "type": data.get("type", "osm" if data.get("source") else "unknown"),
        "path": str(path),
        "created_utc": data.get("created_utc"),
        "generation": data.get("generation", {}),
        "inputs": {"existing": existing_inputs, "total": len(inputs)},
        "instance_progress": {
            "completed": completed_instances,
            "total": total_instances,
            "ratio": completed_instances / total_instances if total_instances else 0,
        },
        "preview": previews.get("all") or next(iter(previews.values()), None),
        "previews": previews,
        "instance_previews": result_preview_list(path, data),
        "has_preview": (bool(total_instances) and inputs_available) or any((path / preview).exists() for preview in previews.values()),
        "run_index": run_index,
        "benchmark_runs": data.get("benchmark_runs", []),
        "version": campaign_file.stat().st_mtime_ns,
    }


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/touringpolygons-pycache")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def ensure_live_solver_binary() -> None:
    configured_target = None
    if SOLVER_BUILD_CACHE.exists():
        for line in SOLVER_BUILD_CACHE.read_text().splitlines():
            if line.startswith("TARGET:STRING="):
                configured_target = line.split("=", 1)[1]
                break

    if SOLVER_BINARY.exists() and configured_target == "main-visualizer_solve":
        return

    subprocess.run(
        [
            "cmake",
            "--preset",
            "nonconvex-release",
            "-DTARGET=main-visualizer_solve",
            "-DTPP_ENABLE_GUROBI=OFF",
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", "--preset", "nonconvex-release"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def live_solver_input(case: CaseData, max_calls: int, max_seconds: float) -> str:
    start, target, polygons = counter_clockwise_case(case)
    lines = [
        f"{start[0]} {start[1]}",
        f"{target[0]} {target[1]}",
        f"{len(polygons)} {max_calls} {max_seconds}",
    ]
    for polygon in polygons:
        lines.append(str(len(polygon)))
        lines.extend(f"{x} {y}" for x, y in polygon)
    return "\n".join(lines) + "\n"


def parse_live_solver_output(output: str) -> dict[str, Any]:
    lines = output.strip().splitlines()
    if not lines:
        raise HTTPException(status_code=500, detail="Solver produced no output.")
    header = lines[0].split()
    if header[0] == "ERR":
        raise HTTPException(status_code=422, detail=" ".join(header[1:]))
    if len(header) != 5 or header[0] != "OK":
        raise HTTPException(status_code=500, detail="Invalid solver output.")
    exact = header[1] == "1"
    calls = int(header[2])
    seconds = float(header[3])
    point_count = int(header[4])
    if len(lines) != point_count + 1:
        raise HTTPException(status_code=500, detail="Truncated solver output.")
    path = []
    for line in lines[1:]:
        x, y = line.split()
        path.append([float(x), float(y)])
    return {"path": path, "exact": exact, "calls": calls, "seconds": seconds}


def import_binary_suite(
    *,
    name: str,
    source_suite: Path,
    input_filename: str,
    campaign_type: str,
    format_name: str,
    overwrite: bool,
    extra_generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = campaign_path(name)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    elif (path / "campaign.json").exists():
        raise HTTPException(status_code=400, detail="Campaign already exists.")

    input_path = path / "inputs" / input_filename
    input_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_suite, input_path)
    instances = binary_case_count(input_path)
    preview_cases = read_binary_cases(input_path, limit=instances)
    previews, instance_previews = write_imported_previews(path, preview_cases)
    generation = {
        "source_suite": str(source_suite),
        "instances": instances,
        "polygons": None,
        "format": format_name,
    }
    if extra_generation:
        generation.update(extra_generation)
    data = {
        "schema_version": 1,
        "name": path.name,
        "type": campaign_type,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation": generation,
        "inputs": [
            {
                "file": f"inputs/{input_filename}",
                "instances": instances,
                "polygons_per_instance": None,
                "source_suite": str(source_suite),
            }
        ],
        "preview": previews["all"] if previews else None,
        "previews": previews,
        "instance_previews": instance_previews,
        "benchmark_runs": [],
    }
    (path / "campaign.json").write_text(json.dumps(data, indent=2) + "\n")
    return {"ok": True, "campaign": campaign_summary(path)}


def append_generated_cases_to_campaign(
    destination_name: str,
    generated_path: Path,
    source_kind: str,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    destination_path = campaign_path(destination_name)
    if not (destination_path / "campaign.json").exists():
        raise HTTPException(status_code=404, detail="Append target campaign does not exist.")
    generated_data = read_json(generated_path / "campaign.json")
    generated_cases = read_campaign_cases(generated_path, generated_data)
    if not generated_cases:
        raise HTTPException(status_code=400, detail="Generated campaign has no instances to append.")

    existing_cases = read_editable_case_requests(destination_path)
    offset = len(existing_cases)
    appended_cases = [
        ManualCaseRequest(
            **manual_case_to_data(
                case,
                name=f"{source_kind} generated {offset + index + 1}",
                generated=True,
            )
        )
        for index, case in enumerate(generated_cases)
    ]
    next_cases = existing_cases + appended_cases
    validate_manual_cases(next_cases)
    rebuild_manual_campaign(destination_path, next_cases)

    campaign_file = destination_path / "campaign.json"
    data = read_json(campaign_file)
    edit_history = data.get("edit_history")
    if not isinstance(edit_history, list):
        edit_history = []
    edit_history.append(
        {
            "action": "append-generated",
            "source": source_kind,
            "appended_instances": len(appended_cases),
            "start_index": offset,
            "generator": generator_config,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    data["edit_history"] = edit_history
    campaign_file.write_text(json.dumps(data, indent=2) + "\n")
    _json_cache.pop(campaign_file, None)
    return campaign_summary(destination_path)


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


job_controller = JobController(
    jobs,
    jobs_path=JOBS_PATH,
    repo_root=REPO_ROOT,
    campaign_path=campaign_path,
    read_run_index=read_run_index,
    solvers=SOLVERS,
)
active_job = job_controller.active_job
persist_jobs = job_controller.persist_jobs
refresh_job_progress_from_logs = job_controller.refresh_job_progress_from_logs
run_job = job_controller.run_job
job_controller.load_jobs()


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/runs")
async def run_campaign(request: RunCampaignRequest):
    ensure_manual_binary_cache(campaign_path(request.name))
    existing_job = active_job("run", request.name)
    if existing_job is not None:
        return {"job": existing_job.id, "command": existing_job.command}

    command = [sys.executable, str(BENCHMARK_CLI), "run", request.name]
    if request.threads is not None:
        command.extend(["--threads", str(request.threads)])
    if request.solver:
        solver = SOLVERS.get(request.solver)
        if solver is None:
            raise HTTPException(status_code=400, detail="Unknown solver.")
        command.extend(["--solver", solver])
    if request.max_instances is not None:
        command.extend(["--max-instances", str(request.max_instances)])
    command.extend(["--max-calls", request.max_calls])
    if request.max_seconds:
        command.extend(["--max-seconds", request.max_seconds])
    if request.timeout is not None:
        command.extend(["--timeout", str(request.timeout)])
    if request.force:
        command.append("--force")
    if request.no_build:
        command.append("--no-build")
    if request.dry_run:
        command.append("--dry-run")

    job = Job(id=str(uuid.uuid4()), command=command, kind="run", campaign=request.name)
    jobs[job.id] = job
    persist_jobs()
    asyncio.create_task(run_job(job))
    return {"job": job.id, "command": command}


@app.post("/api/comparisons")
async def compare_solvers(request: CompareSolversRequest):
    if not request.solvers:
        raise HTTPException(status_code=400, detail="Select at least one solver.")
    existing_job = active_job("comparison", request.name)
    if existing_job is not None:
        return {"job": existing_job.id, "command": existing_job.command}

    path = campaign_path(request.name)
    ensure_manual_binary_cache(path)
    suite = first_input_file(path)
    command = [
        sys.executable,
        str(BENCHMARK_CLI),
        "compare-solvers",
        "--suite",
        str(suite),
        "--output",
        str(path / "results" / "comparisons"),
        "--max-calls",
        request.max_calls,
        "--max-instances",
        str(request.max_instances) if request.max_instances is not None else "-1",
        "--max-polygons",
        "-1",
        "--max-branching",
        "-1",
        "--keep-going",
    ]
    for solver_name in request.solvers:
        solver = SOLVERS.get(solver_name)
        if solver is None:
            raise HTTPException(status_code=400, detail=f"Unknown solver: {solver_name}")
        command.extend(["--solver", solver])
    if request.threads is not None:
        command.extend(["--threads", str(request.threads)])
    if request.max_seconds:
        command.extend(["--max-seconds", request.max_seconds])
    if request.no_build:
        command.append("--no-build")

    job = Job(
        id=str(uuid.uuid4()),
        command=command,
        kind="comparison",
        campaign=request.name,
        solver_progress_completed=0,
        solver_progress_total=len(request.solvers),
    )
    jobs[job.id] = job
    persist_jobs()
    asyncio.create_task(run_job(job))
    return {"job": job.id, "command": command}


register_campaign_routes(
    app,
    campaigns_root=CAMPAIGNS_ROOT,
    campaign_path=campaign_path,
    campaign_summary=campaign_summary,
    read_json=read_json,
    read_run_index=read_run_index,
    preview_map=preview_map,
    refresh_stale_previews=refresh_stale_previews,
    ensure_instance_preview=ensure_instance_preview,
    ensure_manual_binary_cache=ensure_manual_binary_cache,
    solution_preview_path=solution_preview_path,
    run_command=run_command,
    rewrite_dashboard_previews=rewrite_dashboard_previews,
    import_binary_suite=import_binary_suite,
    create_manual_campaign_data=create_manual_campaign_data,
    read_editable_case_requests=read_editable_case_requests,
    manual_case_request_to_json=manual_case_request_to_json,
    validate_manual_cases=validate_manual_cases,
    rebuild_manual_campaign=rebuild_manual_campaign,
    append_generated_cases_to_campaign=append_generated_cases_to_campaign,
    read_manual_cases=read_manual_cases,
    manual_case_from_request=manual_case_from_request,
    manual_case_to_data=manual_case_to_data,
    ensure_live_solver_binary=ensure_live_solver_binary,
    live_solver_input=live_solver_input,
    parse_live_solver_output=parse_live_solver_output,
    benchmark_cli=BENCHMARK_CLI,
    repo_root=REPO_ROOT,
    convert_instances_script=CONVERT_INSTANCES_SCRIPT,
    canonical_suite=CANONICAL_SUITE,
    tracked_nonconvex_suite=TRACKED_NONCONVEX_SUITE,
    german_instances_zip=GERMAN_INSTANCES_ZIP,
    solver_binary=SOLVER_BINARY,
)

register_support_routes(
    app,
    jobs=jobs,
    results_root=RESULTS_ROOT,
    campaigns_root=CAMPAIGNS_ROOT,
    repo_root=REPO_ROOT,
    persist_jobs=persist_jobs,
    refresh_job_progress_from_logs=refresh_job_progress_from_logs,
    campaign_path=campaign_path,
    find_osm_files=find_osm_files,
    summary_files=summary_files,
    parse_markdown_tables=parse_markdown_tables,
    summary_result_rows=summary_result_rows,
    campaign_input_label=campaign_input_label,
    benchmarked_instances=benchmarked_instances,
    comparison_data=comparison_data,
)
