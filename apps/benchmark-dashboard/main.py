from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
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
PREVIEW_VERSION = 6
FILE_CACHE_LIMIT = 128
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
	"gurobi": "gurobi",
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
from dashboard_models import (
	CaseData,
	CompareSolversRequest,
	CreateOsmRequest,
	CreateSyntheticRequest,
	ImportCanonicalRequest,
	ImportGermanRequest,
	Job,
	ManualCampaignRequest,
	ManualCaseRequest,
	ManualCasesRequest,
	MAX_MANUAL_CASES,
	MAX_POLYGONS_PER_CASE,
	MAX_VERTICES_PER_CASE,
	Point,
	RunCampaignRequest,
)
from dashboard_binary import (
	_binary_offset_cache,
	binary_case_count,
	binary_case_offsets,
	read_binary_case,
	read_binary_case_from_file,
	read_binary_cases,
	read_exact,
	read_point,
	read_size,
	skip_binary_case,
	skip_bytes,
	trim_binary_cache,
	write_binary_cases,
	write_size,
	write_vector,
)


app = FastAPI(title="TPP Benchmark Dashboard")
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")
if VISUALIZER_STATIC_ROOT.exists():
	app.mount("/visualizer-static", StaticFiles(directory=VISUALIZER_STATIC_ROOT), name="visualizer-static")
templates = Jinja2Templates(directory=str(APP_ROOT / "templates"))


_preview_lock = threading.RLock()


jobs: dict[str, Job] = {}
_json_cache: OrderedDict[Path, tuple[int, int, dict[str, Any]]] = OrderedDict()
_csv_cache: OrderedDict[tuple[Path, str], tuple[int, int, list[dict[str, str]]]] = OrderedDict()
PROGRESS_PATTERN = re.compile(r"cases\s+\|\s+\[[^\]]*\]\s+(\d+)\s*/\s*(\d+)")
SOLVER_SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def active_job(kind: str, campaign: str) -> Job | None:
	for job in jobs.values():
		if job.kind == kind and job.campaign == campaign and job.status in {"running", "stopping"}:
			return job
	return None


def campaign_path(name: str) -> Path:
	if "/" in name or "\\" in name or name in {"", ".", ".."}:
		raise HTTPException(status_code=400, detail="Invalid campaign name.")
	return CAMPAIGNS_ROOT / name


def file_signature(path: Path) -> tuple[int, int]:
	stat = path.stat()
	return stat.st_mtime_ns, stat.st_size


def clone_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
	return [row.copy() for row in rows]


def trim_file_caches() -> None:
	while len(_json_cache) > FILE_CACHE_LIMIT:
		_json_cache.popitem(last=False)
	while len(_csv_cache) > FILE_CACHE_LIMIT:
		_csv_cache.popitem(last=False)
	while len(_binary_offset_cache) > FILE_CACHE_LIMIT:
		_binary_offset_cache.popitem(last=False)


def read_json(path: Path) -> dict[str, Any]:
	try:
		signature = file_signature(path)
		cached = _json_cache.get(path)
		if cached and cached[:2] == signature:
			_json_cache.move_to_end(path)
			return dict(cached[2])
		data = json.loads(path.read_text())
		_json_cache[path] = (*signature, data)
		trim_file_caches()
		return dict(data)
	except FileNotFoundError as error:
		raise HTTPException(status_code=404, detail=f"Missing file: {path}") from error
	except json.JSONDecodeError as error:
		raise HTTPException(status_code=500, detail=f"Invalid JSON: {path}") from error


def read_run_index(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {"exists": False, "rows": [], "counts": {}}

	rows = read_csv_rows(path)

	counts: dict[str, int] = {}
	for row in rows:
		status = row.get("status", "unknown")
		counts[status] = counts.get(status, 0) + 1

	return {"exists": True, "rows": rows, "counts": counts}


def read_csv_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
	if not path.exists():
		return []
	signature = file_signature(path)
	cache_key = (path, delimiter)
	cached = _csv_cache.get(cache_key)
	if cached and cached[:2] == signature:
		_csv_cache.move_to_end(cache_key)
		return clone_rows(cached[2])
	with path.open(newline="") as file:
		rows = list(csv.DictReader(file, delimiter=delimiter))
	_csv_cache[cache_key] = (*signature, rows)
	trim_file_caches()
	return clone_rows(rows)


def read_result_rows(path: Path) -> list[dict[str, str]]:
	return read_csv_rows(path, delimiter=";")


def persist_jobs() -> None:
	JOBS_PATH.write_text(json.dumps([job.snapshot() for job in jobs.values()], indent=2) + "\n")


def load_jobs() -> None:
	if not JOBS_PATH.exists():
		return
	try:
		raw_jobs = json.loads(JOBS_PATH.read_text())
	except (OSError, json.JSONDecodeError):
		return
	if not isinstance(raw_jobs, list):
		return
	for raw_job in raw_jobs[-100:]:
		if not isinstance(raw_job, dict):
			continue
		job = Job(
			id=str(raw_job.get("id") or uuid.uuid4()),
			command=[str(part) for part in raw_job.get("command", [])],
			kind=str(raw_job.get("kind") or "run"),
			campaign=raw_job.get("campaign") if isinstance(raw_job.get("campaign"), str) else None,
			started_at=float(raw_job.get("started_at") or time.time()),
			finished_at=raw_job.get("finished_at") if isinstance(raw_job.get("finished_at"), float | int) else time.time(),
			returncode=raw_job.get("returncode") if isinstance(raw_job.get("returncode"), int) else 130,
			output=str(raw_job.get("output") or ""),
			progress_completed=raw_job.get("progress_completed") if isinstance(raw_job.get("progress_completed"), int) else None,
			progress_total=raw_job.get("progress_total") if isinstance(raw_job.get("progress_total"), int) else None,
			solver_progress_completed=raw_job.get("solver_progress_completed") if isinstance(raw_job.get("solver_progress_completed"), int) else None,
			solver_progress_total=raw_job.get("solver_progress_total") if isinstance(raw_job.get("solver_progress_total"), int) else None,
			current_solver=raw_job.get("current_solver") if isinstance(raw_job.get("current_solver"), str) else None,
			cancel_requested=bool(raw_job.get("cancel_requested")),
		)
		if raw_job.get("returncode") is None:
			job.returncode = 130
			job.finished_at = time.time()
			job.output = (job.output + "\nServer restarted before this job finished.").strip()
		jobs[job.id] = job


def parse_float(value: str | None) -> float:
	try:
		return float(value or "0")
	except ValueError:
		return 0.0


def validate_manual_cases(cases: list[ManualCaseRequest]) -> None:
	if len(cases) > MAX_MANUAL_CASES:
		raise HTTPException(status_code=413, detail=f"A campaign may contain at most {MAX_MANUAL_CASES} cases.")
	for case_index, case in enumerate(cases):
		if len(case.polygons) > MAX_POLYGONS_PER_CASE:
			raise HTTPException(status_code=413, detail=f"Case {case_index} has too many polygons.")
		vertex_count = sum(len(polygon) for polygon in case.polygons)
		if vertex_count > MAX_VERTICES_PER_CASE:
			raise HTTPException(status_code=413, detail=f"Case {case_index} has too many vertices.")


def case_bounds(case: CaseData) -> tuple[float, float, float, float]:
	start, target, polygons = case
	points = [start, target, *(point for polygon in polygons for point in polygon)]
	return (
		min(point[0] for point in points),
		min(point[1] for point in points),
		max(point[0] for point in points),
		max(point[1] for point in points),
	)


def svg_points(points: list[Point], offset_x: float, offset_y: float, scale: float) -> str:
	return " ".join(f"{offset_x + x * scale:.2f},{offset_y - y * scale:.2f}" for x, y in points)


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str, opacity: float = 1.0) -> str:
	return (
		f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
		f'stroke="{color}" stroke-opacity="{opacity:.2f}" stroke-width="1"/>'
	)


def preview_grid_metrics(scale: float) -> tuple[float, int]:
	decision_value = 83 / scale
	exponent = math.ceil(math.log10(decision_value)) if decision_value > 0 else 0
	multiplier = 1
	sub_grid_count = 4
	grid_scale = 10 ** exponent
	if grid_scale / 5 > decision_value:
		sub_grid_count = 3
		exponent -= 1
		multiplier = 2
	elif grid_scale / 2 > decision_value:
		exponent -= 1
		multiplier = 5
	return (10 ** exponent) * multiplier, sub_grid_count


def write_case_preview(
	path: Path,
	cases: list[CaseData],
	*,
	cell_width: int,
	cell_height: int,
	columns: int,
	padding: int = 10,
) -> None:
	if not cases:
		return
	columns = min(columns, len(cases))
	rows = (len(cases) + columns - 1) // columns
	width = columns * cell_width
	height = rows * cell_height
	colors = ["#38bdf8", "#a3e635", "#f97316", "#f472b6", "#c084fc"]
	elements = [
		f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
		f'viewBox="0 0 {width} {height}" data-preview-version="{PREVIEW_VERSION}">',
		'<rect width="100%" height="100%" fill="#121417"/>',
	]
	for case_index, case in enumerate(cases):
		start, target, polygons = case
		col = case_index % columns
		row = case_index // columns
		x0 = col * cell_width
		y0 = row * cell_height
		min_x, min_y, max_x, max_y = case_bounds(case)
		span_x = max(max_x - min_x, 1e-9)
		span_y = max(max_y - min_y, 1e-9)
		scale = min((cell_width - 2 * padding) / span_x, (cell_height - 2 * padding) / span_y)
		draw_width = span_x * scale
		draw_height = span_y * scale
		offset_x = x0 + (cell_width - draw_width) / 2 - min_x * scale
		offset_y = y0 + (cell_height + draw_height) / 2 + min_y * scale
		elements.append(f'<rect x="{x0}" y="{y0}" width="{cell_width}" height="{cell_height}" fill="#121417"/>')
		grid_step, sub_grid_count = preview_grid_metrics(scale)
		visible_min_x = (x0 - offset_x) / scale
		visible_max_x = (x0 + cell_width - offset_x) / scale
		visible_min_y = (offset_y - (y0 + cell_height)) / scale
		visible_max_y = (offset_y - y0) / scale
		first_x = math.floor(visible_min_x / grid_step) * grid_step
		first_y = math.floor(visible_min_y / grid_step) * grid_step
		x = first_x
		while x <= visible_max_x + grid_step:
			screen_x = offset_x + x * scale
			if x0 <= screen_x <= x0 + cell_width:
				elements.append(svg_line(screen_x, y0, screen_x, y0 + cell_height, "#515a67", 0.62))
				for index in range(sub_grid_count):
					sub_x = screen_x + (index + 1) * grid_step * scale / (sub_grid_count + 1)
					if x0 <= sub_x <= x0 + cell_width:
						elements.append(svg_line(sub_x, y0, sub_x, y0 + cell_height, "#2a2f38", 0.74))
			x += grid_step
		y = first_y
		while y <= visible_max_y + grid_step:
			screen_y = offset_y - y * scale
			if y0 <= screen_y <= y0 + cell_height:
				elements.append(svg_line(x0, screen_y, x0 + cell_width, screen_y, "#515a67", 0.62))
				for index in range(sub_grid_count):
					sub_y = screen_y - (index + 1) * grid_step * scale / (sub_grid_count + 1)
					if y0 <= sub_y <= y0 + cell_height:
						elements.append(svg_line(x0, sub_y, x0 + cell_width, sub_y, "#2a2f38", 0.74))
			y += grid_step
		origin_x = offset_x
		origin_y = offset_y
		if y0 <= origin_y <= y0 + cell_height:
			elements.append(svg_line(x0, origin_y, x0 + cell_width, origin_y, "#9aa3ad", 0.86))
		if x0 <= origin_x <= x0 + cell_width:
			elements.append(svg_line(origin_x, y0, origin_x, y0 + cell_height, "#9aa3ad", 0.86))
		for polygon_index, polygon in enumerate(polygons):
			color = colors[polygon_index % len(colors)]
			elements.append(
				f'<polygon points="{svg_points(polygon, offset_x, offset_y, scale)}" '
				f'fill="{color}" fill-opacity="0.20" stroke="{color}" stroke-width="2"/>'
			)
			for point_x, point_y in polygon:
				screen_x = offset_x + point_x * scale
				screen_y = offset_y - point_y * scale
				elements.append(f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="1.35" fill="{color}"/>')
		start_x = offset_x + start[0] * scale
		start_y = offset_y - start[1] * scale
		target_x = offset_x + target[0] * scale
		target_y = offset_y - target[1] * scale
		elements.append(f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="5" fill="#22c55e"/>')
		elements.append(f'<text x="{start_x + 10:.2f}" y="{start_y + 4:.2f}" font-size="12" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" fill="#f8fafc">s</text>')
		elements.append(f'<circle cx="{target_x:.2f}" cy="{target_y:.2f}" r="5" fill="#ef4444"/>')
		elements.append(f'<text x="{target_x + 10:.2f}" y="{target_y + 4:.2f}" font-size="12" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" fill="#f8fafc">t</text>')
	elements.append("</svg>")
	path.parent.mkdir(parents=True, exist_ok=True)
	content = "\n".join(elements) + "\n"
	with _preview_lock:
		with tempfile.NamedTemporaryFile(
			mode="w",
			encoding="utf-8",
			dir=path.parent,
			prefix=f".{path.name}.",
			suffix=".tmp",
			delete=False,
		) as temporary:
			temporary.write(content)
		temporary_path = Path(temporary.name)
		try:
			os.replace(temporary_path, path)
		finally:
			temporary_path.unlink(missing_ok=True)


def write_imported_previews(path: Path, cases: list[CaseData]) -> tuple[dict[str, str], list[str]]:
	preview_dir = path / "previews"
	overview_cases = sample_cases(cases, 20)
	overview_columns = 5 if len(overview_cases) <= 10 else 7
	overview_rows = (len(overview_cases) + overview_columns - 1) // overview_columns
	overview_cell_height = round((overview_columns * 150) / max(overview_rows, 1) / 2.52)
	write_case_preview(preview_dir / "selected.svg", cases[:1], cell_width=420, cell_height=320, columns=1, padding=8)
	write_case_preview(preview_dir / "four.svg", cases[:4], cell_width=210, cell_height=160, columns=2, padding=6)
	write_case_preview(preview_dir / "all.svg", overview_cases, cell_width=150, cell_height=overview_cell_height, columns=overview_columns, padding=6)
	instance_paths = [
		f"previews/instances/case-{index:04}.svg"
		for index in range(len(cases))
	]
	previews = {
		"selected": "previews/selected.svg",
		"four": "previews/four.svg",
		"all": "previews/all.svg",
	}
	return previews, instance_paths


def ensure_instance_preview(path: Path, data: dict[str, Any], index: int) -> Path | None:
	if index < 0:
		return None
	instance_previews = instance_preview_list(data)
	if index >= len(instance_previews):
		return None
	preview_path = path / instance_previews[index]
	if preview_path.exists() and not preview_svg_is_stale(preview_path):
		return preview_path
	case = read_campaign_case(path, data, index)
	if case is None:
		return None
	write_case_preview(preview_path, [case], cell_width=260, cell_height=180, columns=1, padding=6)
	return preview_path


def sample_cases(cases: list[CaseData], limit: int) -> list[CaseData]:
	if len(cases) <= limit:
		return cases
	return [cases[round(index * (len(cases) - 1) / (limit - 1))] for index in range(limit)]


def manual_cases_path(path: Path) -> Path:
	return path / "manual-cases.json"


def manual_input_path(path: Path) -> Path:
	return path / "inputs" / "manual.bin"


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
	polygons = [
		[(float(x), float(y)) for x, y in polygon]
		for polygon in request.polygons
		if len(polygon) >= 3
	]
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
	manual_cases_path(path).write_text(json.dumps({
		"schema_version": 1,
		"cases": [
			{
				**manual_case_to_data(manual_case_from_request(case), name=case.name),
				**({"generated": True} if getattr(case, "generated", False) else {}),
			}
			for case in cases
		],
	}, indent=2) + "\n")


def write_manual_cases(path: Path, cases: list[CaseData]) -> None:
	write_manual_case_requests(path, [ManualCaseRequest(**manual_case_to_data(case)) for case in cases])


def rebuild_manual_campaign(
	path: Path,
	cases: list[ManualCaseRequest] | list[CaseData],
	*,
	rebuild_previews: bool = True,
) -> dict[str, Any]:
	case_requests = [
		case if isinstance(case, ManualCaseRequest) else ManualCaseRequest(**manual_case_to_data(case))
		for case in cases
	]
	case_data = [manual_case_from_request(case) for case in case_requests]
	write_manual_case_requests(path, case_requests)
	write_binary_cases(manual_input_path(path), case_data)
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
	data["inputs"] = [{
		"file": "inputs/manual.bin",
		"instances": len(case_data),
		"polygons_per_instance": None,
		"source": "manual-editor",
	}]
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
		"inputs": [{
			"file": "inputs/manual.bin",
			"instances": 0,
			"polygons_per_instance": None,
			"source": "manual-editor",
		}],
		"preview": None,
		"previews": {},
		"instance_previews": [],
		"benchmark_runs": [],
	}
	(path / "campaign.json").write_text(json.dumps(data, indent=2) + "\n")
	write_manual_cases(path, [])
	write_binary_cases(manual_input_path(path), [])


def total_instance_count(data: dict[str, Any]) -> int:
	input_total = 0
	for input_record in data.get("inputs", []):
		instances = input_record.get("instances")
		if isinstance(instances, int):
			input_total += instances
	if input_total > 0:
		return input_total
	generation_instances = data.get("generation", {}).get("instances")
	return generation_instances if isinstance(generation_instances, int) else 0


def completed_instance_count(path: Path, run_index: dict[str, Any] | None = None) -> int:
	index = run_index or read_run_index(path / "results/run-index.csv")
	completed_cases: set[tuple[str, str]] = set()
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
			case_index = result_row.get("case_index")
			if case_index is not None:
				completed_cases.add((str(csv_path), case_index))
	return len(completed_cases)


def refresh_job_progress_from_logs(job: Job) -> None:
	if job.campaign is None or job.kind != "run" or job.status != "running":
		return
	path = campaign_path(job.campaign)
	index = read_run_index(path / "results/run-index.csv")
	for run_row in reversed(index["rows"]):
		if run_row.get("action") != "running":
			continue
		log_value = run_row.get("log_output", "")
		if not log_value:
			continue
		log_path = Path(log_value)
		if not log_path.is_absolute():
			log_path = path / log_path
		if log_path.exists():
			update_job_progress(job, log_path.read_text(errors="replace")[-12000:])
			return


def parse_markdown_tables(text: str) -> list[dict[str, Any]]:
	tables: list[dict[str, Any]] = []
	section = ""
	lines = text.splitlines()
	index = 0
	while index < len(lines):
		line = lines[index].strip()
		if line.startswith("## "):
			section = line.removeprefix("## ").strip()
			index += 1
			continue
		if not line.startswith("|") or index + 1 >= len(lines):
			index += 1
			continue
		separator = lines[index + 1].strip()
		if not separator.startswith("|") or "---" not in separator:
			index += 1
			continue
		headers = [cell.strip() for cell in line.strip("|").split("|")]
		rows: list[dict[str, str]] = []
		index += 2
		while index < len(lines) and lines[index].strip().startswith("|"):
			cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
			if len(cells) == len(headers):
				rows.append(dict(zip(headers, cells, strict=True)))
			index += 1
		title = headers[0] if headers else "Table"
		if section == "Distributions" and title == "Metric":
			title = "Metric:distributions"
		tables.append({"section": section, "title": title, "headers": headers, "rows": rows})
	return tables


def summary_files(path: Path) -> list[Path]:
	results_dir = path / "results"
	if not results_dir.exists():
		return []
	return sorted(
		(file for file in results_dir.glob("*.md") if file.is_file()),
		key=lambda file: file.stat().st_mtime,
		reverse=True,
	)


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


def preview_map(data: dict[str, Any]) -> dict[str, str]:
	previews = data.get("previews")
	if isinstance(previews, dict):
		return {
			str(name): str(value)
			for name, value in previews.items()
			if isinstance(value, str)
		}
	preview = data.get("preview")
	if isinstance(preview, str) and preview:
		return {"all": preview}
	return {}


def instance_preview_list(data: dict[str, Any]) -> list[str]:
	previews = data.get("instance_previews")
	if isinstance(previews, list):
		return [str(value) for value in previews if isinstance(value, str)]
	return []


def result_preview_list(path: Path, data: dict[str, Any]) -> list[str]:
	previews = instance_preview_list(data)
	if previews:
		return previews

	found: list[Path] = []
	for input_record in data.get("inputs", []):
		file_value = input_record.get("file")
		if not isinstance(file_value, str):
			continue
		stem = Path(file_value).stem
		for directory in (
			path / "previews" / "instances",
			path / "inputs" / f"{stem}-instances",
			path / "previews" / f"{stem}-instances",
		):
			if directory.exists():
				found.extend(sorted(directory.glob("case-*.*")))

	return [str(preview.relative_to(path)) for preview in found if preview.suffix.lower() in {".png", ".svg"}]


def preview_svg_is_stale(path: Path) -> bool:
	if path.suffix.lower() != ".svg" or not path.exists():
		return False
	try:
		text = path.read_text(errors="ignore")
	except OSError:
		return False
	return (
		">case " in text
		or 'fill="#ffffff"' in text
		or f'data-preview-version="{PREVIEW_VERSION}"' not in text
	)


def campaign_previews_are_stale(path: Path, data: dict[str, Any]) -> bool:
	candidates = list(preview_map(data).values()) + instance_preview_list(data)[:1]
	if not candidates:
		return total_instance_count(data) > 0
	return any(preview_svg_is_stale(path / preview) for preview in candidates)


def read_campaign_cases(path: Path, data: dict[str, Any]) -> list[CaseData]:
	cases: list[CaseData] = []
	total = total_instance_count(data)
	for input_record in data.get("inputs", []):
		file_value = input_record.get("file")
		if not isinstance(file_value, str):
			continue
		input_path = path / file_value
		if not input_path.exists():
			continue
		remaining = max(0, total - len(cases)) if total else 10_000
		if remaining == 0:
			break
		cases.extend(read_binary_cases(input_path, remaining))
	return cases


def read_campaign_case(path: Path, data: dict[str, Any], index: int) -> CaseData | None:
	if index < 0:
		return None
	seen = 0
	for input_record in data.get("inputs", []):
		file_value = input_record.get("file")
		if not isinstance(file_value, str):
			continue
		input_path = path / file_value
		if not input_path.exists():
			continue
		instances = input_record.get("instances")
		input_count = instances if isinstance(instances, int) and instances >= 0 else binary_case_count(input_path)
		if index < seen + input_count:
			return read_binary_case(input_path, index - seen)
		seen += input_count
	return None


def refresh_stale_previews(path: Path, data: dict[str, Any]) -> dict[str, Any]:
	if not campaign_previews_are_stale(path, data):
		return data
	cases = read_campaign_cases(path, data)
	if not cases:
		return data
	previews, instance_previews = write_imported_previews(path, cases)
	data["preview"] = previews.get("all")
	data["previews"] = previews
	data["instance_previews"] = instance_previews
	campaign_file = path / "campaign.json"
	campaign_file.write_text(json.dumps(data, indent=2) + "\n")
	_json_cache.pop(campaign_file, None)
	return data


def rewrite_dashboard_previews(path: Path) -> None:
	campaign_file = path / "campaign.json"
	if not campaign_file.exists():
		return
	data = read_json(campaign_file)
	cases = read_campaign_cases(path, data)
	if not cases:
		return
	previews, instance_previews = write_imported_previews(path, cases)
	data["preview"] = previews.get("all")
	data["previews"] = previews
	data["instance_previews"] = instance_previews
	campaign_file.write_text(json.dumps(data, indent=2) + "\n")
	_json_cache.pop(campaign_file, None)


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
			instances.append({
				"case_index": case_index,
				"repeat_index": result_row.get("repeat_index", "0"),
				"status": "solved" if exhausted and not branch_limited and not time_limited else "capped",
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
				"solution_preview": str(solution_preview.relative_to(path)) if solution_preview and solution_preview.exists() else None,
				"solution_available": bool(solution_preview and solution_preview.exists()),
			})
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
	matches = sorted(results_dir.glob(f"*-solutions/case-{case_index:04}-repeat-{repeat:03}.svg")) if results_dir.exists() else []
	return matches[-1] if matches else solution_path


def first_input_file(path: Path) -> Path:
	data = read_json(path / "campaign.json")
	for input_record in data.get("inputs", []):
		file_value = input_record.get("file")
		if not isinstance(file_value, str):
			continue
		input_path = path / file_value
		if input_path.exists():
			return input_path
	raise HTTPException(status_code=400, detail="Campaign has no generated input file.")


def campaign_input_label(path: Path) -> str | None:
	data = read_json(path / "campaign.json")
	for input_record in data.get("inputs", []):
		file_value = input_record.get("file")
		if isinstance(file_value, str) and file_value:
			return file_value
	return None


def comparison_rows(path: Path) -> list[dict[str, str]]:
	comparison_root = path / "results" / "comparisons"
	if not comparison_root.exists():
		return []
	candidates = sorted(
		comparison_root.glob("*/comparison.csv"),
		key=lambda file: file.stat().st_mtime,
		reverse=True,
	)
	if not candidates:
		return []
	return read_csv_rows(candidates[0])


def comparison_data(path: Path) -> dict[str, Any]:
	comparison_root = path / "results" / "comparisons"
	if not comparison_root.exists():
		return {"rows": [], "input_file": campaign_input_label(path), "path": None}
	candidates = sorted(
		comparison_root.glob("*/comparison.csv"),
		key=lambda file: file.stat().st_mtime,
		reverse=True,
	)
	if not candidates:
		return {"rows": [], "input_file": campaign_input_label(path), "path": None}
	rows = read_csv_rows(candidates[0])
	return {
		"rows": rows,
		"input_file": campaign_input_label(path),
		"path": str(candidates[0].relative_to(path)),
	}


def summary_result_rows(summary_path: Path) -> list[dict[str, str]]:
	return read_result_rows(summary_path.with_suffix(".csv"))


def campaign_summary(path: Path) -> dict[str, Any]:
	data = read_json(path / "campaign.json")
	campaign_file = path / "campaign.json"
	inputs = data.get("inputs", [])
	existing_inputs = sum((path / record["file"]).exists() for record in inputs)
	previews = preview_map(data)
	run_index = read_run_index(path / "results/run-index.csv")
	total_instances = total_instance_count(data)
	completed_instances = completed_instance_count(path, run_index)
	return {
		"name": data.get("name", path.name),
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
		"has_preview": any((path / preview).exists() for preview in previews.values()),
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
	start, target, polygons = case
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
		"inputs": [{
			"file": f"inputs/{input_filename}",
			"instances": instances,
			"polygons_per_instance": None,
			"source_suite": str(source_suite),
		}],
		"preview": previews["all"] if previews else None,
		"previews": previews,
		"instance_previews": instance_previews,
		"benchmark_runs": [],
	}
	(path / "campaign.json").write_text(json.dumps(data, indent=2) + "\n")
	return {"ok": True, "campaign": campaign_summary(path)}


def clamp_float(value: float, minimum: float, maximum: float) -> float:
	return max(minimum, min(maximum, value))


def update_job_progress(job: Job, output: str) -> None:
	for match in PROGRESS_PATTERN.finditer(output):
		job.progress_completed = int(match.group(1))
		job.progress_total = int(match.group(2))
	if job.kind == "comparison" and job.solver_progress_total is not None:
		known_solvers = set(SOLVERS.values())
		solvers = [
			match.group(1).strip()
			for match in SOLVER_SECTION_PATTERN.finditer(output)
			if match.group(1).strip() in known_solvers
		]
		if solvers:
			job.current_solver = solvers[-1]
			job.solver_progress_completed = min(job.solver_progress_total, max(0, len(solvers) - 1))


async def run_job(job: Job) -> None:
	env = os.environ.copy()
	env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/touringpolygons-pycache")
	process = await asyncio.create_subprocess_exec(
		*job.command,
		cwd=REPO_ROOT,
		env=env,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.STDOUT,
		start_new_session=True,
	)
	job.process = process
	assert process.stdout is not None
	output_parts: list[str] = []
	while True:
		chunk = await process.stdout.read(1024)
		if not chunk:
			break
		text = chunk.decode(errors="replace")
		output_parts.append(text)
		job.output = "".join(output_parts)[-20000:]
		update_job_progress(job, job.output)
	job.returncode = await process.wait()
	if job.returncode == 0 and job.progress_total is not None:
		job.progress_completed = job.progress_total
	if job.returncode == 0 and job.solver_progress_total is not None:
		job.solver_progress_completed = job.solver_progress_total
	job.finished_at = time.time()
	job.process = None
	persist_jobs()


load_jobs()


@app.get("/")
async def index(request: Request):
	return templates.TemplateResponse(request, "index.html")


@app.get("/api/campaigns")
async def list_campaigns():
	CAMPAIGNS_ROOT.mkdir(parents=True, exist_ok=True)
	campaigns = [
		campaign_summary(path)
		for path in sorted(CAMPAIGNS_ROOT.iterdir())
		if path.is_dir() and (path / "campaign.json").exists()
	]
	return {"campaigns": campaigns}


@app.get("/api/campaigns/{name}")
async def get_campaign(name: str):
	return campaign_summary(campaign_path(name))


@app.get("/api/campaigns/{name}/preview")
async def get_preview(name: str):
	return await get_preview_kind(name, "all")


@app.get("/api/campaigns/{name}/preview/{kind}")
async def get_preview_kind(name: str, kind: str):
	path = campaign_path(name)
	data = read_json(path / "campaign.json")
	if not kind.startswith("instance-"):
		data = refresh_stale_previews(path, data)
	previews = preview_map(data)
	if kind.startswith("instance-"):
		try:
			index = int(kind.removeprefix("instance-"))
		except ValueError as error:
			raise HTTPException(status_code=404, detail="Invalid instance preview.") from error
		preview_path = ensure_instance_preview(path, data, index)
		if not preview_path:
			raise HTTPException(status_code=404, detail="Campaign has no preview.")
		return FileResponse(preview_path)
	else:
		preview = previews.get(kind) or previews.get("all") or next(iter(previews.values()), None)
	if not preview:
		raise HTTPException(status_code=404, detail="Campaign has no preview.")
	preview_path = path / preview
	if not preview_path.exists():
		raise HTTPException(status_code=404, detail="Preview file does not exist.")
	return FileResponse(preview_path)


@app.get("/api/campaigns/{name}/solution-preview/{case_index}")
async def get_solution_preview(name: str, case_index: int, repeat_index: int = 0):
	path = campaign_path(name)
	index = read_run_index(path / "results/run-index.csv")
	for run_row in reversed(index["rows"]):
		csv_value = run_row.get("csv_output", "")
		if not csv_value:
			continue
		csv_path = Path(csv_value)
		if not csv_path.is_absolute():
			csv_path = path / csv_path
		preview_path = solution_preview_path(path, csv_path, case_index, str(repeat_index))
		if preview_path and preview_path.exists():
			return FileResponse(preview_path)
	raise HTTPException(status_code=404, detail="Solution preview does not exist.")


@app.post("/api/campaigns/synthetic")
async def create_synthetic(request: CreateSyntheticRequest):
	command = [
		sys.executable,
		str(BENCHMARK_CLI),
		"create",
		request.name,
		"--vertices", request.vertices,
		"--polygons", str(request.polygons),
		"--instances", str(request.instances),
		"--shape", request.shape,
		"--seed", str(request.seed),
	]
	if request.no_preview:
		command.append("--no-preview")
	if request.overwrite:
		command.append("--overwrite")

	completed = run_command(command)
	if completed.returncode != 0:
		return JSONResponse(
			{"ok": False, "output": completed.stdout},
			status_code=400,
		)
	path = campaign_path(request.name)
	if not request.no_preview:
		rewrite_dashboard_previews(path)
	return {"ok": True, "output": completed.stdout, "campaign": campaign_summary(path)}


@app.post("/api/campaigns/osm")
async def create_osm(request: CreateOsmRequest):
	simplify_tolerance = clamp_float(request.simplify_tolerance, 0.0, 10.0)
	scale = clamp_float(request.scale, 0.1, 10.0)
	grid_polygon_size = clamp_float(request.grid_polygon_size, 0.1, 20.0)
	grid_cell_size = clamp_float(request.grid_cell_size, 0.2, 40.0)
	if grid_cell_size <= grid_polygon_size:
		grid_cell_size = min(40.0, grid_polygon_size + 0.1)
	convex_replacement_fraction = clamp_float(request.convex_replacement_fraction, 0.0, 1.0)
	command = [
		sys.executable,
		str(BENCHMARK_CLI),
		"generate-matrix",
		request.name,
		request.pbf_path,
		"--instances", str(request.instances),
		"--polygon-count", str(request.polygon_counts),
		"--layout", request.layout,
		"--grid-cell-size", str(grid_cell_size),
		"--simplify-tolerance", str(simplify_tolerance),
		"--normalization", request.normalization,
		"--scale", str(scale),
		"--sampling", request.sampling,
		"--local-pool-size", str(request.local_pool_size),
		"--grid-polygon-size", str(grid_polygon_size),
		"--grid-placement", request.grid_placement,
		"--convex-replacement-fraction", str(convex_replacement_fraction),
		"--convex-replacement-vertices", str(request.convex_replacement_vertices),
		"--convex-replacement-position", request.convex_replacement_position,
		"--order", request.order,
		"--endpoint-mode", request.endpoint_mode,
		"--single-preview-count", str(request.instances),
		"--seed", str(request.seed),
		"--with-manifest",
	]
	if request.grid_columns is not None:
		command.extend(["--grid-columns", str(request.grid_columns)])
	if request.sample_size is not None:
		command.extend(["--sample-size", str(request.sample_size)])
	if not request.no_preview:
		command.append("--with-preview")
	if request.overwrite:
		command.append("--overwrite")

	completed = run_command(command)
	if completed.returncode != 0:
		return JSONResponse(
			{"ok": False, "output": completed.stdout},
			status_code=400,
		)
	path = campaign_path(request.name)
	if not request.no_preview:
		rewrite_dashboard_previews(path)
	return {"ok": True, "output": completed.stdout, "campaign": campaign_summary(path)}


@app.post("/api/campaigns/canonical")
async def import_canonical(request: ImportCanonicalRequest):
	source_suite = CANONICAL_SUITE if CANONICAL_SUITE.exists() else TRACKED_NONCONVEX_SUITE
	if not source_suite.exists():
		raise HTTPException(status_code=404, detail="No canonical or tracked nonconvex suite exists.")
	return import_binary_suite(
		name=request.name,
		source_suite=source_suite,
		input_filename="canonical-v1.bin",
		campaign_type="canonical",
		format_name="canonical-v1",
		overwrite=request.overwrite,
	)


@app.post("/api/campaigns/german")
async def import_german(request: ImportGermanRequest):
	if GERMAN_INSTANCES_ZIP.exists():
		completed = run_command([
			sys.executable,
			str(CONVERT_INSTANCES_SCRIPT),
			"--input", str(GERMAN_INSTANCES_ZIP),
			"--output", str(TRACKED_NONCONVEX_SUITE),
		])
		if completed.returncode != 0:
			return JSONResponse(
				{"ok": False, "output": completed.stdout},
				status_code=400,
			)
	elif not TRACKED_NONCONVEX_SUITE.exists():
		raise HTTPException(status_code=404, detail="No German instances zip or converted nonconvex suite exists.")

	return import_binary_suite(
		name=request.name,
		source_suite=TRACKED_NONCONVEX_SUITE,
		input_filename="german-instances.bin",
		campaign_type="german",
		format_name="socg-simplified",
		overwrite=request.overwrite,
		extra_generation={"source_zip": str(GERMAN_INSTANCES_ZIP) if GERMAN_INSTANCES_ZIP.exists() else None},
	)


@app.post("/api/campaigns/manual")
async def create_manual_campaign(request: ManualCampaignRequest):
	path = campaign_path(request.name)
	if path.exists() and request.overwrite:
		shutil.rmtree(path)
	elif (path / "campaign.json").exists():
		raise HTTPException(status_code=400, detail="Campaign already exists.")
	create_manual_campaign_data(path)
	return {"ok": True, "campaign": campaign_summary(path), "cases": []}


@app.get("/api/campaigns/{name}/cases")
async def get_manual_cases(name: str):
	path = campaign_path(name)
	read_json(path / "campaign.json")
	return {"cases": [manual_case_request_to_json(case) for case in read_editable_case_requests(path)]}


@app.put("/api/campaigns/{name}/cases")
async def replace_manual_cases(name: str, request: ManualCasesRequest, refresh_previews: bool = True):
	path = campaign_path(name)
	read_json(path / "campaign.json")
	validate_manual_cases(request.cases)
	campaign = rebuild_manual_campaign(path, request.cases, rebuild_previews=refresh_previews)
	return {"ok": True, "campaign": campaign, "cases": [manual_case_request_to_json(case) for case in request.cases]}


@app.post("/api/campaigns/{name}/cases")
async def append_manual_case(name: str, request: ManualCaseRequest):
	path = campaign_path(name)
	cases = read_manual_cases(path)
	cases.append(manual_case_from_request(request))
	validate_manual_cases([ManualCaseRequest(**manual_case_to_data(case)) for case in cases])
	campaign = rebuild_manual_campaign(path, cases)
	return {"ok": True, "index": len(cases) - 1, "campaign": campaign}


@app.put("/api/campaigns/{name}/cases/{case_index}")
async def update_manual_case(name: str, case_index: int, request: ManualCaseRequest):
	path = campaign_path(name)
	cases = read_manual_cases(path)
	if case_index < 0 or case_index >= len(cases):
		raise HTTPException(status_code=404, detail="Case does not exist.")
	cases[case_index] = manual_case_from_request(request)
	validate_manual_cases([ManualCaseRequest(**manual_case_to_data(case)) for case in cases])
	campaign = rebuild_manual_campaign(path, cases)
	return {"ok": True, "campaign": campaign}


@app.delete("/api/campaigns/{name}/cases/{case_index}")
async def delete_manual_case(name: str, case_index: int):
	path = campaign_path(name)
	cases = read_manual_cases(path)
	if case_index < 0 or case_index >= len(cases):
		raise HTTPException(status_code=404, detail="Case does not exist.")
	del cases[case_index]
	campaign = rebuild_manual_campaign(path, cases)
	return {"ok": True, "campaign": campaign}


@app.post("/api/editor/solve")
async def solve_editor_case(request: ManualCaseRequest):
	validate_manual_cases([request])
	case = manual_case_from_request(request)
	if len(case[2]) == 0:
		return {"path": [list(case[0]), list(case[1])], "exact": True, "calls": 0, "seconds": 0}
	try:
		ensure_live_solver_binary()
		completed = subprocess.run(
			[SOLVER_BINARY],
			input=live_solver_input(case, max_calls=200000, max_seconds=3.0),
			cwd=REPO_ROOT,
			check=False,
			capture_output=True,
			text=True,
			timeout=8.0,
		)
	except subprocess.TimeoutExpired as error:
		raise HTTPException(status_code=504, detail="Solver timed out.") from error
	except subprocess.CalledProcessError as error:
		raise HTTPException(status_code=500, detail=error.stderr.strip() or "Solver build failed.") from error
	if completed.returncode != 0 and not completed.stdout:
		raise HTTPException(status_code=500, detail=completed.stderr.strip() or "Solver failed.")
	return JSONResponse(parse_live_solver_output(completed.stdout))


@app.delete("/api/campaigns/{name}")
async def delete_campaign(name: str):
	path = campaign_path(name)
	if not (path / "campaign.json").exists():
		raise HTTPException(status_code=404, detail="Campaign does not exist.")
	shutil.rmtree(path)
	return {"ok": True}


@app.post("/api/runs")
async def run_campaign(request: RunCampaignRequest):
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
	suite = first_input_file(path)
	command = [
		sys.executable,
		str(BENCHMARK_CLI),
		"compare-solvers",
		"--suite", str(suite),
		"--output", str(path / "results" / "comparisons"),
		"--max-calls", request.max_calls,
		"--max-instances", str(request.max_instances) if request.max_instances is not None else "-1",
		"--max-polygons", "-1",
		"--max-branching", "-1",
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


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
	job = jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail="Unknown job.")
	refresh_job_progress_from_logs(job)
	elapsed_seconds = (job.finished_at or time.time()) - job.started_at
	return {
		"id": job.id,
		"status": job.status,
		"returncode": job.returncode,
		"started_at": job.started_at,
		"finished_at": job.finished_at,
		"elapsed_seconds": elapsed_seconds,
		"command": job.command,
		"output": job.output,
		"kind": job.kind,
		"campaign": job.campaign,
		"progress_completed": job.progress_completed,
		"progress_total": job.progress_total,
		"solver_progress_completed": job.solver_progress_completed,
		"solver_progress_total": job.solver_progress_total,
		"current_solver": job.current_solver,
	}


@app.get("/api/jobs/{job_id}/progress")
async def get_job_progress(job_id: str):
	job = jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail="Unknown job.")
	refresh_job_progress_from_logs(job)
	return job.progress_snapshot()


@app.get("/api/jobs")
async def list_jobs():
	items = sorted(jobs.values(), key=lambda item: item.started_at, reverse=True)
	return {"jobs": [job.snapshot() for job in items[:100]]}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
	job = jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail="Unknown job.")
	if job.returncode is not None:
		return {"ok": True, "status": job.status}
	job.cancel_requested = True
	if job.process is not None:
		try:
			os.killpg(job.process.pid, signal.SIGTERM)
		except ProcessLookupError:
			pass
	persist_jobs()
	return {"ok": True, "status": job.status}


@app.get("/api/system")
async def get_system():
	return {"cpu_count": os.cpu_count() or 1}


@app.get("/api/osm-files")
async def get_osm_files():
	return {"files": find_osm_files()}


@app.get("/api/campaigns/{name}/logs")
async def get_campaign_logs(name: str):
	path = campaign_path(name)
	results_dir = path / "results"
	logs = []
	log_paths = sorted(results_dir.rglob("*.log")) if results_dir.exists() else []
	for log_path in log_paths:
		text = log_path.read_text(errors="replace")
		logs.append({
			"path": str(log_path.relative_to(path)),
			"tail": text[-8000:],
		})
	return {"logs": logs}


@app.get("/api/campaigns/{name}/summaries")
async def get_campaign_summaries(name: str):
	path = campaign_path(name)
	files = []
	for summary_path in summary_files(path):
		text = summary_path.read_text(errors="replace")
		files.append({
			"path": str(summary_path.relative_to(path)),
			"mtime": summary_path.stat().st_mtime,
			"tables": parse_markdown_tables(text),
			"rows": summary_result_rows(summary_path),
		})
	return {
		"files": files,
		"tables": files[0]["tables"] if files else [],
		"input_file": campaign_input_label(path),
	}


@app.get("/api/campaigns/{name}/benchmarked-instances")
async def get_benchmarked_instances(name: str, limit: int = 200):
	return {"instances": benchmarked_instances(campaign_path(name), limit=limit)}


@app.get("/api/campaigns/{name}/comparisons")
async def get_comparisons(name: str):
	return comparison_data(campaign_path(name))


@app.get("/api/results")
async def list_results():
	roots = [RESULTS_ROOT, CAMPAIGNS_ROOT]
	files = [
		{
			"path": str(path.relative_to(REPO_ROOT)),
			"size": path.stat().st_size,
			"mtime": path.stat().st_mtime,
		}
		for root in roots
		if root.exists()
		for path in sorted(root.rglob("*"))
		if path.is_file() and path.suffix in {".csv", ".md", ".log"}
	]
	files.sort(key=lambda file: file["mtime"])
	return {"files": files[-200:]}
