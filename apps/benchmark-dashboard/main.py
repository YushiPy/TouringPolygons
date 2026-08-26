from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
BENCHMARK_CLI = REPO_ROOT / "benchmarks/tpp.py"
CONVERT_INSTANCES_SCRIPT = REPO_ROOT / "benchmarks/scripts/convert_instances.py"
CAMPAIGNS_ROOT = REPO_ROOT / "benchmarks/campaigns"
RESULTS_ROOT = REPO_ROOT / "benchmarks/results"
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
	"gurobi": "gurobi",
}
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


app = FastAPI(title="TPP Benchmark Dashboard")
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(APP_ROOT / "templates"))


class CreateSyntheticRequest(BaseModel):
	name: str
	vertices: str = "8"
	polygons: int = 20
	instances: int = 100
	shape: str = "star"
	seed: int = 42
	no_preview: bool = False
	overwrite: bool = False


class CreateOsmRequest(BaseModel):
	name: str
	pbf_path: str
	instances: int = 100
	polygon_counts: int = 20
	sample_size: int | None = None
	seed: int = 42
	simplify_tolerance: float = 1.0
	normalization: str = "instance"
	scale: float = 1.0
	sampling: str = "local"
	local_pool_size: int = 80
	layout: str = "geographic"
	grid_polygon_size: float = 1.0
	grid_cell_size: float = 3.0
	grid_columns: int | None = None
	grid_placement: str = "random"
	convex_replacement_fraction: float = 0.0
	convex_replacement_vertices: int = 64
	convex_replacement_position: str = "middle"
	order: str = "spatial"
	endpoint_mode: str = "ordered"
	no_preview: bool = False
	overwrite: bool = False


class RunCampaignRequest(BaseModel):
	name: str
	threads: int | None = None
	solver: str | None = None
	max_instances: int | None = None
	max_calls: str = "1000000"
	max_seconds: str | None = None
	timeout: int | None = None
	force: bool = False
	no_build: bool = False
	dry_run: bool = False


class CompareSolversRequest(BaseModel):
	name: str
	solvers: list[str]
	threads: int | None = None
	max_instances: int | None = None
	max_calls: str = "1000000"
	max_seconds: str | None = None
	timeout: int | None = None
	no_build: bool = False


class ImportCanonicalRequest(BaseModel):
	name: str = "canonical-v1"
	overwrite: bool = False


class ImportGermanRequest(BaseModel):
	name: str = "german-instances"
	overwrite: bool = False


@dataclass
class Job:
	id: str
	command: list[str]
	kind: str = "run"
	campaign: str | None = None
	started_at: float = field(default_factory=time.time)
	finished_at: float | None = None
	returncode: int | None = None
	output: str = ""
	progress_completed: int | None = None
	progress_total: int | None = None
	solver_progress_completed: int | None = None
	solver_progress_total: int | None = None
	current_solver: str | None = None
	cancel_requested: bool = False
	process: asyncio.subprocess.Process | None = field(default=None, repr=False)

	@property
	def status(self) -> str:
		if self.returncode is None:
			if self.cancel_requested:
				return "stopping"
			return "running"
		if self.cancel_requested:
			return "canceled"
		if self.returncode == 0:
			return "completed"
		return "failed"


jobs: dict[str, Job] = {}
PROGRESS_PATTERN = re.compile(r"cases\s+\|\s+\[[^\]]*\]\s+(\d+)\s*/\s*(\d+)")
SOLVER_SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
Point = tuple[float, float]
CaseData = tuple[Point, Point, list[list[Point]]]


def campaign_path(name: str) -> Path:
	if "/" in name or "\\" in name or name in {"", ".", ".."}:
		raise HTTPException(status_code=400, detail="Invalid campaign name.")
	return CAMPAIGNS_ROOT / name


def read_json(path: Path) -> dict[str, Any]:
	try:
		return json.loads(path.read_text())
	except FileNotFoundError as error:
		raise HTTPException(status_code=404, detail=f"Missing file: {path}") from error
	except json.JSONDecodeError as error:
		raise HTTPException(status_code=500, detail=f"Invalid JSON: {path}") from error


def read_run_index(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {"exists": False, "rows": [], "counts": {}}

	with path.open(newline="") as file:
		rows = list(csv.DictReader(file))

	counts: dict[str, int] = {}
	for row in rows:
		status = row.get("status", "unknown")
		counts[status] = counts.get(status, 0) + 1

	return {"exists": True, "rows": rows, "counts": counts}


def read_result_rows(path: Path) -> list[dict[str, str]]:
	if not path.exists():
		return []
	with path.open(newline="") as file:
		return list(csv.DictReader(file, delimiter=";"))


def parse_float(value: str | None) -> float:
	try:
		return float(value or "0")
	except ValueError:
		return 0.0


def binary_case_count(path: Path) -> int:
	import struct

	data = path.read_bytes()
	offset = 0
	count = 0
	size = struct.calcsize("<Q")
	vector_size = struct.calcsize("<dd")
	while offset < len(data):
		if offset + 2 * vector_size + size > len(data):
			break
		offset += 2 * vector_size
		polygon_count = struct.unpack_from("<Q", data, offset)[0]
		offset += size
		for _ in range(polygon_count):
			if offset + size > len(data):
				return count
			vertex_count = struct.unpack_from("<Q", data, offset)[0]
			offset += size + vertex_count * vector_size
			if offset > len(data):
				return count
		if offset + size > len(data):
			return count
		offset += size
		count += 1
	return count


def read_binary_cases(path: Path, limit: int) -> list[CaseData]:
	import struct

	data = path.read_bytes()
	offset = 0
	cases: list[CaseData] = []
	size = struct.calcsize("<Q")
	vector_size = struct.calcsize("<dd")
	while offset < len(data) and len(cases) < limit:
		if offset + 2 * vector_size + size > len(data):
			break
		start = struct.unpack_from("<dd", data, offset)
		offset += vector_size
		target = struct.unpack_from("<dd", data, offset)
		offset += vector_size
		polygon_count = struct.unpack_from("<Q", data, offset)[0]
		offset += size
		polygons: list[list[Point]] = []
		for _ in range(polygon_count):
			if offset + size > len(data):
				return cases
			vertex_count = struct.unpack_from("<Q", data, offset)[0]
			offset += size
			polygon: list[Point] = []
			for _ in range(vertex_count):
				if offset + vector_size > len(data):
					return cases
				polygon.append(struct.unpack_from("<dd", data, offset))
				offset += vector_size
			polygons.append(polygon)
		if offset + size > len(data):
			return cases
		offset += size
		cases.append((start, target, polygons))
	return cases


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


def write_case_preview(path: Path, cases: list[CaseData], *, cell_size: int, columns: int) -> None:
	if not cases:
		return
	columns = min(columns, len(cases))
	rows = (len(cases) + columns - 1) // columns
	width = columns * cell_size
	height = rows * cell_size
	padding = 18
	colors = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]
	elements = [
		f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
		'<rect width="100%" height="100%" fill="#ffffff"/>',
	]
	for case_index, case in enumerate(cases):
		start, target, polygons = case
		col = case_index % columns
		row = case_index // columns
		x0 = col * cell_size
		y0 = row * cell_size
		min_x, min_y, max_x, max_y = case_bounds(case)
		span = max(max_x - min_x, max_y - min_y, 1e-9)
		scale = (cell_size - 2 * padding) / span
		offset_x = x0 + padding - min_x * scale
		offset_y = y0 + cell_size - padding + min_y * scale
		elements.append(f'<text x="{x0 + 9}" y="{y0 + 17}" font-size="12" fill="#334155">case {case_index + 1}</text>')
		for polygon_index, polygon in enumerate(polygons):
			color = colors[polygon_index % len(colors)]
			elements.append(
				f'<polygon points="{svg_points(polygon, offset_x, offset_y, scale)}" '
				f'fill="{color}" fill-opacity="0.32" stroke="#111827" stroke-width="1"/>'
			)
		start_x = offset_x + start[0] * scale
		start_y = offset_y - start[1] * scale
		target_x = offset_x + target[0] * scale
		target_y = offset_y - target[1] * scale
		elements.append(f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="5" fill="#22c55e" stroke="#111827"/>')
		elements.append(f'<text x="{start_x + 8:.2f}" y="{start_y + 4:.2f}" font-size="13" font-weight="700" fill="#166534">s</text>')
		elements.append(f'<circle cx="{target_x:.2f}" cy="{target_y:.2f}" r="5" fill="#ef4444" stroke="#111827"/>')
		elements.append(f'<text x="{target_x + 8:.2f}" y="{target_y + 4:.2f}" font-size="13" font-weight="700" fill="#991b1b">t</text>')
	elements.append("</svg>")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(elements) + "\n")


def write_imported_previews(path: Path, cases: list[CaseData]) -> tuple[dict[str, str], list[str]]:
	preview_dir = path / "previews"
	write_case_preview(preview_dir / "selected.svg", cases[:1], cell_size=320, columns=1)
	write_case_preview(preview_dir / "four.svg", cases[:4], cell_size=160, columns=2)
	write_case_preview(preview_dir / "all.svg", cases[:100], cell_size=120, columns=10)
	instance_dir = preview_dir / "instances"
	instance_paths = []
	for index, case in enumerate(cases):
		instance_path = instance_dir / f"case-{index:04}.svg"
		write_case_preview(instance_path, [case], cell_size=260, columns=1)
		instance_paths.append(str(instance_path.relative_to(path)))
	previews = {
		"selected": "previews/selected.svg",
		"four": "previews/four.svg",
		"all": "previews/all.svg",
	}
	return previews, instance_paths


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
	with candidates[0].open(newline="") as file:
		return list(csv.DictReader(file))


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
	with candidates[0].open(newline="") as file:
		rows = list(csv.DictReader(file))
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
	previews = preview_map(data)
	if kind.startswith("instance-"):
		try:
			index = int(kind.removeprefix("instance-"))
		except ValueError as error:
			raise HTTPException(status_code=404, detail="Invalid instance preview.") from error
		instance_previews = result_preview_list(path, data)
		preview = instance_previews[index] if 0 <= index < len(instance_previews) else None
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
	return {"ok": True, "output": completed.stdout, "campaign": campaign_summary(campaign_path(request.name))}


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
	return {"ok": True, "output": completed.stdout, "campaign": campaign_summary(campaign_path(request.name))}


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


@app.delete("/api/campaigns/{name}")
async def delete_campaign(name: str):
	path = campaign_path(name)
	if not (path / "campaign.json").exists():
		raise HTTPException(status_code=404, detail="Campaign does not exist.")
	shutil.rmtree(path)
	return {"ok": True}


@app.post("/api/runs")
async def run_campaign(request: RunCampaignRequest):
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
	asyncio.create_task(run_job(job))
	return {"job": job.id, "command": command}


@app.post("/api/comparisons")
async def compare_solvers(request: CompareSolversRequest):
	if not request.solvers:
		raise HTTPException(status_code=400, detail="Select at least one solver.")
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
