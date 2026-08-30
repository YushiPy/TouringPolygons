from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

JobMap = dict[str, Any]


def register_support_routes(
	app: FastAPI,
	*,
	jobs: JobMap,
	results_root: Path,
	campaigns_root: Path,
	repo_root: Path,
	persist_jobs: Callable[[], None],
	refresh_job_progress_from_logs: Callable[[Any], None],
	campaign_path: Callable[[str], Path],
	find_osm_files: Callable[[], list[dict[str, Any]]],
	summary_files: Callable[[Path], list[Path]],
	parse_markdown_tables: Callable[[str], list[dict[str, Any]]],
	summary_result_rows: Callable[[Path], list[dict[str, Any]]],
	campaign_input_label: Callable[[Path], str],
	benchmarked_instances: Callable[[Path], list[dict[str, Any]]],
	comparison_data: Callable[[Path], dict[str, Any]],
) -> None:
	router = APIRouter()

	@router.get("/api/jobs/{job_id}")
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

	@router.get("/api/jobs/{job_id}/progress")
	async def get_job_progress(job_id: str):
		job = jobs.get(job_id)
		if job is None:
			raise HTTPException(status_code=404, detail="Unknown job.")
		refresh_job_progress_from_logs(job)
		return job.progress_snapshot()

	@router.get("/api/jobs")
	async def list_jobs():
		items = sorted(jobs.values(), key=lambda item: item.started_at, reverse=True)
		return {"jobs": [job.snapshot() for job in items[:100]]}

	@router.post("/api/jobs/{job_id}/cancel")
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

	@router.get("/api/system")
	async def get_system():
		return {"cpu_count": os.cpu_count() or 1}

	@router.get("/api/osm-files")
	async def get_osm_files():
		return {"files": find_osm_files()}

	@router.get("/api/campaigns/{name}/logs")
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

	@router.get("/api/campaigns/{name}/summaries")
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

	@router.get("/api/campaigns/{name}/benchmarked-instances")
	async def get_benchmarked_instances(name: str, limit: int = 200):
		return {"instances": benchmarked_instances(campaign_path(name), limit=limit)}

	@router.get("/api/campaigns/{name}/comparisons")
	async def get_comparisons(name: str):
		return comparison_data(campaign_path(name))

	@router.get("/api/results")
	async def list_results():
		roots = [results_root, campaigns_root]
		files = [
			{
				"path": str(path.relative_to(repo_root)),
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

	app.include_router(router)
