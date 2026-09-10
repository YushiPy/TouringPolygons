from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dashboard.dashboard_models import Job

PROGRESS_PATTERN = re.compile(r"cases\s+\|\s+\[[^\]]*\]\s+(\d+)\s*/\s*(\d+)")
SOLVER_SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
INPUT_START_PATTERN = re.compile(r"^\[(\d+)/(\d+)\]\s+(?:run|skip)\s+(.+)$", re.MULTILINE)
BUILD_NINJA_PATTERN = re.compile(r"^\[(\d+)/(\d+)\]\s+", re.MULTILINE)
BUILD_PERCENT_PATTERN = re.compile(r"^\[\s*(\d+)%\]", re.MULTILINE)
FREE_INSTANCE_PATTERN = re.compile(r"^instance\s+\|\s+\[free\]\s+(\d+)\s*/\s*(\d+)\s+started$", re.MULTILINE)
JOB_OUTPUT_LIMIT = 20_000
JOB_READ_SIZE = 16_384


class JobController:
    def __init__(
        self,
        jobs: dict[str, Job],
        *,
        jobs_path: Path,
        repo_root: Path,
        campaign_path: Callable[[str], Path],
        read_run_index: Callable[[Path], dict[str, Any]],
        solvers: dict[str, str],
    ) -> None:
        self.jobs = jobs
        self.jobs_path = jobs_path
        self.repo_root = repo_root
        self.campaign_path = campaign_path
        self.read_run_index = read_run_index
        self.solvers = solvers

    def active_job(self, kind: str, campaign: str) -> Job | None:
        for job in self.jobs.values():
            if job.kind == kind and job.campaign == campaign and job.status in {"running", "stopping"}:
                return job
        return None

    def persist_jobs(self) -> None:
        self.jobs_path.write_text(json.dumps([job.snapshot() for job in self.jobs.values()], indent=2) + "\n")

    def load_jobs(self) -> None:
        if not self.jobs_path.exists():
            return
        try:
            raw_jobs = json.loads(self.jobs_path.read_text())
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
                campaign=(raw_job.get("campaign") if isinstance(raw_job.get("campaign"), str) else None),
                started_at=float(raw_job.get("started_at") or time.time()),
                finished_at=(
                    raw_job.get("finished_at") if isinstance(raw_job.get("finished_at"), float | int) else time.time()
                ),
                returncode=(raw_job.get("returncode") if isinstance(raw_job.get("returncode"), int) else 130),
                output=str(raw_job.get("output") or ""),
                progress_completed=(
                    raw_job.get("progress_completed") if isinstance(raw_job.get("progress_completed"), int) else None
                ),
                progress_total=(
                    raw_job.get("progress_total") if isinstance(raw_job.get("progress_total"), int) else None
                ),
                solver_progress_completed=(
                    raw_job.get("solver_progress_completed")
                    if isinstance(raw_job.get("solver_progress_completed"), int)
                    else None
                ),
                solver_progress_total=(
                    raw_job.get("solver_progress_total")
                    if isinstance(raw_job.get("solver_progress_total"), int)
                    else None
                ),
                current_solver=(
                    raw_job.get("current_solver") if isinstance(raw_job.get("current_solver"), str) else None
                ),
                phase=str(raw_job.get("phase") or "starting"),
                current_item=(raw_job.get("current_item") if isinstance(raw_job.get("current_item"), str) else None),
                current_item_started_at=(raw_job.get("current_item_started_at") if isinstance(raw_job.get("current_item_started_at"), int | float) else None),
                build_completed=(raw_job.get("build_completed") if isinstance(raw_job.get("build_completed"), int) else None),
                build_total=(raw_job.get("build_total") if isinstance(raw_job.get("build_total"), int) else None),
                cancel_requested=bool(raw_job.get("cancel_requested")),
            )
            if raw_job.get("returncode") is None:
                job.returncode = 130
                job.finished_at = time.time()
                job.output = (job.output + "\nServer restarted before this job finished.").strip()
            self.jobs[job.id] = job

    def refresh_job_progress_from_logs(self, job: Job) -> None:
        if job.campaign is None or job.kind != "run" or job.status != "running" or "free-order" in job.command:
            return
        path = self.campaign_path(job.campaign)
        index = self.read_run_index(path / "results/run-index.csv")
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
                self.update_job_progress(job, log_path.read_text(errors="replace")[-12000:])
                return

    def update_job_progress(self, job: Job, output: str) -> None:
        if "+ cmake --build" in output or "cmake --build" in output and job.progress_total is None:
            job.phase = "compile"
        for match in BUILD_NINJA_PATTERN.finditer(output):
            if job.phase == "compile":
                job.build_completed = int(match.group(1))
                job.build_total = int(match.group(2))
        for match in BUILD_PERCENT_PATTERN.finditer(output):
            if job.phase == "compile":
                job.build_completed = int(match.group(1))
                job.build_total = 100
        for match in INPUT_START_PATTERN.finditer(output):
            item = match.group(3).strip()
            if job.current_item != item:
                job.current_item = item
                job.current_item_started_at = time.time()
            job.phase = "benchmark"
            job.progress_completed = max(0, int(match.group(1)) - (0 if "skip" in match.group(0) else 1))
            job.progress_total = int(match.group(2))
        for match in FREE_INSTANCE_PATTERN.finditer(output):
            item = f"instance {match.group(1)}/{match.group(2)}"
            if job.current_item != item:
                job.current_item = item
                job.current_item_started_at = time.time()
            job.phase = "benchmark"
            job.progress_completed = max(0, int(match.group(1)) - 1)
            job.progress_total = int(match.group(2))
        for match in PROGRESS_PATTERN.finditer(output):
            if job.progress_completed != int(match.group(1)):
                job.current_item_started_at = time.time()
            job.phase = "benchmark"
            job.progress_completed = int(match.group(1))
            job.progress_total = int(match.group(2))
            next_instance = min(job.progress_total, job.progress_completed + 1)
            job.current_item = f"instance {next_instance}/{job.progress_total}"
        if job.kind == "comparison" and job.solver_progress_total is not None:
            known_solvers = set(self.solvers.values()) | {"unordered", "tspn"}
            solvers = [
                match.group(1).strip()
                for match in SOLVER_SECTION_PATTERN.finditer(output)
                if match.group(1).strip() in known_solvers
            ]
            if solvers:
                job.current_solver = solvers[-1]
                job.solver_progress_completed = min(job.solver_progress_total, max(0, len(solvers) - 1))

    async def run_job(self, job: Job) -> None:
        env = os.environ.copy()
        env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/touringpolygons-pycache")
        try:
            process = await asyncio.create_subprocess_exec(
                *job.command,
                cwd=self.repo_root,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            job.process = process
            if job.cancel_requested:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            assert process.stdout is not None
            while True:
                chunk = await process.stdout.read(JOB_READ_SIZE)
                if not chunk:
                    break
                text = chunk.decode(errors="replace")
                job.output = (job.output + text)[-JOB_OUTPUT_LIMIT:]
                self.update_job_progress(job, job.output)
            job.returncode = await process.wait()
            if job.returncode == 0 and job.progress_total is not None:
                job.progress_completed = job.progress_total
            if job.returncode == 0 and job.solver_progress_total is not None:
                job.solver_progress_completed = job.solver_progress_total
        except asyncio.CancelledError:
            if job.process is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(job.process.pid, signal.SIGKILL)
            job.returncode = 130
            job.cancel_requested = True
            raise
        except (OSError, RuntimeError) as error:
            job.returncode = 127
            job.output = (job.output + f"\nCould not start benchmark process: {error}").strip()[-JOB_OUTPUT_LIMIT:]
        finally:
            job.finished_at = time.time()
            job.process = None
            self.persist_jobs()
