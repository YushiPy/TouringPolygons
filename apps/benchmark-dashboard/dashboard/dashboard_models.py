from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Point = tuple[float, float]
CaseData = tuple[Point, Point, list[list[Point]]]


class CreateSyntheticRequest(BaseModel):
    name: str
    append_to: str | None = None
    vertices: str = "8"
    polygons: int = Field(default=20, ge=1, le=500)
    instances: int = Field(default=100, ge=1, le=5000)
    shape: Literal["star", "convex"] = "star"
    seed: int = 42
    no_preview: bool = False
    overwrite: bool = False


class CreateOsmRequest(BaseModel):
    name: str
    append_to: str | None = None
    pbf_path: str
    instances: int = Field(default=100, ge=1, le=5000)
    polygon_counts: int = Field(default=20, ge=1, le=500)
    sample_size: int | None = Field(default=None, ge=1)
    seed: int = 42
    simplify_tolerance: float = Field(default=1.0, ge=0.0, le=10.0)
    normalization: Literal["instance", "dataset", "none"] = "instance"
    scale: float = Field(default=1.0, ge=0.1, le=10.0)
    sampling: Literal["local", "uniform"] = "local"
    local_pool_size: int = Field(default=80, ge=1)
    layout: Literal["geographic", "grid"] = "geographic"
    grid_polygon_size: float = Field(default=1.0, ge=0.1, le=20.0)
    grid_cell_size: float = Field(default=3.0, ge=0.2, le=40.0)
    grid_columns: int | None = Field(default=None, ge=1)
    grid_placement: Literal["row-major", "random"] = "random"
    convex_replacement_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    convex_replacement_vertices: int = Field(default=64, ge=3)
    convex_replacement_position: Literal["middle", "random", "alternating"] = "middle"
    order: Literal["spatial", "left-to-right", "random", "angle"] = "spatial"
    endpoint_mode: Literal["ordered", "bbox"] = "ordered"
    no_preview: bool = False
    overwrite: bool = False


class RunCampaignRequest(BaseModel):
    visit_order: Literal["fixed", "free"] = "fixed"
    name: str
    threads: int | None = Field(default=None, ge=1, le=128)
    solver: str | None = None
    max_instances: int | None = Field(default=None, ge=1, le=5000)
    max_calls: str = "1000000"
    max_seconds: str | None = None
    timeout: int | None = Field(default=None, ge=1)
    force: bool = False
    no_build: bool = False
    dry_run: bool = False


class CompareSolversRequest(BaseModel):
    visit_order: Literal["fixed", "free"] = "fixed"
    name: str
    solvers: list[str]
    threads: int | None = Field(default=None, ge=1, le=128)
    max_instances: int | None = Field(default=None, ge=1, le=5000)
    max_calls: str = "1000000"
    max_seconds: str | None = None
    timeout: int | None = Field(default=None, ge=1)
    no_build: bool = False


class ImportCanonicalRequest(BaseModel):
    name: str = "canonical-v1"
    overwrite: bool = False


class ImportGermanRequest(BaseModel):
    name: str = "german-instances"
    overwrite: bool = False


class ManualCampaignRequest(BaseModel):
    name: str
    overwrite: bool = False


class BackgroundImage(BaseModel):
    data_url: str = Field(max_length=16_000_000)
    opacity: float = Field(default=0.45, ge=0.05, le=1.0)
    bounds: tuple[float, float, float, float]

    @field_validator("data_url")
    @classmethod
    def validate_data_url(cls, value: str) -> str:
        if not value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
            raise ValueError("Background must be a PNG, JPEG, or WebP data URL.")
        return value

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        min_x, min_y, max_x, max_y = value
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("Background bounds must have positive width and height.")
        return value


class MapView(BaseModel):
    units_per_pixel: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    latitude: float = Field(ge=-85.0, le=85.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    zoom: int = Field(default=19, ge=1, le=20)


class ManualCaseRequest(BaseModel):
    name: str | None = None
    generated: bool = False
    start: Point = (0.0, 0.0)
    target: Point = (1.0, 0.0)
    polygons: list[list[Point]] = Field(default_factory=list)
    background: BackgroundImage | None = None
    map_view: MapView | None = None


class LiveSolveRequest(ManualCaseRequest):
    visit_order: Literal["fixed", "free"] = "fixed"


class ManualCasesRequest(BaseModel):
    cases: list[ManualCaseRequest] = Field(default_factory=list)


class CampaignOrderRequest(BaseModel):
    names: list[str]


class CampaignRenameRequest(BaseModel):
    name: str


MAX_MANUAL_CASES = 2000
MAX_POLYGONS_PER_CASE = 500
MAX_VERTICES_PER_CASE = 20000


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
    phase: str = "starting"
    current_item: str | None = None
    current_item_started_at: float | None = None
    build_completed: int | None = None
    build_total: int | None = None
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "kind": self.kind,
            "campaign": self.campaign,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "output": self.output,
            "progress_completed": self.progress_completed,
            "progress_total": self.progress_total,
            "solver_progress_completed": self.solver_progress_completed,
            "solver_progress_total": self.solver_progress_total,
            "current_solver": self.current_solver,
            "phase": self.phase,
            "current_item": self.current_item,
            "current_item_started_at": self.current_item_started_at,
            "build_completed": self.build_completed,
            "build_total": self.build_total,
            "cancel_requested": self.cancel_requested,
            "status": self.status,
        }

    def progress_snapshot(self) -> dict[str, Any]:
        elapsed = max(0.0, (self.finished_at or time.time()) - self.started_at)
        item_elapsed = max(0.0, time.time() - self.current_item_started_at) if self.current_item_started_at else None
        live = ""
        if self.status in {"running", "stopping"}:
            if self.phase == "compile":
                count = f" {self.build_completed}/{self.build_total} build steps" if self.build_total else ""
                live = f"\n\nLive: compiling{count} · {elapsed:.1f}s elapsed"
            elif self.current_item:
                live = f"\n\nLive: {self.current_item} · {(item_elapsed or 0):.1f}s on current input · {elapsed:.1f}s total"
        return {
            "id": self.id,
            "kind": self.kind,
            "campaign": self.campaign,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "output": self.output + live,
            "command": self.command,
            "progress_completed": self.progress_completed,
            "progress_total": self.progress_total,
            "solver_progress_completed": self.solver_progress_completed,
            "solver_progress_total": self.solver_progress_total,
            "current_solver": self.current_solver,
            "elapsed_seconds": elapsed,
            "phase": self.phase,
            "current_item": self.current_item,
            "current_item_elapsed_seconds": item_elapsed,
            "build_completed": self.build_completed,
            "build_total": self.build_total,
        }
