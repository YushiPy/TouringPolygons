from __future__ import annotations

import json
import subprocess
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class PartitionRequest(BaseModel):
	model_config = ConfigDict(allow_inf_nan=False)
	polygons: list[list[tuple[float, float]]] = Field(min_length=1, max_length=100)


ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "apps/benchmark-dashboard"
LIB = ROOT / "packages/optimal-convex-partition/cpp"
BINARY = ROOT / ".build/dashboard-partition/partition"
LOCK = Lock()


def native_partition(polygons: list[list[list[float]]]) -> list:
	from shapely.geometry import Polygon

	if not polygons or any(len(p) < 3 or not Polygon(p).is_valid for p in polygons):
		raise ValueError("A decomposição requer polígonos simples válidos.")
	sources = [APP / "scripts/partition_cli.cpp", LIB / "src/optimal_convex_partition.cpp", LIB / "include/optimal_convex_partition/optimal_convex_partition.h"]
	with LOCK:
		if not BINARY.exists() or BINARY.stat().st_mtime < max(p.stat().st_mtime for p in sources):
			BINARY.parent.mkdir(parents=True, exist_ok=True)
			subprocess.run(["c++", "-std=c++20", "-O3", "-I", str(LIB / "include"), str(sources[0]), str(sources[1]), "-o", str(BINARY)], check=True, capture_output=True, timeout=120)
	payload = "\n".join(str(len(p)) + " " + " ".join(str(v) for point in p for v in point) for p in polygons)
	result = subprocess.run([str(BINARY)], input=payload, text=True, capture_output=True, timeout=30, check=True)
	pieces = [json.loads(line) for line in result.stdout.splitlines()]
	if len(pieces) != len(polygons):
		raise ValueError("Incomplete native decomposition")
	return pieces


def partition_router() -> APIRouter:
	router = APIRouter()

	@router.post("/api/geometry/partition")
	def partition(request: PartitionRequest):
		if any(len(p) > 300 for p in request.polygons):
			raise HTTPException(422, "Maximum 300 vertices per polygon")
		try:
			return {"algorithm": "optimal_convex_partition::decompose_polygon", "pieces": native_partition(request.polygons)}
		except ValueError as error:
			raise HTTPException(422, str(error)) from error
		except (subprocess.SubprocessError, OSError) as error:
			raise HTTPException(503, "Native partition unavailable") from error

	return router
