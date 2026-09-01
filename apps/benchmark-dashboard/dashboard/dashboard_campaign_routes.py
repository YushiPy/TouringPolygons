from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from dashboard.dashboard_models import (
    CampaignOrderRequest,
    CampaignRenameRequest,
    CreateOsmRequest,
    CreateSyntheticRequest,
    ImportCanonicalRequest,
    ImportGermanRequest,
    ManualCampaignRequest,
    ManualCaseRequest,
    ManualCasesRequest,
)


def register_campaign_routes(
    app: FastAPI,
    *,
    campaigns_root: Path,
    campaign_path: Callable[[str], Path],
    campaign_summary: Callable[[Path], dict[str, Any]],
    read_json: Callable[[Path], dict[str, Any]],
    read_run_index: Callable[[Path], dict[str, Any]],
    preview_map: Callable[[dict[str, Any]], dict[str, str]],
    refresh_stale_previews: Callable[[Path, dict[str, Any]], dict[str, Any]],
    ensure_instance_preview: Callable[[Path, dict[str, Any], int], Path | None],
    ensure_manual_binary_cache: Callable[[Path], Path],
    solution_preview_path: Callable[[Path, Path, int, str], Path | None],
    run_command: Callable[[list[str]], subprocess.CompletedProcess[str]],
    rewrite_dashboard_previews: Callable[[Path], Any],
    import_binary_suite: Callable[..., dict[str, Any]],
    create_manual_campaign_data: Callable[[Path], None],
    read_editable_case_requests: Callable[[Path], list[ManualCaseRequest]],
    manual_case_request_to_json: Callable[[ManualCaseRequest], dict[str, Any]],
    validate_manual_cases: Callable[[list[ManualCaseRequest]], None],
    rebuild_manual_campaign: Callable[..., dict[str, Any]],
    append_generated_cases_to_campaign: Callable[[str, Path, str, dict[str, Any]], dict[str, Any]],
    read_manual_cases: Callable[[Path], list[Any]],
    manual_case_from_request: Callable[[ManualCaseRequest], Any],
    manual_case_to_data: Callable[..., dict[str, Any]],
    ensure_live_solver_binary: Callable[[], None],
    live_solver_input: Callable[..., str],
    parse_live_solver_output: Callable[[str], dict[str, Any]],
    benchmark_cli: Path,
    repo_root: Path,
    convert_instances_script: Path,
    canonical_suite: Path,
    tracked_nonconvex_suite: Path,
    german_instances_zip: Path,
    solver_binary: Path,
) -> None:
    router = APIRouter()

    @router.get("/api/campaigns")
    async def list_campaigns():
        campaigns_root.mkdir(parents=True, exist_ok=True)
        paths = [
            path
            for path in sorted(campaigns_root.iterdir())
            if path.is_dir() and (path / "campaign.json").exists()
        ]
        campaigns = [campaign_summary(path) for path in paths]
        campaigns.sort(key=lambda campaign: (
            campaign["order"] is None,
            campaign["order"] if campaign["order"] is not None else 0,
            campaign["name"].lower(),
        ))
        return {"campaigns": campaigns}

    @router.put("/api/campaigns/order")
    async def reorder_campaigns(request: CampaignOrderRequest):
        campaigns_root.mkdir(parents=True, exist_ok=True)
        paths = {
            path.name: path
            for path in campaigns_root.iterdir()
            if path.is_dir() and (path / "campaign.json").exists()
        }
        if len(request.names) != len(set(request.names)) or set(request.names) != set(paths):
            raise HTTPException(status_code=400, detail="Campaign order must include every campaign exactly once.")
        for order, name in enumerate(request.names):
            path = paths[name]
            data = read_json(path / "campaign.json")
            data["display_order"] = order
            (path / "campaign.json").write_text(json.dumps(data, indent=2) + "\n")
        return {"ok": True, "campaigns": [campaign_summary(paths[name]) for name in request.names]}

    @router.get("/api/campaigns/{name}")
    async def get_campaign(name: str):
        return campaign_summary(campaign_path(name))

    @router.put("/api/campaigns/{name}/rename")
    async def rename_campaign(name: str, request: CampaignRenameRequest):
        path = campaign_path(name)
        target = campaign_path(request.name)
        if not (path / "campaign.json").exists():
            raise HTTPException(status_code=404, detail="Campaign does not exist.")
        if target != path and target.exists():
            raise HTTPException(status_code=400, detail="Campaign already exists.")
        data = read_json(path / "campaign.json")
        data["name"] = request.name
        (path / "campaign.json").write_text(json.dumps(data, indent=2) + "\n")
        if target != path:
            path.rename(target)
        return {"ok": True, "campaign": campaign_summary(target)}

    @router.delete("/api/campaigns/{name}")
    async def delete_campaign(name: str):
        path = campaign_path(name)
        if not (path / "campaign.json").exists():
            raise HTTPException(status_code=404, detail="Campaign does not exist.")
        shutil.rmtree(path)
        return {"ok": True}

    @router.get("/api/campaigns/{name}/preview")
    async def get_preview(name: str):
        return await get_preview_kind(name, "all")

    @router.get("/api/campaigns/{name}/preview/{kind}")
    async def get_preview_kind(name: str, kind: str):
        path = campaign_path(name)
        data = read_json(path / "campaign.json")
        if data.get("type") == "manual" or (path / "manual-cases.json").exists():
            ensure_manual_binary_cache(path)
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
        preview = previews.get(kind) or previews.get("all") or next(iter(previews.values()), None)
        if not preview:
            raise HTTPException(status_code=404, detail="Campaign has no preview.")
        preview_path = path / preview
        if not preview_path.exists():
            raise HTTPException(status_code=404, detail="Preview file does not exist.")
        return FileResponse(preview_path)

    @router.get("/api/campaigns/{name}/solution-preview/{case_index}")
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

    @router.post("/api/campaigns/synthetic")
    async def create_synthetic(request: CreateSyntheticRequest):
        append_to = request.append_to.strip() if request.append_to else ""
        generated_name = f"__append_synthetic_{uuid.uuid4().hex}" if append_to else request.name
        command = [
            sys.executable,
            str(benchmark_cli),
            "create",
            generated_name,
            "--vertices",
            request.vertices,
            "--polygons",
            str(request.polygons),
            "--instances",
            str(request.instances),
            "--shape",
            request.shape,
            "--seed",
            str(request.seed),
        ]
        if request.no_preview:
            command.append("--no-preview")
        if request.overwrite or append_to:
            command.append("--overwrite")
        path = campaign_path(generated_name)
        try:
            completed = run_command(command)
            if completed.returncode != 0:
                return JSONResponse({"ok": False, "output": completed.stdout}, status_code=400)
            if append_to:
                summary = append_generated_cases_to_campaign(
                    append_to,
                    path,
                    "synthetic",
                    {
                        "vertices": request.vertices,
                        "polygons": request.polygons,
                        "instances": request.instances,
                        "shape": request.shape,
                        "seed": request.seed,
                    },
                )
                return {"ok": True, "output": completed.stdout, "campaign": summary}
            if not request.no_preview:
                rewrite_dashboard_previews(path)
            return {"ok": True, "output": completed.stdout, "campaign": campaign_summary(path)}
        finally:
            if append_to and path.exists():
                shutil.rmtree(path)

    @router.post("/api/campaigns/osm")
    async def create_osm(request: CreateOsmRequest):
        append_to = request.append_to.strip() if request.append_to else ""
        generated_name = f"__append_osm_{uuid.uuid4().hex}" if append_to else request.name
        simplify_tolerance = max(0.0, min(10.0, request.simplify_tolerance))
        scale = max(0.1, min(10.0, request.scale))
        grid_polygon_size = max(0.1, min(20.0, request.grid_polygon_size))
        grid_cell_size = max(0.2, min(40.0, request.grid_cell_size))
        if grid_cell_size <= grid_polygon_size:
            grid_cell_size = min(40.0, grid_polygon_size + 0.1)
        convex_replacement_fraction = max(0.0, min(1.0, request.convex_replacement_fraction))
        command = [
            sys.executable,
            str(benchmark_cli),
            "generate-matrix",
            generated_name,
            request.pbf_path,
            "--instances",
            str(request.instances),
            "--polygon-count",
            str(request.polygon_counts),
            "--layout",
            request.layout,
            "--grid-cell-size",
            str(grid_cell_size),
            "--simplify-tolerance",
            str(simplify_tolerance),
            "--normalization",
            request.normalization,
            "--scale",
            str(scale),
            "--sampling",
            request.sampling,
            "--local-pool-size",
            str(request.local_pool_size),
            "--grid-polygon-size",
            str(grid_polygon_size),
            "--grid-placement",
            request.grid_placement,
            "--convex-replacement-fraction",
            str(convex_replacement_fraction),
            "--convex-replacement-vertices",
            str(request.convex_replacement_vertices),
            "--convex-replacement-position",
            request.convex_replacement_position,
            "--order",
            request.order,
            "--endpoint-mode",
            request.endpoint_mode,
            "--single-preview-count",
            str(request.instances),
            "--seed",
            str(request.seed),
            "--with-manifest",
        ]
        if request.grid_columns is not None:
            command.extend(["--grid-columns", str(request.grid_columns)])
        if request.sample_size is not None:
            command.extend(["--sample-size", str(request.sample_size)])
        if not request.no_preview:
            command.append("--with-preview")
        if request.overwrite or append_to:
            command.append("--overwrite")
        path = campaign_path(generated_name)
        try:
            completed = run_command(command)
            if completed.returncode != 0:
                return JSONResponse({"ok": False, "output": completed.stdout}, status_code=400)
            if append_to:
                summary = append_generated_cases_to_campaign(
                    append_to,
                    path,
                    "osm",
                    {
                        "pbf_path": request.pbf_path,
                        "instances": request.instances,
                        "polygon_counts": request.polygon_counts,
                        "sample_size": request.sample_size,
                        "seed": request.seed,
                        "layout": request.layout,
                        "sampling": request.sampling,
                        "order": request.order,
                        "endpoint_mode": request.endpoint_mode,
                    },
                )
                return {"ok": True, "output": completed.stdout, "campaign": summary}
            if not request.no_preview:
                rewrite_dashboard_previews(path)
            return {"ok": True, "output": completed.stdout, "campaign": campaign_summary(path)}
        finally:
            if append_to and path.exists():
                shutil.rmtree(path)

    @router.post("/api/campaigns/canonical")
    async def import_canonical(request: ImportCanonicalRequest):
        source_suite = canonical_suite if canonical_suite.exists() else tracked_nonconvex_suite
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

    @router.post("/api/campaigns/german")
    async def import_german(request: ImportGermanRequest):
        if german_instances_zip.exists():
            completed = run_command(
                [
                    sys.executable,
                    str(convert_instances_script),
                    "--input",
                    str(german_instances_zip),
                    "--output",
                    str(tracked_nonconvex_suite),
                ]
            )
            if completed.returncode != 0:
                return JSONResponse({"ok": False, "output": completed.stdout}, status_code=400)
        elif not tracked_nonconvex_suite.exists():
            raise HTTPException(status_code=404, detail="No German instances zip or converted nonconvex suite exists.")
        return import_binary_suite(
            name=request.name,
            source_suite=tracked_nonconvex_suite,
            input_filename="german-instances.bin",
            campaign_type="german",
            format_name="socg-simplified",
            overwrite=request.overwrite,
            extra_generation={"source_zip": str(german_instances_zip) if german_instances_zip.exists() else None},
        )

    @router.post("/api/campaigns/manual")
    async def create_manual_campaign(request: ManualCampaignRequest):
        path = campaign_path(request.name)
        if path.exists() and request.overwrite:
            shutil.rmtree(path)
        elif (path / "campaign.json").exists():
            raise HTTPException(status_code=400, detail="Campaign already exists.")
        create_manual_campaign_data(path)
        return {"ok": True, "campaign": campaign_summary(path), "cases": []}

    @router.get("/api/campaigns/{name}/cases")
    async def get_manual_cases(name: str):
        path = campaign_path(name)
        read_json(path / "campaign.json")
        return {"cases": [manual_case_request_to_json(case) for case in read_editable_case_requests(path)]}

    @router.put("/api/campaigns/{name}/cases")
    async def replace_manual_cases(name: str, request: ManualCasesRequest, refresh_previews: bool = True):
        path = campaign_path(name)
        read_json(path / "campaign.json")
        validate_manual_cases(request.cases)
        campaign = rebuild_manual_campaign(path, request.cases, rebuild_previews=refresh_previews)
        return {
            "ok": True,
            "campaign": campaign,
            "cases": [manual_case_request_to_json(case) for case in request.cases],
        }

    @router.post("/api/campaigns/{name}/cases")
    async def append_manual_case(name: str, request: ManualCaseRequest):
        path = campaign_path(name)
        cases = read_manual_cases(path)
        cases.append(manual_case_from_request(request))
        validate_manual_cases([ManualCaseRequest(**manual_case_to_data(case)) for case in cases])
        campaign = rebuild_manual_campaign(path, cases)
        return {"ok": True, "index": len(cases) - 1, "campaign": campaign}

    @router.put("/api/campaigns/{name}/cases/{case_index}")
    async def update_manual_case(name: str, case_index: int, request: ManualCaseRequest):
        path = campaign_path(name)
        cases = read_manual_cases(path)
        if case_index < 0 or case_index >= len(cases):
            raise HTTPException(status_code=404, detail="Case does not exist.")
        cases[case_index] = manual_case_from_request(request)
        validate_manual_cases([ManualCaseRequest(**manual_case_to_data(case)) for case in cases])
        campaign = rebuild_manual_campaign(path, cases)
        return {"ok": True, "campaign": campaign}

    @router.delete("/api/campaigns/{name}/cases/{case_index}")
    async def delete_manual_case(name: str, case_index: int):
        path = campaign_path(name)
        cases = read_manual_cases(path)
        if case_index < 0 or case_index >= len(cases):
            raise HTTPException(status_code=404, detail="Case does not exist.")
        del cases[case_index]
        campaign = rebuild_manual_campaign(path, cases)
        return {"ok": True, "campaign": campaign}

    @router.post("/api/editor/solve")
    async def solve_editor_case(request: ManualCaseRequest):
        validate_manual_cases([request])
        case = manual_case_from_request(request)
        if len(case[2]) == 0:
            return {"path": [list(case[0]), list(case[1])], "exact": True, "calls": 0, "seconds": 0}
        try:
            ensure_live_solver_binary()
            completed = subprocess.run(
                [solver_binary],
                input=live_solver_input(case, max_calls=200000, max_seconds=3.0),
                cwd=repo_root,
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

    app.include_router(router)
