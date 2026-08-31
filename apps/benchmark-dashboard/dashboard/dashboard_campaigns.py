from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from dashboard.dashboard_binary import binary_case_count, read_binary_case, read_binary_cases
from dashboard.dashboard_files import read_json
from dashboard.dashboard_models import CaseData


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


def preview_map(data: dict[str, Any]) -> dict[str, str]:
    previews = data.get("previews")
    if isinstance(previews, dict):
        return {str(name): str(value) for name, value in previews.items() if isinstance(value, str)}
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
