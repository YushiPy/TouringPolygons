from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from dashboard.dashboard_campaigns import (
    instance_preview_list,
    preview_map,
    read_campaign_case,
    read_campaign_cases,
    total_instance_count,
)
from dashboard.dashboard_files import _json_cache, read_json
from dashboard.dashboard_models import CaseData, Point

PREVIEW_VERSION = 6

_preview_lock = threading.RLock()


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
    grid_scale = 10**exponent
    if grid_scale / 5 > decision_value:
        sub_grid_count = 3
        exponent -= 1
        multiplier = 2
    elif grid_scale / 2 > decision_value:
        exponent -= 1
        multiplier = 5
    return (10**exponent) * multiplier, sub_grid_count


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
        elements.append(
            f'<text x="{start_x + 10:.2f}" y="{start_y + 4:.2f}" font-size="12" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" fill="#f8fafc">s</text>'
        )
        elements.append(f'<circle cx="{target_x:.2f}" cy="{target_y:.2f}" r="5" fill="#ef4444"/>')
        elements.append(
            f'<text x="{target_x + 10:.2f}" y="{target_y + 4:.2f}" font-size="12" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" fill="#f8fafc">t</text>'
        )
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


def sample_cases(cases: list[CaseData], limit: int) -> list[CaseData]:
    if len(cases) <= limit:
        return cases
    return [cases[round(index * (len(cases) - 1) / (limit - 1))] for index in range(limit)]


def write_imported_previews(path: Path, cases: list[CaseData]) -> tuple[dict[str, str], list[str]]:
    preview_dir = path / "previews"
    overview_cases = sample_cases(cases, 20)
    overview_columns = 5 if len(overview_cases) <= 10 else 7
    overview_rows = (len(overview_cases) + overview_columns - 1) // overview_columns
    overview_cell_height = round((overview_columns * 150) / max(overview_rows, 1) / 2.52)
    write_case_preview(
        preview_dir / "selected.svg",
        cases[:1],
        cell_width=420,
        cell_height=320,
        columns=1,
        padding=8,
    )
    write_case_preview(
        preview_dir / "four.svg",
        cases[:4],
        cell_width=210,
        cell_height=160,
        columns=2,
        padding=6,
    )
    write_case_preview(
        preview_dir / "all.svg",
        overview_cases,
        cell_width=150,
        cell_height=overview_cell_height,
        columns=overview_columns,
        padding=6,
    )
    instance_paths = [f"previews/instances/case-{index:04}.svg" for index in range(len(cases))]
    previews = {
        "selected": "previews/selected.svg",
        "four": "previews/four.svg",
        "all": "previews/all.svg",
    }
    return previews, instance_paths


def preview_svg_is_stale(path: Path) -> bool:
    if path.suffix.lower() != ".svg" or not path.exists():
        return False
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False
    return ">case " in text or 'fill="#ffffff"' in text or f'data-preview-version="{PREVIEW_VERSION}"' not in text


def campaign_previews_are_stale(path: Path, data: dict[str, Any]) -> bool:
    candidates = list(preview_map(data).values()) + instance_preview_list(data)[:1]
    if not candidates:
        return total_instance_count(data) > 0
    return any(preview_svg_is_stale(path / preview) for preview in candidates)


def ensure_instance_preview(path: Path, data: dict[str, Any], index: int) -> Path | None:
    if index < 0:
        return None
    instance_previews = instance_preview_list(data)
    if index >= len(instance_previews) and index >= total_instance_count(data):
        return None
    preview_path = (
        path / instance_previews[index]
        if index < len(instance_previews)
        else path / "previews" / "instances" / f"case-{index:04}.svg"
    )
    if preview_path.exists() and not preview_svg_is_stale(preview_path):
        return preview_path
    case = read_campaign_case(path, data, index)
    if case is None:
        return None
    write_case_preview(preview_path, [case], cell_width=260, cell_height=180, columns=1, padding=6)
    return preview_path


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
