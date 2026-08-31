from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from dashboard.dashboard_files import (
    FILE_CACHE_LIMIT,
    file_signature,
    read_csv_rows,
    read_result_rows,
    read_run_index,
)

_completed_count_cache: OrderedDict[Path, tuple[tuple[tuple[str, int, int], ...], int]] = OrderedDict()


def parse_float(value: str | None) -> float:
    try:
        return float(value or "0")
    except ValueError:
        return 0.0


def completed_instance_count(path: Path, run_index: dict[str, Any] | None = None) -> int:
    index = run_index or read_run_index(path / "results/run-index.csv")
    csv_paths: list[Path] = []
    for run_row in index["rows"]:
        if run_row.get("status") != "completed":
            continue
        csv_value = run_row.get("csv_output", "")
        if not csv_value:
            continue
        csv_path = Path(csv_value)
        if not csv_path.is_absolute():
            csv_path = path / csv_path
        csv_paths.append(csv_path)

    signature_parts: list[tuple[str, int, int]] = []
    run_index_path = path / "results/run-index.csv"
    if run_index_path.exists():
        mtime, size = file_signature(run_index_path)
        signature_parts.append((str(run_index_path), mtime, size))
    for csv_path in csv_paths:
        if csv_path.exists():
            mtime, size = file_signature(csv_path)
            signature_parts.append((str(csv_path), mtime, size))
        else:
            signature_parts.append((str(csv_path), -1, -1))
    signature = tuple(signature_parts)

    cached = _completed_count_cache.get(path)
    if cached and cached[0] == signature:
        _completed_count_cache.move_to_end(path)
        return cached[1]

    completed_cases: set[tuple[str, str]] = set()
    for csv_path in csv_paths:
        for result_row in read_result_rows(csv_path):
            case_index = result_row.get("case_index")
            if case_index is not None:
                completed_cases.add((str(csv_path), case_index))
    count = len(completed_cases)
    _completed_count_cache[path] = (signature, count)
    while len(_completed_count_cache) > FILE_CACHE_LIMIT:
        _completed_count_cache.popitem(last=False)
    return count


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


def summary_result_rows(summary_path: Path) -> list[dict[str, str]]:
    return read_result_rows(summary_path.with_suffix(".csv"))


def latest_comparison_path(path: Path) -> Path | None:
    comparison_root = path / "results" / "comparisons"
    if not comparison_root.exists():
        return None
    candidates = sorted(
        comparison_root.glob("*/comparison.csv"),
        key=lambda file: file.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def comparison_rows(path: Path) -> list[dict[str, str]]:
    comparison_path = latest_comparison_path(path)
    return read_csv_rows(comparison_path) if comparison_path else []
