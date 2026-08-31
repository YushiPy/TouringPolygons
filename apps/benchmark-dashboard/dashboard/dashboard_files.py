from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from dashboard.dashboard_binary import _binary_offset_cache

FILE_CACHE_LIMIT = 128

_json_cache: OrderedDict[Path, tuple[int, int, dict[str, Any]]] = OrderedDict()
_csv_cache: OrderedDict[tuple[Path, str], tuple[int, int, list[dict[str, str]]]] = OrderedDict()


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


def read_run_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": [], "counts": {}}

    rows = read_csv_rows(path)

    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    return {"exists": True, "rows": rows, "counts": counts}
