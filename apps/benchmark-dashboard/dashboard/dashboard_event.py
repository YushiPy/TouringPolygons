from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "static/event/siicusp34.json"
TRACE_PATH = DATA_PATH.with_name("siicusp34-traces.json")
GERMAN_DATA_PATH = Path(__file__).resolve().parents[1] / "static/event/german-instances-exact-20260918.json"
GERMAN_PARTITIONS_PATH = GERMAN_DATA_PATH.with_name("german-instances-exact-20260918-partitions.json")


@lru_cache(maxsize=1)
def event_data() -> dict:
    return json.loads(DATA_PATH.read_text())


@lru_cache(maxsize=1)
def trace_data() -> dict:
    if not TRACE_PATH.exists():
        return {"schema_version": 1, "cases": {}}
    return json.loads(TRACE_PATH.read_text())


@lru_cache(maxsize=1)
def visual_data() -> dict:
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import nearest_points

    data = json.loads(DATA_PATH.read_text())
    partitions = json.loads(DATA_PATH.with_name("siicusp34-partitions.json").read_text())["cases"]
    for row in data["rows"]:
        path = row["path"]
        lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
        total = sum(lengths)
        contacts = []
        pieces = [[piece for piece in group if Polygon(piece).area > 0] for group in partitions[str(row["case"])]]
        for vertices in row["geometry"]["polygons"]:
            polygon = Polygon(vertices)
            elapsed = 0.0
            contact = None
            for a, b, length in zip(path, path[1:], lengths):
                segment = LineString([a, b])
                intersection = segment.intersection(polygon)
                if not intersection.is_empty:
                    point = nearest_points(Point(a), intersection)[1]
                elif segment.distance(polygon) <= 1e-7:
                    point = nearest_points(segment, polygon)[0]
                else:
                    elapsed += length
                    continue
                contact = {
                    "point": [point.x, point.y],
                    "fraction": (elapsed + Point(a).distance(point)) / total if total else 0,
                }
                break
            contacts.append(contact)
        row["visualization"] = {"contacts": contacts, "decomposition": pieces}
    return data


@lru_cache(maxsize=1)
def german_visual_data() -> dict:
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import nearest_points

    data = json.loads(GERMAN_DATA_PATH.read_text())
    partitions = json.loads(GERMAN_PARTITIONS_PATH.read_text())["cases"]
    data.setdefault("corpus", "german")
    for row in data["rows"]:
        path = row["path"]
        lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
        total = sum(lengths)
        contacts = []
        for vertices in row["geometry"]["polygons"]:
            polygon = Polygon(vertices)
            elapsed = 0.0
            contact = None
            for a, b, length in zip(path, path[1:], lengths):
                segment = LineString([a, b])
                intersection = segment.intersection(polygon)
                if not intersection.is_empty:
                    point = nearest_points(Point(a), intersection)[1]
                elif segment.distance(polygon) <= 1e-7:
                    point = nearest_points(segment, polygon)[0]
                else:
                    elapsed += length
                    continue
                contact = {
                    "point": [point.x, point.y],
                    "fraction": (elapsed + Point(a).distance(point)) / total if total else 0,
                }
                break
            contacts.append(contact)
        row["visualization"] = {
            "contacts": contacts,
            "decomposition": partitions.get(str(row["case"]), [[] for _ in row["geometry"]["polygons"]]),
        }
    return data


def event_context() -> dict:
    data = visual_data()
    row = next(row for row in data["rows"] if row["case"] == 2)
    points = [point for polygon in row["geometry"]["polygons"] for point in polygon] + row["path"]
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    scale = min(760 / max(xmax - xmin, 1e-9), 400 / max(ymax - ymin, 1e-9))
    ox, oy = (840 - (xmax - xmin) * scale) / 2, (480 - (ymax - ymin) * scale) / 2

    def project(point: list[float]) -> tuple[float, float]:
        return ox + (point[0] - xmin) * scale, 480 - oy - (point[1] - ymin) * scale

    return {
        "data": data,
        "challenge": json.loads(DATA_PATH.with_name("siicusp34-challenge.json").read_text()),
        "traces": trace_data(),
        "initial": row,
        "polygons": [" ".join(f"{x},{y}" for x, y in map(project, polygon)) for polygon in row["geometry"]["polygons"]],
        "path": " ".join(f"{x},{y}" for x, y in map(project, row["path"])),
        "start": project(row["geometry"]["start"]),
        "target": project(row["geometry"]["target"]),
        "start_label": endpoint_label(row["path"], project, False),
        "target_label": endpoint_label(row["path"], project, True),
    }


def german_context() -> dict:
    data = german_visual_data()
    row = next(row for row in data["rows"] if row["case"] == 2)
    points = [point for polygon in row["geometry"]["polygons"] for point in polygon] + row["path"]
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    scale = min(760 / max(xmax - xmin, 1e-9), 400 / max(ymax - ymin, 1e-9))
    ox, oy = (840 - (xmax - xmin) * scale) / 2, (480 - (ymax - ymin) * scale) / 2

    def project(point: list[float]) -> tuple[float, float]:
        return ox + (point[0] - xmin) * scale, 480 - oy - (point[1] - ymin) * scale

    return {
        "data": data,
		"challenge": json.loads(DATA_PATH.with_name("siicusp34-challenge.json").read_text()),
        "initial": row,
        "polygons": [" ".join(f"{x},{y}" for x, y in map(project, polygon)) for polygon in row["geometry"]["polygons"]],
        "path": " ".join(f"{x},{y}" for x, y in map(project, row["path"])),
        "start": project(row["geometry"]["start"]),
        "target": project(row["geometry"]["target"]),
        "start_label": endpoint_label(row["path"], project, False),
        "target_label": endpoint_label(row["path"], project, True),
    }


def endpoint_label(
    path: list[list[float]], project: Callable[[list[float]], tuple[float, float]], target: bool
) -> tuple[float, float]:
    points = list(reversed(path)) if target else path
    a = project(points[0])
    for point in points[1:]:
        b = project(point)
        length = math.dist(a, b)
        if length > 1e-10:
            return a[0] + 22 * (a[0] - b[0]) / length, a[1] + 22 * (a[1] - b[1]) / length
    return a[0] + (22 if target else -22), a[1]


def _inline_event_assets(html: str, data_path: Path, data_href: str) -> str:
    static = DATA_PATH.parents[1]
    css = (static / "event.css").read_text()
    modules = [(static / name).read_text() for name in ("dom.js", "event-geometry.js", "event.js")]
    javascript = "\n".join(
        re.sub(r"^export ", "", re.sub(r"^import .*;\n", "", source, flags=re.M), flags=re.M) for source in modules
    )
    html = html.replace('<link rel="stylesheet" href="/static/event.css?v=20260918-3">', f"<style>{css}</style>")
    html = html.replace(
        '<script type="module" src="/static/event.js?v=20260918-3"></script>',
        f'<script type="module">{javascript}</script>',
    )
    encoded = base64.b64encode(data_path.read_bytes()).decode("ascii")
    return html.replace(f'href="{data_href}"', f'href="data:application/json;base64,{encoded}"')


def inline_event_assets(html: str) -> str:
    return _inline_event_assets(html, DATA_PATH, "/static/event/siicusp34.json")


def inline_german_assets(html: str) -> str:
    return _inline_event_assets(html, GERMAN_DATA_PATH, "/static/event/german-instances-exact-20260918.json")
