#!/usr/bin/env python3
"""Build the independent, static USP teaching example from reviewed OSM outlines.

This is an app-local exporter, not a benchmark entry point. Run from the repository
root with ``--solver .build/unordered/tpp``. It never changes the 558-case corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/usp-footprints.json"
OUTPUT = ROOT / "data/usp-demo.js"
PREVIEW = ROOT / "data/usp-preview.svg"
PREVIEW_MOBILE = ROOT / "data/usp-preview-mobile.svg"
LATITUDE_ORIGIN = -23.557
LONGITUDE_ORIGIN = -46.732
MAX_CALLS = 5_000_000
MAX_SECONDS = 60


def project(lon_lat: list[float]) -> list[float]:
    """WGS84 tangent-plane linearization, in metres around the Poli/IME area."""
    lon, lat = lon_lat
    phi = math.radians(LATITUDE_ORIGIN)
    eccentricity_squared = 0.0066943799901413165
    radius = 6_378_137.0
    denominator = 1 - eccentricity_squared * math.sin(phi) ** 2
    east_radius = radius / math.sqrt(denominator)
    north_radius = radius * (1 - eccentricity_squared) / denominator**1.5
    return [
        math.radians(lon - LONGITUDE_ORIGIN) * east_radius * math.cos(phi),
        math.radians(lat - LATITUDE_ORIGIN) * north_radius,
    ]


def centroid(points: list[list[float]]) -> list[float]:
    double_area = 0.0
    cx = cy = 0.0
    for index, (x, y) in enumerate(points):
        nx, ny = points[(index + 1) % len(points)]
        cross = x * ny - nx * y
        double_area += cross
        cx += (x + nx) * cross
        cy += (y + ny) * cross
    if abs(double_area) < 1e-12:
        raise ValueError("Degenerate endpoint footprint")
    return [cx / (3 * double_area), cy / (3 * double_area)]


def cross(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(point: list[float], a: list[float], b: list[float], tolerance: float) -> bool:
    return abs(cross(a, b, point)) <= tolerance * max(1.0, math.dist(a, b)) and all(
        min(a[axis], b[axis]) - tolerance <= point[axis] <= max(a[axis], b[axis]) + tolerance
        for axis in (0, 1)
    )


def point_in_polygon(point: list[float], polygon: list[list[float]], tolerance: float) -> bool:
    inside = False
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        if on_segment(point, a, b, tolerance):
            return True
        if (a[1] > point[1]) != (b[1] > point[1]):
            x_at_y = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if point[0] < x_at_y:
                inside = not inside
    return inside


def intersection_fraction(a: list[float], b: list[float], c: list[float], d: list[float], tolerance: float) -> float | None:
    ab = [b[0] - a[0], b[1] - a[1]]
    cd = [d[0] - c[0], d[1] - c[1]]
    denominator = ab[0] * cd[1] - ab[1] * cd[0]
    if abs(denominator) <= tolerance:
        candidates = []
        length_squared = ab[0] ** 2 + ab[1] ** 2
        for point in (c, d):
            if on_segment(point, a, b, tolerance) and length_squared > 0:
                candidates.append(max(0.0, min(1.0, ((point[0] - a[0]) * ab[0] + (point[1] - a[1]) * ab[1]) / length_squared)))
        return min(candidates) if candidates else None
    offset = [c[0] - a[0], c[1] - a[1]]
    along_ab = (offset[0] * cd[1] - offset[1] * cd[0]) / denominator
    along_cd = (offset[0] * ab[1] - offset[1] * ab[0]) / denominator
    if -tolerance <= along_ab <= 1 + tolerance and -tolerance <= along_cd <= 1 + tolerance:
        return max(0.0, min(1.0, along_ab))
    return None


def first_contact(path: list[list[float]], polygon: list[list[float]], tolerance: float) -> dict | None:
    lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
    total = sum(lengths)
    traversed = 0.0
    for a, b, segment_length in zip(path, path[1:], lengths):
        entry = 0.0 if point_in_polygon(a, polygon, tolerance) else None
        for index, c in enumerate(polygon):
            candidate = intersection_fraction(a, b, c, polygon[(index + 1) % len(polygon)], tolerance)
            if candidate is not None and (entry is None or candidate < entry):
                entry = candidate
        if entry is not None:
            return {
                "point": [a[axis] + entry * (b[axis] - a[axis]) for axis in (0, 1)],
                "fraction": (traversed + entry * segment_length) / total if total else 0.0,
            }
        traversed += segment_length
    return None


def build(solver: Path) -> dict:
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    polygons = [[project(point) for point in building["lon_lat"]] for building in source["buildings"]]
    ime_polygon = polygons[0]
    ime_center = centroid(ime_polygon)
    entrance = project(source["depot"]["entrance_lon_lat"])
    if not any(on_segment(entrance, point, ime_polygon[(index + 1) % len(ime_polygon)], 1e-7) for index, point in enumerate(ime_polygon)):
        raise ValueError("The mapped entrance is not on the IME footprint")
    outward = [entrance[axis] - ime_center[axis] for axis in (0, 1)]
    direction_length = math.hypot(*outward)
    if direction_length < 1e-9:
        raise ValueError("IME entrance and centroid coincide")
    offset = source["depot"]["outside_offset_metres"]
    if offset <= 0:
        raise ValueError("The depot offset must be positive")
    start = [entrance[axis] + offset * outward[axis] / direction_length for axis in (0, 1)]
    if point_in_polygon(start, ime_polygon, 1e-7):
        raise ValueError("The depot is not outside the IME footprint")
    target = start
    lines = [f"{start[0]:.12f} {start[1]:.12f} {target[0]:.12f} {target[1]:.12f} {len(polygons)} {MAX_CALLS} {MAX_SECONDS}"]
    for polygon in polygons:
        lines.append(str(len(polygon)) + " " + " ".join(f"{x:.12f} {y:.12f}" for x, y in polygon))
    encoded_input = ("\n".join(lines) + "\n").encode()
    run = subprocess.run([str(solver.resolve())], input=encoded_input, capture_output=True, check=True, timeout=MAX_SECONDS + 30)
    result = json.loads(run.stdout)
    path = result["path"]
    lower, upper = float(result["lower_bound"]), float(result["upper_bound"])
    if result.get("termination") != "optimal" or not result.get("exact"):
        raise ValueError(f"USP example is not numerically certified: {result.get('termination')}")
    if upper - lower > 1e-7 + 1e-9 * abs(upper):
        raise ValueError("The numerical gap exceeds the documented tolerance")
    if math.dist(path[0], start) > 1e-7 or math.dist(path[-1], target) > 1e-7:
        raise ValueError("Route endpoints differ from the input")
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    if abs(length - upper) > 1e-7 + 1e-9 * abs(upper):
        raise ValueError("Route length differs from the upper bound")
    contacts = [first_contact(path, polygon, 1e-7) for polygon in polygons]
    if any(contact is None for contact in contacts):
        raise ValueError("The route misses at least one building outline")
    order = sorted(range(len(polygons)), key=lambda index: (contacts[index]["fraction"], index))
    if order != result["order"]:
        raise ValueError(f"First-contact order {order} differs from solver order {result['order']}")
    return {
        "schema_version": 1,
        "case": "usp",
        "title": "Rota entre edifícios da USP",
        "corpus": "demonstration",
        "polygons": len(polygons),
        "geometry": {"start": start, "target": target, "polygons": polygons},
        "path": path,
        "order": order,
        "length": upper,
        "lower_bound": lower,
        "upper_bound": upper,
        "exact": True,
        "termination": "optimal",
        "seconds": result["seconds"],
        "visualization": {"contacts": contacts, "decomposition": []},
        "buildings": [{"id": building["id"], "label": building["label"], "osm_way": building["osm_way"]} for building in source["buildings"]],
        "depot": {"label": source["depot"]["label"], "osm_entrance_node": source["depot"]["osm_entrance_node"], "entrance": entrance, "outside_offset_metres": offset},
        "provenance": {
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "solver_sha256": hashlib.sha256(solver.read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(encoded_input).hexdigest(),
            "projection": "WGS84 local tangent plane at 23.557 S, 46.732 W; metres",
            "absolute_gap": 1e-7,
            "relative_gap": 1e-9,
            "visit_tolerance_metres": 1e-7,
            "max_calls": MAX_CALLS,
            "max_seconds": MAX_SECONDS,
        },
    }


def write_preview(demo: dict, width: int, height: int, output: Path) -> None:
    """Keep the opening route visible while the large corpus scripts load."""
    points = [*demo["geometry"]["polygons"], demo["path"]]
    flattened = [point for group in points for point in group]
    min_x, max_x = min(point[0] for point in flattened), max(point[0] for point in flattened)
    min_y, max_y = min(point[1] for point in flattened), max(point[1] for point in flattened)
    scale = min((width - 80) / (max_x - min_x), (height - 80) / (max_y - min_y))
    offset_x = (width - (max_x - min_x) * scale) / 2
    offset_y = (height - (max_y - min_y) * scale) / 2

    def map_point(point: list[float]) -> tuple[float, float]:
        return offset_x + (point[0] - min_x) * scale, height - offset_y - (point[1] - min_y) * scale

    def path_points(path: list[list[float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in map(map_point, path))

    shapes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Rota fechada por {demo["polygons"]} edifícios da USP com partida perto da entrada do IME">',
        f'<rect width="{width}" height="{height}" fill="#102c38"/>',
    ]
    for index, polygon in enumerate(demo["geometry"]["polygons"]):
        rank = demo["order"].index(index)
        hue = round(105 + 70 * rank / (demo["polygons"] - 1))
        fill = "#b97732" if demo["buildings"][index]["id"] == "ime" else f"hsl({hue} 58% 42% / .68)"
        stroke = "#ffdcaa" if demo["buildings"][index]["id"] == "ime" else "#b6e8dc"
        shapes.append(f'<polygon points="{path_points(polygon)}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        if demo["polygons"] <= 15:
            x, y = map_point(centroid(polygon))
            shapes.append(f'<text x="{x:.2f}" y="{y:.2f}" fill="white" stroke="#102c38" stroke-width="1.4" paint-order="stroke" font-family="system-ui,sans-serif" font-weight="650" font-size="15" text-anchor="middle" dominant-baseline="central">{rank + 1}</text>')
    shapes.append(f'<polyline points="{path_points(demo["path"])}" fill="none" stroke="#ffad66" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
    x, y = map_point(demo["geometry"]["start"])
    shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#fff" stroke="#102c38" stroke-width="2"/>')
    shapes.append(f'<text x="{x + 13:.2f}" y="{y + 5:.2f}" fill="white" font-family="system-ui,sans-serif" font-weight="700" font-size="15">IME · S = T</text>')
    shapes.append("</svg>")
    output.write_text("\n".join(shapes) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    args = parser.parse_args()
    demo = build(args.solver)
    OUTPUT.write_text("window.TPPUspDemo = " + json.dumps(demo, ensure_ascii=False, separators=(",", ":")) + ";\n")
    write_preview(demo, 840, 480, PREVIEW)
    write_preview(demo, 420, 440, PREVIEW_MOBILE)
    print(f"{OUTPUT}: {demo['polygons']} regions; {demo['upper_bound']:.3f} m; certified={demo['exact']}")
