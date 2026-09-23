#!/usr/bin/env python3
"""Build the independent, static USP example from the user-drawn 50-region suite.

This is an app-local exporter, not a benchmark entry point. Run from the repository
root with ``--solver .build/unordered/tpp``. It never changes the 558-case corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
SUITE = REPOSITORY / "benchmarks/suites/usp-butanta-50"
SOURCE = SUITE / "usp-butanta-50.bin"
MAPPING = SUITE / "polygons.csv"
QGIS_GEOMETRY = SUITE / "qgis/predios.gpkg"
OUTPUT = ROOT / "data/usp-demo.js"
PREVIEW = ROOT / "data/usp-preview.svg"
PREVIEW_MOBILE = ROOT / "data/usp-preview-mobile.svg"
MAX_CALLS = 5_000_000
MAX_SECONDS = 60
PARTITION_SOURCE = REPOSITORY / "packages/optimal-convex-partition/cpp"


def google_footprint() -> tuple[int, list[list[float]]]:
    """Read the drawn Google contour from the QGIS project's GeoPackage."""
    with sqlite3.connect(QGIS_GEOMETRY.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT fid, geometry FROM predios WHERE id = 'google' AND geometry IS NOT NULL"
        ).fetchall()
    if len(rows) != 1:
        raise ValueError(f"Expected one drawn Google footprint, found {len(rows)}")
    fid, blob = rows[0]
    if blob[:2] != b"GP" or blob[2] != 0:
        raise ValueError("Invalid GeoPackage geometry header")
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    envelope = (blob[3] >> 1) & 7
    if envelope not in envelope_bytes:
        raise ValueError("Unsupported GeoPackage envelope")
    position = 8 + envelope_bytes[envelope]
    endian = "<" if blob[position] == 1 else ">" if blob[position] == 0 else None
    if endian is None or struct.unpack_from(endian + "II", blob, position + 1) != (3, 1):
        raise ValueError("Google footprint must be a simple polygon with one ring")
    vertex_count = struct.unpack_from(endian + "I", blob, position + 9)[0]
    if vertex_count < 4 or position + 13 + 16 * vertex_count != len(blob):
        raise ValueError("Invalid Google footprint vertex count")
    wgs84 = [struct.unpack_from(endian + "dd", blob, position + 13 + 16 * i)
             for i in range(vertex_count)]
    if wgs84[0] != wgs84[-1]:
        raise ValueError("Google footprint ring is not closed")
    wgs84.pop()
    latitude_origin, longitude_origin = -23.557, -46.732
    phi = math.radians(latitude_origin)
    eccentricity_squared = 0.0066943799901413165
    radius = 6_378_137.0
    denominator = 1 - eccentricity_squared * math.sin(phi) ** 2
    east_radius = radius / math.sqrt(denominator)
    north_radius = radius * (1 - eccentricity_squared) / denominator**1.5
    projected = [
        [math.radians(lon - longitude_origin) * east_radius * math.cos(phi),
         math.radians(lat - latitude_origin) * north_radius]
        for lon, lat in wgs84
    ]
    if not all(math.isfinite(value) for point in projected for value in point) or area(projected) < 1:
        raise ValueError("Google footprint is degenerate")
    return fid, projected


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


def area(points: list[list[float]]) -> float:
    return abs(sum(
        x * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * y
        for index, (x, y) in enumerate(points)
    )) / 2


def optimal_decomposition(polygons: list[list[list[float]]]) -> list[list[list[list[float]]]]:
    """Export the solver's C++ optimal convex partition, without a runtime dependency."""
    with tempfile.TemporaryDirectory(prefix="siicusp34-partition-") as directory:
        executable = Path(directory) / "partition_usp"
        compiler = os.environ.get("CXX", "c++")
        subprocess.run([
            compiler, "-std=c++20", "-O3", "-I", str(PARTITION_SOURCE / "include"),
            str(ROOT / "scripts/partition_usp.cpp"),
            str(PARTITION_SOURCE / "src/optimal_convex_partition.cpp"),
            "-o", str(executable),
        ], check=True, capture_output=True, timeout=120)
        encoded = "\n".join(
            str(len(polygon)) + " " + " ".join(f"{x:.12f} {y:.12f}" for x, y in polygon)
            for polygon in polygons
        ) + "\n"
        run = subprocess.run([str(executable)], input=encoded, text=True,
                             capture_output=True, check=True, timeout=180)
    partitions = [json.loads(line) for line in run.stdout.splitlines()]
    if len(partitions) != len(polygons):
        raise ValueError("Partition output does not match the polygon count")
    for index, (polygon, pieces) in enumerate(zip(polygons, partitions)):
        if not pieces or any(len(piece) < 3 for piece in pieces):
            raise ValueError(f"Empty convex piece in region {index}")
        for piece in pieces:
            turns = [cross(piece[vertex], piece[(vertex + 1) % len(piece)],
                           piece[(vertex + 2) % len(piece)]) for vertex in range(len(piece))]
            if min(turns) < -1e-7 and max(turns) > 1e-7:
                raise ValueError(f"Nonconvex partition piece in region {index}")
        if abs(sum(area(piece) for piece in pieces) - area(polygon)) > 1e-5:
            raise ValueError(f"Partition area differs from region {index}")
    return partitions


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
    sys.path.insert(0, str(REPOSITORY / "benchmarks/_internal"))
    from benchmark_cases import read_encoded_cases

    source_bytes = SOURCE.read_bytes()
    cases = read_encoded_cases(SOURCE)
    if len(cases) != 1 or cases[0].polygon_count != 50:
        raise ValueError("Expected one USP suite case with 50 polygons")
    case = cases[0]
    with MAPPING.open(encoding="utf-8-sig", newline="") as file:
        mapping = list(csv.DictReader(file))
    if len(mapping) != 50 or any(int(row["polygon_index"]) != index for index, row in enumerate(mapping)):
        raise ValueError("The CSV mapping does not match the 50 polygons")
    if mapping[0]["qgis_id"] != "ime-b":
        raise ValueError("The first region must be the IME entrance building")
    polygons = [[list(point) for point in polygon] for polygon in case.polygons]
    google_fid, google_polygon = google_footprint()
    sx, sy, tx, ty = struct.unpack_from("<dddd", case.data)
    start, target = [sx, sy], [tx, ty]
    if math.dist(start, target) > 1e-7:
        raise ValueError("The USP route must return to its starting point")
    if point_in_polygon(start, polygons[0], 1e-7):
        raise ValueError("The depot is not outside the IME footprint")
    position = 40 + sum(8 + 16 * len(polygon) for polygon in polygons)
    reference_count = struct.unpack_from("<Q", case.data, position)[0]
    reference_path = [struct.unpack_from("<dd", case.data, position + 8 + 16 * index) for index in range(reference_count)]
    if len(reference_path) < 2 or math.dist(reference_path[0], start) > 1e-7 or math.dist(reference_path[-1], target) > 1e-7:
        raise ValueError("The suite reference route does not start and end at the depot")
    polygons.append(google_polygon)
    decomposition = optimal_decomposition(polygons)
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
        raise ValueError("The route misses at least one USP region")
    order = sorted(range(len(polygons)), key=lambda index: (contacts[index]["fraction"], index))
    if order != result["order"]:
        raise ValueError(f"First-contact order {order} differs from solver order {result['order']}")
    # A closed Euclidean tour has the same length in either direction. Present
    # the IME near the start of the drone story rather than just before return.
    if order.index(0) > len(order) // 2:
        path = list(reversed(path))
        contacts = [first_contact(path, polygon, 1e-7) for polygon in polygons]
        if any(contact is None for contact in contacts):
            raise ValueError("Reversing the tour lost a USP region")
        order = sorted(range(len(polygons)), key=lambda index: (contacts[index]["fraction"], index))
    if order[0] != 0:
        raise ValueError("The IME must be the first visited region in the presented tour")
    return {
        "schema_version": 1,
        "case": "usp",
        "title": "Rota entre regiões da USP",
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
        "visualization": {"contacts": contacts, "decomposition": decomposition},
        "buildings": [
            {
                "id": "ime" if row["qgis_id"] == "ime-b" else row["qgis_id"],
                "qgis_id": row["qgis_id"],
                "label": row["name"],
                "name_status": row["name_status"],
                "fid": int(row["fid"]),
            }
            for row in mapping
        ] + [{"id": "google", "qgis_id": "google", "label": "Google", "name_status": "confirmed", "fid": google_fid}],
        "depot": {"label": "Entrada do IME", "outside_offset_metres": 3},
        "provenance": {
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "mapping_sha256": hashlib.sha256(MAPPING.read_bytes()).hexdigest(),
            "qgis_geo_package_sha256": hashlib.sha256(QGIS_GEOMETRY.read_bytes()).hexdigest(),
            "partition_source_sha256": hashlib.sha256((PARTITION_SOURCE / "src/optimal_convex_partition.cpp").read_bytes()).hexdigest(),
            "source": "50 regions from usp-butanta-50.bin plus the Google footprint drawn in qgis/predios.gpkg",
            "source_license": "No redistributable license declared; author authorized this event instance",
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
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Rota fechada por {demo["polygons"]} regiões da USP com partida perto da entrada do IME">',
        f'<rect width="{width}" height="{height}" fill="#102c38"/>',
    ]
    for index, polygon in enumerate(demo["geometry"]["polygons"]):
        rank = demo["order"].index(index)
        hue = round(105 + 70 * rank / (demo["polygons"] - 1))
        fill = "#b97732" if demo["buildings"][index]["qgis_id"] in {"ime-a", "ime-b", "ime-c"} else f"hsl({hue} 58% 42% / .68)"
        stroke = "#ffdcaa" if demo["buildings"][index]["qgis_id"] in {"ime-a", "ime-b", "ime-c"} else "#b6e8dc"
        shapes.append(f'<polygon points="{path_points(polygon)}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        if demo["polygons"] <= 15:
            x, y = map_point(centroid(polygon))
            shapes.append(f'<text x="{x:.2f}" y="{y:.2f}" fill="white" stroke="#102c38" stroke-width="1.4" paint-order="stroke" font-family="system-ui,sans-serif" font-weight="650" font-size="15" text-anchor="middle" dominant-baseline="central">{rank + 1}</text>')
    shapes.append(f'<polyline points="{path_points(demo["path"])}" fill="none" stroke="#ffad66" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
    x, y = map_point(demo["geometry"]["start"])
    shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#fff" stroke="#102c38" stroke-width="2"/>')
    shapes.append(f'<text x="{x + 13:.2f}" y="{y + 5:.2f}" fill="white" font-family="system-ui,sans-serif" font-weight="700" font-size="15">IME · s = t</text>')
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
