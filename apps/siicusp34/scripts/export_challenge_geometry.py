#!/usr/bin/env python3
"""Read the three author-drawn SIICUSP cases and export display-scale geometry.

The C++ optimal convex partition is used for the two nonconvex challenges.
The affine transform is a uniform scale, translation, and vertical reflection;
it preserves Euclidean route comparisons.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from build_usp_demo import optimal_decomposition

REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY / "benchmarks/campaigns/SIICUSP34 - Instances/manual-cases.json"


def transform_case(case: dict, pieces: list | None) -> dict:
    points = [case["start"], case["target"], *(point for polygon in case["polygons"] for point in polygon)]
    min_x, max_x = min(point[0] for point in points), max(point[0] for point in points)
    min_y, max_y = min(point[1] for point in points), max(point[1] for point in points)
    scale = min(360 / (max_x - min_x), 200 / (max_y - min_y))
    offset_x = (420 - (max_x - min_x) * scale) / 2
    offset_y = (260 - (max_y - min_y) * scale) / 2

    def project(point: list[float]) -> list[float]:
        return [offset_x + (point[0] - min_x) * scale,
                offset_y + (max_y - point[1]) * scale]

    return {
        "geometry": {
            "start": project(case["start"]),
            "target": project(case["target"]),
            "polygons": [[project(point) for point in polygon] for polygon in case["polygons"]],
        },
        **({"pieces": [[[project(point) for point in piece] for piece in group] for group in pieces]} if pieces else {}),
    }


def main() -> None:
    source = json.loads(SOURCE.read_text())
    cases = source["cases"]
    if len(cases) != 3 or any(len(case["polygons"]) < 2 for case in cases):
        raise ValueError("Expected three manual SIICUSP cases with multiple regions")
    exported = [
        transform_case(case, None if index == 0 else optimal_decomposition(case["polygons"]))
        for index, case in enumerate(cases)
    ]
    print(json.dumps({
        "cases": exported,
        "source": "benchmarks/campaigns/SIICUSP34 - Instances/manual-cases.json",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
