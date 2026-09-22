"""Shared process runner for the free-order C++ solver."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Sequence

Point = Sequence[float]
Polygon = Sequence[Point]


def encode_instance(
	start: Point,
	target: Point,
	polygons: Sequence[Polygon],
	max_calls: int,
	max_seconds: float,
) -> str:
	seconds_text = "1e308" if math.isinf(max_seconds) else str(max_seconds)
	lines = [' '.join(list(map(str, (*start, *target, len(polygons), max_calls))) + [seconds_text])]
	lines.extend(f'{len(polygon)} ' + ' '.join(str(coordinate) for vertex in polygon for coordinate in vertex)
		for polygon in polygons)
	return '\n'.join(lines) + '\n'


def run_unordered_solver(
	solver: Path,
	start: Point,
	target: Point,
	polygons: Sequence[Polygon],
	max_calls: int,
	max_seconds: float,
	arguments: Sequence[str] = (),
) -> dict:
	command = [str(solver.resolve()), *arguments]
	kwargs = {
		"input": encode_instance(start, target, polygons, max_calls, max_seconds),
		"text": True,
		"capture_output": True,
	}
	if math.isfinite(max_seconds):
		kwargs["timeout"] = max(30, max_seconds + 30)
	process = subprocess.run(command, **kwargs)
	if process.returncode:
		raise RuntimeError(process.stderr.strip() or 'Free-order solver failed.')
	return json.loads(process.stdout)
