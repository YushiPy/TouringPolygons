from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


def solve(start: list[int], target: list[int], polygons: list) -> dict:
	from shapely.geometry import Point, Polygon

	process = subprocess.run([main.SOLVER_BINARY], input=main.live_solver_input((start, target, polygons), 200000, 3), text=True, capture_output=True, check=True, timeout=8)
	result = main.parse_live_solver_output(process.stdout)
	path = result["path"]
	assert result["exact"] and path[0] == start and path[-1] == target
	assert len(path) == len(polygons) + 2
	assert all(Point(point).distance(Polygon(polygon)) <= 1e-7 for point, polygon in zip(path[1:-1], polygons))
	return {"path": path, "length": sum(math.dist(a, b) for a, b in zip(path, path[1:]))}


def piece_challenge() -> dict:
	start, target = [25, 200], [395, 180]
	origins = [(65, 35), (170, 140), (280, 30)]
	outline = [(0, 0), (20, 0), (20, 75), (60, 75), (60, 0), (80, 0), (80, 95), (0, 95)]
	polygons = [[[x + dx, y + dy] for dx, dy in outline] for x, y in origins]
	rectangles = [(0, 0, 20, 75), (0, 75, 80, 95), (60, 0, 80, 75)]
	pieces = [[[[x + a, y + b], [x + c, y + b], [x + c, y + d], [x + a, y + d]] for a, b, c, d in rectangles] for x, y in origins]
	solutions = [{"choices": choice, **solve(start, target, [pieces[index][piece] for index, piece in enumerate(choice)])} for choice in itertools.product(range(3), repeat=3)]
	return {"geometry": {"start": start, "target": target, "polygons": polygons}, "pieces": pieces, "solutions": solutions, "reference": min(range(27), key=lambda index: solutions[index]["length"])}


def export(destination: Path) -> None:
	from shapely.geometry import LineString, Point, Polygon

	start, target = [25, 130], [395, 130]
	boxes = [(85, 45, 135, 95), (235, 160, 285, 210), (85, 165, 135, 215), (260, 35, 310, 85)]
	polygons = [[[x, y], [xx, y], [xx, yy], [x, yy]] for x, y, xx, yy in boxes]
	main.ensure_live_solver_binary()
	solutions = []
	for order in itertools.permutations(range(4)):
		case = start, target, [polygons[index] for index in order]
		process = subprocess.run([main.SOLVER_BINARY], input=main.live_solver_input(case, 200000, 3), text=True, capture_output=True, check=True, timeout=8)
		result = main.parse_live_solver_output(process.stdout)
		path = result["path"]
		assert result["exact"] and path[0] == start and path[-1] == target
		assert all(LineString(path).distance(Polygon(polygon)) <= 1e-7 for polygon in polygons)
		assert len(path) == len(order) + 2
		assert all(Point(point).distance(Polygon(polygons[index])) <= 1e-7 for point, index in zip(path[1:-1], order))
		solutions.append({"order": order, "path": path, "length": sum(math.dist(a, b) for a, b in zip(path, path[1:]))})
	data = {"geometry": {"start": start, "target": target, "polygons": polygons}, "solutions": solutions, "reference": min(range(24), key=lambda index: solutions[index]["length"]), "provenance": "Native fixed-order solver, all 24 permutations; numerical solutions. Synthetic teaching example, separate from the research suite."}
	data["piece_challenge"] = piece_challenge()
	with destination.open("x") as output:
		json.dump(data, output, ensure_ascii=False, indent=2)
		output.write("\n")


if __name__ == "__main__":
	export(Path(sys.argv[1]))
